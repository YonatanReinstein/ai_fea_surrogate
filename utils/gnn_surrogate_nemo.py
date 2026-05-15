"""PhysicsNeMo-based variant of utils/gnn_surrogate.py.

Same architecture as the original (single global virtual node for `GNN`;
real <-> tile <-> global hierarchy for `HierarchicalGNN`), but the model is
rebuilt on NVIDIA PhysicsNeMo components:

  * `physicsnemo.Module`        -> base class (registration + save/load + AMP metadata)
  * `MeshGraphMLP`              -> every MLP (node encoder, edge encoder, message
                                   MLPs, decoder head, tile input encoder)
  * MeshGraphNet message block  -> replaces the old static-edge `EdgeAttrConv`:
                                   edge features are now *persistent and updated
                                   every layer*, with residual connections on
                                   both the edge and node updates.

The dynamic graph construction (virtual nodes, 6-connected tile adjacency,
fine/coarse multi-scale schedule) is kept identical to the original and still
runs on plain `edge_index` tensors -- PhysicsNeMo's `MeshEdgeBlock`/`MeshNodeBlock`
require a DGL/CuGraph graph object, which does not fit per-forward virtual-node
construction, so the scatter is done with `index_add` here.

Requires `nvidia-physicsnemo` to be installed. Import paths below follow the
post-rename PhysicsNeMo layout; on older Modulus installs replace `physicsnemo`
with `modulus`.

Drop-in: exposes `GNN` and `HierarchicalGNN` with the same constructor and
forward signatures as utils/gnn_surrogate.py, so training/gnn_training.py only
needs its import line changed.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import global_max_pool

from physicsnemo.models.module import Module as PhysicsNeMoModule
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.gnn_layers import MeshGraphMLP


@dataclass
class _SurrogateMeta(ModelMetaData):
    name: str = "HierarchicalMeshGraphNet"
    amp_cpu: bool = True
    amp_gpu: bool = True


def _mlp(in_dim, out_dim, hidden_dim, norm=True):
    """MeshGraphMLP wrapper. norm=True applies output LayerNorm (PhysicsNeMo default)."""
    return MeshGraphMLP(
        input_dim=in_dim,
        output_dim=out_dim,
        hidden_dim=hidden_dim,
        hidden_layers=1,
        norm_type="LayerNorm" if norm else None,
    )


class MeshGraphNetBlock(nn.Module):
    """MeshGraphNet-style edge + node update block built from `MeshGraphMLP`.

    Unlike the original `EdgeAttrConv`, edge features are persistent state:
    each layer updates them with a learned edge MLP and a residual. The node
    update aggregates the *updated* edge features (sum) and applies its own
    residual MLP -- this is the encode-process-decode block of MeshGraphNet.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # edge update: [edge, src_node, dst_node] -> edge
        self.edge_mlp = _mlp(3 * hidden_dim, hidden_dim, hidden_dim, norm=True)
        # node update: [node, aggregated_edges] -> node
        self.node_mlp = _mlp(2 * hidden_dim, hidden_dim, hidden_dim, norm=True)

    def forward(self, h, edge_index, e):
        src, dst = edge_index[0], edge_index[1]
        e_upd = e + self.edge_mlp(torch.cat([e, h[src], h[dst]], dim=-1))
        agg = torch.zeros_like(h).index_add(0, dst, e_upd)
        h_upd = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h_upd, e_upd


class GNN(PhysicsNeMoModule):
    """Single global virtual node, PhysicsNeMo build of the original `GNN`."""

    def __init__(self, node_in_dim, edge_in_dim=1, hidden_dim=128, num_layers=6,
                 use_checkpoint=False):
        super().__init__(meta=_SurrogateMeta())
        self.use_checkpoint = use_checkpoint

        self.encoder = _mlp(node_in_dim, hidden_dim, hidden_dim, norm=False)
        self.edge_encoder = _mlp(edge_in_dim, hidden_dim, hidden_dim, norm=False)
        self.convs = nn.ModuleList(MeshGraphNetBlock(hidden_dim) for _ in range(num_layers))
        self.head = _mlp(hidden_dim, 1, hidden_dim, norm=False)

        self.virtual_node_emb = nn.Parameter(torch.zeros(hidden_dim))
        self.virtual_edge_emb = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.virtual_node_emb, std=0.02)
        nn.init.normal_(self.virtual_edge_emb, std=0.02)

    def forward(self, x, edge_index, edge_attr, batch,
                tile_idx=None, tile_NX=None, tile_NY=None, tile_NZ=None, tile_x=None,
                num_graphs=None):
        h = self.encoder(x)
        N = h.size(0)
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if N > 0 else 0
        device = h.device

        h_virtual = self.virtual_node_emb.unsqueeze(0).expand(num_graphs, -1)
        h_aug = torch.cat([h, h_virtual], dim=0)

        real_idx = torch.arange(N, device=device)
        virtual_idx = N + batch
        v_to_r = torch.stack([virtual_idx, real_idx], dim=0)
        r_to_v = torch.stack([real_idx, virtual_idx], dim=0)
        virtual_edges = torch.cat([v_to_r, r_to_v], dim=1)
        edge_index_aug = torch.cat([edge_index, virtual_edges], dim=1)

        e_real = self.edge_encoder(edge_attr)
        e_virtual = self.virtual_edge_emb.unsqueeze(0).expand(virtual_edges.size(1), -1)
        e = torch.cat([e_real, e_virtual], dim=0)

        for conv in self.convs:
            if self.training and self.use_checkpoint:
                h_aug, e = checkpoint(conv, h_aug, edge_index_aug, e, use_reentrant=False)
            else:
                h_aug, e = conv(h_aug, edge_index_aug, e)

        h = h_aug[:N]
        node_pred = self.head(h)
        graph_pred = global_max_pool(node_pred, batch)
        return graph_pred, node_pred


class HierarchicalGNN(PhysicsNeMoModule):
    """3-level GNN: real nodes <-> tile virtual nodes <-> global virtual node.

    PhysicsNeMo build of the original `HierarchicalGNN`. Graph construction and
    the fine/coarse multi-scale layer schedule are unchanged; only the MLPs and
    the message block come from PhysicsNeMo.
    """

    def __init__(self, node_in_dim, edge_in_dim=1, hidden_dim=128, num_layers=6,
                 tile_in_dim=1, use_checkpoint=False, checkpoint_fine_only=False):
        super().__init__(meta=_SurrogateMeta())
        self.use_checkpoint = use_checkpoint
        self.checkpoint_fine_only = checkpoint_fine_only

        self.encoder = _mlp(node_in_dim, hidden_dim, hidden_dim, norm=False)
        self.edge_encoder = _mlp(edge_in_dim, hidden_dim, hidden_dim, norm=False)
        self.convs = nn.ModuleList(MeshGraphNetBlock(hidden_dim) for _ in range(num_layers))
        self.head = _mlp(hidden_dim, 1, hidden_dim, norm=False)

        # Level 1 -- tile virtual nodes
        self.tile_emb = nn.Parameter(torch.zeros(hidden_dim))
        self.tile_input_encoder = _mlp(tile_in_dim, hidden_dim, hidden_dim, norm=False)
        self.real_tile_edge_emb = nn.Parameter(torch.zeros(hidden_dim))
        self.tile_tile_edge_emb = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.tile_emb, std=0.02)
        nn.init.normal_(self.real_tile_edge_emb, std=0.02)
        nn.init.normal_(self.tile_tile_edge_emb, std=0.02)

        # Level 2 -- global virtual node
        self.global_emb = nn.Parameter(torch.zeros(hidden_dim))
        self.tile_global_edge_emb = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.global_emb, std=0.02)
        nn.init.normal_(self.tile_global_edge_emb, std=0.02)

        # Cache for local tile adjacency pairs, keyed by (NX, NY, NZ, device).
        self._tile_adj_cache: dict = {}

    @staticmethod
    def _build_local_tile_adj(NX, NY, NZ):
        """Build local (intra-graph) 6-connected tile pairs. Called once per grid shape."""
        pairs = []
        for ix in range(NX):
            for iy in range(NY):
                for iz in range(NZ):
                    flat = ix * NY * NZ + iy * NZ + iz
                    for dix, diy, diz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        jx, jy, jz = ix + dix, iy + diy, iz + diz
                        if jx < NX and jy < NY and jz < NZ:
                            nb = jx * NY * NZ + jy * NZ + jz
                            pairs += [(flat, nb), (nb, flat)]
        if not pairs:
            return None
        return torch.tensor(pairs, dtype=torch.long).t()  # [2, E_local]

    def _tile_adjacency(self, NX, NY, NZ, num_graphs, tile_start, device):
        """Bidirectional 6-connected grid edges between tile virtual nodes."""
        key = (NX, NY, NZ, device)
        if key not in self._tile_adj_cache:
            local = self._build_local_tile_adj(NX, NY, NZ)
            self._tile_adj_cache[key] = local.to(device) if local is not None else None
        local = self._tile_adj_cache[key]

        if local is None:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        K = NX * NY * NZ
        offsets = torch.arange(num_graphs, device=device) * K
        src = (local[0].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1) + tile_start
        dst = (local[1].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1) + tile_start
        return torch.stack([src, dst], dim=0)

    def _run_layer(self, conv, h_aug, ei, e, is_fine):
        ckpt_this = (
            self.training
            and self.use_checkpoint
            and (is_fine or not self.checkpoint_fine_only)
        )
        if ckpt_this:
            return checkpoint(conv, h_aug, ei, e, use_reentrant=False)
        return conv(h_aug, ei, e)

    def forward(self, x, edge_index, edge_attr, batch,
                tile_idx=None, tile_NX=None, tile_NY=None, tile_NZ=None, tile_x=None,
                num_graphs=None):
        h = self.encoder(x)
        N = h.size(0)
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if N > 0 else 0
        device = h.device
        real_idx = torch.arange(N, device=device)
        e_real = self.edge_encoder(edge_attr)

        if tile_idx is not None:
            NX = int(tile_NX[0])
            NY = int(tile_NY[0])
            NZ = int(tile_NZ[0])
            K = NX * NY * NZ

            tile_start = N
            global_start = N + num_graphs * K

            # --- Build augmented node set ---
            tile_h = self.tile_emb.unsqueeze(0).expand(num_graphs * K, -1)
            if tile_x is not None:
                tile_h = tile_h + self.tile_input_encoder(tile_x.float())
            global_h = self.global_emb.unsqueeze(0).expand(num_graphs, -1)
            h_aug = torch.cat([h, tile_h, global_h], dim=0)

            # Real <-> Tile  (each real node -> its tile's virtual node)
            tile_node_abs = tile_start + batch * K + tile_idx
            rt_edges = torch.cat([
                torch.stack([real_idx, tile_node_abs], dim=0),
                torch.stack([tile_node_abs, real_idx], dim=0),
            ], dim=1)

            # Tile <-> Tile  (6-connected spatial adjacency)
            tt_edges = self._tile_adjacency(NX, NY, NZ, num_graphs, tile_start, device)

            # Tile <-> Global  (every tile node -> its graph's global node)
            all_tile_nodes = torch.arange(num_graphs * K, device=device)
            graph_of_tile = all_tile_nodes // K
            tg_src = tile_start + all_tile_nodes
            tg_dst = global_start + graph_of_tile
            tg_edges = torch.cat([
                torch.stack([tg_src, tg_dst], dim=0),
                torch.stack([tg_dst, tg_src], dim=0),
            ], dim=1)

            rt_e = self.real_tile_edge_emb.unsqueeze(0).expand(rt_edges.size(1), -1)
            tt_e = self.tile_tile_edge_emb.unsqueeze(0).expand(tt_edges.size(1), -1)
            tg_e = self.tile_global_edge_emb.unsqueeze(0).expand(tg_edges.size(1), -1)

            # Multi-scale schedule: first + last layer touch the fine mesh
            # (real<->real + real<->tile); middle layers operate only on the
            # coarse graph (tile<->tile + tile<->global). Edge features are
            # persistent per edge set: fine_e evolves on the fine layers,
            # coarse_e on the coarse layers.
            fine_ei = torch.cat([edge_index, rt_edges], dim=1)
            fine_e = torch.cat([e_real, rt_e], dim=0)

            coarse_ei = torch.cat([tt_edges, tg_edges], dim=1)
            coarse_e = torch.cat([tt_e, tg_e], dim=0)

            num_layers = len(self.convs)
            for i, conv in enumerate(self.convs):
                is_fine = (i == 0 or i == num_layers - 1 or num_layers <= 2)
                if is_fine:
                    h_aug, fine_e = self._run_layer(conv, h_aug, fine_ei, fine_e, True)
                else:
                    h_aug, coarse_e = self._run_layer(conv, h_aug, coarse_ei, coarse_e, False)

        else:
            # Fallback: single global virtual node (original GNN behaviour)
            global_start = N
            global_h = self.global_emb.unsqueeze(0).expand(num_graphs, -1)
            h_aug = torch.cat([h, global_h], dim=0)

            global_node_abs = global_start + batch
            virt_edges = torch.cat([
                torch.stack([real_idx, global_node_abs], dim=0),
                torch.stack([global_node_abs, real_idx], dim=0),
            ], dim=1)
            edge_index_aug = torch.cat([edge_index, virt_edges], dim=1)
            tg_e = self.tile_global_edge_emb.unsqueeze(0).expand(virt_edges.size(1), -1)
            e = torch.cat([e_real, tg_e], dim=0)

            for conv in self.convs:
                h_aug, e = self._run_layer(conv, h_aug, edge_index_aug, e, True)

        h = h_aug[:N]
        node_pred = self.head(h)
        graph_pred = global_max_pool(node_pred, batch)
        return graph_pred, node_pred
