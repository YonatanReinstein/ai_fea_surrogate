import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_geometric.nn.models import MLP


class EdgeAttrConv(MessagePassing):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr="mean")
        self.mlp = MLP([2 * hidden_dim + edge_dim, hidden_dim, hidden_dim], norm="layer_norm")

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))


class GNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim=1, hidden_dim=128, num_layers=6,
                 use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.encoder = MLP([node_in_dim, hidden_dim, hidden_dim], norm=None)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EdgeAttrConv(hidden_dim, edge_in_dim))
        self.head = MLP([hidden_dim, hidden_dim, 1], norm=None)

        # Virtual node: one per graph, bidirectionally connected to every real
        # node, providing global context during message passing.
        self.virtual_node_emb = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.virtual_node_emb, std=0.02)
        self.virtual_edge_attr = nn.Parameter(torch.zeros(edge_in_dim))

    def forward(self, x, edge_index, edge_attr, batch,
                tile_idx=None, tile_NX=None, tile_NY=None, tile_NZ=None, tile_x=None,
                num_graphs=None, pe=None, node_id=None):
        # pe/node_id accepted for API parity with HierarchicalGNN; unused here.
        h = self.encoder(x)

        N = h.size(0)
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if N > 0 else 0
        device = h.device

        h_virtual = self.virtual_node_emb.unsqueeze(0).expand(num_graphs, -1)
        h_aug = torch.cat([h, h_virtual], dim=0)

        real_idx = torch.arange(N, device=device)
        virtual_idx = N + batch  # virtual node id for each real node
        v_to_r = torch.stack([virtual_idx, real_idx], dim=0)
        r_to_v = torch.stack([real_idx, virtual_idx], dim=0)
        virtual_edges = torch.cat([v_to_r, r_to_v], dim=1)
        edge_index_aug = torch.cat([edge_index, virtual_edges], dim=1)

        virtual_edge_attr = self.virtual_edge_attr.unsqueeze(0).expand(
            virtual_edges.size(1), -1
        )
        edge_attr_aug = torch.cat([edge_attr, virtual_edge_attr], dim=0)

        for conv in self.convs:
            if self.training and self.use_checkpoint:
                h_aug = h_aug + checkpoint(conv, h_aug, edge_index_aug, edge_attr_aug,
                                           use_reentrant=False)
            else:
                h_aug = h_aug + conv(h_aug, edge_index_aug, edge_attr_aug)

        h = h_aug[:N]
        node_pred = self.head(h)
        graph_pred = global_max_pool(node_pred, batch)
        return graph_pred, node_pred


class HierarchicalGNN(nn.Module):
    """3-level GNN: real nodes ↔ tile virtual nodes ↔ global virtual node.

    Tile nodes are connected to their assigned real nodes (via tile_idx) and
    to spatially adjacent tiles (6-connected grid from NX×NY×NZ).  The global
    node aggregates from all tile nodes, giving a clean multi-scale hierarchy.
    """

    def __init__(self, node_in_dim, edge_in_dim=1, hidden_dim=128, num_layers=6, tile_in_dim=1,
                 use_checkpoint=False, checkpoint_fine_only=False,
                 pe_dim=0, num_pos_nodes=0, pe_sign_flip=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # When True, only the encode/decode (fine-mesh) layers are checkpointed.
        # The coarse middle layers run without checkpoint because their
        # activations are tiny anyway. Requires use_checkpoint=True to have effect.
        self.checkpoint_fine_only = checkpoint_fine_only

        # --- Fixed-topology exploits (hollow_cube etc.) -------------------
        # The graph is frozen and node ordering is stable, so we can feed
        # precomputed Laplacian positional encodings (pe_dim>0) and learn a
        # per-node embedding table (num_pos_nodes>0) indexed by the stable
        # node id. Both default to off, preserving the generic GNN behaviour.
        self.pe_dim = pe_dim
        self.pe_sign_flip = pe_sign_flip
        self.node_emb = nn.Embedding(num_pos_nodes, hidden_dim) if num_pos_nodes else None
        if self.node_emb is not None:
            nn.init.normal_(self.node_emb.weight, std=0.02)

        self.encoder = MLP([node_in_dim + pe_dim, hidden_dim, hidden_dim], norm=None)
        self.convs = nn.ModuleList(
            [EdgeAttrConv(hidden_dim, edge_in_dim) for _ in range(num_layers)]
        )
        self.head = MLP([hidden_dim, hidden_dim, 1], norm=None)

        # Level 1 — tile virtual nodes
        self.tile_emb            = nn.Parameter(torch.zeros(hidden_dim))
        self.tile_input_encoder  = MLP([tile_in_dim, hidden_dim, hidden_dim], norm=None)
        self.real_tile_edge_attr = nn.Parameter(torch.zeros(edge_in_dim))
        self.tile_tile_edge_attr = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.tile_emb, std=0.02)

        # Level 2 — global virtual node
        self.global_emb             = nn.Parameter(torch.zeros(hidden_dim))
        self.tile_global_edge_attr  = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.global_emb, std=0.02)

        # Cache for local tile adjacency pairs, keyed by (NX, NY, NZ).
        # Plain dict — not serialized in state_dict, rebuilt on first forward.
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
        offsets = torch.arange(num_graphs, device=device) * K             # [G]
        src = (local[0].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1) + tile_start
        dst = (local[1].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1) + tile_start
        return torch.stack([src, dst], dim=0)

    def forward(self, x, edge_index, edge_attr, batch,
                tile_idx=None, tile_NX=None, tile_NY=None, tile_NZ=None, tile_x=None,
                num_graphs=None, pe=None, node_id=None):
        N = x.size(0)
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if N > 0 else 0
        device = x.device

        # --- Fixed-topology inputs ---
        if pe is not None and self.pe_dim:
            if self.training and self.pe_sign_flip:
                # Eigenvectors have arbitrary sign; flip per-graph so the model
                # can't latch onto a fixed sign convention.
                signs = torch.randint(0, 2, (num_graphs, pe.size(1)), device=device,
                                      dtype=pe.dtype) * 2 - 1
                pe = pe * signs[batch]
            x = torch.cat([x, pe], dim=-1)

        h = self.encoder(x)
        if self.node_emb is not None and node_id is not None:
            h = h + self.node_emb(node_id)
        real_idx = torch.arange(N, device=device)

        if tile_idx is not None:
            NX = int(tile_NX[0])
            NY = int(tile_NY[0])
            NZ = int(tile_NZ[0])
            K  = NX * NY * NZ

            tile_start  = N
            global_start = N + num_graphs * K

            # --- Build augmented node set ---
            tile_h = self.tile_emb.unsqueeze(0).expand(num_graphs * K, -1)
            if tile_x is not None:
                tile_h = tile_h + self.tile_input_encoder(tile_x.float())
            global_h = self.global_emb.unsqueeze(0).expand(num_graphs, -1)
            h_aug = torch.cat([h, tile_h, global_h], dim=0)

            # Real ↔ Tile  (each real node → its tile's virtual node)
            tile_node_abs = tile_start + batch * K + tile_idx
            rt_edges = torch.cat([
                torch.stack([real_idx, tile_node_abs], dim=0),
                torch.stack([tile_node_abs, real_idx], dim=0),
            ], dim=1)

            # Tile ↔ Tile  (6-connected spatial adjacency)
            tt_edges = self._tile_adjacency(NX, NY, NZ, num_graphs, tile_start, device)

            # Tile ↔ Global  (every tile node → its graph's global node)
            all_tile_nodes = torch.arange(num_graphs * K, device=device)
            graph_of_tile  = all_tile_nodes // K
            tg_src = tile_start   + all_tile_nodes
            tg_dst = global_start + graph_of_tile
            tg_edges = torch.cat([
                torch.stack([tg_src, tg_dst], dim=0),
                torch.stack([tg_dst, tg_src], dim=0),
            ], dim=1)

            rt_ea = self.real_tile_edge_attr.unsqueeze(0).expand(rt_edges.size(1), -1)
            tt_ea = self.tile_tile_edge_attr.unsqueeze(0).expand(tt_edges.size(1), -1)
            tg_ea = self.tile_global_edge_attr.unsqueeze(0).expand(tg_edges.size(1), -1)

            # Multi-scale schedule: first + last layer touch the fine mesh
            # (real↔real + real↔tile); middle layers operate only on the coarse
            # graph (tile↔tile + tile↔global), which has vastly fewer edges.
            # Encode: gather local geometry and push it up to tiles.
            # Process: cheap global reasoning at the coarse scale.
            # Decode: pull tile info back to real nodes + final local refinement.
            fine_ei = torch.cat([edge_index, rt_edges], dim=1)
            fine_ea = torch.cat([edge_attr, rt_ea], dim=0)

            coarse_ei = torch.cat([tt_edges, tg_edges], dim=1)
            coarse_ea = torch.cat([tt_ea, tg_ea], dim=0)

            num_layers = len(self.convs)
            for i, conv in enumerate(self.convs):
                is_fine = (i == 0 or i == num_layers - 1 or num_layers <= 2)
                if is_fine:
                    ei, ea = fine_ei, fine_ea
                else:
                    ei, ea = coarse_ei, coarse_ea

                ckpt_this = (
                    self.training
                    and self.use_checkpoint
                    and (is_fine or not self.checkpoint_fine_only)
                )
                if ckpt_this:
                    h_aug = h_aug + checkpoint(conv, h_aug, ei, ea, use_reentrant=False)
                else:
                    h_aug = h_aug + conv(h_aug, ei, ea)

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
            tg_ea = self.tile_global_edge_attr.unsqueeze(0).expand(virt_edges.size(1), -1)
            edge_attr_aug = torch.cat([edge_attr, tg_ea], dim=0)

            for conv in self.convs:
                if self.training and self.use_checkpoint:
                    h_aug = h_aug + checkpoint(conv, h_aug, edge_index_aug, edge_attr_aug,
                                               use_reentrant=False)
                else:
                    h_aug = h_aug + conv(h_aug, edge_index_aug, edge_attr_aug)

        h = h_aug[:N]
        node_pred = self.head(h)
        graph_pred = global_max_pool(node_pred, batch)
        return graph_pred, node_pred
