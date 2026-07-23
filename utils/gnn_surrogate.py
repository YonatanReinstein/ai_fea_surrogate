import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.utils import scatter


def ks_pool(node_pred, batch, num_graphs, node_target_mean, node_target_std, rho_mpa):
    """KS (Kreisselmeier-Steinhauser) smooth-max pool over per-graph node
    predictions, matching core/mesh.py:get_aggregated_stress.

    node_pred lives in node-normalized space, but unlike a hard max, KS
    aggregation is nonlinear and does not commute with that normalization --
    pooling it directly would apply the wrong effective smoothing bandwidth
    (off by a factor of node_target_std). So: denormalize to raw MPa, KS
    aggregate with rho_mpa (the same sharpness the dataset's ks_stress
    target was built with), then renormalize the scalar result back to
    node-normalized space, so downstream denorm logic (gnn_training.py,
    gnn_evaluator.py) is unaffected -- it just sees a different graph_pred.
    """
    raw = node_pred.squeeze(-1) * node_target_std.squeeze() + node_target_mean.squeeze()
    per_graph_max = scatter(raw, batch, dim=0, dim_size=num_graphs, reduce="max")
    shifted = raw - per_graph_max[batch]
    sum_exp = scatter(torch.exp(rho_mpa * shifted), batch, dim=0, dim_size=num_graphs, reduce="sum")
    ks_raw = per_graph_max + torch.log(sum_exp) / rho_mpa
    return ((ks_raw - node_target_mean.squeeze()) / node_target_std.squeeze()).unsqueeze(-1)


class EdgeAttrConv(MessagePassing):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr="mean")
        self.mlp = MLP([2 * hidden_dim + edge_dim, hidden_dim, hidden_dim], norm="layer_norm")

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))


class BipartiteEdgeAttrConv(MessagePassing):
    """EdgeAttrConv variant for edges crossing two differently-sized node
    spaces (e.g. the fine/coarse boundary of a hierarchical GNN). Source
    features are projected to the target width before the message MLP, so
    the MLP itself -- and the target update -- live entirely at dst_dim;
    src_dim is never upsampled beyond this one projection."""
    def __init__(self, src_dim, dst_dim, edge_dim):
        super().__init__(aggr="mean")
        self.src_proj = nn.Linear(src_dim, dst_dim)
        self.mlp = MLP([2 * dst_dim + edge_dim, dst_dim, dst_dim], norm="layer_norm")

    def forward(self, x_src, x_dst, edge_index, edge_attr):
        return self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr,
                               size=(x_src.size(0), x_dst.size(0)))

    def message(self, x_i, x_j, edge_attr):
        x_j = self.src_proj(x_j)
        return self.mlp(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))


class GNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim=1, hidden_dim=128, num_layers=6,
                 use_checkpoint=False, node_target_mean=0.0, node_target_std=1.0,
                 stress_ks_rho_mpa=0.15):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.stress_ks_rho_mpa = stress_ks_rho_mpa
        self.register_buffer("node_target_mean", torch.as_tensor(node_target_mean, dtype=torch.float), persistent=False)
        self.register_buffer("node_target_std", torch.as_tensor(node_target_std, dtype=torch.float), persistent=False)

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
                num_graphs=None, node_id=None):
        # node_id accepted for API parity with HierarchicalGNN; unused here.
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
        graph_pred = ks_pool(node_pred, batch, num_graphs, self.node_target_mean,
                              self.node_target_std, self.stress_ks_rho_mpa)
        return graph_pred, node_pred


class HierarchicalGNN(nn.Module):
    """3-level GNN: real nodes ↔ tile virtual nodes ↔ global virtual node.

    Tile nodes are connected to their assigned real nodes (via tile_idx) and
    to spatially adjacent tiles (6-connected grid from NX×NY×NZ).  The global
    node aggregates from all tile nodes, giving a clean multi-scale hierarchy.
    """

    def __init__(self, node_in_dim, edge_in_dim=1, hidden_dim=128, num_layers=6, tile_in_dim=1,
                 use_checkpoint=False, checkpoint_fine_only=False,
                 num_pos_nodes=0, transformer_heads=4, transformer_ff_mult=2,
                 transformer_dropout=0.1, node_target_mean=0.0, node_target_std=1.0,
                 stress_ks_rho_mpa=0.15):
        super().__init__()
        assert num_layers >= 2, "need >=2 layers: fine encode + fine decode around the coarse transformer"
        self.use_checkpoint = use_checkpoint
        # When True, only the encode/decode (fine-mesh) layers are checkpointed.
        # The coarse transformer stage runs without checkpoint because its
        # activations are tiny anyway. Requires use_checkpoint=True to have effect.
        self.checkpoint_fine_only = checkpoint_fine_only
        self.stress_ks_rho_mpa = stress_ks_rho_mpa
        self.register_buffer("node_target_mean", torch.as_tensor(node_target_mean, dtype=torch.float), persistent=False)
        self.register_buffer("node_target_std", torch.as_tensor(node_target_std, dtype=torch.float), persistent=False)

        # --- Fixed-topology exploit (hollow_cube etc.) ---------------------
        # The graph is frozen and node ordering is stable, so we can learn a
        # per-node embedding table (num_pos_nodes>0) indexed by the stable
        # node id. Defaults to off, preserving the generic GNN behaviour.
        self.node_emb = nn.Embedding(num_pos_nodes, hidden_dim) if num_pos_nodes else None
        if self.node_emb is not None:
            nn.init.normal_(self.node_emb.weight, std=0.02)

        self.encoder = MLP([node_in_dim, hidden_dim, hidden_dim], norm=None)
        self.head = MLP([hidden_dim, hidden_dim, 1], norm=None)

        # Fine stage: two local convs (encode real->tile, decode tile->real).
        self.convs = nn.ModuleList(
            [EdgeAttrConv(hidden_dim, edge_in_dim) for _ in range(2)]
        )

        # Coarse stage: full self-attention over the tile + global tokens.
        # The tile grid is small (tens to low-thousands of tokens), so full
        # attention is cheap and gives every tile a direct path to every
        # other tile in one layer, instead of relying on many 6-connected
        # message-passing hops to cover the whole grid.
        num_coarse_layers = max(num_layers - 2, 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=transformer_heads,
            dim_feedforward=hidden_dim * transformer_ff_mult,
            dropout=transformer_dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.coarse_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_coarse_layers)
        # Attention has no notion of the 6-connected grid the local convs
        # relied on, so this tells tiles apart spatially: grid coords
        # (ix, iy, iz) normalized to [0, 1], pushed through a small MLP.
        self.tile_pos_encoder = MLP([3, hidden_dim, hidden_dim], norm=None)

        # Level 1 — tile virtual nodes
        self.tile_emb            = nn.Parameter(torch.zeros(hidden_dim))
        self.tile_input_encoder  = MLP([tile_in_dim, hidden_dim, hidden_dim], norm=None)
        self.real_tile_edge_attr = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.tile_emb, std=0.02)

        # Level 2 — global virtual node
        self.global_emb             = nn.Parameter(torch.zeros(hidden_dim))
        self.tile_global_edge_attr  = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.global_emb, std=0.02)

        # Tile grid positions, fixed for the lifetime of a training run (grid
        # shape never changes for a given geometry). Populated once by
        # set_tile_grid() -- NOT lazily inside forward() via a dict keyed by
        # (NX, NY, NZ, device): torch.compile/dynamo re-guards a plain dict's
        # contents by object identity on every trace and eventually hits its
        # recompile cache limit, silently falling back to eager mid-training
        # (see the identical fix in utils/gnn_hierarchical_mp.py).
        self.register_buffer("tile_pos_raw", None, persistent=False)
        self._tile_grid = None  # (NX, NY, NZ) tile_pos_raw was built for

    @staticmethod
    def _build_tile_positions(NX, NY, NZ, device):
        """Normalized (ix, iy, iz) grid coords, one row per tile, in the same
        flat order as tile_idx (ix*NY*NZ + iy*NZ + iz)."""
        ix = torch.arange(NX, device=device).view(NX, 1, 1).expand(NX, NY, NZ)
        iy = torch.arange(NY, device=device).view(1, NY, 1).expand(NX, NY, NZ)
        iz = torch.arange(NZ, device=device).view(1, 1, NZ).expand(NX, NY, NZ)
        pos = torch.stack([ix, iy, iz], dim=-1).reshape(-1, 3).float()
        norm = torch.tensor(
            [max(NX - 1, 1), max(NY - 1, 1), max(NZ - 1, 1)], device=device, dtype=torch.float
        )
        return pos / norm

    def set_tile_grid(self, NX, NY, NZ):
        """Precompute the fixed tile position encoding for this geometry's
        grid. Call once, right after construction."""
        self.tile_pos_raw = self._build_tile_positions(NX, NY, NZ, self.tile_emb.device)
        self._tile_grid = (NX, NY, NZ)

    def _tile_positions(self, NX, NY, NZ, device):
        assert self._tile_grid == (NX, NY, NZ), (
            f"set_tile_grid{self._tile_grid} was called, but forward() saw "
            f"grid {(NX, NY, NZ)} -- fixed-topology assumption violated."
        )
        return self.tile_pos_raw

    def forward(self, x, edge_index, edge_attr, batch,
                tile_idx=None, tile_NX=None, tile_NY=None, tile_NZ=None, tile_x=None,
                num_graphs=None, node_id=None):
        N = x.size(0)
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if N > 0 else 0
        device = x.device

        h = self.encoder(x)
        if self.node_emb is not None and node_id is not None:
            h = h + self.node_emb(node_id)
        real_idx = torch.arange(N, device=device)

        if tile_idx is not None:
            # tile_NX/NY/NZ arrive as plain Python ints (extracted once in the
            # caller, outside any compiled region) rather than GPU tensors --
            # calling int(tensor[0]) here was a torch.compile graph break that
            # triggered a recompilation storm, since these never actually vary
            # across samples of a fixed-topology geometry.
            NX, NY, NZ = tile_NX, tile_NY, tile_NZ
            K  = NX * NY * NZ

            tile_start  = N
            global_start = N + num_graphs * K

            # --- Build augmented node set ---
            pos_h = self.tile_pos_encoder(self._tile_positions(NX, NY, NZ, device))  # [K, H]
            tile_h = self.tile_emb.unsqueeze(0).expand(num_graphs * K, -1) + pos_h.repeat(num_graphs, 1)
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
            rt_ea = self.real_tile_edge_attr.unsqueeze(0).expand(rt_edges.size(1), -1)

            # Encode/decode schedule: the fine convs touch the real mesh
            # (real↔real + real↔tile); the coarse transformer in between
            # attends over every tile + the global token in one shot.
            # Encode: gather local geometry and push it up to tiles.
            # Process: full attention across all tiles — cheap global reasoning.
            # Decode: pull tile info back to real nodes + final local refinement.
            fine_ei = torch.cat([edge_index, rt_edges], dim=1)
            fine_ea = torch.cat([edge_attr, rt_ea], dim=0)

            conv_encode, conv_decode = self.convs

            ckpt_fine = self.training and self.use_checkpoint
            if ckpt_fine:
                h_aug = h_aug + checkpoint(conv_encode, h_aug, fine_ei, fine_ea, use_reentrant=False)
            else:
                h_aug = h_aug + conv_encode(h_aug, fine_ei, fine_ea)

            tile_tok = h_aug[tile_start:tile_start + num_graphs * K].view(num_graphs, K, -1)
            global_tok = h_aug[global_start:global_start + num_graphs].unsqueeze(1)
            tokens = torch.cat([tile_tok, global_tok], dim=1)  # [G, K+1, H]

            ckpt_coarse = self.training and self.use_checkpoint and not self.checkpoint_fine_only
            if ckpt_coarse:
                tokens = checkpoint(self.coarse_transformer, tokens, use_reentrant=False)
            else:
                tokens = self.coarse_transformer(tokens)

            tile_new   = tokens[:, :K, :].reshape(num_graphs * K, -1)
            global_new = tokens[:, K, :]
            h_aug = torch.cat([h_aug[:tile_start], tile_new, global_new], dim=0)

            if ckpt_fine:
                h_aug = h_aug + checkpoint(conv_decode, h_aug, fine_ei, fine_ea, use_reentrant=False)
            else:
                h_aug = h_aug + conv_decode(h_aug, fine_ei, fine_ea)

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
        graph_pred = ks_pool(node_pred, batch, num_graphs, self.node_target_mean,
                              self.node_target_std, self.stress_ks_rho_mpa)
        return graph_pred, node_pred
