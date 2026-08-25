import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.models import MLP

from utils.gnn_surrogate import EdgeAttrConv, BipartiteEdgeAttrConv, ks_pool


class HierarchicalGNNMP(nn.Module):
    """3-level GNN: real nodes ↔ tile virtual nodes ↔ global virtual node.

    Tile nodes are connected to their assigned real nodes (via tile_idx) and
    to spatially adjacent tiles (6-connected grid from NX×NY×NZ).  The global
    node aggregates from all tile nodes, giving a clean multi-scale hierarchy.

    Coarse stage is plain message passing over the tile grid (as opposed to
    the full-attention coarse transformer in HierarchicalGNN) — kept as the
    pre-transformer architecture for comparison.

    Fine (real-node) and coarse (tile/global) message passing run at their
    own widths -- hidden_dim and coarse_dim respectively -- rather than
    sharing one residual stream. Two bipartite convs (encode: real->tile,
    decode: tile->real) are the only bridge between the two, each doing its
    own dimension projection internally; nothing else is ever downcast to
    hidden_dim or upcast to coarse_dim.
    """

    def __init__(self, node_in_dim, edge_in_dim=1, hidden_dim=128, num_layers=6, tile_in_dim=1,
                 use_checkpoint=False, checkpoint_fine_only=False,
                 num_pos_nodes=0, dropout=0.0, node_target_mean=0.0, node_target_std=1.0,
                 stress_ks_rho_mpa=0.15, coarse_dim=None):
        super().__init__()
        # coarse_dim decouples the tile/global message-passing width from
        # hidden_dim (the fine real-node width), so sweeping hidden_dim down for
        # speed doesn't also shrink coarse-scale capacity. Defaults to hidden_dim
        # (old behaviour) when unset.
        self.coarse_dim = coarse_dim = coarse_dim if coarse_dim is not None else hidden_dim
        assert num_layers >= 2, "HierarchicalGNNMP needs at least 2 conv layers (encode + decode pass)"
        self.use_checkpoint = use_checkpoint
        # When True, only the fine-mesh layers (fine_convs + encode_conv/decode_conv)
        # are checkpointed. The coarse middle layers run without checkpoint because
        # their activations are tiny anyway (few tile/global nodes). Requires
        # use_checkpoint=True to have effect.
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

        self.encoder = MLP([node_in_dim, hidden_dim, hidden_dim], norm=None, dropout=dropout)
        self.head = MLP([hidden_dim, hidden_dim, 1], norm=None, dropout=dropout)
        self.conv_dropout = nn.Dropout(dropout)

        # Fine mesh (real<->real): one pass right after encoding (paired with
        # the encode bridge below), one final pass right after decoding.
        self.fine_convs = nn.ModuleList([EdgeAttrConv(hidden_dim, edge_in_dim) for _ in range(2)])
        # Coarse mesh (tile<->tile, tile<->global): native coarse_dim throughout,
        # never downcast to hidden_dim.
        self.coarse_convs = nn.ModuleList(
            [EdgeAttrConv(coarse_dim, edge_in_dim) for _ in range(num_layers - 2)]
        )
        # Fine/coarse bridges -- the only two places a projection happens.
        self.encode_conv = BipartiteEdgeAttrConv(hidden_dim, coarse_dim, edge_in_dim)  # real -> tile
        self.decode_conv = BipartiteEdgeAttrConv(coarse_dim, hidden_dim, edge_in_dim)  # tile -> real

        # Level 1 — tile virtual nodes (native coarse_dim)
        self.tile_emb            = nn.Parameter(torch.zeros(coarse_dim))
        self.tile_input_encoder  = MLP([tile_in_dim, coarse_dim, coarse_dim], norm=None)
        self.real_tile_edge_attr = nn.Parameter(torch.zeros(edge_in_dim))
        self.tile_tile_edge_attr = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.tile_emb, std=0.02)

        # Level 2 — global virtual node (native coarse_dim)
        self.global_emb             = nn.Parameter(torch.zeros(coarse_dim))
        self.tile_global_edge_attr  = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.global_emb, std=0.02)
        # Degenerate fallback path (no tile grid at all -- see forward()): there's
        # no coarse mesh to give its own width, so the global node just projects
        # straight into hidden_dim as it always did.
        self.global_proj_fallback = nn.Linear(coarse_dim, hidden_dim) if coarse_dim != hidden_dim else nn.Identity()

        # Local tile-tile adjacency, fixed for the lifetime of a training run
        # (grid shape never changes for a given geometry). Populated once by
        # set_tile_grid() -- NOT lazily inside forward() via a dict keyed by
        # (NX, NY, NZ, device): torch.compile/dynamo re-guards a plain dict's
        # contents by object identity on every trace and eventually hits its
        # recompile cache limit, silently falling back to eager mid-training.
        self.register_buffer("tile_adj_local", None, persistent=False)
        self._tile_grid = None  # (NX, NY, NZ) tile_adj_local was built for

    @staticmethod
    def _build_local_tile_adj(NX, NY, NZ):
        """Build local (intra-graph) 6-connected tile pairs. Called once per grid shape.

        Flat indices here must match tile_idx's own convention: IRIT's
        tiling order is X-fastest/Z-slowest (see UserMicroEvalTilingIndex
        in irit/user_lib/micro_strct.c), the opposite nesting from
        dims.json's ShellThicknesses array (X-outermost).
        """
        pairs = []
        for ix in range(NX):
            for iy in range(NY):
                for iz in range(NZ):
                    flat = iz * NY * NX + iy * NX + ix
                    for dix, diy, diz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                        jx, jy, jz = ix + dix, iy + diy, iz + diz
                        if jx < NX and jy < NY and jz < NZ:
                            nb = jz * NY * NX + jy * NX + jx
                            pairs += [(flat, nb), (nb, flat)]
        if not pairs:
            return None
        return torch.tensor(pairs, dtype=torch.long).t()  # [2, E_local]

    def set_tile_grid(self, NX, NY, NZ):
        """Precompute the fixed tile-tile adjacency for this geometry's grid.
        Call once, right after construction (before/after torch.compile
        wrapping, either is fine -- the underlying module is shared)."""
        local = self._build_local_tile_adj(NX, NY, NZ)
        device = self.tile_emb.device
        self.tile_adj_local = local.to(device) if local is not None else None
        self._tile_grid = (NX, NY, NZ)

    def _tile_adjacency(self, NX, NY, NZ, num_graphs, tile_start, device):
        """Bidirectional 6-connected grid edges between tile virtual nodes."""
        assert self._tile_grid == (NX, NY, NZ), (
            f"set_tile_grid{self._tile_grid} was called, but forward() saw "
            f"grid {(NX, NY, NZ)} -- fixed-topology assumption violated."
        )
        local = self.tile_adj_local
        if local is None:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        K = NX * NY * NZ
        offsets = torch.arange(num_graphs, device=device) * K             # [G]
        src = (local[0].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1) + tile_start
        dst = (local[1].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1) + tile_start
        return torch.stack([src, dst], dim=0)

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

            tile_h = self.tile_emb.unsqueeze(0).expand(num_graphs * K, -1)
            if tile_x is not None:
                tile_h = tile_h + self.tile_input_encoder(tile_x.float())
            global_h = self.global_emb.unsqueeze(0).expand(num_graphs, -1)

            # Real <-> tile bipartite edges, as two directed edge sets (each
            # local to its own pair of tensors -- no shared augmented tensor,
            # so no index offset needed).
            tile_local = batch * K + tile_idx
            rt_encode_ei = torch.stack([real_idx, tile_local], dim=0)   # real -> tile
            rt_decode_ei = torch.stack([tile_local, real_idx], dim=0)   # tile -> real
            rt_ea = self.real_tile_edge_attr.unsqueeze(0).expand(N, -1)

            fine_ckpt = self.training and self.use_checkpoint

            # --- Encode: first real-real refinement + push real info up to tiles ---
            if fine_ckpt:
                h = h + self.conv_dropout(
                    checkpoint(self.fine_convs[0], h, edge_index, edge_attr, use_reentrant=False))
                tile_h = tile_h + self.conv_dropout(
                    checkpoint(self.encode_conv, h, tile_h, rt_encode_ei, rt_ea, use_reentrant=False))
            else:
                h = h + self.conv_dropout(self.fine_convs[0](h, edge_index, edge_attr))
                tile_h = tile_h + self.conv_dropout(self.encode_conv(h, tile_h, rt_encode_ei, rt_ea))

            # --- Process: coarse mesh (tile<->tile, tile<->global), native coarse_dim ---
            tile_start = 0
            global_start = num_graphs * K
            coarse_h = torch.cat([tile_h, global_h], dim=0)

            tt_edges = self._tile_adjacency(NX, NY, NZ, num_graphs, tile_start, device)
            all_tile_nodes = torch.arange(num_graphs * K, device=device)
            graph_of_tile  = all_tile_nodes // K
            tg_src = tile_start   + all_tile_nodes
            tg_dst = global_start + graph_of_tile
            tg_edges = torch.cat([
                torch.stack([tg_src, tg_dst], dim=0),
                torch.stack([tg_dst, tg_src], dim=0),
            ], dim=1)
            tt_ea = self.tile_tile_edge_attr.unsqueeze(0).expand(tt_edges.size(1), -1)
            tg_ea = self.tile_global_edge_attr.unsqueeze(0).expand(tg_edges.size(1), -1)
            coarse_ei = torch.cat([tt_edges, tg_edges], dim=1)
            coarse_ea = torch.cat([tt_ea, tg_ea], dim=0)

            coarse_ckpt = fine_ckpt and not self.checkpoint_fine_only
            for conv in self.coarse_convs:
                if coarse_ckpt:
                    coarse_h = coarse_h + self.conv_dropout(
                        checkpoint(conv, coarse_h, coarse_ei, coarse_ea, use_reentrant=False))
                else:
                    coarse_h = coarse_h + self.conv_dropout(conv(coarse_h, coarse_ei, coarse_ea))

            tile_h = coarse_h[:num_graphs * K]

            # --- Decode: pull tile info back to real nodes + final local refinement ---
            if fine_ckpt:
                h = h + self.conv_dropout(
                    checkpoint(self.decode_conv, tile_h, h, rt_decode_ei, rt_ea, use_reentrant=False))
                h = h + self.conv_dropout(
                    checkpoint(self.fine_convs[1], h, edge_index, edge_attr, use_reentrant=False))
            else:
                h = h + self.conv_dropout(self.decode_conv(tile_h, h, rt_decode_ei, rt_ea))
                h = h + self.conv_dropout(self.fine_convs[1](h, edge_index, edge_attr))

        else:
            # Fallback: single global virtual node, no tile grid (original GNN
            # behaviour). No coarse mesh exists here, so there's nothing to give
            # its own width -- the global node just projects into hidden_dim and
            # rides along with the fine_convs.
            global_start = N
            global_h = self.global_proj_fallback(self.global_emb.unsqueeze(0).expand(num_graphs, -1))
            h_aug = torch.cat([h, global_h], dim=0)

            global_node_abs = global_start + batch
            virt_edges = torch.cat([
                torch.stack([real_idx, global_node_abs], dim=0),
                torch.stack([global_node_abs, real_idx], dim=0),
            ], dim=1)
            edge_index_aug = torch.cat([edge_index, virt_edges], dim=1)
            tg_ea = self.tile_global_edge_attr.unsqueeze(0).expand(virt_edges.size(1), -1)
            edge_attr_aug = torch.cat([edge_attr, tg_ea], dim=0)

            for conv in self.fine_convs:
                if self.training and self.use_checkpoint:
                    h_aug = h_aug + self.conv_dropout(checkpoint(conv, h_aug, edge_index_aug, edge_attr_aug,
                                               use_reentrant=False))
                else:
                    h_aug = h_aug + self.conv_dropout(conv(h_aug, edge_index_aug, edge_attr_aug))
            h = h_aug[:N]

        node_pred = self.head(h)
        graph_pred = ks_pool(node_pred, batch, num_graphs, self.node_target_mean,
                              self.node_target_std, self.stress_ks_rho_mpa)
        return graph_pred, node_pred
