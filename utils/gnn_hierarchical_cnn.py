import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.models import MLP

from utils.gnn_surrogate import EdgeAttrConv, BipartiteEdgeAttrConv, ks_pool


def _group_norm(dim):
    """GroupNorm with the largest group count <=8 that evenly divides dim
    (falls back to 1, i.e. full-channel normalization, if none does)."""
    groups = min(8, dim)
    while dim % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, dim)


class _ConvBlock(nn.Module):
    """Pre-norm residual 3x3x3 conv, same spatial resolution in and out."""
    def __init__(self, dim):
        super().__init__()
        self.norm = _group_norm(dim)
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv(F.gelu(self.norm(x)))


class _DownBlock(nn.Module):
    """One U-Net encoder level: residual refine at the input resolution (the
    skip connection), then a strided conv. Axes already at size 1 (e.g. a
    plate-like tile grid with NZ=1) use stride 1 on that axis and are left
    alone, so the U-Net degrades gracefully to 2D/1D instead of erroring."""
    def __init__(self, dim, stride):
        super().__init__()
        self.refine = _ConvBlock(dim)
        self.down_norm = _group_norm(dim)
        self.down = nn.Conv3d(dim, dim, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        skip = self.refine(x)
        x = self.down(F.gelu(self.down_norm(skip)))
        return x, skip


class _UpBlock(nn.Module):
    """Upsample to the matching encoder resolution, merge with its skip
    connection, then residually refine."""
    def __init__(self, dim):
        super().__init__()
        self.refine = _ConvBlock(dim)

    def forward(self, x, skip, target_size):
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        return self.refine(x + skip)


class HierarchicalGNNCNN(nn.Module):
    """3-level model: real nodes <-> tile virtual nodes <-> a dense 3D U-Net
    over the tile grid.

    Tile nodes form a regular NX x NY x NZ grid (fixed per training run, see
    set_tile_grid). Rather than message-passing over 6-connected tile-tile
    edges plus a global virtual node (HierarchicalGNNMP) or full attention
    over tile+global tokens (HierarchicalGNN), this reshapes the tile grid
    into a dense tensor and runs a small 3D U-Net over it: the
    downsample/bottleneck/upsample path gives every tile a route to
    whole-grid context in a handful of cheap strided-conv layers, while its
    skip connections keep a coarse *spatial* map of that context at every
    scale (not a single pooled vector) -- important when e.g. a load applied
    at one tile needs to propagate, with its spatial profile intact, to a
    distant anchor.

    Fine (real-node) and coarse (tile) processing run at their own widths --
    hidden_dim and coarse_dim respectively -- exactly as in
    HierarchicalGNNMP; only the coarse stage itself differs.
    """

    def __init__(self, node_in_dim, edge_in_dim=1, hidden_dim=128, num_layers=6, tile_in_dim=1,
                 use_checkpoint=False, checkpoint_fine_only=False,
                 num_pos_nodes=0, dropout=0.0, node_target_mean=0.0, node_target_std=1.0,
                 stress_ks_rho_mpa=0.15, coarse_dim=None, cnn_depth=3,
                 predict_node_grad=False, node_out_dim=1):
        super().__init__()
        # node_out_dim: per-node outputs. 1 = von Mises stress, pooled to the
        # graph KS by ks_pool (every checkpoint before 2026-10-13). >1 (the
        # multi-load surrogate: displacement 3 + stress tensor 6) returns the
        # raw per-node prediction and graph_pred=None -- the caller derives
        # von Mises / KS from the predicted tensor.
        self.node_out_dim = int(node_out_dim)
        # num_layers is accepted (unused) for constructor-signature parity with
        # HierarchicalGNN/HierarchicalGNNMP -- checkpoint reconstruction always
        # passes it regardless of coarse_arch. U-Net depth is its own knob
        # (cnn_depth) since "how many down/up levels" doesn't map onto "how
        # many message-passing hops" the way it does for the other two archs.
        self.coarse_dim = coarse_dim = coarse_dim if coarse_dim is not None else hidden_dim
        assert cnn_depth >= 1, "HierarchicalGNNCNN needs at least 1 U-Net down/up level"
        self.cnn_depth = cnn_depth
        self.use_checkpoint = use_checkpoint
        self.checkpoint_fine_only = checkpoint_fine_only
        self.stress_ks_rho_mpa = stress_ks_rho_mpa
        self.register_buffer("node_target_mean", torch.as_tensor(node_target_mean, dtype=torch.float), persistent=False)
        self.register_buffer("node_target_std", torch.as_tensor(node_target_std, dtype=torch.float), persistent=False)

        # --- Fixed-topology exploit (hollow_cube etc.) ---------------------
        self.node_emb = nn.Embedding(num_pos_nodes, hidden_dim) if num_pos_nodes else None
        if self.node_emb is not None:
            nn.init.normal_(self.node_emb.weight, std=0.02)

        self.encoder = MLP([node_in_dim, hidden_dim, hidden_dim], norm=None, dropout=dropout)
        self.head = MLP([hidden_dim, hidden_dim, self.node_out_dim], norm=None, dropout=dropout)
        # Per-node dKS/dX head. Predicts the ADJOINT NODAL gradient field
        # (3 components/node) off the same final hidden state the stress
        # head reads, so it costs one small MLP and no second forward.
        #
        # Why per-node rather than per-tile: the tile sensitivity is
        # dKS/dt_i = sum_{n in tile i} <dKS/dX_n, dX_n/dt_i>, and dX/dt is
        # EXACT, block-diagonal geometry from core/geometric_jacobian.py
        # (two mesh builds, ~10s) -- there is no reason to spend network
        # capacity approximating a closed-form chain rule. Measured
        # 2026-09-07: composing stored node gradients through a freshly
        # computed Jacobian reproduces the stored per-tile labels to
        # cosine +1.00000002 / rel_l2 3.8e-08 on two very different
        # designs, and the composition is strikingly error-tolerant --
        # 20% per-node relative error still gives tile cosine +0.997 with
        # 0.0% wrong-way mass, versus the constant-direction control's
        # +0.613 / 12.4%. Each tile sums ~57 node contributions and the
        # largest single node carries only ~10-14% of its tile's total, so
        # per-node error averages down instead of accumulating.
        #
        # Built only when asked so existing checkpoints keep loading and
        # runs that do not use it pay nothing.
        self.grad_head = (MLP([hidden_dim, hidden_dim, 3], norm=None, dropout=dropout)
                          if predict_node_grad else None)
        self.conv_dropout = nn.Dropout(dropout)

        # Fine mesh (real<->real): one pass right after encoding (paired with
        # the encode bridge below), one final pass right after decoding.
        self.fine_convs = nn.ModuleList([EdgeAttrConv(hidden_dim, edge_in_dim) for _ in range(2)])
        # Fine/coarse bridges -- the only two places a dimension projection
        # happens, same as HierarchicalGNNMP.
        self.encode_conv = BipartiteEdgeAttrConv(hidden_dim, coarse_dim, edge_in_dim)  # real -> tile
        self.decode_conv = BipartiteEdgeAttrConv(coarse_dim, hidden_dim, edge_in_dim)  # tile -> real

        # Level 1 — tile virtual nodes (native coarse_dim)
        self.tile_emb            = nn.Parameter(torch.zeros(coarse_dim))
        self.tile_input_encoder  = MLP([tile_in_dim, coarse_dim, coarse_dim], norm=None)
        self.real_tile_edge_attr = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.tile_emb, std=0.02)

        # Degenerate fallback path (no tile grid at all -- see forward()):
        # there's no grid to run a CNN over, so this collapses to a single
        # global virtual node projected into hidden_dim, same as
        # HierarchicalGNNMP's fallback.
        self.global_emb            = nn.Parameter(torch.zeros(coarse_dim))
        self.tile_global_edge_attr = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.global_emb, std=0.02)
        self.global_proj_fallback = nn.Linear(coarse_dim, hidden_dim) if coarse_dim != hidden_dim else nn.Identity()

        # U-Net down/bottleneck/up blocks -- built lazily in set_tile_grid()
        # once NX/NY/NZ are known, since per-axis strides depend on the grid
        # shape. Same pattern as tile_adj_local in HierarchicalGNNMP: call
        # set_tile_grid() once, right after construction, before
        # optimizer/compile/DDP wrap the model.
        self.down_blocks = None
        self.bottleneck = None
        self.up_blocks = None
        self._dims_per_level = None  # [(NX,NY,NZ), ..., bottleneck dims]
        self._tile_grid = None       # (NX, NY, NZ) the above was built for

    def set_tile_grid(self, NX, NY, NZ):
        """Precompute the fixed U-Net down/up blocks for this geometry's
        grid shape. Call once, right after construction.

        tile_idx (and hence the flat tile_h layout reshaped in forward())
        is ordered X-fastest/Z-slowest -- IRIT's own tiling order (see
        UserMicroEvalTilingIndex in irit/user_lib/micro_strct.c), the
        opposite nesting from dims.json's ShellThicknesses array. The U-Net
        is therefore built over a (NZ, NY, NX)-shaped volume so its axes
        line up with a row-major reshape of that flat layout.
        """
        dims = [(NZ, NY, NX)]
        strides = []
        cur = (NZ, NY, NX)
        for _ in range(self.cnn_depth):
            stride = tuple(2 if d > 1 else 1 for d in cur)
            # Conv3d output size for kernel=3, padding=1: floor((d-1)/s)+1.
            nxt = tuple((d - 1) // s + 1 for d, s in zip(cur, stride))
            strides.append(stride)
            dims.append(nxt)
            cur = nxt

        device = self.tile_emb.device
        self.down_blocks = nn.ModuleList(
            [_DownBlock(self.coarse_dim, s) for s in strides]).to(device)
        self.bottleneck = nn.ModuleList(
            [_ConvBlock(self.coarse_dim), _ConvBlock(self.coarse_dim)]).to(device)
        self.up_blocks = nn.ModuleList(
            [_UpBlock(self.coarse_dim) for _ in range(self.cnn_depth)]).to(device)
        self._dims_per_level = dims
        self._tile_grid = (NX, NY, NZ)

    def _unet_forward(self, G):
        skips = []
        x = G
        for down in self.down_blocks:
            x, skip = down(x)
            skips.append(skip)
        for block in self.bottleneck:
            x = block(x)
        targets = list(reversed(self._dims_per_level[:-1]))
        for up, skip, size in zip(self.up_blocks, reversed(skips), targets):
            x = up(x, skip, size)
        return x

    def forward(self, x, edge_index, edge_attr, batch,
                tile_idx=None, tile_NX=None, tile_NY=None, tile_NZ=None, tile_x=None,
                num_graphs=None, node_id=None, return_node_grad=False, ks_mask=None):
        N = x.size(0)
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if N > 0 else 0
        device = x.device

        h = self.encoder(x)
        if self.node_emb is not None and node_id is not None:
            h = h + self.node_emb(node_id)
        real_idx = torch.arange(N, device=device)

        if tile_idx is not None:
            NX, NY, NZ = tile_NX, tile_NY, tile_NZ
            assert self._tile_grid == (NX, NY, NZ), (
                f"set_tile_grid{self._tile_grid} was called, but forward() saw "
                f"grid {(NX, NY, NZ)} -- fixed-topology assumption violated."
            )
            K = NX * NY * NZ

            tile_h = self.tile_emb.unsqueeze(0).expand(num_graphs * K, -1)
            if tile_x is not None:
                tile_h = tile_h + self.tile_input_encoder(tile_x.float())

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

            # --- Process: 3D U-Net over the dense tile grid, native coarse_dim ---
            # tile_h's K rows are ordered X-fastest/Z-slowest (tile_idx's
            # native order, see set_tile_grid), so reshape to (NZ,NY,NX).
            G = tile_h.view(num_graphs, NZ, NY, NX, -1).permute(0, 4, 1, 2, 3).contiguous()
            coarse_ckpt = fine_ckpt and not self.checkpoint_fine_only
            if coarse_ckpt:
                G = G + checkpoint(self._unet_forward, G, use_reentrant=False)
            else:
                G = G + self._unet_forward(G)
            tile_h = G.permute(0, 2, 3, 4, 1).reshape(num_graphs * K, -1)

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
            # Fallback: single global virtual node, no tile grid -- there's
            # nothing to run a CNN over.
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
        if self.node_out_dim != 1:
            graph_pred = None
        else:
            graph_pred = ks_pool(node_pred, batch, num_graphs, self.node_target_mean,
                                  self.node_target_std, self.stress_ks_rho_mpa,
                                  ks_mask=ks_mask)
        if return_node_grad:
            assert self.grad_head is not None, (
                "return_node_grad=True but the model was built with "
                "predict_node_grad=False -- no grad_head exists")
            return graph_pred, node_pred, self.grad_head(h)
        return graph_pred, node_pred
