"""Load-linear surrogate: stress is EXACTLY linear in the applied load, by
construction, and nonlinear in the geometry.

Linear elasticity is sigma = A(t) f with A(t) = D B K(t)^-1: any function of
the design, but a linear operator on the nodal forces. This model has the same
shape. Two streams:

    geometry stream  h = HierarchicalGNNCNN(design) -- unrestricted, nonlinear;
                     it sees no load at all.
    load stream      u starts as the nodal force vector f and only ever goes
                     through operations that are linear in u: bias-free
                     Linear/Conv, sums over edges and tiles, interpolation, and
                     elementwise products with gates computed from h. No
                     bias, no activation, no normalization ever touches u.

So model(t, a f1 + b f2) == a model(t, f1) + b model(t, f2) to round-off, for
any loads, and every load case is the same learned operator A(t) applied to a
different f -- the 11 load rows are channels of ONE forward, not 11 separately
conditioned predictions. d sigma / dt runs through the gates only.

The load stream mirrors the geometry stream's hierarchy so a load can reach
the anchor: fine linear message passing, a pool to the tile grid, a linear
U-Net over the tiles (gated by the geometry stream's tile features pooled to
each level), an unpool, more fine message passing, and a gated linear readout.
Gates are 1 + MLP(.) with the MLP's last layer zero-initialized: the model
starts as a geometry-independent linear GNN and learns the modulation. With
gate_bound a > 0 a gate is 1 + a tanh(MLP / a), smoothly saturating in
(1 - a, 1 + a): the load stream is a product of many gates, and unbounded ones
let that product grow until the operator is badly conditioned (run v1 peaked at
epoch 7 and degraded from epoch 8 with the gate weights still growing).

Every d sigma / dt has to pass through these gates, and the Sobolev loss
differentiates through that a second time. With gate_bound 1 half the tile gates
ended up saturated (|z| > 2), and the Sobolev gradient into the geometry tile
CNN grew 0.7 -> 117 before every such run destabilized. Two knobs against that:
tile_gate_bound b > 0 makes the tile gates exp(b tanh(z/b)) -- positive, range
(e^-b, e^b), slope 1 at zero like 1 + z -- and geo_spectral_norm puts spectral
normalization on the geometry tile CNN's convs and its thickness encoder, which
bounds how steeply the tile features (hence the gates) can respond to t.

Neither stopped the blow-ups. What the gates were straining for: tiles at the
0.03 floor carry ~59% of all |dS/dt| in the dataset, and at their own nodes
dS/dt ~ -n S / t (cos^2 ~0.7, n ~0.3-0.45) -- 10-15x the local stress -- which
a product of gates can only produce in a steep, weight-sensitive regime (the
per-design Sobolev gradient on thin designs was ~20-50x v5's). thick_power
n0 > 0 multiplies each node's stress by (t_node / 0.2)^-n, n > 0 learned per
stress component from n0, so that part of dS/dt is analytic and the gates only
learn a correction. Still exactly linear in f: the factor depends on t only.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP

from utils.gnn_hierarchical_cnn import HierarchicalGNNCNN


def _gate_mlp(dims):
    m = MLP(dims, norm=None)
    nn.init.zeros_(m.lins[-1].weight)
    nn.init.zeros_(m.lins[-1].bias)
    return m


class LoadLinearGNNCNN(nn.Module):
    load_linear = True

    def __init__(self, node_in_dim, edge_in_dim=4, hidden_dim=128, coarse_dim=128, cnn_depth=2, tile_in_dim=1,
                 num_pos_nodes=0, load_dim=3, lin_dim=32, lin_layers=2, out_dim=6, use_checkpoint=False,
                 gate_bound=0.0, full_readout=False, lin_act=False, tile_gate_bound=0.0, geo_spectral_norm=False,
                 thick_power=0.0):
        super().__init__()
        self.thick_log_n = (nn.Parameter(torch.full((out_dim,), math.log(thick_power))) if thick_power > 0
                            else None)
        self.tile_gate_bound = float(tile_gate_bound)
        self.geo_spectral_norm = bool(geo_spectral_norm)
        # lin_act: DIAGNOSTIC ONLY -- a GELU after every load-stream block, which
        # breaks the linearity; isolates "is the linear structure the limit" from
        # everything else about this model
        self.lin_act = bool(lin_act)
        self.gate_bound = float(gate_bound)
        self.geo = HierarchicalGNNCNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, hidden_dim=hidden_dim,
                                      tile_in_dim=tile_in_dim, num_pos_nodes=num_pos_nodes, coarse_dim=coarse_dim,
                                      cnn_depth=cnn_depth, node_out_dim=hidden_dim, use_checkpoint=use_checkpoint,
                                      norm_mode="node")       # each design independent of its batch
        C, L = lin_dim, lin_layers
        self.C, self.L, self.cnn_depth, self.out_dim = C, L, cnn_depth, out_dim
        # node gates: lift, L fine (before), pool, unpool, L fine (after), readout
        self.n_node_gates = 2 * L + 4
        self.node_gates = _gate_mlp([hidden_dim, hidden_dim, C * self.n_node_gates])
        self.edge_gates = _gate_mlp([2 * hidden_dim + edge_in_dim, hidden_dim, C * 2 * L])
        self.lift = nn.Linear(load_dim, C, bias=False)
        self.fine = nn.ModuleList([nn.Linear(C, C, bias=False) for _ in range(2 * L)])
        self.unpool = nn.Linear(C, C, bias=False)
        self.out = nn.Linear(C, out_dim, bias=False)
        # full_readout: stress = R(h_i) u_i with a full (out_dim x C) matrix predicted
        # per node from the geometry, instead of a per-channel gate and one shared
        # matrix -- stress recovery (D B in FE) depends on the local geometry as a
        # whole matrix, not a channel scaling (diagonal gates plateaued at ~3x v5's
        # value loss). Still linear in u: R depends on h only.
        self.readout = MLP([hidden_dim, hidden_dim, C * out_dim], norm=None) if full_readout else None
        self.coarse_dim = coarse_dim
        self._dims_per_level = None

    def set_tile_grid(self, NX, NY, NZ):
        """Build the geometry U-Net and the linear tile U-Net (same strides).
        Call once, right after construction."""
        self.geo.set_tile_grid(NX, NY, NZ)
        if self.geo_spectral_norm:
            sn = nn.utils.parametrizations.spectral_norm
            for part in (self.geo.down_blocks, self.geo.up_blocks, self.geo.bottleneck):
                for mod in part.modules():
                    if isinstance(mod, nn.Conv3d):
                        sn(mod)
            for lin in self.geo.tile_input_encoder.lins:
                sn(lin)
        dims, strides = self.geo._dims_per_level, []
        for a, b in zip(dims[:-1], dims[1:]):
            strides.append(tuple(2 if x > 1 and y < x else 1 for x, y in zip(a, b)))
        C, D = self.C, self.cnn_depth
        conv = lambda s=1: nn.Conv3d(C, C, kernel_size=3, stride=s, padding=1, bias=False)
        dev = self.lift.weight.device
        self.down_refine = nn.ModuleList([conv() for _ in range(D)]).to(dev)
        self.down = nn.ModuleList([conv(s) for s in strides]).to(dev)
        self.bottleneck = nn.ModuleList([conv(), conv()]).to(dev)
        self.up_refine = nn.ModuleList([conv() for _ in range(D)]).to(dev)
        # tile gates per level from the geometry tile features pooled to that level:
        # level l < D: down-refine, down, up-refine (3); bottleneck level D: 2
        self.tile_gates = nn.ModuleList(
            [nn.Conv3d(self.coarse_dim, C * (3 if l < D else 2), kernel_size=1) for l in range(D + 1)]).to(dev)
        for g in self.tile_gates:
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)
        self._dims_per_level = dims
        self._tile_grid = (NX, NY, NZ)

    def _gate(self, z):
        """1 + z, or 1 + a tanh(z / a) with gate_bound a > 0; float32."""
        z, a = z.float(), self.gate_bound
        return 1.0 + (a * torch.tanh(z / a) if a > 0 else z)

    def _tile_gate(self, z):
        """exp(b tanh(z / b)) with tile_gate_bound b > 0, else the node/edge gate."""
        b = self.tile_gate_bound
        return torch.exp(b * torch.tanh(z.float() / b)) if b > 0 else self._gate(z)

    def _linear_unet(self, U, gates, B, R):
        """U (B*R, C, *grid) -> same; gates[l] (B, 1, n_l, C, *dims_l), 1 + pooled geometry."""
        def gate(U, l, k):
            d = U.shape[2:]
            return (U.view(B, R, self.C, *d) * gates[l][:, :, k]).view(B * R, self.C, *d)

        skips = []
        for l in range(self.cnn_depth):
            skip = U + self.down_refine[l](gate(U, l, 0))
            U = self.down[l](gate(skip, l, 1))
            skips.append(skip)
        D = self.cnn_depth
        for k, c in enumerate(self.bottleneck):
            U = U + c(gate(U, D, k))
        for l in reversed(range(D)):
            U = F.interpolate(U, size=self._dims_per_level[l], mode="trilinear", align_corners=False) + skips[l]
            U = U + self.up_refine[l](gate(U, l, 2))
        return U

    def forward(self, x, edge_index, edge_attr, batch, f, tile_idx, tile_NX, tile_NY, tile_NZ, tile_x=None,
                num_graphs=None, node_id=None, t_node=None):
        """f (B*N, R, load_dim) nodal forces of R load cases -> stress (B*N, R, out_dim),
        linear in f. t_node (B*N,): each node's own tile thickness, raw (needed with
        thick_power)."""
        BN, R = f.shape[0], f.shape[1]
        B, C = num_graphs, self.C
        NX, NY, NZ = tile_NX, tile_NY, tile_NZ
        K = NX * NY * NZ
        _, h, tile_h = self.geo(x, edge_index, edge_attr, batch, tile_idx=tile_idx, tile_NX=NX, tile_NY=NY,
                                tile_NZ=NZ, tile_x=tile_x, num_graphs=B, node_id=node_id, return_tile_h=True)
        gn = self._gate(self.node_gates(h)).view(BN, 1, self.n_node_gates, C)
        src, dst = edge_index
        ge = self._gate(self.edge_gates(torch.cat([h[src], h[dst], edge_attr.to(h.dtype)], dim=1))).view(
            -1, 1, 2 * self.L, C)
        deg = torch.bincount(dst, minlength=BN).clamp_min(1).to(torch.float32)[:, None, None]

        def fine(u, k, g):
            agg = torch.zeros_like(u).index_add_(0, dst, u[src] * ge[:, :, k]) / deg
            u = u + self.fine[k](agg * gn[:, :, g]).float()
            return F.gelu(u) if self.lin_act else u

        u = self.lift(f.float()).float() * gn[:, :, 0]                          # (BN,R,C)
        for k in range(self.L):
            u = fine(u, k, 1 + k)

        # --- pool to tiles, linear U-Net, unpool ---
        tl = batch * K + tile_idx
        cnt = torch.bincount(tl, minlength=B * K).clamp_min(1).to(torch.float32)[:, None, None]
        U = torch.zeros(B * K, R, C, device=u.device, dtype=u.dtype).index_add_(0, tl, u * gn[:, :, self.L + 1]) / cnt
        U = U.view(B, NZ, NY, NX, R, C).permute(0, 4, 5, 1, 2, 3).reshape(B * R, C, NZ, NY, NX)
        G = tile_h.view(B, NZ, NY, NX, -1).permute(0, 4, 1, 2, 3)
        gates = []
        for l, conv in enumerate(self.tile_gates):
            Gl = F.adaptive_avg_pool3d(G, self._dims_per_level[l])
            gl = self._tile_gate(conv(Gl))
            gates.append(gl.view(B, 1, -1, C, *gl.shape[2:]))
        U = U + self._linear_unet(U, gates, B, R).float()
        if self.lin_act:
            U = F.gelu(U)
        U = U.view(B, R, C, NZ, NY, NX).permute(0, 3, 4, 5, 1, 2).reshape(B * K, R, C)
        u = u + self.unpool(U[tl] * gn[:, :, self.L + 2]).float()

        for k in range(self.L, 2 * self.L):
            u = fine(u, k, 3 + k)
        if self.readout is not None:
            Rm = self.readout(h).float().view(BN, C, self.out_dim) / C ** 0.5
            out = torch.einsum("nrc,nco->nro", u, Rm)
        else:
            out = self.out(u * gn[:, :, 2 * self.L + 3])
        if self.thick_log_n is not None:
            out = out * (t_node.float()[:, None, None] / 0.2) ** -self.thick_log_n.exp()
        return out
