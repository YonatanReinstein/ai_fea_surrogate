"""Sobolev training of the hierarchical CNN surrogate on the cross4
stress-sensitivity dataset (training/build_cross4_dsig_dataset.py).

Predicts the nodal stress tensor (6) of one (design, load row) at a time, as a
function of the tile thicknesses t only (training.dsig_data). Because stress
is linear in the load, the 11 per-row predictions superpose at inference into
any load combination, and dKS/dt of the combination is one backward pass.

    python -m training.train_dsig --run_dir training/runs/cross4_dsig_sobolev_v1

--arch load_linear: the model is exactly linear in the load by construction
(utils/gnn_load_linear.py) -- one learned operator A(t) applied to each row's
force vector, all rows of a design in one forward (--designs_per_step whole
designs per step) -- instead of a per-row prediction conditioned on the row.

Data: designs whose KS is <= --ks_max_mpa on EVERY row (a weak row would make
that design's basis wrong in every combination); the filter runs before the
train / validation split, which is by design.

Loss, per case:
    value    MSE of the standardized stress components
    Sobolev  for --sob_dirs random Gaussian directions v over the K tiles, the
             model's directional derivative J v of the stress field (double
             backward through autograd) against sum_j v_j dS_j from the labels;
             relative L2 error r = |Jv - Jv*| / |Jv*|, smoothly saturated as
             r / (1 + r/2); weight --sob_weight.
LR: AdamW, linear warmup, then x0.99 per epoch and x0.5 when the epoch's
training loss stops improving (no cosine).

Validation every epoch on --val_max_designs held-out designs, single loads AND
load combinations (equal weights + Dirichlet(1) draws): von Mises relative L2,
KS relative error, and dKS/dt from one backward pass against the exact label
(cosine, wrong-way mass, sign agreement) next to the constant-direction
control; single loads also report the directional dS relative error.
Every epoch is saved as epoch_NNN.pt; best.pt = best combination median cosine.
"""
import argparse
import collections
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch

from training.dsig_data import (DsigStatic, DesignDataset, available_designs, design_ks, design_path,
                                mirror_case, mirror_maps)
from training.train_multiload import von_mises
from utils.gnn_hierarchical_cnn import HierarchicalGNNCNN
from utils.gnn_load_linear import LoadLinearGNNCNN


def build_model(model_kwargs, **extra):
    """Model from a checkpoint's model_kwargs; arch "load_linear" (stress exactly
    linear in the load, utils/gnn_load_linear.py) or the per-row hierarchical CNN."""
    kw = dict(model_kwargs)
    cls = LoadLinearGNNCNN if kw.pop("arch", "hier") == "load_linear" else HierarchicalGNNCNN
    return cls(**kw, **extra)


class Batcher:
    """Builds model inputs for B load rows of one design from a differentiable
    t (B,K) in dims order. The graph is the same for every design, so the
    batched topology is built once per B."""

    def __init__(self, static, st, dev):
        self.s, self.st, self.dev = static, st, dev
        self.perm = torch.as_tensor(static.perm, device=dev)
        self.node_tile = torch.as_tensor(static.node_tile, device=dev)
        ref = torch.as_tensor(static.ref_coords, dtype=torch.float32, device=dev)
        self.ref = (ref - st["ref_mean"]) / st["ref_std"]
        self.anch = torch.as_tensor(static.anchored, dtype=torch.float32, device=dev)[:, None]
        self.keep = torch.as_tensor(~static.pad, device=dev)
        self.edge_attr = ((static.edge_attr.to(dev) - st["edge_mean"]) / st["edge_std"])
        self.load_tile = {r: t for r, t in static.load_tile.items()}
        # each tile's row index (IRIT order ix + NX*iy), for the load-position features
        self.tile_row = torch.arange(static.K, device=dev) // static.NX
        self._topo = {}

    def topo(self, B):
        if B not in self._topo:
            N, E = self.s.N, self.s.edge_index.size(1)
            ei = self.s.edge_index.to(self.dev)
            off = torch.arange(B, device=self.dev).repeat_interleave(E) * N
            self._topo[B] = dict(
                edge_index=ei.repeat(1, B) + off, edge_attr=self.edge_attr.repeat(B, 1),
                batch=torch.arange(B, device=self.dev).repeat_interleave(N),
                tile_idx=self.node_tile.repeat(B), node_id=torch.arange(N, device=self.dev).repeat(B))
        return self._topo[B]

    def forward(self, model, t, load_w, rows):
        """t (B,K) dims order, load_w (B,N) -> predicted stress (B,N,6) in MPa."""
        if getattr(model, "load_linear", False):
            return self.forward_rows(model, t, load_w[:, None])[:, 0]
        st, B, N, K = self.st, t.size(0), self.s.N, self.s.K
        tn = (t[:, self.perm] - st["t_mean"]) / st["t_std"]                      # (B,K) IRIT order
        x = torch.cat([self.ref.expand(B, N, 3), (load_w / st["lw_scale"])[..., None],
                       self.anch.expand(B, N, 1), tn[:, self.node_tile][..., None]], dim=2).reshape(B * N, 6)
        flag = torch.zeros(B, K, device=self.dev, dtype=tn.dtype)   # explicit: core.fem_adjoint sets float64 as default
        for b, r in enumerate(rows):
            flag[b, self.load_tile[r]] = 1.0
        feats = [tn, flag]
        if model.tile_input_encoder.lins[0].in_channels == 4:
            # load position on EVERY tile (--load_pos_features): the tile's signed
            # row offset from the loaded row and the loaded row itself, both / (NY-1).
            # Without them only the loaded tile knows where the load is, and the
            # anchor side has to infer it 10 tiles away.
            r = torch.as_tensor(rows, dtype=tn.dtype, device=self.dev)[:, None]
            feats += [(self.tile_row[None].to(tn.dtype) - r) / (self.s.NY - 1),
                      (r / (self.s.NY - 1)).expand(B, K)]
        tile_x = torch.stack(feats, dim=2).reshape(B * K, len(feats))
        g = self.topo(B)
        _, pred = model(x, g["edge_index"], g["edge_attr"], g["batch"], tile_idx=g["tile_idx"],
                        tile_NX=self.s.NX, tile_NY=self.s.NY, tile_NZ=1, tile_x=tile_x,
                        num_graphs=B, node_id=g["node_id"])
        return pred.float().view(B, N, 6) * st["y_std"] + st["y_mean"]

    def forward_rows(self, model, t, load_w):
        """Load-linear model (utils/gnn_load_linear.py): t (B,K) dims order, load_w
        (B,R,N) -> stress (B,R,N,6) in MPa, R load cases from ONE geometry forward
        per design. The geometry stream sees no load; the force (-y, the dataset's
        direction) enters only the linear stream, and the output is only scaled
        (no y_mean shift), so the prediction is exactly linear in load_w."""
        st, B, R, N, K = self.st, t.size(0), load_w.size(1), self.s.N, self.s.K
        tn = (t[:, self.perm] - st["t_mean"]) / st["t_std"]                      # (B,K) IRIT order
        x = torch.cat([self.ref.expand(B, N, 3), self.anch.expand(B, N, 1),
                       tn[:, self.node_tile][..., None]], dim=2).reshape(B * N, 5)
        lw = (load_w / st["lw_scale"]).permute(0, 2, 1).reshape(B * N, R)
        f = torch.stack([torch.zeros_like(lw), -lw, torch.zeros_like(lw)], dim=2)   # (B*N,R,3)
        g = self.topo(B)
        pred = model(x, g["edge_index"], g["edge_attr"], g["batch"], f, tile_idx=g["tile_idx"],
                     tile_NX=self.s.NX, tile_NY=self.s.NY, tile_NZ=1, tile_x=tn.reshape(B * K, 1),
                     num_graphs=B, node_id=g["node_id"], t_node=t[:, self.perm][:, self.node_tile].reshape(B * N))
        return pred.float().view(B, N, R, 6).permute(0, 2, 1, 3) * st["y_std"]

    def ks(self, S, rho):
        """KS (rho per MPa) of von Mises over non-pad nodes; S (...,N,6) MPa."""
        vm = von_mises(S)
        vm = torch.where(self.keep, vm, torch.full_like(vm, -float("inf")))
        m = vm.max(-1, keepdim=True).values.detach()
        return m[..., 0] + torch.logsumexp(rho * (vm - m), -1) / rho


def directional(S, t, dirs):
    """J v of S(t) for each v in dirs (each (B,K)) by double backward:
    g(u) = J^T u is linear in u, and d<g(u), v>/du = J v."""
    u = torch.zeros_like(S, requires_grad=True)
    g = torch.autograd.grad(S, t, u, create_graph=True)[0]
    return [torch.autograd.grad(g, u, v, create_graph=True)[0] for v in dirs]


def rel_l2(p, q):
    """per graph |p - q| / |q| over (N,6); p, q (B,N,6)."""
    return (p - q).flatten(1).norm(dim=1) / q.flatten(1).norm(dim=1).clamp_min(1e-12)


def rel_err(p, q, power=1.0):
    """per graph relative error of p against q over (N,6), p, q (B,N,6). With
    power < 1 each node is weighted by |q_n|^(2 power - 2) (|q_n| = norm over the 6
    components, floored at 1% of the graph's mean), so a node's share of the loss
    grows like |q_n|^(2 power) instead of |q_n|^2 and the stress hot spots around
    thin tiles stop dominating it (power 0.3: top 10% of nodes 73% -> 25% of the
    loss). power = 1 is the plain relative L2."""
    if power == 1.0:
        return rel_l2(p, q)
    a = q.norm(dim=2)                                                           # (B,N)
    w = (a + 0.01 * a.mean(1, keepdim=True)).pow(2 * power - 2).detach()
    return ((w * (p - q).pow(2).sum(2)).sum(1) / (w * a.pow(2)).sum(1).clamp_min(1e-30)).sqrt()


def ccos_np(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def uniform_check(model, bt, uni, rho):
    """Per load row on the exact uniform-t design (dataset's exact_uniform_*.npz):
    KS error, centered cosine of dKS/dt against exact, and mirror consistency
    (row r against the tile-mirrored row NY-1-r; exact is +1). The random
    validation designs do not show load-position errors; this does."""
    s, dev = bt.s, bt.dev
    C, K, NX, NY = len(s.rows), s.K, s.NX, s.NY
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    tmir = (ix * NY + (NY - 1 - iy)).reshape(-1)
    model.eval()
    lw = torch.as_tensor(np.stack([s.load_weights(uni["coords"], r) for r in s.rows]), dtype=torch.float32, device=dev)
    if getattr(model, "load_linear", False):        # one geometry forward, per-row grads from it
        t = torch.full((1, K), float(uni["t"]), dtype=torch.float32, device=dev, requires_grad=True)
        ks = bt.ks(bt.forward_rows(model, t, lw[None])[0], rho)
        g = torch.stack([torch.autograd.grad(ks[c], t, retain_graph=True)[0][0] for c in range(C)])
    else:
        t = torch.full((C, K), float(uni["t"]), dtype=torch.float32, device=dev, requires_grad=True)
        ks = bt.ks(bt.forward(model, t, lw, s.rows), rho)
        g, = torch.autograd.grad(ks.sum(), t)
    model.train()
    g, ks = g.double().cpu().numpy(), ks.detach().cpu().numpy()
    cc = np.array([ccos_np(g[c], uni["dks"][c]) for c in range(C)])
    mir = np.array([ccos_np(g[c], g[C - 1 - c][tmir]) for c in range(C)])
    ke = np.abs(ks / uni["ks"] - 1)
    mid = slice(2, C - 2)
    return {"uniform/ccos_mean": float(cc.mean()), "uniform/ccos_min": float(cc.min()),
            "uniform/ccos_mid_mean": float(cc[mid].mean()), "uniform/mirror_mid_mean": float(mir[mid].mean()),
            "uniform/ks_rel_max": float(ke.max()), "uniform/ccos_rows": cc.round(3).tolist()}


def grad_metrics(p, t):
    """cosine, wrong-way mass, sign agreement of predicted vs true dKS/dt; (n,K)."""
    cos = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1)).clamp_min(1e-30)
    wrong = torch.sign(p) != torch.sign(t)
    mass = (t.abs() * wrong).sum(1) / t.abs().sum(1).clamp_min(1e-30)
    return cos, mass, (~wrong).float().mean(1)


TILE_CNN = ("down_blocks", "up_blocks", "bottleneck", "tile_input_encoder", "encode_conv", "decode_conv", "tile_emb")


def grad_split(model, g_value):
    """Pre-clip gradient norms by loss term and module group. g_value: the value
    term's grads by parameter name; the parameters' .grad hold the Sobolev term's.
    Groups: the geometry tile CNN (where the Sobolev gradient blew up in every
    load_linear run), the rest of the geometry stream, and the load stream with
    its gate heads (load_linear); for the per-row model everything outside the
    tile CNN is "rest"."""
    ll = getattr(model, "load_linear", False)

    def group(n):
        parts = n.split(".")
        if (parts[1] if ll and parts[0] == "geo" else parts[0]) in TILE_CNN:
            return "tile_cnn"
        return ("geo_rest" if parts[0] == "geo" else "load_stream") if ll else "rest"

    sq = {}
    for n, p_ in model.named_parameters():
        for term, g in (("value", g_value.get(n)), ("sob", p_.grad)):
            if g is not None:
                k = f"{term}:{group(n)}"
                sq[k] = sq.get(k, 0.0) + float(g.float().pow(2).sum())
    return {k: v ** 0.5 for k, v in sorted(sq.items())}


class RollbackGuard:
    """Every load_linear run trained to a good peak, then the Sobolev gradient
    grew 10-1000x and the run never recovered: halving the lr only slowed the
    slide. The pre-clip grad norm starts rising 1-2k steps before validation
    collapses, so: keep a CPU snapshot of model + optimizer every `every` steps
    (the last 3); once the mean grad norm over the last `window` steps exceeds k
    x the median of the healthy history, restore the OLDEST snapshot (2-3k steps
    back, before the rise began) and halve the lr. Steps with a grad norm above
    2x the healthy median are kept out of that history, so a slow drift cannot
    raise its own threshold."""

    def __init__(self, k, every=1000, window=200, hist=2000, min_hist=1000):
        self.k, self.every = k, every
        self.recent = collections.deque(maxlen=window)
        self.hist = collections.deque(maxlen=hist)
        self.min_hist = min_hist
        self.snaps = collections.deque(maxlen=3)
        self.count = 0

    def snapshot(self, step, model, opt):
        cpu = lambda d: {k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in d.items()}
        opt_sd = opt.state_dict()
        self.snaps.append((step, cpu(model.state_dict()),
                           {"state": {i: cpu(st) for i, st in opt_sd["state"].items()},
                            "param_groups": copy.deepcopy(opt_sd["param_groups"])}))

    def check(self, gnorm, step, active):
        """True when a rollback is due."""
        self.recent.append(gnorm)
        if not active:
            return False
        base = float(np.median(self.hist)) if len(self.hist) >= self.min_hist else None
        if base is None or gnorm < 2 * base:
            self.hist.append(gnorm)
        return (base is not None and len(self.recent) == self.recent.maxlen and self.snaps
                and float(np.mean(self.recent)) > self.k * base)

    def restore(self, model, opt):
        step0, msd, osd = self.snaps[0]
        model.load_state_dict(msd)
        opt.load_state_dict(osd)
        self.snaps = collections.deque([self.snaps[0]], maxlen=3)
        self.recent.clear()
        self.count += 1
        return step0


def case_clipped_backward(model, per_case, hist, k, active):
    """Backward of per_case.mean() with each case's gradient smoothly capped:
    g_b <- g_b / sqrt(1 + (|g_b| / c)^2), c = k x the median of the last
    len(hist) per-case norms (uncapped while not `active`, i.e. during the lr
    warmup, when the norms move too fast for a running median). The Sobolev
    gradient is heavy-tailed across designs -- at the best load_linear checkpoint
    the top 1% of designs (thin tiles, |dS/dt| up to ~1e4) carried 27% of its
    squared norm, ~10x the median design, while fit no worse than average -- so
    one such case set the direction of a whole clipped step. Costs one backward
    per case. Returns the uncapped norms and the scales applied."""
    P = [p for p in model.parameters() if p.requires_grad]
    B = per_case.numel()
    c = k * float(np.median(hist)) if active and hist else float("inf")
    acc = [torch.zeros_like(p) for p in P]
    norms, scales = [], []
    for b in range(B):
        gs = torch.autograd.grad(per_case[b] / B, P, retain_graph=b < B - 1, allow_unused=True)
        n = float(torch.linalg.vector_norm(torch.stack([x.norm() for x in gs if x is not None]))) * B
        sc = 1.0 / (1.0 + (n / c) ** 2) ** 0.5
        for a, x in zip(acc, gs):
            if x is not None:
                a.add_(x, alpha=sc)
        norms.append(n); scales.append(sc)
    hist.extend(norms)
    for p, a in zip(P, acc):
        p.grad = a
    return norms, scales


def case_batches(loader, rows, B, pool_size, rng, mirror=None):
    """Training batches of B (design, load row) cases, CPU tensors
    (t (B,K), load_w (B,N), S (B,N,6), dS (B,K,N,6) f16, rows).

    pool_size <= 1: the old order -- all of one design's rows, B at a time, so
    consecutive steps see the same design. Otherwise designs are MIXED: a pool of
    pool_size designs is kept, each step takes B different designs from it and one
    unused row of each, and a design leaves the pool (the loader's next one comes
    in) once all its rows are used -- every case still once per epoch, but no
    step is pulled toward a single design.

    mirror = (prob, tmir, nmir, NY): each case is replaced by its exact mirror
    image (dsig_data.mirror_case: tile rows and the load row flipped in y) with
    probability prob, so the model sees both halves of the y-symmetry equally."""
    C = len(rows)

    def one(d, c):
        case = (d["t"], d["load_w"][c], d["S"][c], d["dS"][c], rows[c])
        if mirror is not None and rng.random() < mirror[0]:
            case = mirror_case(*case, *mirror[1:])
        return case

    def pack(cases):
        cs = [one(d, c) for d, c in cases]
        return tuple(torch.stack([x[i] for x in cs]) for i in range(4)) + ([x[4] for x in cs],)
    if pool_size <= 1:
        for d in loader:
            order = list(range(C))
            rng.shuffle(order)
            for j in range(0, C, B):
                yield pack([(d, c) for c in order[j:j + B]])
        return
    it = iter(loader)
    pool = []                                    # [design dict, remaining rows]
    def refill():
        while len(pool) < pool_size:
            d = next(it, None)
            if d is None:
                return
            left = list(range(C))
            rng.shuffle(left)
            pool.append([d, left])
    refill()
    while pool:
        pick = rng.sample(range(len(pool)), min(B, len(pool)))
        cases = [(pool[k][0], pool[k][1].pop()) for k in pick]
        pool[:] = [e for e in pool if e[1]]
        refill()
        yield pack(cases)


def design_batches(loader, B):
    """Load-linear training: batches of B whole designs, every row of each
    (t (B,K), load_w (B,C,N), S (B,C,N,6), dS (B,C,K,N,6) f16). The loader is
    shuffled, so each step mixes B different designs."""
    buf = []
    for d in loader:
        buf.append(d)
        if len(buf) == B:
            yield tuple(torch.stack([x[k] for x in buf]) for k in ("t", "load_w", "S", "dS"))
            buf = []
    if buf:
        yield tuple(torch.stack([x[k] for x in buf]) for k in ("t", "load_w", "S", "dS"))


def compute_stats(static, ids, rng):
    ys, ts, lws = [], [], []
    for i in rng.sample(ids, min(40, len(ids))):
        z = np.load(design_path(static.root, i))
        ys.append(torch.as_tensor(z["S"][:, ::3]).reshape(-1, 6).double())
        ts.append(torch.as_tensor(z["t"]).double())
        c = z["coords"].astype(np.float64)
        lws.append(np.concatenate([static.load_weights(c, r)[static.load_nodes[r]] for r in static.rows]))
    y, t = torch.cat(ys), torch.cat(ts)
    ref = torch.as_tensor(static.ref_coords)
    e = static.edge_attr.double()
    return {"y_mean": y.mean(0).float(), "y_std": y.std(0).float(),
            "t_mean": t.mean().float(), "t_std": t.std().float(),
            "lw_scale": torch.tensor(float(np.mean(np.concatenate(lws)))),
            "ref_mean": ref.mean(0).float(), "ref_std": ref.std(0).float(),
            "edge_mean": e.mean(0).float(), "edge_std": e.std(0).clamp_min(1e-12).float()}


def control_directions(static, ids):
    """(C,K) per-row mean of the unit-normalized dKS/dt over training designs:
    the input-independent constant-direction predictor."""
    acc = np.zeros((len(static.rows), static.K))
    for i in ids:
        d = np.load(design_path(static.root, i))["dks"].astype(np.float64)
        acc += d / np.linalg.norm(d, axis=1, keepdims=True).clip(1e-30)
    return torch.as_tensor(acc / len(ids), dtype=torch.float32)


def combo_weights(n, C, seed):
    W = np.random.default_rng(seed).dirichlet(np.ones(C), size=max(n - 1, 0))
    return torch.as_tensor(np.concatenate([np.full((1, C), 1.0 / C), W]), dtype=torch.float32)


def validate(model, loader, bt, args, dev, ctrl):
    model.eval()
    rho = args.ks_rho_mpa
    keys = ("vm_rel", "ks_rel", "cos", "mass", "sign", "ctrl_cos", "ctrl_mass", "ctrl_sign")
    agg = {"single": {k: [] for k in keys + ("dS_rel",)}, "combo": {k: [] for k in keys}}
    rows = bt.s.rows
    C = len(rows)
    for nd, d in enumerate(loader):
        if nd >= args.val_max_designs:
            break
        t0 = d["t"].to(dev)
        lw, S_t = d["load_w"].to(dev), d["S"].to(dev)
        dS = d["dS"].to(dev)                                                     # (C,K,N,6) f16
        lin = getattr(model, "load_linear", False)
        # load_linear: one geometry forward, every row from one t (1,K) -- as in
        # training; 11 per-row forwards with retained graphs ran out of memory
        t = (t0[None] if lin else t0.expand(C, -1)).clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
            S_p = bt.forward_rows(model, t, lw[None])[0] if lin else bt.forward(model, t, lw, rows)
        # --- single loads ---
        ks_p = bt.ks(S_p, rho)
        if lin:
            g_p = torch.stack([torch.autograd.grad(ks_p[c], t, retain_graph=True)[0][0] for c in range(C)])
        else:
            g_p = torch.autograd.grad(ks_p.sum(), t, retain_graph=True)[0]       # (C,K)
        ks_t = d["ks"].to(dev)
        vp, vt = von_mises(S_p), von_mises(S_t)
        a = agg["single"]
        a["vm_rel"] += ((vp - vt).norm(dim=1) / vt.norm(dim=1)).tolist()
        a["ks_rel"] += ((ks_p - ks_t).abs() / ks_t).tolist()
        g_t = d["dks"].to(dev)
        for k, v in zip(("cos", "mass", "sign"), grad_metrics(g_p, g_t)):
            a[k] += v.tolist()
        for k, v in zip(("ctrl_cos", "ctrl_mass", "ctrl_sign"), grad_metrics(ctrl.to(dev), g_t)):
            a[k] += v.tolist()
        gen = torch.Generator(device=dev).manual_seed(int(d["id"]))
        dirs = [torch.randn(t.size(0), bt.s.K, device=dev, generator=gen) for _ in range(args.sob_dirs)]
        for v, Jv in zip(dirs, directional(S_p[None] if lin else S_p, t, dirs)):
            Jv = Jv[0] if lin else Jv                                            # (C,N,6); lin: one v for all rows
            Jt = torch.stack([torch.einsum("k,knc->nc", v[0 if lin else c], dS[c].float()) for c in range(C)])
            a["dS_rel"] += rel_l2(Jv.detach(), Jt).tolist()
        # --- combinations ---
        W = combo_weights(args.val_combos, C, int(d["id"])).to(dev)              # (n,C)
        Sc_p = torch.einsum("wc,cnk->wnk", W, S_p)
        ksc_p = bt.ks(Sc_p, rho)
        gc_p = torch.stack([torch.autograd.grad(ksc_p[w], t, retain_graph=True)[0].sum(0)
                            for w in range(len(W))])                             # (n,K)
        Sc_t = torch.einsum("wc,cnk->wnk", W, S_t).requires_grad_(True)
        ksc_t = bt.ks(Sc_t, rho)
        gS = torch.autograd.grad(ksc_t.sum(), Sc_t)[0]                           # (n,N,6) dKS_w/dS_w
        gc_t = torch.zeros(len(W), bt.s.K, device=dev)
        for c in range(C):
            gc_t += W[:, c:c + 1] * torch.einsum("wnk,jnk->wj", gS, dS[c].float())
        b = agg["combo"]
        vcp, vct = von_mises(Sc_p), von_mises(Sc_t.detach())
        b["vm_rel"] += ((vcp - vct).norm(dim=1) / vct.norm(dim=1)).tolist()
        b["ks_rel"] += ((ksc_p - ksc_t).abs() / ksc_t).tolist()
        for k, v in zip(("cos", "mass", "sign"), grad_metrics(gc_p, gc_t)):
            b[k] += v.tolist()
        for k, v in zip(("ctrl_cos", "ctrl_mass", "ctrl_sign"), grad_metrics(W @ ctrl.to(dev), gc_t)):
            b[k] += v.tolist()
    model.train()
    out = {}
    for part, a in agg.items():
        for k, v in a.items():
            v = np.asarray(v)
            if k.endswith("cos"):
                out[f"{part}/{k}_median"] = float(np.median(v))
            elif k.endswith("rel"):
                out[f"{part}/{k}_median"] = float(np.median(v))
                out[f"{part}/{k}_p90"] = float(np.percentile(v, 90))
            else:
                out[f"{part}/{k}"] = float(np.mean(v))
        out[f"{part}/n"] = len(a["cos"])
    return out


def fmt_val(v):
    lines = []
    for part in ("single", "combo"):
        g = lambda k: v.get(f"{part}/{k}", float("nan"))
        lines.append(
            f"  {part:6s} n={v[part + '/n']}: control cos {g('ctrl_cos_median'):+.3f} wrong-mass {g('ctrl_mass'):.1%} "
            f"sign {g('ctrl_sign'):.1%} | model cos {g('cos_median'):+.3f} wrong-mass {g('mass'):.1%} sign {g('sign'):.1%} "
            f"| KS rel {g('ks_rel_median'):.4f} (p90 {g('ks_rel_p90'):.4f}) vm rel {g('vm_rel_median'):.4f}"
            + (f" dS-dir rel {g('dS_rel_median'):.4f} (p90 {g('dS_rel_p90'):.4f})" if part == "single" else ""))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", default="data/cross4_11x11/dataset_dsig_2500_tmin003")
    p.add_argument("--run_dir", default="training/runs/cross4_dsig_sobolev_v1")
    p.add_argument("--ks_max_mpa", type=float, default=800.0, help="keep designs with KS <= this on every row")
    p.add_argument("--ks_rho_mpa", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--graphs_per_step", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--coarse_dim", type=int, default=128)
    p.add_argument("--cnn_depth", type=int, default=2)
    p.add_argument("--sob_weight", type=float, default=1.0)
    p.add_argument("--sob_dirs", type=int, default=2)
    p.add_argument("--sob_node_power", type=float, default=1.0,
                   help="Sobolev node weighting: a node's loss share ~ |Jv*_n|^(2 power); 1 = plain relative L2 (see rel_err)")
    p.add_argument("--mix_pool", type=int, default=0,
                   help="0: each step's cases come from one design (old order); N: mix designs, B different "
                        "designs per step from a pool of N (case_batches)")
    p.add_argument("--mirror_aug", type=float, default=0.0,
                   help="probability each training case is replaced by its exact y-mirror image (0 = off)")
    p.add_argument("--arch", default="hier", choices=["hier", "load_linear"],
                   help="hier: one forward per (design, row), load as an input feature; load_linear: stress "
                        "exactly linear in the load (utils/gnn_load_linear.py), all rows of a design in one forward")
    p.add_argument("--lin_dim", type=int, default=32, help="load_linear: channels per load case in the linear stream")
    p.add_argument("--lin_layers", type=int, default=2, help="load_linear: fine linear message-passing layers before "
                                                              "and after the tile U-Net (each)")
    p.add_argument("--gate_bound", type=float, default=0.0,
                   help="load_linear: gates 1 + a tanh(z/a) in (1-a, 1+a); 0 = unbounded 1 + z")
    p.add_argument("--tile_gate_bound", type=float, default=0.0,
                   help="load_linear: tile gates exp(b tanh(z/b)) in (e^-b, e^b); 0 = same form as --gate_bound")
    p.add_argument("--thick_power", type=float, default=0.0,
                   help="load_linear: stress x (t_node/0.2)^-n, n > 0 learned per component from this init (0 = off)")
    p.add_argument("--geo_spectral_norm", type=int, default=0,
                   help="load_linear: spectral norm on the geometry tile CNN convs + thickness encoder")
    p.add_argument("--full_readout", type=int, default=0,
                   help="load_linear: stress = R(geometry) u with a full per-node matrix instead of gate + shared matrix")
    p.add_argument("--lin_act", type=int, default=0,
                   help="load_linear DIAGNOSTIC: GELU in the load stream (breaks linearity)")
    p.add_argument("--rollback_k", type=float, default=0.0,
                   help="roll back when the 200-step mean pre-clip grad norm exceeds k x its healthy median: "
                        "restore the snapshot from 2-3k steps earlier and halve the lr (0 = off); see RollbackGuard")
    p.add_argument("--case_clip", type=float, default=0.0,
                   help="--case_batching: smoothly cap each case's gradient at k x the running median of "
                        "per-case gradient norms before averaging the batch (0 = off); see case_clipped_backward")
    p.add_argument("--case_batching", type=int, default=0,
                   help="load_linear: batch like the per-row model (--graphs_per_step cases from different designs, "
                        "--mix_pool) instead of whole designs; one geometry forward per case")
    p.add_argument("--designs_per_step", type=int, default=2, help="load_linear: whole designs (all rows) per step")
    p.add_argument("--load_pos_features", type=int, default=1,
                   help="1: every tile also gets its row offset from the loaded row and the loaded row (tile_in_dim 4)")
    p.add_argument("--val_designs", type=int, default=200)
    p.add_argument("--val_max_designs", type=int, default=60)
    p.add_argument("--val_combos", type=int, default=4, help="per design: equal weights + (n-1) Dirichlet draws")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", type=int, default=1, help="bf16 autocast")
    p.add_argument("--use_checkpoint", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0, help="stop after this many steps (smoke tests)")
    args = p.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    split_path = os.path.join(args.run_dir, "split.json")
    if os.path.exists(split_path):
        split = json.load(open(split_path))
    else:
        ids = available_designs(args.data)
        ks = design_ks(args.data, ids)
        kept = [i for i, k in zip(ids, ks) if k.max() <= args.ks_max_mpa]
        rng.shuffle(kept)
        split = {"val": sorted(kept[:args.val_designs]), "train": sorted(kept[args.val_designs:]),
                 "ks_max_mpa": args.ks_max_mpa, "available": len(ids)}
        json.dump(split, open(split_path, "w"))
    train_ids, val_ids = split["train"], split["val"]
    print(f"[data] {split['available']} designs, {len(train_ids) + len(val_ids)} with KS <= {split['ks_max_mpa']} MPa "
          f"on every row -> {len(train_ids)} train / {len(val_ids)} val (validating on {min(args.val_max_designs, len(val_ids))})",
          flush=True)

    static = DsigStatic(args.data)
    stats_path = os.path.join(args.run_dir, "stats.pt")
    if os.path.exists(stats_path):
        st = torch.load(stats_path)
    else:
        st = compute_stats(static, train_ids, rng)
        st["ctrl"] = control_directions(static, train_ids)
        torch.save(st, stats_path)
    ctrl = st.pop("ctrl")
    st = {k: v.to(dev) for k, v in st.items()}
    bt = Batcher(static, st, dev)

    if args.arch == "load_linear":
        assert args.mirror_aug == 0, "--mirror_aug is per (design, row) case; not wired for load_linear"
        model_kwargs = dict(arch="load_linear", node_in_dim=5, edge_in_dim=4, hidden_dim=args.hidden_dim,
                            tile_in_dim=1, num_pos_nodes=static.N, coarse_dim=args.coarse_dim,
                            cnn_depth=args.cnn_depth, lin_dim=args.lin_dim, lin_layers=args.lin_layers,
                            gate_bound=args.gate_bound, full_readout=bool(args.full_readout),
                            lin_act=bool(args.lin_act), tile_gate_bound=args.tile_gate_bound,
                            geo_spectral_norm=bool(args.geo_spectral_norm), thick_power=args.thick_power)
    else:
        model_kwargs = dict(node_in_dim=6, edge_in_dim=4, hidden_dim=args.hidden_dim,
                            tile_in_dim=4 if args.load_pos_features else 2, num_pos_nodes=static.N,
                            coarse_dim=args.coarse_dim, cnn_depth=args.cnn_depth, node_out_dim=6)
    model = build_model(model_kwargs, use_checkpoint=bool(args.use_checkpoint)).to(dev)
    model.set_tile_grid(static.NX, static.NY, 1)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ck_path = os.path.join(args.run_dir, "last.pt")
    start_epoch, step, best, prev_epoch_loss, lr_scale = 0, 0, -float("inf"), float("inf"), 1.0
    best_uni = -float("inf")
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, step, best = ck["epoch"], ck["step"], ck["best"]
        prev_epoch_loss, lr_scale = ck["prev_epoch_loss"], ck["lr_scale"]
        print(f"[resume] epoch {start_epoch}, step {step}, best combo cos {best:.4f}", flush=True)
    json.dump(vars(args), open(os.path.join(args.run_dir, "args.json"), "w"), indent=1)

    def save(path, epoch, with_opt=True, extra=None):
        torch.save({"model": model.state_dict(), **({"opt": opt.state_dict()} if with_opt else {}),
                    "epoch": epoch, "step": step, "best": best, "prev_epoch_loss": prev_epoch_loss,
                    "lr_scale": lr_scale, "args": vars(args), "model_kwargs": model_kwargs,
                    "stats": {k: v.cpu() for k, v in st.items()}, "ctrl": ctrl,
                    "ref_coords": static.ref_coords, **(extra or {})}, path + ".tmp")
        os.replace(path + ".tmp", path)

    mk_loader = lambda ds, shuffle, workers: torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=shuffle, num_workers=workers, collate_fn=lambda x: x[0],
        persistent_workers=True, prefetch_factor=2)
    train_loader = mk_loader(DesignDataset(static, train_ids), True, args.workers)
    val_loader = mk_loader(DesignDataset(static, val_ids[:args.val_max_designs]), False, 3)
    log = open(os.path.join(args.run_dir, "log.jsonl"), "a")
    rows = static.rows

    uni_path = os.path.join(args.data, "exact_uniform_0.2.npz")
    uni = dict(np.load(uni_path), t=0.2) if os.path.exists(uni_path) else None

    def run_val(epoch):
        t_v = time.time()
        v = validate(model, val_loader, bt, args, dev, ctrl)
        if uni is not None:
            u = uniform_check(model, bt, uni, args.ks_rho_mpa)
            v.update(u)
            print(f"  uniform t=0.2 (exact): centered cos mean {u['uniform/ccos_mean']:+.3f}, min {u['uniform/ccos_min']:+.3f}, "
                  f"rows 2-8 {u['uniform/ccos_mid_mean']:+.3f} (control ~+0.90); mirror rows 2-8 {u['uniform/mirror_mid_mean']:+.3f}; "
                  f"KS err max {u['uniform/ks_rel_max']:.1%}; per row {u['uniform/ccos_rows']}", flush=True)
        print(f"[val] epoch {epoch} step {step} ({time.time() - t_v:.0f} s):\n" + fmt_val(v), flush=True)
        if getattr(model, "thick_log_n", None) is not None:
            print("  thickness power n per stress component: "
                  + " ".join(f"{x:.3f}" for x in model.thick_log_n.detach().exp().tolist()), flush=True)
        log.write(json.dumps({"step": step, "epoch": epoch, "val": v}) + "\n"); log.flush()
        return v

    mirror = (args.mirror_aug, *mirror_maps(static), static.NY) if args.mirror_aug > 0 else None
    assert args.case_clip == 0 or args.case_batching or args.arch != "load_linear", "--case_clip needs case batches"
    guard = RollbackGuard(args.rollback_k) if args.rollback_k > 0 else None
    case_hist = collections.deque(maxlen=400)          # ~100 steps of per-case gradient norms
    model.train()
    for epoch in range(start_epoch, args.epochs):
        t_ep, ep_loss, ep_n, t_log = time.time(), 0.0, 0, time.time()
        run = {"loss": 0.0, "value": 0.0, "sob": 0.0, "r": 0.0, "gnorm": 0.0, "gmax": 0.0, "cmed": 0.0, "capped": 0.0,
               "n": 0}
        if args.arch == "load_linear" and not args.case_batching:
            batches = ((t_b, lw_b, S_b, dS_b, None) for t_b, lw_b, S_b, dS_b in
                       design_batches(train_loader, args.designs_per_step))
        else:
            batches = case_batches(train_loader, rows, args.graphs_per_step, args.mix_pool, rng, mirror)
        for t_b, lw_b, S_b, dS_b, rows_b in batches:
            B = t_b.size(0)
            lr = args.lr * lr_scale * min(1.0, (step + 1) / args.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            t = t_b.to(dev).requires_grad_(True)
            S_t = S_b.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
                lw = lw_b.to(dev, non_blocking=True)
                S_p = bt.forward_rows(model, t, lw) if rows_b is None else bt.forward(model, t, lw, rows_b)
            value = (((S_p - S_t) / st["y_std"]) ** 2).mean()
            loss, sob, rmean = value, torch.zeros((), device=dev), 0.0
            per_case = (((S_p - S_t) / st["y_std"]) ** 2).flatten(1).mean(1) if args.case_clip > 0 else None
            if args.sob_weight > 0:
                dirs = [torch.randn(B, static.K, device=dev) for _ in range(args.sob_dirs)]
                dS = dS_b.to(dev, non_blocking=True)                             # (B,[C,]K,N,6) f16
                for v, Jv in zip(dirs, directional(S_p, t, dirs)):
                    if rows_b is None:          # load_linear: every row of each design, one direction per design
                        Jt = torch.stack([torch.einsum("k,cknx->cnx", v[b], dS[b].float()) for b in range(B)])
                        Jv, Jt = Jv.flatten(0, 1), Jt.flatten(0, 1)
                    else:
                        Jt = torch.einsum("bk,bknc->bnc", v, dS.float())
                    r = rel_err(Jv, Jt, args.sob_node_power)
                    sob = sob + (r / (1 + r / 2)).mean() / args.sob_dirs
                    if per_case is not None:
                        per_case = per_case + args.sob_weight * (r / (1 + r / 2)) / args.sob_dirs
                    rmean += float(r.detach().mean()) / args.sob_dirs
                loss = value + args.sob_weight * sob
            opt.zero_grad(set_to_none=True)
            split = None
            if (step + 1) % 100 == 0 and args.sob_weight > 0:
                # the logging step: backward each loss term separately for the split
                # below, then sum, so the update is the same as loss.backward()
                value.backward(retain_graph=True)
                gv = {n: p_.grad.clone() for n, p_ in model.named_parameters() if p_.grad is not None}
                opt.zero_grad(set_to_none=True)
                (args.sob_weight * sob).backward(retain_graph=per_case is not None)
                split = grad_split(model, gv)
                for n, p_ in model.named_parameters():
                    if n in gv:
                        p_.grad = gv[n] if p_.grad is None else p_.grad + gv[n]
            if per_case is not None:            # replaces the plain sum (the split above is still pre-cap)
                opt.zero_grad(set_to_none=True)
                cn, cs = case_clipped_backward(model, per_case, case_hist, args.case_clip, step >= args.warmup_steps)
                run["cmed"] += float(np.median(cn)); run["capped"] += float(np.mean(np.array(cs) < 0.5))
            elif split is None:
                loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            step += 1
            lv = float(loss)
            ep_loss += lv; ep_n += 1
            for k, x in (("loss", lv), ("value", float(value)), ("sob", float(sob)), ("r", rmean), ("gnorm", gnorm)):
                run[k] += x
            run["gmax"] = max(run["gmax"], gnorm)
            if guard is not None:
                if step % guard.every == 0 and step >= args.warmup_steps and float(np.mean(guard.recent or [0])) < 2 * float(
                        np.median(guard.hist or [float("inf")])):
                    guard.snapshot(step, model, opt)
                if guard.check(gnorm, step, step >= args.warmup_steps):
                    m200, base = float(np.mean(guard.recent)), float(np.median(guard.hist))
                    step0 = guard.restore(model, opt)
                    lr_scale *= 0.5
                    ep_loss, ep_n = 0.0, 0          # the epoch loss (plateau rule) counts only post-rollback steps
                    print(f"[rollback] step {step}: 200-step mean grad {m200:.2f} > {args.rollback_k:g} x healthy median "
                          f"{base:.2f}; restored the step-{step0} snapshot, lr x0.5 -> {args.lr * lr_scale:.2e} "
                          f"(rollback #{guard.count})", flush=True)
                    log.write(json.dumps({"step": step, "rollback_to": step0, "grad_mean": m200, "grad_base": base,
                                          "lr_scale": lr_scale}) + "\n"); log.flush()
            run["n"] += 1
            if step % 100 == 0:
                dt, n = time.time() - t_log, run["n"]
                print(f"[train] epoch {epoch} step {step} loss {run['loss'] / n:.4f} (value {run['value'] / n:.4f}, "
                      f"sob {run['sob'] / n:.4f}, dS-dir rel {run['r'] / n:.3f}) lr {lr:.2e} "
                      f"grad {run['gnorm'] / n:.2f} (max {run['gmax']:.1f}) "
                      + (f"case-grad median {run['cmed'] / n:.2f}, {100 * run['capped'] / n:.1f}% cut >2x "
                         if args.case_clip > 0 else "") + f"{n / dt:.2f} steps/s "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
                rec = {"step": step, "epoch": epoch, **{k: run[k] / n for k in ("loss", "value", "sob", "r", "gnorm", "cmed",
                                                                                   "capped")},
                       "gmax": run["gmax"], "lr": lr}
                if split is not None:
                    print("  grad split (pre-clip, this step): " + ", ".join(f"{k} {v:.2f}" for k, v in split.items()),
                          flush=True)
                    rec["grad_split"] = split
                log.write(json.dumps(rec) + "\n"); log.flush()
                t_log, run = time.time(), {k: 0.0 for k in run} | {"n": 0}
            if args.max_steps and step >= args.max_steps:
                run_val(epoch)
                print(f"[smoke] stopping at step {step}; peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
                      flush=True)
                return
        ep_loss /= max(ep_n, 1)
        if ep_loss > prev_epoch_loss * 0.995:
            lr_scale *= 0.5
            print(f"[lr] epoch loss {ep_loss:.4f} did not improve on {prev_epoch_loss:.4f}: lr x0.5", flush=True)
        lr_scale *= 0.99
        prev_epoch_loss = min(prev_epoch_loss, ep_loss)
        print(f"[epoch] {epoch} done: loss {ep_loss:.4f}, {ep_n} steps, {(time.time() - t_ep) / 60:.1f} min", flush=True)
        v = run_val(epoch)
        save(os.path.join(args.run_dir, f"epoch_{epoch:03d}.pt"), epoch + 1, with_opt=False, extra={"val": v})
        if "uniform/ccos_mean" in v and v["uniform/ccos_mean"] > best_uni:
            best_uni = v["uniform/ccos_mean"]
            save(os.path.join(args.run_dir, "best_uniform.pt"), epoch + 1, with_opt=False, extra={"val": v})
        if v["combo/cos_median"] > best:
            best = v["combo/cos_median"]
            save(os.path.join(args.run_dir, "best.pt"), epoch + 1, with_opt=False, extra={"val": v})
        save(ck_path, epoch + 1)


if __name__ == "__main__":
    main()
