"""Per-load-row diagnostics for a train_dsig checkpoint on held-out designs.

For every load row r: KS relative error, and dKS/dt against the exact label --
cosine, CENTERED cosine (Pearson: each design's mean tile gradient removed, so a
shared sign or offset scores nothing; what is left is the tile-to-tile pattern
the optimizer actually ranks tiles by), wrong-way mass -- each next to the
constant-direction control (per-row mean unit dKS/dt over training designs).

Mirror test: cross4 is mirror-symmetric in y, so design t loaded at row r is the
mirror image of design t' (tiles iy -> NY-1-iy) loaded at row NY-1-r. The exact
dKS/dt therefore satisfies g(t', NY-1-r) = mirror(g(t, r)). The model is scored
on how far it breaks that: centered cosine between the two predictions.
Mirrored coordinates come from reflecting the design's own nodes and matching
them to the reference mesh (nearest node).

    python -m analysis.probe_dsig_rows --ckpt training/runs/cross4_dsig_sobolev_v1/epoch_026.pt
"""
import argparse
import json

import numpy as np
import torch
from scipy.spatial import cKDTree

from training.dsig_data import DsigStatic, design_path
from training.train_dsig import Batcher, build_model


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def ccos(a, b):
    return cos(a - a.mean(), b - b.mean())


def wrong(p, t):
    return float(np.abs(t)[np.sign(p) != np.sign(t)].sum() / np.abs(t).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/cross4_11x11/dataset_dsig_2500_tmin003")
    p.add_argument("--designs", type=int, default=60)
    args = p.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    s = DsigStatic(args.data)
    s.ref_coords = ck["ref_coords"]
    model = build_model(ck["model_kwargs"]).to(dev)
    model.set_tile_grid(s.NX, s.NY, 1)
    model.float()
    model.load_state_dict(ck["model"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    st = {k: v.to(dev).float() for k, v in ck["stats"].items()}
    bt = Batcher(s, st, dev)
    ctrl = ck["ctrl"].numpy()
    rho = float(ck["args"]["ks_rho_mpa"])
    NX, NY, C = s.NX, s.NY, len(s.rows)
    # dims order is ix*NY + iy: mirror iy -> NY-1-iy
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    tmir = (ix * NY + (NY - 1 - iy)).reshape(-1)
    refl = lambda X: X * np.array([1.0, -1.0, 1.0]) + np.array([0.0, NY * s.ty, 0.0])
    # one-to-one node matching: the averaged reference mesh is only nearly
    # symmetric, so a few nearest-node lookups collide; keep each target's
    # closest claimant and assign the leftovers exactly (small Hungarian)
    from scipy.optimize import linear_sum_assignment
    M = refl(s.ref_coords)
    d0, nmir = cKDTree(s.ref_coords).query(M)
    best = {}
    for j in np.argsort(d0):
        best.setdefault(nmir[j], j)
    keep = np.zeros(s.N, bool); keep[list(best.values())] = True
    src = np.where(~keep)[0]
    free = np.setdiff1d(np.arange(s.N), nmir[keep])
    r_, c_ = linear_sum_assignment(np.linalg.norm(M[src][:, None] - s.ref_coords[free][None], axis=2))
    nmir[src[r_]] = free[c_]
    assert len(np.unique(nmir)) == s.N
    dist = np.linalg.norm(refl(s.ref_coords) - s.ref_coords[nmir], axis=1)
    print(f"[mirror] node match: max dist {dist.max():.4f}, median {np.median(dist):.5f}", flush=True)

    val = json.load(open(f"{ck['args']['run_dir']}/split.json"))["val"][:args.designs]

    def predict(t, X):
        lw = torch.as_tensor(np.stack([s.load_weights(X, r) for r in s.rows]), dtype=torch.float32, device=dev)
        tt = torch.as_tensor(t, dtype=torch.float32, device=dev).expand(C, -1).clone().requires_grad_(True)
        S = bt.forward(model, tt, lw, s.rows)
        ks = bt.ks(S, rho)
        g, = torch.autograd.grad(ks.sum(), tt)
        return ks.detach().cpu().numpy(), g.cpu().numpy()

    rows = {r: {k: [] for k in ("ks", "cos", "ccos", "wrong", "c_cos", "c_ccos", "c_wrong", "mir")} for r in s.rows}
    for i in val:
        z = np.load(design_path(args.data, i))
        X, t = z["coords"].astype(np.float64), z["t"]
        ks_p, g_p = predict(t, X)
        Xm = np.empty_like(X)
        Xm[nmir] = refl(X)                      # node nmir[j] of t' sits at the mirror of node j of t
        _, g_m = predict(t[tmir], Xm)
        for c, r in enumerate(s.rows):
            gt, gp = z["dks"][c].astype(np.float64), g_p[c].astype(np.float64)
            d = rows[r]
            d["ks"].append(abs(ks_p[c] / z["ks"][c] - 1))
            d["cos"].append(cos(gp, gt)); d["ccos"].append(ccos(gp, gt)); d["wrong"].append(wrong(gp, gt))
            d["c_cos"].append(cos(ctrl[c], gt)); d["c_ccos"].append(ccos(ctrl[c], gt)); d["c_wrong"].append(wrong(ctrl[c], gt))
            cm = s.rows.index(NY - 1 - r)
            d["mir"].append(ccos(gp, g_m[cm][tmir]))
    print(f"\n{len(val)} held-out designs, {args.ckpt}")
    print("row | KS err med | cos med (ctrl) | CENTERED cos med (ctrl) | wrong-way mass (ctrl) | mirror centered cos med")
    allk = {k: [] for k in rows[0]}
    for r in s.rows:
        d = {k: np.array(v) for k, v in rows[r].items()}
        for k in allk:
            allk[k] += list(d[k])
        print(f"{r:3d} | {np.median(d['ks']):9.1%} | {np.median(d['cos']):+.3f} ({np.median(d['c_cos']):+.3f}) | "
              f"{np.median(d['ccos']):+.3f} ({np.median(d['c_ccos']):+.3f}) | {d['wrong'].mean():6.1%} ({d['c_wrong'].mean():5.1%}) | "
              f"{np.median(d['mir']):+.3f} (p10 {np.percentile(d['mir'], 10):+.3f})")
    d = {k: np.array(v) for k, v in allk.items()}
    print(f"all | {np.median(d['ks']):9.1%} | {np.median(d['cos']):+.3f} ({np.median(d['c_cos']):+.3f}) | "
          f"{np.median(d['ccos']):+.3f} ({np.median(d['c_ccos']):+.3f}) | {d['wrong'].mean():6.1%} ({d['c_wrong'].mean():5.1%}) | "
          f"{np.median(d['mir']):+.3f} (p10 {np.percentile(d['mir'], 10):+.3f})")


if __name__ == "__main__":
    main()
