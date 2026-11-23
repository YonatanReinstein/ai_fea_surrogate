"""Uniform-design check for a train_dsig checkpoint: solve t = T uniform exactly
(S, dS/dt for all 11 load rows, the dataset builder's solver) and score the model
per row -- KS error, cosine, centered cosine, wrong-way mass, and the mirror
symmetry error of both (a uniform design is mirror-symmetric, so the row-r
gradient must equal the mirrored row-(NY-1-r) gradient). Also reports how far
the uniform design sits from the dataset's designs.

    python -m analysis.probe_dsig_uniform --ckpt ... --t 0.2
"""
import argparse
import importlib
import json

import numpy as np
import torch

from analysis.probe_dsig_rows import ccos, cos, wrong
from training.dsig_data import DsigStatic, available_designs, design_path
from training.train_dsig import Batcher, build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/cross4_11x11/dataset_dsig_2500_tmin003")
    p.add_argument("--t", type=float, default=0.2)
    args = p.parse_args()
    dev = torch.device("cuda")
    s = DsigStatic(args.data)
    NX, NY, C, K = s.NX, s.NY, len(s.rows), s.K
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    tmir = (ix * NY + (NY - 1 - iy)).reshape(-1)

    # distance of the uniform design to the dataset
    ids = available_designs(args.data)
    T = np.array([np.load(design_path(args.data, i))["t"] for i in ids])
    tu = np.full(K, args.t)
    d_u = np.sort(np.linalg.norm(T - tu, axis=1))
    sub = T[np.random.default_rng(0).choice(len(T), 300, replace=False)]
    nn = [np.sort(np.linalg.norm(T - x, axis=1))[1] for x in sub]
    print(f"[distribution] uniform {args.t}: nearest dataset design at L2 {d_u[0]:.3f} (5th nearest {d_u[4]:.3f}); "
          f"dataset designs' own nearest-neighbour L2: median {np.median(nn):.3f}, p90 {np.percentile(nn, 90):.3f}; "
          f"dataset per-design std of t: median {np.median(T.std(1)):.3f}", flush=True)

    # exact
    from datasets.build_cross4_dsig_dataset import solve_design
    bc = importlib.import_module("data.cross4_11x11.boundary_conditions")
    keys = list(json.load(open("data/cross4_11x11/CAD_model/dims.json")))
    import os
    cache = os.path.join(args.data, f"exact_uniform_{args.t:g}.npz")       # one exact solve, reused
    if os.path.exists(cache):
        out = dict(np.load(cache))
    else:
        out, _ = solve_design({k: args.t for k in keys}, bc, s.rows, float(s.cfg["load_N"]), 0.25, workers=8)
        np.savez(cache, coords=out["coords"], ks=out["ks"], dks=out["dks"])
    X, ks_t, g_t = out["coords"], out["ks"], out["dks"]
    print(f"[exact] mirror check of the exact gradient: centered cos row0 vs mirrored row10 "
          f"{ccos(g_t[0], g_t[-1][tmir]):+.4f}, row5 vs its own mirror {ccos(g_t[5], g_t[5][tmir]):+.4f}", flush=True)

    for path in args.ckpt.split(","):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        s.ref_coords = ck["ref_coords"]
        model = build_model(ck["model_kwargs"]).to(dev)
        model.set_tile_grid(NX, NY, 1)
        model.float()
        model.load_state_dict(ck["model"])
        model.eval()
        for q in model.parameters():
            q.requires_grad_(False)
        st = {k: v.to(dev).float() for k, v in ck["stats"].items()}
        bt = Batcher(s, st, dev)
        lw = torch.as_tensor(np.stack([s.load_weights(X, r) for r in s.rows]), dtype=torch.float32, device=dev)
        tt = torch.full((C, K), args.t, dtype=torch.float32, device=dev, requires_grad=True)
        S = bt.forward(model, tt, lw, s.rows)
        ks = bt.ks(S, 0.25)
        g, = torch.autograd.grad(ks.sum(), tt)
        g, ks = g.cpu().numpy().astype(np.float64), ks.detach().cpu().numpy()
        ctrl = ck["ctrl"].numpy()
        print(f"\n{path}  (uniform t = {args.t})")
        print("row | exact KS | KS err | cos (ctrl) | CENTERED cos (ctrl) | wrong-way | model mirror ccos")
        for c, r in enumerate(s.rows):
            print(f"{r:3d} | {ks_t[c]:8.1f} | {ks[c] / ks_t[c] - 1:+6.1%} | {cos(g[c], g_t[c]):+.3f} ({cos(ctrl[c], g_t[c]):+.3f}) | "
                  f"{ccos(g[c], g_t[c]):+.3f} ({ccos(ctrl[c], g_t[c]):+.3f}) | {wrong(g[c], g_t[c]):6.1%} | "
                  f"{ccos(g[c], g[C - 1 - c][tmir]):+.3f}")


if __name__ == "__main__":
    main()
