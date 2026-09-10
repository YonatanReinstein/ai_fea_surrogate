"""Per-tile sensitivities from the exact shape adjoint, scored against the
remesh-FD map that evaluators/tile_gradient_map.py wrote.

One factorization and two back-substitutions produce every tile's
d(ks_stress)/d(thickness), at any tile count -- the same O(1) promise ccx's
*SENSITIVITY makes, but with d(K_e)/d(x) autodiffed instead of finite-
differenced, which is the term that carries ccx's error here.

    python -m evaluators.adjoint_tile_gradient --geometry hollow_cube \
        --mesh_res 3,3,3 --npz evaluators/reports/tilegrad_hollow_cube_333_C3D8.npz
"""
import argparse
import importlib
import json
import os
import time

import numpy as np

from core.IritModel import IritCModel
from core.fem_adjoint import ShapeAdjoint
from core.geometric_jacobian import compute_thickness_jacobian
from evaluators.fd_reference import _build_mesh


def _spearman(x, y):
    n = len(x)
    rx = np.empty(n); ry = np.empty(n)
    rx[np.argsort(x)] = np.arange(n); ry[np.argsort(y)] = np.arange(n)
    return float(1 - 6 * np.sum((rx - ry) ** 2) / (n * (n * n - 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="hollow_cube")
    p.add_argument("--mesh_res", type=str, default="3,3,3")
    p.add_argument("--stress_ks_rho_mpa", type=float, default=0.15)
    p.add_argument("--plain_c3d8", dest="incompatible", action="store_false",
                   help="Drop the Wilson incompatible modes and run plain "
                        "C3D8. Off by default: the incompatible modes now "
                        "reproduce ccx's own C3D8I to 1.7e-5%%, which is the "
                        "element the rest of the pipeline solves on. Plain "
                        "C3D8 agrees at a uniform design and reads ~30%% low "
                        "on thinned ones, so it is a reference, not a choice.")
    p.set_defaults(incompatible=True)
    p.add_argument("--mesh_threads", type=int, default=4)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--signal_frac", type=float, default=0.10)
    p.add_argument("--npz", type=str, default="",
                   help="tile_gradient_map .npz to score against.")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    base = f"data/{args.geometry}"
    exe = os.path.join(base, "CAD_model", "model")
    dims_t = json.load(open(os.path.join(base, "CAD_model", "dims.json")))
    mat = json.load(open(os.path.join(base, "CAD_model", "material_properties.json")))
    bc = importlib.import_module(f"data.{args.geometry}.boundary_conditions")
    fixed = getattr(bc, "fixed_dims", lambda: {})()
    U, V, W = (int(t) for t in args.mesh_res.split(","))
    keys = list(dims_t)

    t0 = time.time()
    dX_dt, nodes0, elements0, e2t = compute_thickness_jacobian(
        exe, dims_t, U, V, W, fixed_dims=fixed, mesh_threads=args.mesh_threads)
    t_jac = time.time() - t0

    m = _build_mesh(nodes0, elements0, bc.anchor_condition, bc.force_pattern)
    anchored = [n.id for n in m.all_nodes() if n.anchored]
    fnodes = [n.id for n in m.all_nodes() if any(n.forces)]
    total = [sum(n.forces[j] for n in m.all_nodes()) for j in range(3)]

    sa = ShapeAdjoint(nodes0, elements0, anchored, fnodes, total,
                      mat["young_modulus"], mat["poisson_ratio"],
                      incompatible=args.incompatible, chunk=args.chunk)
    sa.assert_tie_is_fixed(dX_dt)
    print(f"{args.geometry} @ {args.mesh_res}: {sa.N} nodes, {len(sa.conn)} elements, "
          f"{sa.n_red} reduced DOF, min detJ {sa.min_detJ:.4g}; jacobian {t_jac:.0f}s",
          flush=True)

    t0 = time.time(); sa.solve(); t_solve = time.time() - t0
    ks = sa.ks(args.stress_ks_rho_mpa)
    print(f"  factorize + solve {t_solve:.1f}s -> KS = {ks/1e6:.4f} MPa", flush=True)

    t0 = time.time()
    NX, NY, NZ = bc.tile_grid()
    tile = sa.tile_gradient(dX_dt, e2t, keys, NX, NY, NZ)
    print(f"  adjoint + contraction {time.time()-t0:.1f}s -> {len(tile)} tile "
          f"sensitivities", flush=True)

    adj = np.array([tile[k] for k in keys])
    if args.npz:
        z = np.load(args.npz, allow_pickle=True)
        assert list(z["keys"]) == keys
        fd, ccx = z["fd"], z["ccx"]
        cut = args.signal_frac * np.max(np.abs(fd))
        print(f"\n  {'':22s} {'sign':>10s} {'spearman':>10s} {'ratio med':>11s}")
        for label, mask in (("all tiles", np.ones(len(fd), bool)),
                            (f"|FD| >= {args.signal_frac:g}x max", np.abs(fd) >= cut)):
            for name, v in (("ccx axis", ccx), ("shape adjoint", adj)):
                s = int(np.sum(v[mask] * fd[mask] > 0))
                r = np.sort(v[mask] / fd[mask])
                print(f"  {label:>22s} {name:>16s} {s:>4d}/{int(mask.sum()):<5d} "
                      f"{_spearman(fd[mask], v[mask]):>+10.4f} {np.median(r):>11.4f}")
        print(f"\n  sum of per-tile: FD {fd.sum():+.6g}  ccx {ccx.sum():+.6g}  "
              f"adjoint {adj.sum():+.6g}   (adjoint/FD {adj.sum()/fd.sum():.4f})")

    out = args.out or (f"evaluators/reports/adjoint_{args.geometry}_"
                       f"{U}{V}{W}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, keys=np.array(keys), adjoint=adj, ks=ks)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
