"""Full discrete node-coordinate gradient, brute-forced as ground truth.

Measures the full discrete gradient

    dKS/dX[n][j]   for every moving node n and every direction j in {x,y,z}

by central-differencing ks_stress against each node coordinate directly -- 6
solves per moving node -- and contracts it with the tile-parameter ->
node-displacement map:

    dKS/dt_tile = sum_n sum_j  dKS/dX[n][j] * dX[n][j]/dt_tile

That contraction is, to first order, EXACTLY the derivative of the same
discrete KS the remesh finite difference measures: it keeps the tangential
and interior-node motion, and needs no normal and no surface area.

Stock CCX cannot produce this gradient -- TYPE=COORDINATE gives one scalar per
node along NORM only -- but the patched ccx from tools/build_ccx_axis.sh can,
in three solves total regardless of node count: see
evaluators/validate_axis_gradient.py, which reproduces this script's brute
force to a median magnitude ratio of 0.999 (hollow_cube_one at mesh 5,5,5).
This script remains the independent ground truth for that one.

Usage:
    python -m evaluators.validate_node_gradient --geometry hollow_cube_small \
        --tiles d1,d5,d11
"""
import argparse
import importlib
import json
import math
import os
import time

from core.IritModel import IritCModel
from evaluators.fd_reference import _build_mesh, _solve_ks


def _mesh_nodes(exe_path, dims, U, V, W, fixed_dims, mesh_threads):
    m = IritCModel(exe_path, dims_dict=dims, fixed_dims=fixed_dims,
                   mesh_threads=mesh_threads)
    return m.create_mesh(U=U, V=V, W=W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="hollow_cube_small")
    p.add_argument("--tiles", type=str, default="d1")
    p.add_argument("--mesh_res", type=str, default=None)
    p.add_argument("--node_step", type=float, default=3e-4,
                   help="Central-difference step on a node coordinate.")
    p.add_argument("--jac_eps", type=float, default=1e-4)
    p.add_argument("--fd_frac", type=float, default=0.02)
    p.add_argument("--total_force", type=float, default=None)
    p.add_argument("--stress_ks_rho_mpa", type=float, default=0.15)
    p.add_argument("--ccx_path", type=str, default="ccx")
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--mesh_threads", type=int, default=1)
    p.add_argument("--out", type=str, default="node_gradient_result.json")
    args = p.parse_args()

    base = f"data/{args.geometry}"
    exe = os.path.join(base, "CAD_model", "model")
    dims_t = json.load(open(os.path.join(base, "CAD_model", "dims.json")))
    mat = json.load(open(os.path.join(base, "CAD_model", "material_properties.json")))
    young, poisson = mat["young_modulus"], mat["poisson_ratio"]
    bc = importlib.import_module(f"data.{args.geometry}.boundary_conditions")
    U, V, W = bc.mesh_resolution()
    if args.mesh_res:
        U, V, W = (int(x) for x in args.mesh_res.split(","))
    fixed = getattr(bc, "fixed_dims", lambda: {})()
    dims = {k: v["default"] for k, v in dims_t.items()}
    rho = args.stress_ks_rho_mpa

    nodes0, elements0, _ = _mesh_nodes(exe, dims, U, V, W, fixed, args.mesh_threads)

    def ks_of(mesh):
        mesh.solve_ccx(young, poisson, ccx_path=args.ccx_path, nproc=args.nproc)
        return mesh.get_aggregated_stress(rho=rho * 1e-6)

    def ks_with_moved_node(nid, j, delta):
        # BCs assigned from the NOMINAL coordinates, then frozen, then the node
        # is moved -- so a node on the anchor/force face cannot change its own
        # boundary condition just by moving.
        m = _build_mesh(nodes0, elements0, bc.anchor_condition, bc.force_pattern,
                        total_force=args.total_force)
        m.nodes[nid].coords[j] += delta
        return ks_of(m)

    results = []
    for key in [k.strip() for k in args.tiles.split(",")]:
        t0 = time.time()
        # Single-tile geometric Jacobian: perturb ONLY this tile, so there is no
        # node->tile assignment step to get wrong.
        dp = dict(dims); dp[key] = dims[key] + args.jac_eps
        nodes1, elements1, _ = _mesh_nodes(exe, dp, U, V, W, fixed, args.mesh_threads)
        if nodes0.keys() != nodes1.keys() or elements0.keys() != elements1.keys():
            raise RuntimeError("mesh topology changed under the single-tile perturbation")
        dX = {n: tuple((nodes1[n][k] - nodes0[n][k]) / args.jac_eps for k in range(3))
              for n in nodes0}
        moving = [n for n, v in dX.items() if any(abs(c) > 1e-9 for c in v)]

        h = args.node_step
        pred, ngrad = 0.0, {}
        for nid in moving:
            g = []
            for j in range(3):
                if abs(dX[nid][j]) <= 1e-9:
                    g.append(0.0)          # direction contributes nothing anyway
                    continue
                g.append((ks_with_moved_node(nid, j, +h)
                          - ks_with_moved_node(nid, j, -h)) / (2.0 * h))
            ngrad[nid] = g
            pred += sum(g[j] * dX[nid][j] for j in range(3))

        # Ground truth: remesh the whole geometry with this tile perturbed.
        lo, hi = dims_t[key]["min"], dims_t[key]["max"]
        step = args.fd_frac * (hi - lo)
        ksp, ksm = (_solve_ks(exe, {**dims, key: dims[key] + s * step}, U, V, W, fixed,
                              args.mesh_threads, bc.anchor_condition, bc.force_pattern,
                              young, poisson, args.ccx_path, args.nproc, rho,
                              1.0, args.total_force)
                    for s in (+1, -1))
        fd = (ksp - ksm) / (2.0 * step)

        nsolves = sum(1 for n in moving for j in range(3) if abs(dX[n][j]) > 1e-9) * 2
        row = {"tile": key, "contracted_node_gradient": pred, "remesh_fd": fd,
               "ratio": pred / fd if fd else None, "moving_nodes": len(moving),
               "ccx_solves": nsolves, "seconds": time.time() - t0}
        results.append(row)
        print(f"{key:>5}  node-grad contraction={pred:>12.5g}  remesh FD={fd:>12.5g}  "
              f"ratio={row['ratio']:.4f}  ({len(moving)} nodes, {nsolves} solves, "
              f"{row['seconds']:.0f}s)", flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
