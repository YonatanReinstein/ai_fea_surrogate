"""Five-solve pass/fail on the axis gradient: does it get d(KS)/d(uniform
thickening) right?

Bumping every tile's thickness by the same amount is the one design change
whose derivative can be measured directly, with two static solves, and whose
predicted value needs no per-tile attribution at all -- by linearity it is
just the sum of the per-tile derivatives the contraction produces. So

    sum_tiles dKS/dt   vs   (KS(t+h) - KS(t-h)) / 2h

is a complete, cheap audit of the nodal gradient: 3 axis solves plus 2 static
ones, against the 1680 remesh solves a full per-tile FD map costs. It cannot
localise an error, but it cannot be fooled by one either -- a gradient that
fails this is wrong no matter how well any subset of tiles correlates.

Measured on hollow_cube at uniform t=0.30 (ratio of the ccx sum to the
directly measured truth):
    hollow_cube_mid (48 tiles)  3,3,3   0.877   usable
    hollow_cube    (840 tiles)  3,3,3  -2.790   wrong sign, 2.8x too large
The per-tile remesh FD sums to 1.0026x and 1.0001x of truth on those same two
cases, so the tile decomposition itself is exact and the defect is in the
nodal dKS/dX -- which is why this test is worth running before any long
optimisation on a new geometry or mesh.
"""
import argparse
import importlib
import json
import os
import time

from core.IritModel import IritCModel
from core.geometric_jacobian import compute_thickness_jacobian
from core.tile_sensitivity import assemble_tile_thickness_sensitivity_axis
from evaluators.fd_reference import _build_mesh
from evaluators.validate_axis_gradient import force_element_type


def run(geom, mesh_res, element_type, fd_frac, rho_mpa, ccx_path, ccx_axis_path,
        nproc, mesh_threads):
    base = f"data/{geom}"
    exe = os.path.join(base, "CAD_model", "model")
    dims_t = json.load(open(os.path.join(base, "CAD_model", "dims.json")))
    mat = json.load(open(os.path.join(base, "CAD_model", "material_properties.json")))
    young, poisson = mat["young_modulus"], mat["poisson_ratio"]
    bc = importlib.import_module(f"data.{geom}.boundary_conditions")
    fixed = getattr(bc, "fixed_dims", lambda: {})()
    U, V, W = (int(t) for t in mesh_res.split(","))
    dims = {k: v["default"] for k, v in dims_t.items()}
    keys = list(dims_t)
    force_element_type(element_type)

    def ks_at(delta, tag):
        d = {k: v + delta for k, v in dims.items()}
        n, e, _ = IritCModel(exe, dims_dict=d, fixed_dims=fixed,
                             mesh_threads=mesh_threads).create_mesh(U=U, V=V, W=W)
        m = _build_mesh(n, e, bc.anchor_condition, bc.force_pattern)
        t0 = time.time()
        m.solve_ccx(young, poisson, ccx_path=ccx_path, nproc=nproc)
        ks = m.get_aggregated_stress(rho=rho_mpa * 1e-6)
        print(f"    {tag}: KS={ks/1e6:.4f} MPa ({time.time()-t0:.0f}s)", flush=True)
        return ks

    step = fd_frac * (dims_t[keys[0]]["max"] - dims_t[keys[0]]["min"])
    print(f"  {geom} @ {mesh_res} {element_type}, uniform t={dims[keys[0]]:.2f} "
          f"+- {step:.4f}", flush=True)
    truth = (ks_at(+step, "t+h") - ks_at(-step, "t-h")) / (2 * step)

    t0 = time.time()
    dX_dt, nodes0, elements0, e2t = compute_thickness_jacobian(
        exe, dims_t, U, V, W, fixed_dims=fixed, mesh_threads=mesh_threads)
    mesh = _build_mesh(nodes0, elements0, bc.anchor_condition, bc.force_pattern)
    ks0, grad = mesh.solve_ccx_axis_gradient(
        young, poisson, sorted(nodes0), stress_ks_rho_mpa=rho_mpa,
        ccx_path=ccx_axis_path, nproc=nproc, interior=True,
        element_type=element_type)
    NX, NY, NZ = bc.tile_grid()
    total = sum(assemble_tile_thickness_sensitivity_axis(
        dX_dt, grad, elements0, e2t, NX, NY, NZ, keys).values())
    print(f"    axis gradient: KS={ks0/1e6:.4f} MPa, {len(nodes0)} nodes, "
          f"3 solves in {time.time()-t0:.0f}s", flush=True)
    print(f"    d(KS)/d(uniform t) truth {truth:+.6g} | ccx sum {total:+.6g} "
          f"| RATIO {total/truth:+.4f}\n", flush=True)
    return {"geometry": geom, "mesh": mesh_res, "element_type": element_type,
            "ks_mpa": ks0 / 1e6, "nodes": len(nodes0),
            "truth": truth, "ccx_sum": total, "ratio": total / truth}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="hollow_cube")
    p.add_argument("--mesh_res", type=str, default="3,3,3",
                   help="Comma-separated U,V,W; repeat with ';' for a sweep, "
                        "e.g. '3,3,3;5,3,3;7,3,3'.")
    p.add_argument("--element_type", type=str, default="C3D8",
                   choices=["C3D8", "C3D8I", "C3D20", "C3D20R"])
    p.add_argument("--fd_frac", type=float, default=0.02)
    p.add_argument("--stress_ks_rho_mpa", type=float, default=0.15)
    p.add_argument("--nproc", type=int, default=16)
    p.add_argument("--mesh_threads", type=int, default=4)
    p.add_argument("--ccx_path", type=str, default="ccx")
    p.add_argument("--ccx_axis_path", type=str,
                   default=os.path.expanduser("~/opt/calculix/root/usr/bin/ccx_axis"))
    p.add_argument("--out", type=str, default="evaluators/reports/sum_rule.json")
    args = p.parse_args()

    rows = [run(args.geometry, mr, args.element_type, args.fd_frac,
                args.stress_ks_rho_mpa, args.ccx_path, args.ccx_axis_path,
                args.nproc, args.mesh_threads)
            for mr in args.mesh_res.split(";")]
    print(f"{'mesh':>10} {'nodes':>8} {'KS MPa':>10} {'truth':>13} "
          f"{'ccx sum':>13} {'ratio':>9}")
    for r in rows:
        print(f"{r['mesh']:>10} {r['nodes']:>8} {r['ks_mpa']:>10.3f} "
              f"{r['truth']:>+13.4g} {r['ccx_sum']:>+13.4g} {r['ratio']:>+9.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
