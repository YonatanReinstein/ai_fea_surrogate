"""Global-axis nodal stress gradient from CalculiX, validated against remesh FD.

Stock ccx's *DESIGNVARIABLES,TYPE=COORDINATE reports ONE scalar per design
node: the derivative of the objective along that node's surface NORMAL, divided
by the node's tributary area. The normal is a mean of adjacent face normals, so
at an edge or corner it points somewhere with no relation to the tile parameter
we differentiate against, and no design-node set can steer it onto a coordinate
axis (see *FILTER,EDGEPRESERVATION -- it only restricts which faces vote).

tools/build_ccx_axis.sh builds a patched ccx honouring three environment
variables, each a no-op when unset (the binary is bit-identical to stock ccx
with none of them set -- verified on hollow_cube_one):

    CCX_DESI_AXIS=1|2|3   replace every design direction by the global x/y/z
                          axis instead of the surface normal
    CCX_NO_AREA_NORM=1    skip distributesens.f's division by weightformgrad,
                          so DFDN is d(objective)/d(nodal displacement) rather
                          than a per-unit-area density
    CCX_DESI_STEP=<len>   override distmin (the smallest element edge length),
                          which is the step of the forward difference ccx takes
                          on the element stiffness
    CCX_DESI_INTERIOR=1   keep interior nodes as design variables (--interior)

Three runs (axis 1, 2, 3) give the full discrete gradient

    dKS/dX[n][j]   for every design node n and direction j in {x,y,z}

which is what evaluators/validate_node_gradient.py brute-forces with 6 solves
per node. Contracted with the tile-parameter -> node-displacement map it gives
dKS/dt_tile with no normal, no surface area and no region logic:

    dKS/dt_tile = sum_n sum_j  dKS/dX[n][j] * dX[n][j]/dt_tile

Accuracy is governed by how many elements resolve a wall, NOT by the tile count:
at 3,3,3 per tile (one element through each wall) the gradient is wrong by ~5x,
at 5,5,5 it matches brute-force FD to a median ratio of 0.999. Use --mesh_res.

On hollow_cube_mid at 5,5,5 with C3D8, over the 26 of 48 tiles whose |remesh FD|
is above 0.1% of max (the rest are FD noise), per-tile ratio to remesh FD:
    stock normal projection : median 1.007  min 0.084  max 2.645  spread 31.4x
    axis, surface only      : median 1.036  min 0.634  max 1.530  spread 2.41x
    axis + --interior       : median 1.000  min 0.981  max 1.094  spread 1.12x
Stock ccx only accepts free-surface nodes as design variables, because a shape
sensitivity is undefined without a surface normal. Under CCX_DESI_AXIS there is
no normal, so --interior lifts that restriction -- and it matters more than
anything else: on hollow_cube_one/d1 at mesh 5,5,5 the interior nodes are 16.6%
of surface magnitude but contribute -9.0% of the tile derivative (opposite sign,
cancelling part of the surface sum), moving the contraction from 1.0755 to
0.9858 of the remesh FD. Always pass --interior unless comparing to stock.

Usage:
    python -m evaluators.validate_axis_gradient --geometry hollow_cube_mid \
        --mesh_res 5,5,5 --element_type C3D8 --tiles all
"""
import argparse
import importlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import core.mesh as core_mesh
from core.IritModel import IritCModel
from core.tile_sensitivity import free_surface_nodes
from evaluators.fd_reference import _build_mesh
from utils.hex20 import is_quadratic

AXES = ("x", "y", "z")


def force_element_type(elt):
    """C3D8I's incompatible modes are statically condensed, and ccx's
    semi-analytical sensitivity differentiates that badly: on hollow_cube_one
    the direction cosine against brute-force FD is -0.68 median with C3D8I and
    +0.90 with plain C3D8 on the same mesh. Let the caller pick."""
    original = core_mesh.Mesh._ccx_base_deck

    def patched(self, young, poisson, element_type="C3D8I"):
        lines, pilot, total = original(self, young, poisson,
                                       element_type=element_type)
        return [l.replace("TYPE=" + element_type, "TYPE=" + elt) for l in lines], pilot, total

    core_mesh.Mesh._ccx_base_deck = patched


def _mesh_nodes(exe, dims, U, V, W, fixed, threads):
    return IritCModel(exe, dims_dict=dims, fixed_dims=fixed,
                      mesh_threads=threads).create_mesh(U=U, V=V, W=W)


def axis_gradient(make_mesh, young, poisson, design_ids, rho_mpa, ccx_path,
                  nproc, step=None, interior=False):
    """Run the patched ccx once per global axis; return (ks, {nid: [gx,gy,gz]})."""
    grad, ks0 = {}, None
    for j in range(3):
        os.environ["CCX_DESI_AXIS"] = str(j + 1)
        os.environ["CCX_NO_AREA_NORM"] = "1"
        if interior:
            os.environ["CCX_DESI_INTERIOR"] = "1"
        if step:
            os.environ["CCX_DESI_STEP"] = repr(step)
        try:
            ks, dfdn, _ = make_mesh().solve_ccx_sensitivity(
                young, poisson, design_ids, stress_ks_rho_mpa=rho_mpa,
                ccx_path=ccx_path, nproc=nproc)
        finally:
            for k in ("CCX_DESI_AXIS", "CCX_NO_AREA_NORM", "CCX_DESI_STEP",
                      "CCX_DESI_INTERIOR"):
                os.environ.pop(k, None)
        ks0 = ks
        for nid, v in dfdn.items():
            grad.setdefault(nid, [0.0, 0.0, 0.0])[j] = v
    return ks0, grad


def _q(sorted_vals, frac):
    return sorted_vals[min(len(sorted_vals) - 1, int(frac * len(sorted_vals)))]


def _tile_metrics(rows, label):
    """Sign agreement, ratio spread and Spearman of the axis contraction
    against the remesh FD, over whichever subset of tiles is handed in."""
    if not rows:
        print(f"  [{label}] no tiles")
        return
    sign_ok = sum(1 for t in rows if t["axis_contraction"] * t["remesh_fd"] > 0)
    rs = sorted(t["ratio"] for t in rows)
    print(f"  [{label}] n={len(rows)}")
    print(f"    sign agreement : {sign_ok}/{len(rows)}")
    print(f"    ratio          : median {_q(rs,.5):.4f}  p10 {_q(rs,.1):.4f}  "
          f"p90 {_q(rs,.9):.4f}  min {rs[0]:.4f}  max {rs[-1]:.4f}")
    xs = [t["remesh_fd"] for t in rows]
    ys = [t["axis_contraction"] for t in rows]
    rx = sorted(range(len(xs)), key=lambda i: xs[i])
    ry = sorted(range(len(ys)), key=lambda i: ys[i])
    px = [0]*len(xs); py = [0]*len(ys)
    for r, i in enumerate(rx): px[i] = r
    for r, i in enumerate(ry): py[i] = r
    n = len(xs)
    if n > 2:
        d2 = sum((px[i]-py[i])**2 for i in range(n))
        print(f"    spearman rho   : {1 - 6*d2/(n*(n*n-1)):.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="hollow_cube_one")
    p.add_argument("--mesh_res", type=str, default=None)
    p.add_argument("--element_type", type=str, default="C3D8",
                   choices=["C3D8", "C3D8I", "C3D20", "C3D20R"],
                   help="C3D20[R] builds a second-order mesh (utils/hex20.py) "
                        "instead of monkeypatching the deck's element name: the "
                        "midside nodes have to exist before the deck is written, "
                        "and every mesh this script builds -- nominal, node-FD, "
                        "remesh-FD -- then gets them consistently.")
    p.add_argument("--node_step", type=float, default=1e-2,
                   help="Central-difference step on a node coordinate. Too small "
                        "and the FD truth is dominated by .frd output precision: "
                        "at mesh 9,9,9 the ratio's 90th percentile falls from "
                        "26.9 at 3e-4 to 1.78 at 1e-2 while the median holds at 1.0.")
    p.add_argument("--max_nodes", type=int, default=40,
                   help="Brute-force at most this many design nodes (0 = all).")
    p.add_argument("--tiles", type=str, default="",
                   help="Comma-separated dims keys, or 'all', to contract "
                        "against dX/dt and compare to remesh FD.")
    p.add_argument("--tiles_from", type=str, default="",
                   help="Reuse the tile list from a previous run's --out JSON, "
                        "so two arms are compared on exactly the same tiles.")
    p.add_argument("--signal_frac", type=float, default=0.10,
                   help="A second metrics block is printed over the tiles whose "
                        "|remesh FD| is at least this fraction of the largest. On "
                        "a large grid most tiles barely move a GLOBAL KS max, so "
                        "their FD is solver noise and an all-tiles Spearman "
                        "mostly ranks noise -- on hollow_cube's 840 tiles |FD| "
                        "spans 1.8 to 5.4e6. The filter is applied on the FD "
                        "(truth) side, so it cannot flatter the method.")
    p.add_argument("--jac_eps", type=float, default=1e-4)
    p.add_argument("--fd_frac", type=float, default=0.02)
    p.add_argument("--total_force", type=float, default=None)
    p.add_argument("--stress_ks_rho_mpa", type=float, default=0.15)
    p.add_argument("--ccx_path", type=str, default="ccx",
                   help="Stock ccx, used for the finite-difference solves.")
    p.add_argument("--ccx_axis_path", type=str,
                   default=os.path.expanduser("~/opt/calculix/root/usr/bin/ccx_axis"))
    p.add_argument("--corners_only", action="store_true",
                   help="Second-order meshes only: offer just the element "
                        "CORNER nodes as design variables, not the midside "
                        "ones. A diagnostic for whether ccx's semi-analytical "
                        "perturbation of a single midside node -- which bends "
                        "an element edge, a deformation a remesh never applies "
                        "-- is what breaks the tile contraction at C3D20.")
    p.add_argument("--interior", action="store_true",
                   help="Keep interior nodes as design variables too. Costs "
                        "nothing extra (still 3 solves) and removes the largest "
                        "error term -- see the module docstring.")
    p.add_argument("--desi_step", type=float, default=None,
                   help="CCX_DESI_STEP override (default: smallest edge length).")
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--mesh_threads", type=int, default=8)
    p.add_argument("--out", type=str, default="axis_gradient_result.json")
    args = p.parse_args()

    if is_quadratic(args.element_type):
        os.environ["FEA_ELEMENT_TYPE"] = args.element_type
    else:
        force_element_type(args.element_type)

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
    print(f"{args.geometry}: {len(nodes0)} nodes, {len(elements0)} {args.element_type} "
          f"elements, mesh {U},{V},{W}, {len(dims_t)} tiles", flush=True)

    def fresh():
        return _build_mesh(nodes0, elements0, bc.anchor_condition, bc.force_pattern,
                           total_force=args.total_force)

    surf = sorted(free_surface_nodes(elements0))
    design = sorted(nodes0) if args.interior else surf
    if args.corners_only:
        corners = {n for nl in elements0.values() for n in list(nl)[:8]}
        design = [n for n in design if n in corners]
    t0 = time.time()
    ks0, grad = axis_gradient(fresh, young, poisson, design, rho,
                              args.ccx_axis_path, args.nproc, args.desi_step,
                              interior=args.interior)
    # A node CCX rejected from the design set (SPC'd, MPC'd, or not on enough
    # design faces -- see getdesiinfo3d.f) is exactly 0 in all three runs.
    live = [n for n in sorted(grad) if any(v != 0.0 for v in grad[n])]
    print(f"3 axis solves in {time.time()-t0:.0f}s: KS={ks0:.6g}, {len(live)} design "
          f"nodes accepted of {len(design)} offered "
          f"({'all nodes' if args.interior else 'free-surface only'}; "
          f"{len(design)-len(live)} rejected by CCX)\n", flush=True)

    def ks_moved(job):
        nid, j, sign = job
        m = fresh()
        m.nodes[nid].coords[j] += sign * args.node_step
        m.solve_ccx(young, poisson, ccx_path=args.ccx_path, nproc=1)
        return m.get_aggregated_stress(rho=rho * 1e-6)

    # ---- per-node check against brute-force FD ---------------------------
    rows = []
    sel = live
    if args.max_nodes and len(live) > args.max_nodes:
        sel = live[::max(1, len(live) // args.max_nodes)][:args.max_nodes]
    if sel:
        t0 = time.time()
        jobs = [(n, j, s) for n in sel for j in range(3) for s in (+1, -1)]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            vals = list(ex.map(ks_moved, jobs))
        h = args.node_step
        for i, nid in enumerate(sel):
            fd = [(vals[6*i+2*j] - vals[6*i+2*j+1]) / (2*h) for j in range(3)]
            g = grad[nid]
            nf = math.sqrt(sum(v*v for v in fd))
            ng = math.sqrt(sum(v*v for v in g))
            rows.append({"node": nid, "ccx": g, "fd": fd, "fd_mag": nf,
                         "cos": (sum(a*b for a, b in zip(fd, g))/(nf*ng)) if nf and ng else None,
                         "mag_ratio": (ng/nf) if nf else None})
        print(f"per-node: {len(sel)} nodes, {len(jobs)} FD solves, {time.time()-t0:.0f}s")
        ok = [r for r in rows if r["cos"] is not None]
        big = sorted(ok, key=lambda r: -r["fd_mag"])[:max(1, len(ok)//2)]
        for label, grp in (("all sampled nodes", ok),
                           ("top half by |dKS/dX|", big)):
            cs = sorted(r["cos"] for r in grp)
            rs = sorted(r["mag_ratio"] for r in grp)
            print(f"  {label:>22} (n={len(grp):>3}): cos median {_q(cs,.5):+.5f} "
                  f"p10 {_q(cs,.1):+.5f} | ratio median {_q(rs,.5):.4f} "
                  f"p10 {_q(rs,.1):.4f} p90 {_q(rs,.9):.4f}")
        print(flush=True)

    # ---- tile selection --------------------------------------------------
    # "top:N" nets the N tiles this arm's own contraction thinks are largest.
    # That is a coarse net, not the verdict: which of them actually carry
    # signal is decided afterwards by --signal_frac on the FD side, and the
    # other arm is run over the same list via --tiles_from, so a selection the
    # method got wrong costs coverage, never correctness.
    ts = args.tiles.strip()
    if args.tiles_from:
        keys = [t["tile"] for t in json.load(open(args.tiles_from))["tiles"]]
        print(f"tile list reused from {args.tiles_from}: {len(keys)} tiles", flush=True)
    elif ts == "all":
        keys = list(dims_t)
    elif ts.startswith("top:"):
        from core.geometric_jacobian import compute_thickness_jacobian
        from core.tile_sensitivity import assemble_tile_thickness_sensitivity_axis
        t0 = time.time()
        # ONE extra remesh gives dX/dt for every tile at once -- tiles share no
        # thickness-sensitive node -- instead of the per-tile jac() below.
        dX_all, _, e0, e2t = compute_thickness_jacobian(
            exe, dims_t, U, V, W, fixed_dims=fixed, mesh_threads=args.mesh_threads)
        NX, NY, NZ = bc.tile_grid()
        dKS = assemble_tile_thickness_sensitivity_axis(
            dX_all, grad, e0, e2t, NX, NY, NZ, list(dims_t))
        ranked = sorted(dKS.items(), key=lambda kv: -abs(kv[1]))
        keys = [k for k, _ in ranked[:int(ts[4:])]]
        print(f"top:{int(ts[4:])} of {len(dKS)} tiles by |contraction| "
              f"({time.time()-t0:.0f}s): |dKS/dt| {abs(ranked[0][1]):.4g} "
              f"(max) .. {abs(ranked[len(keys)-1][1]):.4g} (cut), "
              f"{abs(ranked[-1][1]):.4g} (smallest tile)", flush=True)
    else:
        keys = [k.strip() for k in ts.split(",") if k.strip()]
    tiles = []
    if keys:
        t0 = time.time()

        def remesh_ks(job):
            key, sign = job
            lo, hi = dims_t[key]["min"], dims_t[key]["max"]
            step = args.fd_frac * (hi - lo)
            d = {**dims, key: dims[key] + sign * step}
            n1, e1, _ = _mesh_nodes(exe, d, U, V, W, fixed, 1)
            m = _build_mesh(n1, e1, bc.anchor_condition, bc.force_pattern,
                            total_force=args.total_force)
            m.solve_ccx(young, poisson, ccx_path=args.ccx_path, nproc=1)
            return m.get_aggregated_stress(rho=rho * 1e-6), 2 * step

        def jac(key):
            d = {**dims, key: dims[key] + args.jac_eps}
            n1, _, _ = _mesh_nodes(exe, d, U, V, W, fixed, 1)
            return {n: [(n1[n][k] - nodes0[n][k]) / args.jac_eps for k in range(3)]
                    for n in nodes0}

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fdv = list(ex.map(remesh_ks, [(k, s) for k in keys for s in (+1, -1)]))
            jacs = list(ex.map(jac, keys))
        for i, key in enumerate(keys):
            dX = jacs[i]
            pred = sum(sum(grad[n][j] * dX[n][j] for j in range(3))
                       for n in live if n in dX)
            fd = (fdv[2*i][0] - fdv[2*i+1][0]) / fdv[2*i][1]
            tiles.append({"tile": key, "axis_contraction": pred, "remesh_fd": fd,
                          "ratio": (pred/fd) if fd else None})
        print(f"tiles: {len(keys)} tiles, {2*len(keys)} remesh solves, {time.time()-t0:.0f}s")
        good = [t for t in tiles if t["ratio"] is not None]
        _tile_metrics(good, "all tiles")
        if good:
            cut = args.signal_frac * max(abs(t["remesh_fd"]) for t in good)
            signal = [t for t in good if abs(t["remesh_fd"]) >= cut]
            if len(signal) < len(good):
                _tile_metrics(signal, f"|FD| >= {args.signal_frac:g} x max")

    json.dump({"geometry": args.geometry, "mesh": [U, V, W],
               "element_type": args.element_type, "ks": ks0,
               "design_nodes": len(live), "nodes": rows, "tiles": tiles},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
