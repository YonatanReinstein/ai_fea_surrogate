"""Per-tile d(ks_stress)/d(thickness) over a whole tile grid, computed BOTH
ways, on one design point -- the ccx axis sensitivity that topopt actually
runs, and the remesh central-difference finite differences that are its
ground truth -- saved as an .npz and rendered as heat maps over the grid.

The point is to see WHERE on the grid the two agree, which a single Spearman
number over all tiles cannot show. On hollow_cube's 840 tiles |FD| spans ~7
decades (one tile in 840 barely moves a GLOBAL KS max), so most tiles sit at
solver noise and any all-tiles correlation is mostly ranking noise; the maps
separate "the method is wrong here" from "there is nothing to be right about
here", and the summary repeats every statistic over the signal-carrying
subset for the same reason.

Cost is one CAD remesh + one static solve per FD sample, i.e. 2 per tile,
against 3 solves total for the whole axis gradient -- 1680 vs 3 on
hollow_cube. That asymmetry is the entire reason the axis route exists; this
script pays the 1680 once, deliberately, to audit it.

Tiles are placed on the map by their own MESH CENTROID (averaged over the
elements elem_to_tile assigns to them), never by decoding a tile index into
a row and a column. The two tile orderings in this codebase -- IRIT's
X-fastest elem_to_tile and dims.json's X-outermost ShellThicknesses -- have
silently transposed features before (see core/tile_sensitivity.py's
tile_idx_to_dims_idx), and a heat map drawn from a wrong decode looks
perfectly plausible.

    python -m evaluators.tile_gradient_map --geometry hollow_cube \
        --mesh_res 3,3,3 --element_type C3D8 --workers 44
    python -m evaluators.tile_gradient_map --from_npz <path>   # re-render only
"""
import argparse
import importlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.IritModel import IritCModel
from core.geometric_jacobian import compute_thickness_jacobian
from core.tile_sensitivity import (assemble_tile_thickness_sensitivity_axis,
                                    node_to_tile_map, tile_index_to_dims_key)
from evaluators.fd_reference import _build_mesh
from evaluators.validate_axis_gradient import force_element_type


def _spearman(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    rx = np.empty(n); ry = np.empty(n)
    rx[np.argsort(x)] = np.arange(n)
    ry[np.argsort(y)] = np.arange(n)
    return float(1 - 6 * np.sum((rx - ry) ** 2) / (n * (n * n - 1)))


def compute(args):
    base = f"data/{args.geometry}"
    exe = os.path.join(base, "CAD_model", "model")
    dims_t = json.load(open(os.path.join(base, "CAD_model", "dims.json")))
    mat = json.load(open(os.path.join(base, "CAD_model", "material_properties.json")))
    young, poisson = mat["young_modulus"], mat["poisson_ratio"]
    bc = importlib.import_module(f"data.{args.geometry}.boundary_conditions")
    U, V, W = bc.mesh_resolution()
    if args.mesh_res:
        U, V, W = (int(t) for t in args.mesh_res.split(","))
    fixed = getattr(bc, "fixed_dims", lambda: {})()
    dims = {k: v["default"] for k, v in dims_t.items()}
    keys = list(dims_t)
    rho = args.stress_ks_rho_mpa

    # Both sides on ONE element. The production pipeline deliberately splits
    # them (reported stress on C3D8I, gradient on C3D8, see
    # ccx_evaluator.compute_topopt_gradient), but an audit of the gradient
    # must difference the same discretization it differentiates, or the
    # comparison carries that mismatch too.
    force_element_type(args.element_type)

    # ---- geometric Jacobian + the axis gradient: 1 remesh + 3 solves ------
    t0 = time.time()
    dX_dt, nodes0, elements0, elem_to_tile = compute_thickness_jacobian(
        exe, dims_t, U, V, W, fixed_dims=fixed, mesh_threads=args.mesh_threads)
    t_jac = time.time() - t0
    print(f"{args.geometry} @ {U},{V},{W} {args.element_type}: {len(nodes0)} nodes, "
          f"{len(elements0)} elements, {len(keys)} tiles; jacobian {t_jac:.0f}s",
          flush=True)

    def fresh():
        return _build_mesh(nodes0, elements0, bc.anchor_condition, bc.force_pattern)

    t0 = time.time()
    ks0, grad = fresh().solve_ccx_axis_gradient(
        young, poisson, sorted(nodes0), stress_ks_rho_mpa=rho,
        ccx_path=args.ccx_axis_path, nproc=args.gradient_nproc,
        interior=True, element_type=args.element_type)
    NX, NY, NZ = bc.tile_grid()
    dKS_ccx = assemble_tile_thickness_sensitivity_axis(
        dX_dt, grad, elements0, elem_to_tile, NX, NY, NZ, keys)
    print(f"axis gradient: KS={ks0/1e6:.4f} MPa, 3 solves in {time.time()-t0:.0f}s",
          flush=True)

    # ---- tile centroids, straight off the mesh ---------------------------
    n2t = node_to_tile_map(elements0, elem_to_tile)
    acc = {}
    for nid, t in n2t.items():
        a = acc.setdefault(t, [0.0, 0.0, 0.0, 0])
        c = nodes0[nid]
        a[0] += c[0]; a[1] += c[1]; a[2] += c[2]; a[3] += 1
    centroid = {tile_index_to_dims_key(t, NX, NY, NZ, keys):
                (a[0] / a[3], a[1] / a[3], a[2] / a[3]) for t, a in acc.items()}

    # ---- remesh central-difference FD: 2 solves per tile ------------------
    span = {k: dims_t[k]["max"] - dims_t[k]["min"] for k in keys}

    def fd_one(job):
        key, sign = job
        step = args.fd_frac * span[key]
        d = {**dims, key: dims[key] + sign * step}
        n1, e1, _ = IritCModel(exe, dims_dict=d, fixed_dims=fixed,
                               mesh_threads=1).create_mesh(U=U, V=V, W=W)
        m = _build_mesh(n1, e1, bc.anchor_condition, bc.force_pattern)
        m.solve_ccx(young, poisson, ccx_path=args.ccx_path, nproc=1)
        return m.get_aggregated_stress(rho=rho * 1e-6)

    t0 = time.time()
    jobs = [(k, s) for k in keys for s in (+1, -1)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        vals = list(ex.map(fd_one, jobs))
    dKS_fd = {k: (vals[2 * i] - vals[2 * i + 1]) / (2 * args.fd_frac * span[k])
              for i, k in enumerate(keys)}
    print(f"remesh FD: {len(jobs)} solves in {time.time()-t0:.0f}s", flush=True)

    out = args.out or f"evaluators/reports/tilegrad_{args.geometry}_" \
                      f"{U}{V}{W}_{args.element_type}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out,
             keys=np.array(keys),
             ccx=np.array([dKS_ccx[k] for k in keys]),
             fd=np.array([dKS_fd[k] for k in keys]),
             xyz=np.array([centroid[k] for k in keys]),
             meta=np.array([args.geometry, f"{U},{V},{W}", args.element_type,
                            f"{ks0/1e6:.4f}", str(len(nodes0)), str(len(elements0)),
                            f"{args.fd_frac}", f"{rho}"]))
    print(f"wrote {out}", flush=True)
    return out


def summarize(ccx, fd, signal_frac):
    lines = []
    cut = signal_frac * np.max(np.abs(fd))
    for label, m in (("all tiles", np.ones(len(fd), bool)),
                     (f"|FD| >= {signal_frac:g} x max", np.abs(fd) >= cut)):
        c, f = ccx[m], fd[m]
        if len(c) < 3:
            continue
        sign = int(np.sum(c * f > 0))
        nz = f != 0
        ratio = np.sort(c[nz] / f[nz])
        lines.append(
            f"[{label}] n={len(c)}  sign {sign}/{len(c)} ({100*sign/len(c):.0f}%)  "
            f"spearman {_spearman(f, c):+.4f}  ratio median {np.median(ratio):.4f} "
            f"(p10 {ratio[int(.1*len(ratio))]:.4f}, p90 {ratio[int(.9*len(ratio))]:.4f})")
    return lines


def render(npz_path, signal_frac, adjoint_npz=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    z = np.load(npz_path, allow_pickle=True)
    ccx, fd, xyz = z["ccx"], z["fd"], z["xyz"]
    geom, mesh, elt, ks, nn, ne, fdfrac, rho = [str(v) for v in z["meta"]]

    for line in summarize(ccx, fd, signal_frac):
        print("  " + line, flush=True)

    # The grid is one tile deep, so the two spanning axes are whichever pair
    # of coordinates actually varies -- found from the centroids rather than
    # assumed, so this works for any outline.
    spread = xyz.max(axis=0) - xyz.min(axis=0)
    ax_h, ax_v = np.argsort(spread)[::-1][:2]
    # A tile's centroid does not land exactly on the lattice: nodes are shared
    # between neighbours and node_to_tile_map gives each shared node to the
    # first element that claims it, so a tile's node set is slightly
    # asymmetric and its centroid jitters by a fraction of a tile pitch.
    # Binning on exact coordinates therefore splits one row into two. Cluster
    # instead: sorted gaps are strongly bimodal (jitter << pitch), so a
    # threshold at a quarter of the largest gap separates them cleanly.
    def lattice(vals):
        u = np.unique(vals)
        if len(u) == 1:
            return np.array([u[0]]), np.zeros(len(vals), int)
        gaps = np.diff(u)
        tol = 0.25 * gaps.max()
        centres, cur = [], [u[0]]
        for a, b in zip(u[:-1], u[1:]):
            if b - a > tol:
                centres.append(np.mean(cur)); cur = []
            cur.append(b)
        centres.append(np.mean(cur))
        centres = np.array(centres)
        idx = np.argmin(np.abs(vals[:, None] - centres[None, :]), axis=1)
        return centres, idx

    H, ih = lattice(xyz[:, ax_h])
    Vc, iv = lattice(xyz[:, ax_v])
    iv = len(Vc) - 1 - iv                      # row 0 at the top

    def grid(vals):
        g = np.full((len(Vc), len(H)), np.nan)
        g[iv, ih] = vals
        return g

    # Signed log magnitude: |dKS/dt| spans ~7 decades here, so a linear scale
    # paints every tile but the few largest the same colour. Blue = negative
    # (thickening this tile LOWERS peak stress), red = positive.
    span_dec = 7.0
    def slog(v):
        top = np.log10(np.nanmax(np.abs(v))) if np.nanmax(np.abs(v)) > 0 else 0.0
        m = np.abs(v)
        with np.errstate(divide="ignore"):
            t = np.clip((np.log10(m) - (top - span_dec)) / span_dec, 0, 1)
        t = np.where(m <= 0, 0.0, t)
        return np.sign(v) * t

    cmap = LinearSegmentedColormap.from_list(
        "dks", ["#184f95", "#5598e7", "#cde2fb", "#eceff1",
                "#fbd5d1", "#e06e68", "#8c2828"])
    norm = Normalize(-1, 1)

    adj = None
    if adjoint_npz:
        za = np.load(adjoint_npz, allow_pickle=True)
        assert list(za["keys"]) == list(z["keys"])
        adj = za["adjoint"]
        print("  shape adjoint:", flush=True)
        for line in summarize(adj, fd, signal_frac):
            print("    " + line, flush=True)

    g_fd, g_ccx = grid(slog(fd)), grid(slog(ccx))
    agree = np.sign(fd) * np.sign(ccx)
    cut = signal_frac * np.max(np.abs(fd))
    agree = np.where(np.abs(fd) >= cut, agree, np.nan)   # grey out the noise
    g_ag = grid(agree)

    # One panel is (cols x rows) tiles drawn square; size the figure to that
    # so a 40x21 grid and an 8x6 one both come out without acres of margin.
    cell = min(0.42, 11.0 / max(len(H), 1), 6.0 / max(len(Vc), 1))
    npanel = 3 + (1 if adj is not None else 0)
    fig, axes = plt.subplots(npanel, 1, constrained_layout=True, figsize=(
        max(8.0, cell * len(H) + 2.2),
        max(7.5, npanel * (cell * len(Vc) + 0.9) + 1.0)))
    ext = [H.min(), H.max(), Vc.min(), Vc.max()]
    panels = [(axes[0], g_fd, f"remesh central-difference FD  ({2*len(fd)} solves)"),
               (axes[1], g_ccx, "ccx axis sensitivity  (3 solves, semi-analytical dK/dx)")]
    if adj is not None:
        panels.append((axes[2], grid(slog(adj)),
                       "exact shape adjoint  (1 factorization, autodiffed dK/dx)"))
    for ax, g, title in panels:
        im = ax.imshow(g, cmap=cmap, norm=norm, extent=ext, aspect="equal",
                       interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                          ticks=[-1, -0.5, 0, 0.5, 1])
        cb.ax.set_yticklabels(["−max", "−", "0", "+", "+max"])

    ok = np.nansum(g_ag > 0); bad = np.nansum(g_ag < 0)
    axes = list(axes); last = axes[-1]
    im = last.imshow(g_ag, cmap=LinearSegmentedColormap.from_list(
        "ag", ["#8c2828", "#eceff1", "#184f95"]), vmin=-1, vmax=1,
        extent=ext, aspect="equal", interpolation="nearest")
    last.set_title(
        f"ccx sign agreement, {int(ok+bad)} tiles above the noise floor "
        f"(|FD| >= {signal_frac:g} x max):  blue agrees ({int(ok)}), "
        f"red disagrees ({int(bad)}); blank = no signal to be right about",
        loc="left", fontsize=11)
    fig.colorbar(im, ax=last, fraction=0.02, pad=0.01, ticks=[-1, 1]).ax \
        .set_yticklabels(["opposite", "same"])

    fig.suptitle(
        f"per-tile d(ks_stress)/d(thickness)  ·  {geom}, mesh {mesh}, {elt}\n"
        f"{nn} nodes / {ne} elements  ·  KS {ks} MPa at uniform t=0.30  ·  "
        f"FD step {float(fdfrac)*100:.0f}% of range  ·  "
        f"signed log over {span_dec:.0f} decades, blue = thickening lowers peak stress",
        fontsize=11)
    png = npz_path.replace(".npz", "_with_adjoint.png" if adj is not None else ".png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}", flush=True)

    # scatter: the same data without the spatial layout
    fig2, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    m = (fd != 0) & (ccx != 0)
    ax.scatter(np.abs(fd[m]) * np.sign(fd[m]), np.abs(ccx[m]) * np.sign(ccx[m]),
               s=9, alpha=0.55, color="#2a78d6", edgecolors="none")
    lim = max(np.max(np.abs(fd[m])), np.max(np.abs(ccx[m])))
    ax.plot([-lim, lim], [-lim, lim], color="#828a90", lw=1, ls="--")
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_xlabel("remesh FD  d(ks)/dt"); ax.set_ylabel("ccx axis  d(ks)/dt")
    ax.set_title("per tile, symlog; dashed = exact agreement", fontsize=10)
    png2 = npz_path.replace(".npz", "_scatter.png")
    fig2.savefig(png2, dpi=150)
    print(f"wrote {png2}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="hollow_cube")
    p.add_argument("--mesh_res", type=str, default="3,3,3")
    p.add_argument("--element_type", type=str, default="C3D8",
                   choices=["C3D8", "C3D8I"])
    p.add_argument("--fd_frac", type=float, default=0.02,
                   help="Central-difference step as a fraction of each dim's "
                        "[min,max] range -- the same convention "
                        "validate_axis_gradient.py uses.")
    p.add_argument("--signal_frac", type=float, default=0.10)
    p.add_argument("--stress_ks_rho_mpa", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=44)
    p.add_argument("--gradient_nproc", type=int, default=16)
    p.add_argument("--mesh_threads", type=int, default=4)
    p.add_argument("--ccx_path", type=str, default="ccx")
    p.add_argument("--ccx_axis_path", type=str,
                   default=os.path.expanduser("~/opt/calculix/root/usr/bin/ccx_axis"))
    p.add_argument("--out", type=str, default="")
    p.add_argument("--adjoint_npz", type=str, default="",
                   help="An evaluators.adjoint_tile_gradient .npz to add as a "
                        "third map panel and score alongside.")
    p.add_argument("--from_npz", type=str, default="",
                   help="Skip the solves and re-render an earlier run's .npz.")
    args = p.parse_args()

    npz = args.from_npz or compute(args)
    render(npz, args.signal_frac, args.adjoint_npz)


if __name__ == "__main__":
    main()
