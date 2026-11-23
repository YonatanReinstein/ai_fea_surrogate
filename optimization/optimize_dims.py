import json
import os
from pathlib import Path
from evaluators.factory import get_evaluator


def _load_tile_neighbours(geometry_name, dims_keys):
    """(load_key, [neighbour keys]) for the geometry's loaded tile: the free-end
    tile in boundary_conditions.load_row() (IRIT tile index (NX-1) + NX*row),
    and its in-grid face neighbours."""
    import importlib
    from core.tile_sensitivity import tile_index_to_dims_key
    bc = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
    NX, NY, NZ = bc.tile_grid(None)
    if NZ != 1:
        raise ValueError("load-tile tie implemented for planar (NZ = 1) grids only")
    ix, iy = NX - 1, bc.load_row()
    key = lambda a, b: tile_index_to_dims_key(a + NX * b, NX, NY, NZ, dims_keys)
    nbrs = [key(a, b) for a, b in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1))
            if 0 <= a < NX and 0 <= b < NY]
    return key(ix, iy), nbrs

def _load_tile_block(geometry_name, dims_keys):
    """(load_key, [keys of every in-grid tile touching it]) -- the 3x3 block
    around the loaded tile, diagonals included (5 neighbours on the free-end
    edge)."""
    import importlib
    from core.tile_sensitivity import tile_index_to_dims_key
    bc = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
    NX, NY, NZ = bc.tile_grid(None)
    if NZ != 1:
        raise ValueError("load-tile block implemented for planar (NZ = 1) grids only")
    ix, iy = NX - 1, bc.load_row()
    key = lambda a, b: tile_index_to_dims_key(a + NX * b, NX, NY, NZ, dims_keys)
    nbrs = [key(ix + a, iy + b) for a in (-1, 0, 1) for b in (-1, 0, 1)
            if (a, b) != (0, 0) and 0 <= ix + a < NX and 0 <= iy + b < NY]
    return key(ix, iy), nbrs

def _log_gpu_info():
    import torch
    print(f"[gpu] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[gpu] device_name={torch.cuda.get_device_name(0)}")
        print(f"[gpu] capability={torch.cuda.get_device_capability(0)}")
        print(f"[gpu] torch.version.cuda={torch.version.cuda}")
        print(f"[gpu] arch_list={torch.cuda.get_arch_list()}")


def run_optimization(geometry_name, arch="gnn_dsig", generations=400, processes=None, mesh_threads=4, nproc=1,
                     stress_ks_rho_mpa=0.15, gnn_ckpt_path=None, filter_radius=1.5, move_limit=0.2, oc_eta=0.5,
                     tr_tol=0.05, tr_min_scale=0.02, g_backoff=0.02, stagnation_gens=0, screenshot_every=1,
                     mesh_res=None, init_thickness=None, filter_target="gradient", filter_radius_end=None,
                     filter_decay_gens=0, filter_decay_start=0, sharpen_mode="none", proj_start_gen=100,
                     proj_end_gen=300, proj_w_max=1.0, skel_tau=0.03, skel_tau_end=None, stop_at_gen=None):
    if mesh_res:
        # Set before anything imports the geometry: data/{geometry}/
        # boundary_conditions.mesh_resolution() reads FEA_MESH_RES, and so do
        # the ProcessPoolExecutor workers that inherit this environment.
        os.environ["FEA_MESH_RES"] = ",".join(str(int(t)) for t in str(mesh_res).split(","))
        print(f"[optimize_dims] mesh_resolution override FEA_MESH_RES={os.environ['FEA_MESH_RES']}", flush=True)
    _log_gpu_info()
    evaluator = get_evaluator(geometry_name, arch=arch, processes=processes, mesh_threads=mesh_threads, nproc=nproc,
                              stress_ks_rho_mpa=stress_ks_rho_mpa, gnn_ckpt_path=gnn_ckpt_path)
    dims_path = f"data/{geometry_name}/CAD_model/dims.json"
    dims_dict = json.loads(Path(dims_path).read_text())
    # TOPOPT_T_MIN: lower every free tile's floor (dims.json's min, 0.01 on
    # cross4) for the optimizer only -- dims.json itself is also the dataset
    # builder's sampling range. 0.001 builds and meshes on cross4; 0.0001 does
    # not (the colored Jacobian's 1e-4 bump changes the mesh topology there).
    if os.environ.get("TOPOPT_T_MIN"):
        t_min = float(os.environ["TOPOPT_T_MIN"])
        dims_dict = {k: (dict(v, min=t_min) if v["max"] > v["min"] else v) for k, v in dims_dict.items()}
        print(f"[optimize_dims] tile floor lowered to {t_min} (TOPOPT_T_MIN)", flush=True)
    # TOPOPT_T_MAX: lower every free tile's ceiling the same way (optimizer only).
    if os.environ.get("TOPOPT_T_MAX"):
        t_max = float(os.environ["TOPOPT_T_MAX"])
        dims_dict = {k: (dict(v, max=t_max, default=min(v["default"], t_max)) if v["max"] > v["min"] else v)
                     for k, v in dims_dict.items()}
        print(f"[optimize_dims] tile ceiling lowered to {t_max} (TOPOPT_T_MAX)", flush=True)

    from .topopt_optimizer import TopologyOptimizer
    # Both archs supply exact per-tile sensitivities: gnn_dsig by autodiff,
    # fem by core.fem_adjoint's shape adjoint.
    gradient_func = evaluator.compute_topopt_gradient
    # TOPOPT_LOAD_TILE_MEAN=1: the loaded tile is not a design variable;
    # its thickness is always the mean of its in-grid face neighbours.
    # It is fixed in dims_dict (min == max), every design gets the tie
    # applied (opt.dims_transform below), and its sensitivity is
    # chain-ruled onto the neighbours: d/dt_j += (1/k) d/dt_load.
    # TOPOPT_FIX_LOAD_NBHD=<t>: the loaded tile and every in-grid tile
    # touching it (3x3 block, diagonals included) are fixed at thickness t
    # for the whole run (not design variables). Pair it
    # with TOPOPT_KS_EXCLUDE=load_nbhd so the stress constraint is set by
    # the structure, not by the load introduction these tiles carry.
    # TOPOPT_FIX_LOAD_TILE=<t>: freeze just the loaded tile at t (e.g. with a
    # distributed void load, FEA_LOAD_MODE=void_traction, whose loaded
    # surface must not move under the adjoint's fixed nodal forces).
    if os.environ.get("TOPOPT_FIX_LOAD_TILE"):
        lk, _ = _load_tile_neighbours(geometry_name, list(dims_dict))
        t_fix = float(os.environ["TOPOPT_FIX_LOAD_TILE"])
        dims_dict = {k: dict(v) for k, v in dims_dict.items()}
        dims_dict[lk]["min"] = dims_dict[lk]["max"] = dims_dict[lk]["default"] = t_fix
        print(f"[optimize_dims] loaded tile {lk} fixed at t={t_fix}", flush=True)
    fix_t = os.environ.get("TOPOPT_FIX_LOAD_NBHD")
    if fix_t:
        if os.environ.get("TOPOPT_LOAD_TILE_MEAN") == "1":
            raise ValueError("TOPOPT_FIX_LOAD_NBHD and TOPOPT_LOAD_TILE_MEAN both set the loaded tile")
        lk, nbrs = _load_tile_block(geometry_name, list(dims_dict))
        dims_dict = {k: dict(v) for k, v in dims_dict.items()}
        for k in [lk] + nbrs:
            dims_dict[k]["min"] = dims_dict[k]["max"] = dims_dict[k]["default"] = float(fix_t)
        print(f"[optimize_dims] loaded tile {lk} and neighbours {nbrs} fixed at t={float(fix_t)}", flush=True)
    load_tie = None
    if os.environ.get("TOPOPT_LOAD_TILE_MEAN") == "1":
        load_tie = _load_tile_neighbours(geometry_name, list(dims_dict))
        lk, nbrs = load_tie
        dims_dict = {k: dict(v) for k, v in dims_dict.items()}
        dims_dict[lk]["min"] = dims_dict[lk]["max"] = dims_dict[lk]["default"]
        _raw_grad = gradient_func

        def gradient_func(d, _raw=_raw_grad, _lk=lk, _nbrs=nbrs):
            r = _raw(d)
            if r is None:
                return None
            v, s, y, dV, dG = r
            dV, dG = dict(dV), dict(dG)
            for nb in _nbrs:
                dV[nb] += dV[_lk] / len(_nbrs)
                dG[nb] += dG[_lk] / len(_nbrs)
            dV[_lk] = dG[_lk] = 0.0
            return v, s, y, dV, dG
        print(f"[optimize_dims] loaded tile {lk} tied to the mean of {nbrs} (not a design variable)", flush=True)
    opt = TopologyOptimizer(
        dims_dict=dims_dict,
        gradient_func=gradient_func,
        generations=generations,
        reference_gradient=getattr(evaluator, "reference_gradient", None),
        # TOPOPT_SCREENSHOT_STRESS=surrogate: colour screenshots with the
        # surrogate's nodal stress (no FEM re-solve); TOPOPT_SCREENSHOT_GRAD=0
        # skips the per-tile gradient panel
        screenshot_stress=(getattr(evaluator, "node_von_mises")
                           if os.environ.get("TOPOPT_SCREENSHOT_STRESS") == "surrogate" else None),
        screenshot_grad=os.environ.get("TOPOPT_SCREENSHOT_GRAD", "1") != "0",
        geometry=geometry_name,
        move_limit=move_limit,
        oc_eta=oc_eta,
        filter_target=filter_target,
        filter_radius=filter_radius,
        filter_radius_end=filter_radius_end,
        filter_decay_start=filter_decay_start,
        filter_decay_gens=filter_decay_gens,
        # skeleton sharpening, scheduled by the old proj_* flags
        sharpen="skeleton" if (sharpen_mode == "skeleton" and filter_target in ("design", "density")) else "none",
        sharpen_start=proj_start_gen,
        sharpen_end=proj_end_gen,
        sharpen_w_max=proj_w_max,
        skel_tau=skel_tau,
        skel_tau_end=skel_tau_end,
        tr_tol=tr_tol,
        # TOPOPT_TILE_MOVE=1: per-tile step limits damped on sign reversal
        tile_move_adapt=os.environ.get("TOPOPT_TILE_MOVE") == "1",
        # TOPOPT_FIXED_MOVE=1: full move limit every step, no rejections
        adapt_move=os.environ.get("TOPOPT_FIXED_MOVE") != "1",
        # TOPOPT_UPDATE=mma: Method of Moving Asymptotes instead of OC
        update=os.environ.get("TOPOPT_UPDATE", "oc"),
        # TOPOPT_MMA_ASY_INIT: starting asymptote distance, fraction of the bound span
        mma_asy_init=float(os.environ.get("TOPOPT_MMA_ASY_INIT", "0.5")),
        # TOPOPT_MMA_LOW_MIN: floor for the lower asymptotes (e.g. 0)
        mma_low_min=(float(os.environ["TOPOPT_MMA_LOW_MIN"]) if os.environ.get("TOPOPT_MMA_LOW_MIN") else None),
        tr_min_scale=tr_min_scale,
        g_backoff=g_backoff,
        stagnation_gens=stagnation_gens,
        init_thickness=init_thickness,
        stop_at_gen=stop_at_gen,
        screenshot_every=screenshot_every,
    )
    if load_tie is not None:
        lk, nbrs = load_tie

        def _tie(d, _lk=lk, _nbrs=nbrs):
            d = dict(d)
            d[_lk] = sum(d[nb] for nb in _nbrs) / len(_nbrs)
            return d
        opt.dims_transform = _tie
    try:
        best_dims = opt.run()
    finally:
        # evaluators holding worker pools (gnn_dsig's fine-tune labellers) must
        # release them, or the interpreter waits on them forever at exit
        getattr(evaluator, "close", lambda: None)()

    names = list(dims_dict.keys())
    print("Optimized Dimensions:")
    for name in names:
        print(f"  {name}: {best_dims[name]:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tile-thickness topology optimization for a given geometry.")
    parser.add_argument("--geometry", type=str, default="cross4_11x11", help="Geometry name")
    parser.add_argument("--arch", type=str, default="gnn_dsig", choices=["gnn_dsig", "fem"], help="gnn_dsig: dsig surrogate (--gnn_ckpt_path); fem: exact core.fem_adjoint solve")
    parser.add_argument("--generations", type=int, default=400, help="Number of generations")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes for evaluation")
    parser.add_argument("--mesh_threads", type=int, default=4, help="IRIT mesh-generation threads (per fem worker process; also the gnn_dsig mesh build).")
    parser.add_argument("--nproc", type=int, default=1, help="fem arch: torch threads per solve within each worker process")
    parser.add_argument("--filter_target", choices=["gradient", "design", "density", "none"], default="gradient",
                        help="topopt: apply the spatial filter to the sensitivities (Sigmund, default), to the "
                             "DESIGN after each OC step (density filter; gradients used raw), or to nothing.")
    parser.add_argument("--filter_radius_end", type=float, default=None,
                        help="topopt, filter_target=design: radius to reach after --filter_decay_gens (linear); <=1 = no smoothing.")
    parser.add_argument("--filter_decay_gens", type=int, default=0,
                        help="topopt, filter_target=design: generations over which the design-filter radius goes from --filter_radius to --filter_radius_end (0 = constant).")
    parser.add_argument("--filter_decay_start", type=int, default=0,
                        help="topopt, filter_target=design/density: generation at which the radius decay begins (constant --filter_radius before it).")
    parser.add_argument("--proj_start_gen", type=int, default=100,
                        help="topopt, filter_target=density: beta is 0 up to this generation, then ramps linearly ...")
    parser.add_argument("--proj_end_gen", type=int, default=300,
                        help="... reaching --proj_beta_end at this generation and holding.")
    parser.add_argument("--stop_at_gen", type=int, default=None,
                        help="topopt: absolute generation to stop at (exclusive). Overrides --generations, which counts from the resumed start gen.")
    parser.add_argument("--skel_tau_end", type=float, default=None,
                        help="topopt, sharpen_mode=skeleton: tau ramps linearly from --skel_tau to this over --proj_start_gen..--proj_end_gen (None = constant).")
    parser.add_argument("--skel_tau", type=float, default=0.03,
                        help="topopt, sharpen_mode=skeleton: tiles at or below this thickness never donate material.")
    parser.add_argument("--sharpen_mode", choices=["none", "skeleton"], default="none",
                        help="topopt, filter_target=design/density: 'skeleton' = grey-level skeleton sharpening scheduled by --proj_start_gen/--proj_end_gen/--proj_w_max/--skel_tau.")
    parser.add_argument("--proj_w_max", type=float, default=1.0,
                        help="Ceiling of the gate blend weight (ramped 0 -> this over proj_start_gen..proj_end_gen). 1.0 for filter_target=density. With filter_target=design the gate is applied to the design itself after smoothing each generation, so it compounds: use a small erosion rate (0.02-0.05).")
    parser.add_argument("--filter_radius", type=float, default=1.5, help="topopt: Sigmund sensitivity-filter radius in tile-grid units (via data.{geometry}.boundary_conditions.tile_grid()). 0 disables filtering.")
    parser.add_argument("--move_limit", type=float, default=0.2, help="topopt: OC move limit per generation, as a fraction of each dim's (max - min) span.")
    parser.add_argument("--tr_tol", type=float, default=0.05,
                        help="topopt: trust region on the constraint linearization. _oc_update's lambda bisection always predicts a step landing exactly on g=0; if the measured g comes out more than tr_tol ABOVE that prediction the move limit is halved, and if the step also made an already-violated constraint worse it is rejected outright (revert + retry from the previous design, reusing its cached gradient, no extra solve). Without this the bisection always claims some step wipes out the violation, picks the smallest lambda that does -- the one removing the most material -- and volume falls monotonically while stress rises, forever (jobs 49780 and 50796).")
    parser.add_argument("--tr_min_scale", type=float, default=0.02,
                        help="topopt: floor on the trust region's move-limit multiplier, so repeated rejections terminate. At the floor a step is always accepted.")
    parser.add_argument("--stagnation_gens", type=int, default=0,
                        help="topopt: stop after this many consecutive generations with no improvement to best_feas_vol. Job 54664 found its best design at generation 145 and spent the remaining 455 (about 13 hours) drifting 12%% heavier at constant stress without beating it. The best design is tracked, so stopping only forfeits a late improvement. 0 disables, for a run that must cover a fixed generation count to stay comparable.")
    parser.add_argument("--g_backoff", type=float, default=0.02,
                        help="topopt: constraint back-off. The OC bisection aims at g_lin = -g_backoff instead of 0. Stress is convex in thinning so a linear model under-predicts the rise; aiming exactly at the boundary lands just outside it nearly every generation (job 51500 rode g in +0.008..+0.054 for 140 generations and never tripped feasible once, leaving best_feas_vol frozen at generation 3). 0 restores the old aim-at-the-boundary behavior.")
    parser.add_argument("--stress_ks_rho_mpa", type=float, default=0.15, help="fem arch: KS aggregation sharpness (1/MPa) for the stress reported to optimizers. Larger = closer to the true hard max (noisier per-tile sensitivities); smaller = smoother sensitivities but more conservative (overestimates the true peak).")
    parser.add_argument("--gnn_ckpt_path", type=str, default=None, help="gnn arch: path to the surrogate checkpoint (.pt) to load. Defaults to data/{geometry}/gnn_surrogate.pt if not given.")
    parser.add_argument("--init_thickness", type=float, default=None,
                        help="topopt: uniform starting thickness for every free dim, overriding dims.json's \"default\". Clipped to each dim's [min, max]. Set it when the mesh resolution changes: reported KS climbs steeply with elements-through-wall (94.3 MPa at 3,3,3 vs 132.4 at 5,3,3 for the same t=0.3), so a start that was feasible on a coarse mesh is infeasible on a finer one.")
    parser.add_argument("--oc_eta", type=float, default=0.5,
                        help="topopt: OC damping exponent in x_new = x * B**eta. This is the STEP SIZE, as against --move_limit which is the per-generation cap on it. 0.5 is the classic SIMP value; raising it makes each tile chase its own KKT ratio harder, so the design differentiates faster and is more prone to checkerboarding.")
    parser.add_argument("--mesh_res", type=str, default=None,
                        help="Override the geometry's mesh_resolution(), e.g. '5,5,5'. Exported as FEA_MESH_RES so evaluator worker processes see it too. Each hollow-tube tile wall is subdivided (U-1)x(V-1)x(W-1), so hollow_cube's default (3,3,3) puts only 2 elements through the wall -- enough for the forward stress, short of the >=5 the axis shape sensitivity needs. The applied load is renormalised by mesh_resolution(), so the total force is unchanged.")
    parser.add_argument("--screenshot_every", type=int, default=1, help="topopt: only render/save a screenshot every Nth generation (gen 0 always included). Each screenshot re-solves the design independently (a core.fem_adjoint solve when screenshot_heatmap is on) and typically dominates per-generation wall time, so raise this to speed up runs where only the OC trajectory/final design matters.")
    args = parser.parse_args()

    run_optimization(args.geometry, arch=args.arch, generations=args.generations, processes=args.processes, mesh_threads=args.mesh_threads, nproc=args.nproc, stress_ks_rho_mpa=args.stress_ks_rho_mpa, gnn_ckpt_path=args.gnn_ckpt_path, filter_radius=args.filter_radius, move_limit=args.move_limit, oc_eta=args.oc_eta, tr_tol=args.tr_tol, tr_min_scale=args.tr_min_scale, g_backoff=args.g_backoff, stagnation_gens=args.stagnation_gens, screenshot_every=args.screenshot_every, mesh_res=args.mesh_res, init_thickness=args.init_thickness, filter_target=args.filter_target, filter_radius_end=args.filter_radius_end, filter_decay_gens=args.filter_decay_gens, filter_decay_start=args.filter_decay_start, sharpen_mode=args.sharpen_mode, proj_start_gen=args.proj_start_gen, proj_end_gen=args.proj_end_gen, proj_w_max=args.proj_w_max, skel_tau=args.skel_tau, skel_tau_end=args.skel_tau_end, stop_at_gen=args.stop_at_gen)
