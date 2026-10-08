import json
import os
from pathlib import Path
import numpy as np
from evaluators.factory import get_evaluator
from .genetic_algorithm import GeneticAlgorithm
from .fitness_functions import make_fitness


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

def load_dims_config(geometry_name):
    dims_path = f"data/{geometry_name}/CAD_model/dims.json"
    with open(dims_path, "r") as f:
        dims_data = json.load(f)

    names = list(dims_data.keys())
    mins = [dims_data[k]["min"] for k in names]
    maxs = [dims_data[k]["max"] for k in names]
    defaults = [dims_data[k]["default"] for k in names]

    bounds = np.array(list(zip(mins, maxs)))
    return names, bounds, defaults


def _log_gpu_info():
    import torch
    print(f"[gpu] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[gpu] device_name={torch.cuda.get_device_name(0)}")
        print(f"[gpu] capability={torch.cuda.get_device_capability(0)}")
        print(f"[gpu] torch.version.cuda={torch.version.cuda}")
        print(f"[gpu] arch_list={torch.cuda.get_arch_list()}")


def run_optimization(geometry_name, arch="mlp", pop_size=30, generations=40, screenshots=False, processes=None, batch_size=28, optimizer="ga", penalty_weight=10.0, constraint_mode="auglag", search_space="logit", sigma0=None, reset_on_stagnation=False, max_unproductive_resets=2, filter_radius=1.5, move_limit=0.2, fd_step=0.02, trust_rel_tol=0.5, min_trust_frac=0.05, untrusted_move_frac=0.1, tr_tol=0.05, tr_min_scale=0.02, g_backoff=0.02, calib_every=5, calib_smooth=0.5, calib_max_dev=2.0, stagnation_gens=0, mesh_threads=4, nproc=1, stress_ks_rho_mpa=0.15, gnn_ckpt_path=None, screenshot_every=1, ccx_gradient="adjoint", ccx_axis_path=None, gradient_nproc=None, mesh_res=None, oc_eta=0.5, init_thickness=None, element_type=None, filter_target="gradient", filter_radius_end=None, filter_decay_gens=0, filter_decay_start=0, proj_eta=None, proj_beta_end=16.0, proj_start_gen=100, proj_end_gen=300, proj_w_max=1.0, sharpen_mode="gate", unsharp_radius=None, skel_tau=0.03, skel_tau_end=None, stop_at_gen=None):
    if mesh_res:
        # Set before anything imports the geometry: data/{geometry}/
        # boundary_conditions.mesh_resolution() reads FEA_MESH_RES, and so do
        # the ProcessPoolExecutor workers that inherit this environment.
        os.environ["FEA_MESH_RES"] = ",".join(str(int(t)) for t in str(mesh_res).split(","))
        print(f"[optimize_dims] mesh_resolution override FEA_MESH_RES={os.environ['FEA_MESH_RES']}", flush=True)
    if element_type:
        # Same reason as FEA_MESH_RES: the mesh is built inside evaluator
        # worker processes, which inherit this environment but not a
        # monkeypatch. utils/hex20.py reads it in IritModelBase.create_mesh
        # (to add midside nodes) and in Mesh._ccx_base_deck (to name the
        # element), so one variable switches the whole pipeline.
        os.environ["FEA_ELEMENT_TYPE"] = str(element_type).strip().upper()
        print(f"[optimize_dims] element override FEA_ELEMENT_TYPE={os.environ['FEA_ELEMENT_TYPE']}", flush=True)
    _log_gpu_info()
    material_props_path = f"data/{geometry_name}/CAD_model/material_properties.json"
    material_properties = json.loads(Path(material_props_path).read_text())
    evaluator = get_evaluator(geometry_name, arch=arch, screenshots=screenshots, processes=processes, batch_size=batch_size, mesh_threads=mesh_threads, nproc=nproc,
                               heatmap_base_only=(arch == "gnn" and optimizer == "topopt"), stress_ks_rho_mpa=stress_ks_rho_mpa, gnn_ckpt_path=gnn_ckpt_path,
                               ccx_axis_path=ccx_axis_path, gradient_nproc=gradient_nproc, ccx_gradient=ccx_gradient)
    fitness_func = make_fitness(evaluator, material_properties["yield_strength"])
    dims_path = f"data/{geometry_name}/CAD_model/dims.json"
    dims_dict = json.loads(Path(dims_path).read_text())

    if optimizer == "cma":
        from .cma_optimizer import CMAOptimizer
        opt = CMAOptimizer(
            fitness_func=fitness_func,
            dims_dict=dims_dict,
            pop_size=pop_size,
            generations=generations,
            seed=0,
            geometry=geometry_name,
            penalty_weight=penalty_weight,
            constraint_mode=constraint_mode,
            search_space=search_space,
            sigma0=sigma0,
            reset_on_stagnation=reset_on_stagnation,
            max_unproductive_resets=max_unproductive_resets,
        )
    elif optimizer == "grad":
        from .gradient_optimizer import GradientOptimizer
        opt = GradientOptimizer(
            fitness_func=fitness_func,
            dims_dict=dims_dict,
            pop_size=pop_size,
            generations=generations,
            seed=0,
            geometry=geometry_name,
        )
    elif optimizer == "topopt":
        from .topopt_optimizer import TopologyOptimizer
        # Both gnn and ccx have a constant-cost sensitivity path that
        # replaces topopt's default N+1-evaluator-call finite-difference
        # sweep: the gnn's is its own autodiff, the ccx's is the patched
        # solver's global-axis nodal gradient contracted with the geometric
        # Jacobian (4 solves for any tile count, against 841 on hollow_cube).
        # See each evaluator's compute_topopt_gradient docstring.
        # --ccx_gradient fd restores the FD sweep for the ccx arch, e.g. when
        # the patched binary isn't built on this machine.
        use_grad = arch == "gnn" or (arch == "ccx" and ccx_gradient in ("adjoint", "axis"))
        gradient_func = evaluator.compute_topopt_gradient if use_grad else None
        # TOPOPT_LOAD_TILE_MEAN=1: the loaded tile is not a design variable;
        # its thickness is always the mean of its in-grid face neighbours.
        # It is fixed in dims_dict (min == max), every design gets the tie
        # applied (opt.dims_transform below), and its sensitivity is
        # chain-ruled onto the neighbours: d/dt_j += (1/k) d/dt_load.
        load_tie = None
        if os.environ.get("TOPOPT_LOAD_TILE_MEAN") == "1":
            if gradient_func is None:
                raise ValueError("TOPOPT_LOAD_TILE_MEAN needs a gradient_func path (gnn, or ccx adjoint/axis)")
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
        # Options of the pre-simplification optimizer (commit 7bf07d9). Still
        # accepted on the command line so old job scripts run, but ignored.
        removed = {"fd_step": (fd_step, 0.02), "trust_rel_tol": (trust_rel_tol, 0.5),
                   "min_trust_frac": (min_trust_frac, 0.05), "untrusted_move_frac": (untrusted_move_frac, 0.1),
                   "calib_every": (calib_every, 5), "calib_smooth": (calib_smooth, 0.5),
                   "calib_max_dev": (calib_max_dev, 2.0), "filter_radius_end": (filter_radius_end, None),
                   "filter_decay_gens": (filter_decay_gens, 0), "filter_decay_start": (filter_decay_start, 0),
                   "proj_eta": (proj_eta, None), "proj_beta_end": (proj_beta_end, 16.0),
                   "proj_start_gen": (proj_start_gen, 100), "proj_end_gen": (proj_end_gen, 300),
                   "proj_w_max": (proj_w_max, 1.0), "sharpen_mode": (sharpen_mode, "gate"),
                   "unsharp_radius": (unsharp_radius, None), "skel_tau": (skel_tau, 0.03),
                   "skel_tau_end": (skel_tau_end, None)}
        set_removed = [k for k, (v, d) in removed.items() if v != d]
        if set_removed:
            print(f"[optimize_dims] topopt: ignoring removed options {set_removed} "
                  "(the simplified optimizer has no trust sets, calibration, FD sweep, "
                  "projection or sharpening; commit 7bf07d9 has them)", flush=True)
        if gradient_func is None:
            raise ValueError(f"topopt needs exact gradients: arch=gnn, or arch=ccx with --ccx_gradient "
                             f"adjoint/axis (got arch={arch}, ccx_gradient={ccx_gradient})")
        opt = TopologyOptimizer(
            fitness_func=fitness_func,
            dims_dict=dims_dict,
            gradient_func=gradient_func,
            generations=generations,
            geometry=geometry_name,
            move_limit=move_limit,
            oc_eta=oc_eta,
            filter_target=filter_target,
            filter_radius=filter_radius,
            tr_tol=tr_tol,
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
    else:
        opt = GeneticAlgorithm(
            fitness_func=fitness_func,
            dims_dict=dims_dict,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=0.85,
            mutation_rate=0.85,
            seed=0,
            geometry=geometry_name,
        )

    best_dims = opt.run()

    names = list(dims_dict.keys())
    print("Optimized Dimensions:")
    for name in names:
        print(f"  {name}: {best_dims[name]:.4f}")   



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimize dimensions for a given geometry.")
    parser.add_argument("--geometry", type=str, default="arm", help="Geometry name")
    parser.add_argument("--arch", type=str, default="gnn", help="Architecture type")
    parser.add_argument("--pop_size", type=int, default=None, help="Population size")
    parser.add_argument("--generations", type=int, default=400, help="Number of generations")
    parser.add_argument("--screenshots", action="store_true", help="Enable screenshots")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes for evaluation")
    parser.add_argument("--mesh_threads", type=int, default=4, help="ccx arch: mesh-generation threads per worker process (each worker also uses --nproc cores for the ccx solve itself, so total cores/worker = mesh_threads, and total cores used = processes * max(mesh_threads, nproc)). Set to 1 alongside --nproc 1 to pin each worker to exactly 1 core.")
    parser.add_argument("--nproc", type=int, default=1, help="ccx arch: cores per ccx solve within each worker process")
    parser.add_argument("--batch_size", type=int, default=28, help="Batch size for evaluation")
    parser.add_argument("--optimizer", type=str, default="ga", choices=["ga", "cma", "grad", "topopt"], help="Optimization algorithm")
    parser.add_argument("--penalty_weight", type=float, default=10.0, help="Dimensionless stress-constraint penalty weight (cma, linear mode only)")
    parser.add_argument("--constraint_mode", type=str, default="auglag", choices=["linear", "auglag", "quadratic"], help="CMA stress-constraint handler: 'linear' static penalty, 'quadratic' static squared penalty, or 'auglag' augmented Lagrangian")
    parser.add_argument("--search_space", type=str, default="logit", choices=["logit", "box", "reflect"], help="CMA search variables: 'logit' unbounded R^N via sigmoid to (lo, hi); 'box' normalized [0,1]^N with cma.BoundPenalty; 'reflect' unbounded R^N folded into [lo, hi] by a period-2 triangle wave (reflecting bounds, no penalty)")
    parser.add_argument("--sigma0", type=float, default=None, help="CMA initial step size (sigma). Defaults to 2.0 for logit space, 0.25 for box/reflect. Larger values increase exploration range.")
    parser.add_argument("--reset_on_stagnation", action="store_true", help="On a soft-stagnation stop (tolstagnation/tolflatfitness), reset CMA covariance and sigma while keeping the current mean, then continue. Real convergence (tolfun/tolx) still exits.")
    parser.add_argument("--max_unproductive_resets", type=int, default=2, help="With --reset_on_stagnation: after this many consecutive resets without improvement in best feasible volume, declare hard stagnation and stop.")
    parser.add_argument("--filter_target", choices=["gradient", "design", "density", "none"], default="gradient",
                        help="topopt: apply the spatial filter to the sensitivities (Sigmund, default), to the "
                             "DESIGN after each OC step (density filter; gradients used raw), or to nothing.")
    parser.add_argument("--filter_radius_end", type=float, default=None,
                        help="topopt, filter_target=design: radius to reach after --filter_decay_gens (linear); <=1 = no smoothing.")
    parser.add_argument("--filter_decay_gens", type=int, default=0,
                        help="topopt, filter_target=design: generations over which the design-filter radius goes from --filter_radius to --filter_radius_end (0 = constant).")
    parser.add_argument("--filter_decay_start", type=int, default=0,
                        help="topopt, filter_target=design/density: generation at which the radius decay begins (constant --filter_radius before it).")
    parser.add_argument("--proj_eta", type=float, default=None,
                        help="topopt, filter_target=density: gated-projection threshold as an ABSOLUTE thickness. Filtered tiles below it are driven to the floor as beta grows; above it they keep their thickness. None = no projection (plain density filter with chain rule).")
    parser.add_argument("--proj_beta_end", type=float, default=16.0,
                        help="topopt, filter_target=density: final projection sharpness beta (0 = identity gate).")
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
    parser.add_argument("--sharpen_mode", choices=["gate", "unsharp", "skeleton"], default="gate",
                        help="topopt, filter_target=design: 'gate' = threshold gate at --proj_eta (erosion when applied post-hoc); 'unsharp' = derivative subtraction x + w*(x - blur(x)), i.e. x - w*Laplacian(x): boosts contrast at member edges only, removes blur not material. Amount w ramps 0 -> --proj_w_max over --proj_start_gen..--proj_end_gen.")
    parser.add_argument("--unsharp_radius", type=float, default=None,
                        help="topopt, sharpen_mode=unsharp: radius of the blur subtracted (default: the current design-filter radius).")
    parser.add_argument("--proj_w_max", type=float, default=1.0,
                        help="Ceiling of the gate blend weight (ramped 0 -> this over proj_start_gen..proj_end_gen). 1.0 for filter_target=density. With filter_target=design the gate is applied to the design itself after smoothing each generation, so it compounds: use a small erosion rate (0.02-0.05).")
    parser.add_argument("--filter_radius", type=float, default=1.5, help="topopt: Sigmund sensitivity-filter radius in tile-grid units (via data.{geometry}.boundary_conditions.tile_grid()). 0 disables filtering.")
    parser.add_argument("--move_limit", type=float, default=0.2, help="topopt: OC move limit per generation, as a fraction of each dim's (max - min) span.")
    parser.add_argument("--fd_step", type=float, default=0.02, help="topopt: finite-difference perturbation size per generation, as a fraction of each dim's (max - min) span. Larger values push the sensitivity signal further above solver/mesh noise at the cost of local accuracy.")
    parser.add_argument("--trust_rel_tol", type=float, default=0.5, help="topopt: OC 'trusted' cutoff on the (filtered) constraint sensitivity dg, as a multiple of this generation's std(dg). A tile is trusted (grows via the OC formula) only if dg < -trust_rel_tol*std(dg); otherwise it freezes/shrinks. 0 reduces to the old bare dg<0 cutoff, which lets FD/mesh noise flip neighboring tiles' trust status and shows up as checkerboarding.")
    parser.add_argument("--min_trust_frac", type=float, default=0.05, help="topopt: floor on the trusted-tile fraction. If --trust_rel_tol's cutoff leaves fewer than this fraction of tiles trusted, the min_trust_frac most-negative-dg tiles are trusted unconditionally instead, so an infeasible run can never fully freeze (all tiles frozen + lambda pinned at its cap = permanent deadlock).")
    parser.add_argument("--untrusted_move_frac", type=float, default=0.1, help="topopt: move-limit fraction for untrusted tiles, as a multiple of --move_limit (e.g. 0.1 = 10%% of the trusted step). Untrusted tiles get the same lambda-driven OC step as trusted ones, just capped smaller, instead of being frozen/snapped -- freezing them entirely stalls the design once the trust threshold routinely excludes most tiles (they usually still have a real, if noisy, dg<0 direction). Set to 0 to restore the old hard freeze.")
    parser.add_argument("--tr_tol", type=float, default=0.05,
                        help="topopt: trust region on the constraint linearization. _oc_update's lambda bisection always predicts a step landing exactly on g=0; if the measured g comes out more than tr_tol ABOVE that prediction the move limit is halved, and if the step also made an already-violated constraint worse it is rejected outright (revert + retry from the previous design, reusing its cached gradient, no extra solve). Without this the bisection always claims some step wipes out the violation, picks the smallest lambda that does -- the one removing the most material -- and volume falls monotonically while stress rises, forever (jobs 49780 and 50796).")
    parser.add_argument("--tr_min_scale", type=float, default=0.02,
                        help="topopt: floor on the trust region's move-limit multiplier, so repeated rejections terminate. At the floor a step is always accepted.")
    parser.add_argument("--stagnation_gens", type=int, default=0,
                        help="topopt: stop after this many consecutive generations with no improvement to best_feas_vol. Job 54664 found its best design at generation 145 and spent the remaining 455 (about 13 hours) drifting 12%% heavier at constant stress without beating it. The best design is tracked, so stopping only forfeits a late improvement. 0 disables, for a run that must cover a fixed generation count to stay comparable.")
    parser.add_argument("--g_backoff", type=float, default=0.02,
                        help="topopt: constraint back-off. The OC bisection aims at g_lin = -g_backoff instead of 0. Stress is convex in thinning so a linear model under-predicts the rise; aiming exactly at the boundary lands just outside it nearly every generation (job 51500 rode g in +0.008..+0.054 for 140 generations and never tripped feasible once, leaving best_feas_vol frozen at generation 3). 0 restores the old aim-at-the-boundary behavior.")
    parser.add_argument("--calib_every", type=int, default=5,
                        help="topopt: run one real-solver (fitness_func) solve every N generations and rescale the gradient path's stress AND its sensitivity by the measured ratio. Corrects core/fem_adjoint.py's C3D8 against ccx's C3D8I, which agree to +0.29%% at the uniform starting design but diverge to -5..-12%% (adjoint reads LOW, i.e. optimistically) as tiles thin and C3D8 shear-locks. 0 disables.")
    parser.add_argument("--calib_smooth", type=float, default=0.5,
                        help="topopt: EMA weight for the new ratio in --calib_every's rolling calibration. 1.0 = use the latest ratio outright.")
    parser.add_argument("--calib_max_dev", type=float, default=2.0,
                        help="topopt: reject a calibration sample whose ref/adjoint ratio falls outside [1/x, x] as a bad reference solve rather than folding it into the EMA. The real bias is a few per cent; ccx has twice returned a ratio of 8x-68x on an ordinary design (jobs 48454, 183931), and one such sample pins lambda for ~30 generations. 0 disables.")
    parser.add_argument("--stress_ks_rho_mpa", type=float, default=0.15, help="ccx arch: KS aggregation sharpness (1/MPa) for the stress reported to optimizers. Larger = closer to the true hard max (noisier per-tile sensitivities); smaller = smoother sensitivities but more conservative (overestimates the true peak).")
    parser.add_argument("--gnn_ckpt_path", type=str, default=None, help="gnn arch: path to the surrogate checkpoint (.pt) to load. Defaults to data/{geometry}/gnn_surrogate.pt if not given.")
    parser.add_argument("--ccx_gradient", type=str, default="adjoint", choices=["adjoint", "axis", "fd"], help="ccx arch + topopt: how to get per-tile sensitivities. 'adjoint' (default) is core/fem_adjoint.py's exact shape adjoint -- one factorization of our own C3D8 assembly plus two back-substitutions gives the reported stress AND every tile's dG, at any tile count, with d(K_e)/d(x) autodiffed rather than finite-differenced. Scored against the full 1680-solve remesh FD map on hollow_cube's 840 tiles at (3,3,3): sign 824/840, spearman +0.9997, ratio median 0.9985, sum rule x0.9953, 46 s -- where 'axis' gives 56/840, -0.4716, -19.00, x-2.79 in 433 s. 'axis' uses the patched solver's global-axis nodal gradient contracted with the geometric Jacobian -- 4 solves per generation regardless of tile count, and per-tile agreement with remesh FD inside 1.12x (against 31.4x for the older surface-normal projection). Needs the ccx_axis binary from tools/build_ccx_axis.sh. 'fd' restores the old forward-difference sweep: one extra full solve per tile per generation (841 on hollow_cube), no patched binary required.")
    parser.add_argument("--ccx_axis_path", type=str, default=None, help="ccx arch: path to the patched CalculiX built by tools/build_ccx_axis.sh. Defaults to ~/opt/calculix/root/usr/bin/ccx_axis.")
    parser.add_argument("--init_thickness", type=float, default=None,
                        help="topopt: uniform starting thickness for every free dim, overriding dims.json's \"default\". Clipped to each dim's [min, max]. Set it when the mesh resolution changes: reported KS climbs steeply with elements-through-wall (94.3 MPa at 3,3,3 vs 132.4 at 5,3,3 for the same t=0.3), so a start that was feasible on a coarse mesh is infeasible on a finer one.")
    parser.add_argument("--oc_eta", type=float, default=0.5,
                        help="topopt: OC damping exponent in x_new = x * B**eta. This is the STEP SIZE, as against --move_limit which is the per-generation cap on it. 0.5 is the classic SIMP value; raising it makes each tile chase its own KKT ratio harder, so the design differentiates faster and is more prone to checkerboarding.")
    parser.add_argument("--mesh_res", type=str, default=None,
                        help="Override the geometry's mesh_resolution(), e.g. '5,5,5'. Exported as FEA_MESH_RES so evaluator worker processes see it too. Each hollow-tube tile wall is subdivided (U-1)x(V-1)x(W-1), so hollow_cube's default (3,3,3) puts only 2 elements through the wall -- enough for the forward stress, short of the >=5 the axis shape sensitivity needs. The applied load is renormalised by mesh_resolution(), so the total force is unchanged.")
    parser.add_argument("--element_type", type=str, default=None,
                        choices=["C3D8", "C3D8I", "C3D20", "C3D20R"],
                        help="Override the hex element for BOTH the value and the gradient solve. Exported as FEA_ELEMENT_TYPE so evaluator worker processes see it. Default (unset) keeps the linear pipeline: C3D8I for the reported stress, C3D8 for the sensitivity. C3D20/C3D20R build a second-order mesh instead (utils/hex20.py adds the 12 edge midpoints to every element, no CAD change), which represents wall bending inside a single element -- the accuracy --mesh_res was being raised to buy, at a cost that grows with DOF rather than with the cube of the wall resolution. C3D20R is the reduced-integration variant: ~3x cheaper, not yet checked against the FD reference here.")
    parser.add_argument("--gradient_nproc", type=int, default=None,
                        help="ccx arch + --ccx_gradient axis: OMP threads for each of the 4 solves the gradient path runs. That path solves ONE mesh at a time, so --processes (sized for the old per-tile FD sweep, which ran that many solves side by side) is the wrong number here and is the default only for backwards compatibility -- on a 190-core allocation it would hand a single ccx solve 190 threads. Set this to the per-solve thread count that actually scales (8-32).")
    parser.add_argument("--screenshot_every", type=int, default=1, help="topopt: only render/save a screenshot every Nth generation (gen 0 always included). Each screenshot re-solves the design independently (a real CCX solve when screenshot_heatmap is on) and typically dominates per-generation wall time, so raise this to speed up runs where only the OC trajectory/final design matters.")
    args = parser.parse_args()

    run_optimization(args.geometry, arch=args.arch, pop_size=args.pop_size, generations=args.generations, screenshots=args.screenshots, processes=args.processes, batch_size=args.batch_size, optimizer=args.optimizer, penalty_weight=args.penalty_weight, constraint_mode=args.constraint_mode, search_space=args.search_space, sigma0=args.sigma0, reset_on_stagnation=args.reset_on_stagnation, max_unproductive_resets=args.max_unproductive_resets, filter_radius=args.filter_radius, move_limit=args.move_limit, fd_step=args.fd_step, trust_rel_tol=args.trust_rel_tol, tr_tol=args.tr_tol, tr_min_scale=args.tr_min_scale, g_backoff=args.g_backoff, calib_every=args.calib_every, calib_smooth=args.calib_smooth, calib_max_dev=args.calib_max_dev, stagnation_gens=args.stagnation_gens, min_trust_frac=args.min_trust_frac, untrusted_move_frac=args.untrusted_move_frac, mesh_threads=args.mesh_threads, nproc=args.nproc, stress_ks_rho_mpa=args.stress_ks_rho_mpa, gnn_ckpt_path=args.gnn_ckpt_path, screenshot_every=args.screenshot_every, ccx_gradient=args.ccx_gradient, ccx_axis_path=args.ccx_axis_path, gradient_nproc=args.gradient_nproc, mesh_res=args.mesh_res, oc_eta=args.oc_eta, init_thickness=args.init_thickness, element_type=args.element_type, filter_target=args.filter_target, filter_radius_end=args.filter_radius_end, filter_decay_gens=args.filter_decay_gens, filter_decay_start=args.filter_decay_start, proj_eta=args.proj_eta, proj_beta_end=args.proj_beta_end, proj_start_gen=args.proj_start_gen, proj_end_gen=args.proj_end_gen, proj_w_max=args.proj_w_max, sharpen_mode=args.sharpen_mode, unsharp_radius=args.unsharp_radius, skel_tau=args.skel_tau, skel_tau_end=args.skel_tau_end, stop_at_gen=args.stop_at_gen)