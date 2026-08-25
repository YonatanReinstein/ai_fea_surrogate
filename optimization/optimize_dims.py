import json
from pathlib import Path
import numpy as np
from evaluators.factory import get_evaluator
from .genetic_algorithm import GeneticAlgorithm
from .fitness_functions import make_fitness

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


def run_optimization(geometry_name, arch="mlp", pop_size=30, generations=40, screenshots=False, processes=None, batch_size=28, optimizer="ga", penalty_weight=10.0, constraint_mode="auglag", search_space="logit", sigma0=None, reset_on_stagnation=False, max_unproductive_resets=2, filter_radius=1.5, move_limit=0.2, fd_step=0.02, trust_rel_tol=0.5, min_trust_frac=0.05, untrusted_move_frac=0.1, mesh_threads=4, nproc=1, stress_ks_rho_mpa=0.15, gnn_ckpt_path=None):
    _log_gpu_info()
    material_props_path = f"data/{geometry_name}/CAD_model/material_properties.json"
    material_properties = json.loads(Path(material_props_path).read_text())
    evaluator = get_evaluator(geometry_name, arch=arch, screenshots=screenshots, processes=processes, batch_size=batch_size, mesh_threads=mesh_threads, nproc=nproc,
                               heatmap_base_only=(arch == "gnn" and optimizer == "topopt"), stress_ks_rho_mpa=stress_ks_rho_mpa, gnn_ckpt_path=gnn_ckpt_path)
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
        opt = TopologyOptimizer(
            fitness_func=fitness_func,
            dims_dict=dims_dict,
            pop_size=pop_size,
            generations=generations,
            seed=0,
            geometry=geometry_name,
            filter_radius=filter_radius,
            move_limit=move_limit,
            fd_step=fd_step,
            trust_rel_tol=trust_rel_tol,
            min_trust_frac=min_trust_frac,
            untrusted_move_frac=untrusted_move_frac,
        )
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
    parser.add_argument("--filter_radius", type=float, default=1.5, help="topopt: Sigmund sensitivity-filter radius in tile-grid units (via data.{geometry}.boundary_conditions.tile_grid()). 0 disables filtering.")
    parser.add_argument("--move_limit", type=float, default=0.2, help="topopt: OC move limit per generation, as a fraction of each dim's (max - min) span.")
    parser.add_argument("--fd_step", type=float, default=0.02, help="topopt: finite-difference perturbation size per generation, as a fraction of each dim's (max - min) span. Larger values push the sensitivity signal further above solver/mesh noise at the cost of local accuracy.")
    parser.add_argument("--trust_rel_tol", type=float, default=0.5, help="topopt: OC 'trusted' cutoff on the (filtered) constraint sensitivity dg, as a multiple of this generation's std(dg). A tile is trusted (grows via the OC formula) only if dg < -trust_rel_tol*std(dg); otherwise it freezes/shrinks. 0 reduces to the old bare dg<0 cutoff, which lets FD/mesh noise flip neighboring tiles' trust status and shows up as checkerboarding.")
    parser.add_argument("--min_trust_frac", type=float, default=0.05, help="topopt: floor on the trusted-tile fraction. If --trust_rel_tol's cutoff leaves fewer than this fraction of tiles trusted, the min_trust_frac most-negative-dg tiles are trusted unconditionally instead, so an infeasible run can never fully freeze (all tiles frozen + lambda pinned at its cap = permanent deadlock).")
    parser.add_argument("--untrusted_move_frac", type=float, default=0.1, help="topopt: move-limit fraction for untrusted tiles, as a multiple of --move_limit (e.g. 0.1 = 10%% of the trusted step). Untrusted tiles get the same lambda-driven OC step as trusted ones, just capped smaller, instead of being frozen/snapped -- freezing them entirely stalls the design once the trust threshold routinely excludes most tiles (they usually still have a real, if noisy, dg<0 direction). Set to 0 to restore the old hard freeze.")
    parser.add_argument("--stress_ks_rho_mpa", type=float, default=0.15, help="ccx arch: KS aggregation sharpness (1/MPa) for the stress reported to optimizers. Larger = closer to the true hard max (noisier per-tile sensitivities); smaller = smoother sensitivities but more conservative (overestimates the true peak).")
    parser.add_argument("--gnn_ckpt_path", type=str, default=None, help="gnn arch: path to the surrogate checkpoint (.pt) to load. Defaults to data/{geometry}/gnn_surrogate.pt if not given.")
    args = parser.parse_args()

    run_optimization(args.geometry, arch=args.arch, pop_size=args.pop_size, generations=args.generations, screenshots=args.screenshots, processes=args.processes, batch_size=args.batch_size, optimizer=args.optimizer, penalty_weight=args.penalty_weight, constraint_mode=args.constraint_mode, search_space=args.search_space, sigma0=args.sigma0, reset_on_stagnation=args.reset_on_stagnation, max_unproductive_resets=args.max_unproductive_resets, filter_radius=args.filter_radius, move_limit=args.move_limit, fd_step=args.fd_step, trust_rel_tol=args.trust_rel_tol, min_trust_frac=args.min_trust_frac, untrusted_move_frac=args.untrusted_move_frac, mesh_threads=args.mesh_threads, nproc=args.nproc, stress_ks_rho_mpa=args.stress_ks_rho_mpa, gnn_ckpt_path=args.gnn_ckpt_path)