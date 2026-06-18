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


def run_optimization(geometry_name, arch="mlp", pop_size=30, generations=40, screenshots=False, processes=None, batch_size=28, optimizer="ga", penalty_weight=10.0, constraint_mode="auglag", search_space="logit", reset_on_stagnation=False, max_unproductive_resets=2):
    _log_gpu_info()
    material_props_path = f"data/{geometry_name}/CAD_model/material_properties.json"
    material_properties = json.loads(Path(material_props_path).read_text())
    evaluator = get_evaluator(geometry_name, arch=arch, screenshots=screenshots, processes=processes, batch_size=batch_size)
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
    parser.add_argument("--batch_size", type=int, default=28, help="Batch size for evaluation")
    parser.add_argument("--optimizer", type=str, default="ga", choices=["ga", "cma", "grad"], help="Optimization algorithm")
    parser.add_argument("--penalty_weight", type=float, default=10.0, help="Dimensionless stress-constraint penalty weight (cma, linear mode only)")
    parser.add_argument("--constraint_mode", type=str, default="auglag", choices=["linear", "auglag", "quadratic"], help="CMA stress-constraint handler: 'linear' static penalty, 'quadratic' static squared penalty, or 'auglag' augmented Lagrangian")
    parser.add_argument("--search_space", type=str, default="logit", choices=["logit", "box", "reflect"], help="CMA search variables: 'logit' unbounded R^N via sigmoid to (lo, hi); 'box' normalized [0,1]^N with cma.BoundPenalty; 'reflect' unbounded R^N folded into [lo, hi] by a period-2 triangle wave (reflecting bounds, no penalty)")
    parser.add_argument("--reset_on_stagnation", action="store_true", help="On a soft-stagnation stop (tolstagnation/tolflatfitness), reset CMA covariance and sigma while keeping the current mean, then continue. Real convergence (tolfun/tolx) still exits.")
    parser.add_argument("--max_unproductive_resets", type=int, default=2, help="With --reset_on_stagnation: after this many consecutive resets without improvement in best feasible volume, declare hard stagnation and stop.")
    args = parser.parse_args()

    run_optimization(args.geometry, arch=args.arch, pop_size=args.pop_size, generations=args.generations, screenshots=args.screenshots, processes=args.processes, batch_size=args.batch_size, optimizer=args.optimizer, penalty_weight=args.penalty_weight, constraint_mode=args.constraint_mode, search_space=args.search_space, reset_on_stagnation=args.reset_on_stagnation, max_unproductive_resets=args.max_unproductive_resets)