# optimization/optimize_dims.py
import json
import numpy as np
from evaluators.factory import get_evaluator
from .genetic_algorithm import GeneticAlgorithm
from .fitness_functions import make_fitness

def load_dims_config(geometry_name):
    """Loads the dims.json file and returns ordered names and bounds."""
    dims_path = f"data/{geometry_name}/CAD_model/dims.json"
    with open(dims_path, "r") as f:
        dims_data = json.load(f)

    # Keep order deterministic (sorted by key)
    names = list(dims_data.keys())
    mins = [dims_data[k]["min"] for k in names]
    maxs = [dims_data[k]["max"] for k in names]
    defaults = [dims_data[k]["default"] for k in names]

    bounds = np.array(list(zip(mins, maxs)))
    return names, bounds, defaults


def run_optimization(geometry_name, arch="mlp", pop_size=30, generations=40):
    names, bounds, _ = load_dims_config(geometry_name)
    in_dim = len(names)

    evaluator = get_evaluator(geometry_name, arch=arch)
    fitness_func = make_fitness(evaluator)

    ga = GeneticAlgorithm(
        fitness_func=fitness_func,
        dim=in_dim,
        bounds=bounds,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=0.5,
        mutation_rate=0.2,
        seed=0,
    )

    best_fit, best_ind = ga.run()

    print(f"Best fitness: {best_fit:.4e}")
    print("Best design parameters:")
    for name, val in zip(names, best_ind):
        print(f"  {name}: {val:.3f}")

    result = evaluator.evaluate(best_ind)
    print(f"\nPredicted stress: {result['stress']:.3e}, displacement: {result['disp']:.3e}")
    return names, best_ind, result
