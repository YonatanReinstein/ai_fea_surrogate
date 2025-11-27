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


def run_optimization(geometry_name, arch="mlp", pop_size=30, generations=40, screenshots=False, processes=None):
    material_props_path = f"data/{geometry_name}/CAD_model/material_properties.json"
    material_properties = json.loads(Path(material_props_path).read_text())
    evaluator = get_evaluator(geometry_name, arch=arch, screenshots=screenshots, processes=processes)
    fitness_func = make_fitness(evaluator, material_properties["yield_strength"])
    dims_path = f"data/{geometry_name}/CAD_model/dims.json"
    dims_dict = json.loads(Path(dims_path).read_text())

    #2400 / 16

    ga = GeneticAlgorithm(
        fitness_func=fitness_func,
        dims_dict=dims_dict,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=0.85,
        mutation_rate=0.6,
        seed=0,
    )

    best_dims = ga.run()

    names = list(dims_dict.keys())
    print("Optimized Dimensions:")
    for name in names:
        print(f"  {name}: {best_dims[name]:.4f}")   



if __name__ == "__main__":

    import os

    print("My PID is:", os.getpid())
    #exit(0)
    import argparse
    parser = argparse.ArgumentParser(description="Optimize dimensions for a given geometry.")
    parser.add_argument("--geometry", type=str, default="arm", help="Geometry name")
    parser.add_argument("--arch", type=str, default="gnn", help="Architecture type")
    parser.add_argument("--pop_size", type=int, default=200, help="Population size")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    parser.add_argument("--screenshots", action="store_true", help="Enable screenshots")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes for evaluation")
    args = parser.parse_args()

    run_optimization(args.geometry, arch=args.arch, pop_size=args.pop_size, generations=args.generations, screenshots=args.screenshots, processes=args.processes)