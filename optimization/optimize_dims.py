import json
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


def run_optimization(geometry_name, arch="mlp", pop_size=30, generations=40):
    material_props_path = f"data/{geometry_name}/CAD_model/material_properties.json"
    with open(material_props_path, "r") as f:
        material_properties = json.load(f)

    evaluator = get_evaluator(geometry_name, arch=arch)
    fitness_func = make_fitness(evaluator, material_properties["yield_strength"])
    with open(f"data/{geometry_name}/CAD_model/dims.json", "r") as f:
        dims_dict = json.load(f)

    ga = GeneticAlgorithm(
        fitness_func=fitness_func,
        dims_dict=dims_dict,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=0.85,
        mutation_rate=0.05,
        seed=0,
    )

    best_dims = ga.run()

    names = list(dims_dict.keys())
    print("Optimized Dimensions:")
    for name in names:
        print(f"  {name}: {best_dims[name]:.4f}")   



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimize dimensions for a given geometry.")
    parser.add_argument("--geometry", type=str, default="arm", help="Geometry name")
    parser.add_argument("--arch", type=str, default="gnn", help="Architecture type")
    parser.add_argument("--pop_size", type=int, default=200, help="Population size")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    args = parser.parse_args()

    run_optimization(args.geometry, arch=args.arch, pop_size=args.pop_size, generations=args.generations)