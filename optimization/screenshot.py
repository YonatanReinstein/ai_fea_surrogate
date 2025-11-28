from core.IritModel import IIritModel
from core.component import Component
import importlib
from pathlib import Path
import json

def screenshot(geometry: str, dims: dict, save_path: str, banner: str = None):
    material_props_path = f"data/{geometry}/CAD_model/material_properties.json"
    material_properties = json.loads(Path(material_props_path).read_text())
    young = material_properties["young_modulus"]
    poisson = material_properties["poisson_ratio"]
    cad_model = IIritModel(f"data/{geometry}/CAD_model/model.irt", dims_dict=dims)
    component = Component(cad_model, young, poisson)
    module = importlib.import_module(f"data.{geometry}.boundary_conditions")
    anchor_condition = module.anchor_condition
    force_pattern = module.force_pattern
    mesh_resolution = module.mesh_resolution
    U, V, W = mesh_resolution()
    component.generate_mesh(U=U, V=V, W=W)
    component.mesh.anchor_nodes_by_condition(anchor_condition)
    component.mesh.apply_force_by_pattern(force_pattern)
    component.mesh.plot_mesh(save_path=save_path , banner=banner)

import numpy as np

def vector_to_dict(vec, dim_names):
    """Convert vector → {dim_name: value}"""
    return {name: float(vec[i]) for i, name in enumerate(dim_names)}

def show_gen(geometry: str, gen: int):
    dims = json.loads(Path(f"data/{geometry}/CAD_model/dims.json").read_text())
    dim_names = dims.keys()
    dims_path = f"ga_population_gen_{gen}.npy"
    population = np.load(dims_path)
    for i, vec in enumerate(population):
        if i<=5:
            dims = vector_to_dict(vec,dim_names)
            save_path = f"screenshots/gen_{gen}_ind_{i}.png"
            screenshot(geometry, dims, save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate screenshots for a given generation.")
    parser.add_argument("--geometry", type=str, default="arm", help="Geometry name")
    parser.add_argument("--gen", type=int, default=1, help="Generation number")
    args = parser.parse_args()

    show_gen(args.geometry, args.gen)

# to call:
# python -m optimization.screenshot --geometry arm --gen 100
