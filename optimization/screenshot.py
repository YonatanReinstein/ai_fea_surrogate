from core.IritModel import IritCModel as IIritModel
from core.component import Component
import importlib
from pathlib import Path
import json

def screenshot(geometry: str, dims: dict, save_path: str, banner: str = None,
               heatmap: bool = False, ccx_path: str = "ccx", nproc: int = 1,
               resolution=(5120, 2880), stress_fn=None):
    """Render a design to save_path.

    heatmap=False (default): plain gray wireframe mesh (component geometry
    only, no solve).
    heatmap=True: runs a real CalculiX solve (requires `ccx` on PATH) and
    colors the mesh by nodal von Mises stress, jet colormap, fixed to
    [0, yield_strength] so color is comparable across frames of the same
    geometry instead of auto-scaling per-frame.
    """
    material_props_path = f"data/{geometry}/CAD_model/material_properties.json"
    material_properties = json.loads(Path(material_props_path).read_text())
    young = material_properties["young_modulus"]
    poisson = material_properties["poisson_ratio"]
    model_dir = f"data/{geometry}/CAD_model"
    model_path = f"{model_dir}/model.irt" if Path(f"{model_dir}/model.irt").exists() else f"{model_dir}/model"
    module = importlib.import_module(f"data.{geometry}.boundary_conditions")
    fixed_dims = getattr(module, "fixed_dims", lambda: {})()
    cad_model = IIritModel(model_path, dims_dict=dims, fixed_dims=fixed_dims)
    component = Component(cad_model, young, poisson)
    anchor_condition = module.anchor_condition
    force_pattern = module.force_pattern
    mesh_resolution = module.mesh_resolution
    U, V, W = mesh_resolution()
    component.generate_mesh(U=U, V=V, W=W)
    component.mesh.anchor_nodes_by_condition(anchor_condition)
    component.mesh.apply_force_by_pattern(force_pattern)

    if heatmap:
        if stress_fn is not None:
            # stress_fn(dims, node_ids, coords) -> nodal von Mises MPa (e.g. a
            # surrogate's prediction): colours the mesh without a ccx solve
            ids = list(component.mesh.nodes)
            X = np.array([component.mesh.nodes[i].coords for i in ids], dtype=np.float64)
            vm = stress_fn(dims, ids, X)
            node_stress_mpa = {nid: float(v) for nid, v in zip(ids, vm)}
        else:
            component.ccx_sim(ccx_path=ccx_path, nproc=nproc)
            node_stress_mpa = {nid: node.stress / 1e6 for nid, node in component.mesh.nodes.items()}
        marks, banner = _max_stress_marks(component, model_dir, node_stress_mpa, banner)
        component.mesh.plot_stress_heatmap(
            node_stress_mpa, save_path=save_path, banner=banner,
            resolution=resolution, aa_type="ssaa",
            clim=(0.0, material_properties["yield_strength"]), marks=marks,
        )
    else:
        component.mesh.plot_mesh(save_path=save_path, banner=banner, resolution=resolution)


import numpy as np

def _max_stress_marks(component, model_dir, node_stress_mpa, banner):
    """Marks for the max-stress node among the nodes KS counts (magenta) and,
    if higher, the max in the KS-excluded region (grey; TOPOPT_KS_EXCLUDE,
    e.g. the pads), with the values appended to the banner."""
    from evaluators.ccx_evaluator import _ks_excluded_node_ids
    mesh = component.mesh
    try:
        excl = _ks_excluded_node_ids(
            model_dir, {e.id: [n.id for n in e.nodes] for e in mesh.all_elements()},
            component.elem_to_tile, {n.id: n.coords for n in mesh.all_nodes()})
    except Exception as e:
        print(f"[screenshot] KS exclusion unavailable ({e}); marking the global max", flush=True)
        excl = set()
    counted = [nid for nid in node_stress_mpa if nid not in excl]
    nid = max(counted, key=node_stress_mpa.get)
    x, y, z = mesh.nodes[nid].coords
    v = node_stress_mpa[nid]
    marks = [(nid, f"max {v:.1f} MPa", "magenta")]
    text = f"max stress {v:.1f} MPa at (y {y:.2f}, z {z:.2f}), tile ({int(z)},{int(y)})"
    if excl:
        nx = max(excl, key=lambda k: node_stress_mpa.get(k, 0.0))
        if node_stress_mpa.get(nx, 0.0) > v:
            marks.append((nx, f"excluded {node_stress_mpa[nx]:.1f}", "grey"))
            text += f" | excluded region max {node_stress_mpa[nx]:.1f} MPa"
    return marks, (f"{banner}\n{text}" if banner else text)


def vector_to_dict(vec, dim_names):
    """Convert vector → {dim_name: value}"""
    return {name: float(vec[i]) for i, name in enumerate(dim_names)}

def show_gen(geometry: str, gen: int):
    dims = json.loads(Path(f"data/{geometry}/CAD_model/dims.json").read_text())
    dim_names = dims.keys()
    dims_path = f"optimization/artifacts/ga_population_gen_{gen}.npy"
    population = np.load(dims_path)
    print(population.shape)  # (pop_size, dim)
    for i, vec in enumerate(population):
        if i < 5:
            dims = vector_to_dict(vec,dim_names)
            save_path = f"screenshots/gen_{gen}_ind_{i}.png"
            screenshot(geometry, dims, save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate screenshots for a given generation.")
    parser.add_argument("--geometry", type=str, default="arm", help="Geometry name")
    parser.add_argument("--gen", type=int, default=200, help="Generation number")
    args = parser.parse_args()

    show_gen(args.geometry, args.gen)

# to call:
# python -m optimization.screenshot --geometry arm --gen 100
