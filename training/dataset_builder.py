import json, random, torch, os, importlib
from core.component import Component
from core.IritModel import IIritModel
from ansys.mapdl.core.errors import MapdlRuntimeError

def build_dataset(
    geometry: str,
    num_samples: int = 10,
    seed: int = 45
):
    random.seed(seed)
    torch.manual_seed(seed)


    base_path = f"data/{geometry}"
    model_path = f"{base_path}/CAD_model/model.irt"
    dims_json_path = f"{base_path}/CAD_model/dims.json"
    dataset_dir = f"{base_path}/dataset"
    screenshots_dir = f"{base_path}/dataset/screenshots"
    material_props_path = f"{base_path}/CAD_model/material_properties.json"

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)

    dataset, metadata = [], []

    # if dataset exists, load and continue
    if os.path.exists(f"{dataset_dir}/dataset.pt"):
        print(f"Loading existing dataset from {dataset_dir}/dataset.pt")
        dataset = torch.load(f"{dataset_dir}/dataset.pt", weights_only=False)
        with open(f"{dataset_dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        start_idx = len(dataset)
        print(f"Continuing from sample {start_idx}...")
    else:
        start_idx = 0

    with open(material_props_path, "r") as f:
        material_props = json.load(f)
        young = material_props["young_modulus"]
        poisson = material_props["poisson_ratio"]

    with open(dims_json_path, "r") as f:
        dims_template = json.load(f)

    for i in range(num_samples):
        while True:
            try:
                dims = {k: random.uniform(v["min"], v["max"]) for k, v in dims_template.items()}
                CAD_model = IIritModel(model_path, dims_dict=dims)
                if i < start_idx:
                    print(f"Skipping sample {i + 1} (already in dataset)")
                    break
                comp = Component(CAD_model, young=young, poisson=poisson)
                module = importlib.import_module(f"data.{geometry}.boundary_conditions")
                anchor_condition = module.anchor_condition
                force_pattern = module.force_pattern
                mesh_resolution = module.mesh_resolution
                U, V, W = mesh_resolution()
                comp.generate_mesh(U=U, V=V, W=W)
                comp.mesh.anchor_nodes_by_condition(anchor_condition)
                comp.mesh.apply_force_by_pattern(force_pattern) 
                comp.ansys_sim(screenshot_path=screenshots_dir)
                data = comp.to_graph_with_labels()
                #comp.mesh.plot_mesh(save_path=f"{screenshots_dir}/mesh_{i+1}.png")
                dataset.append(data)
                metadata.append({
                    "id": i,
                    **dims,
                    "volume": comp.get_volume(),
                    "max_stress": comp.mesh.get_max_stress(),
                })
                print(f"[{i+1:02d}/{num_samples}] {geometry}: σmax={comp.mesh.get_max_stress():.2e}")
                if (i+1) % 10 == 0:
                    torch.save(dataset, f"{dataset_dir}/dataset.pt")
                    with open(f"{dataset_dir}/metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)
                    print(f"Dataset saved to {dataset_dir}/dataset.pt")
                break
            except MapdlRuntimeError as e:
                print(f"Sample {i+1} failed: {e}. Retrying...")
                continue
    return dataset, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FEA dataset for given geometry.")
    parser.add_argument("--geometry", type=str, default="arm", help="Geometry name (e.g., 'beam', 'arm').")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate.")
    args = parser.parse_args()
    build_dataset(args.geometry, num_samples=args.num_samples)


