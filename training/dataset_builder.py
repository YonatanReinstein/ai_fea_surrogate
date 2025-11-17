import json, random, torch, os, importlib
from core.component import Component
from core.IritModel import IIritModel




def build_dataset(
    geometry: str,
    num_samples: int = 10,
    young: float = 2.1e11,
    poisson: float = 0.3,
    seed: int = 42
):
    random.seed(seed)

    base_path = f"data/{geometry}"
    model_path = f"{base_path}/CAD_model/model.irt"
    dims_json_path = f"{base_path}/CAD_model/dims.json"
    dataset_dir = f"{base_path}/dataset"

    os.makedirs(dataset_dir, exist_ok=True)

    with open(dims_json_path, "r") as f:
        dims_template = json.load(f)

    dataset, metadata = [], []

    for i in range(num_samples):
        with open(dims_json_path, "r") as f:
            dims_template = json.load(f)
        dims = {k: random.uniform(v["min"], v["max"]) for k, v in dims_template.items()}
        CAD_model = IIritModel(model_path, dims_dict=dims)
        comp = Component(CAD_model, young=young, poisson=poisson)
        module = importlib.import_module(f"data.{geometry}.boundary_conditions")
        anchor_condition = module.anchor_condition
        force_pattern = module.force_pattern
        mesh_resolution = module.mesh_resolution
        U, V, W = mesh_resolution()
        comp.generate_mesh(U=U, V=V, W=W)
        comp.mesh.anchor_nodes_by_condition(anchor_condition)
        comp.mesh.apply_force_by_pattern(force_pattern)
        comp.mesh.solve(young=young, poisson=poisson)
        data = comp.to_graph_with_labels()
        dataset.append(data)
        metadata.append({
            "id": i,
            **dims,
            "volume": comp.get_volume(),
            "max_stress": comp.mesh.get_max_stress(),
        })
        print(f"[{i+1:02d}/{num_samples}] {geometry}: σmax={comp.mesh.get_max_stress():.2e}")
    torch.save(dataset, f"{dataset_dir}/dataset_{num_samples}.pt")
    with open(f"{dataset_dir}/metadata_{num_samples}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Dataset saved to {dataset_dir}/dataset_{num_samples}.pt")
    return dataset, metadata


if __name__ == "__main__":
    build_dataset("beam", num_samples=1)


