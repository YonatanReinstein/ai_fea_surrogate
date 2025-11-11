import json
import random
import torch
import os
import argparse
from core.component import Component

parser = argparse.ArgumentParser(description="Build dataset with randomized geometry.")
parser.add_argument("--model", type=str, required=True, help="Base name of the IRIT model (without .irt/.json). e.g. 'beam'")
parser.add_argument("--samples", type=int, default=1, help="Number of random samples to generate.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
parser.add_argument("--young", type=float, default=2.1e11, help="Young's modulus [Pa]")
parser.add_argument("--poisson", type=float, default=0.3, help="Poisson ratio [-]")
parser.add_argument("--force", type=float, default=1e6, help="Magnitude of applied force [N]")
args = parser.parse_args()

MODEL_NAME = args.model
NUM_SAMPLES = args.samples
SEED = args.seed
YOUNG = args.young
POISSON = args.poisson
FORCE_VAL = args.force

random.seed(SEED)
tmp_path = f"tmp/{MODEL_NAME}_dataset_build"
dataset_path = f"data/{MODEL_NAME}/dataset"
model_path = f"data/{MODEL_NAME}/CAD_model/model.irt"
os.makedirs(tmp_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)

dataset = []
metadata = []
random_dims = {}


base_json_path = f"data/{MODEL_NAME}/CAD_model/dims.json"
if not os.path.exists(base_json_path):
    raise FileNotFoundError(f"Missing base JSON file: {base_json_path}")

with open(base_json_path, "r") as f:
    dims = json.load(f)

if not isinstance(dims, dict):
    raise ValueError(f"{base_json_path} must contain a JSON object with parameter names and default values.")

for i in range(NUM_SAMPLES):
    for key, value in dims.items():
        #random_dims[key] = value["default"]    #random.uniform(value["min"], value["max"])
        random_dims[key] = random.uniform(value["min"], value["max"])

    tmp_json_path = f"{tmp_path}/dims.json"
    with open(tmp_json_path, "w") as f:
        json.dump(random_dims, f, indent=2)

    comp = Component(model_path, tmp_json_path)
    comp.generate_mesh(tmp_json_path) 
    i = 0
    j = 0
    for node in comp.mesh.nodes.values():
        if abs(node.coords[2]) < 1e-6:
            node.anchored = True
            i += 1
            print(f"Anchored node ID: {node.id}, Coords: {node.coords}")
    print(f"Total anchored nodes: {i}")

    for node in comp.mesh.nodes.values():
        if abs(node.coords[2] - 9) < 1e-6:
            node.forces = [ -1e6, 0.0, 0.0]
            j += 1
            print(f"Applying force to node ID: {node.id}, Coords: {node.coords}")
    print(f"Total force nodes: {j}") 
    #comp.mesh.add_anchor(element_id=1, face="-X")
    #comp.mesh.add_force(element_id=9, face="+X", value=FORCE_VAL, dir_code="+X")
    comp.mesh.solve(young=YOUNG, poisson=POISSON)

    # --- Convert to PyG graph ---
    data = comp.mesh.to_graph_with_labels()

    # store geometry + physical scalars
    data.dims = torch.tensor(list(random_dims.values()), dtype=torch.float)
    data.dim_names = list(random_dims.keys())
    data.volume = torch.tensor([comp.volume], dtype=torch.float)
    data.young = torch.tensor([YOUNG], dtype=torch.float)
    data.poisson = torch.tensor([POISSON], dtype=torch.float)
    data.max_stress = torch.tensor([comp.mesh.max_stress], dtype=torch.float)

    dataset.append(data)

    # metadata for inspection
    metadata.append({
        "id": i,
        **random_dims,
        "volume": comp.volume,
        "max_stress": comp.mesh.max_stress
    })

    print(f"[{i+1:02d}/{NUM_SAMPLES}] dims={random_dims}  volume={comp.volume:.3f}  σmax={comp.mesh.max_stress:.2e}")

# ---------------------------------
# Save dataset + metadata
# ---------------------------------
torch.save(dataset, f"data/{MODEL_NAME}/dataset/dataset_{NUM_SAMPLES}.pt")
with open(f"data/{MODEL_NAME}/dataset/metadata_{NUM_SAMPLES}.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved {NUM_SAMPLES} samples for model '{MODEL_NAME}' "
      f"with fixed BCs and random geometry (seed={SEED}).")