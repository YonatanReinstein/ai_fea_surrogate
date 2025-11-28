
import json
from core.IritModel import IIritModel
from core.component import Component
import torch

torch.use_deterministic_algorithms(True)  # make PyTorch forbid nondeterministic ops
torch.set_num_threads(1)                  # single-threaded CPU
torch.set_num_interop_threads(1)


import importlib
module = importlib.import_module(f"data.arm.boundary_conditions")
anchor_condition = module.anchor_condition
force_pattern = module.force_pattern
mesh_resolution = module.mesh_resolution
U, V, W = mesh_resolution()


dims_path = "data/arm/CAD_model/dims.json"
model_path = "data/arm/CAD_model/model.irt"
with open("data/arm/CAD_model/material_properties.json", "r") as f:
    material_properties = json.load(f)
young = material_properties["young_modulus"]
poisson = material_properties["poisson_ratio"]
with open(dims_path, "r") as f:
    dims_dict = json.load(f)

dims = {key: value["default"] for key, value in dims_dict.items()}
cad_model = IIritModel(model_path, dims_dict=dims)
component = Component(cad_model, young, poisson)
component.generate_mesh(U=U, V=V, W=W)
component.mesh.anchor_nodes_by_condition(anchor_condition)
component.mesh.apply_force_by_pattern(force_pattern)
data = component.to_graph_with_labels(with_labels=False)  


x = data.x
edge_index = data.edge_index
batch = torch.zeros(x.size(0), dtype=torch.long)  # Single graph, all nodes in batch 0


model_path = "data/arm/gnn_surrogate.pt"
ckpt = torch.load(model_path, map_location="cpu")

node_in_dim = ckpt["node_in_dim"]
from utils.gnn_surrogate import GNN
model = GNN(
            node_in_dim=node_in_dim,
            hidden_dim=128,
            num_layers=6
        )

model.load_state_dict(ckpt["model_state"])
model.eval()  


with torch.no_grad():
    pred_norm = model(x, edge_index, batch)


#print("Predicted stress:", pred_norm)
#
#with torch.no_grad():
#    pred_norm = model(x, edge_index, batch)
#
#print("Predicted stress:", pred_norm)
#
import torch

print("x shape:", x.shape)
print("edge_index shape:", edge_index.shape)
print("batch shape:", batch.shape)
print("x sum:", float(x.sum()))
print("edge_index sum:", float(edge_index.sum()))
print("batch sum:", float(batch.sum()))

# Make sure nothing will be modified in-place:
x = x.clone()
edge_index = edge_index.clone()
batch = batch.clone()

# Sanity: check param stability in a stronger way
state_before = {k: v.clone() for k, v in model.state_dict().items()}

with torch.no_grad():
    pred1 = model(x, edge_index, batch)
    pred2 = model(x, edge_index, batch)

print("pred1:", pred1)
print("pred2:", pred2)


