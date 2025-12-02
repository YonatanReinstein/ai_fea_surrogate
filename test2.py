# test_dummy.py
import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, global_max_pool
from torch_geometric.nn.models import MLP

# ----------------------------------------------------
# FORCE MAXIMUM DETERMINISM
# ----------------------------------------------------
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(0)


# ----------------------------------------------------
# CREATE DUMMY DATA
# ----------------------------------------------------
N = 300           # number of nodes
F = 7             # feature dimension
E = 1500          # number of edges

torch.manual_seed(42)
x = torch.randn(N, F)
edge_index = torch.randint(0, N, (2, E))
batch = torch.zeros(N, dtype=torch.long)  # single graph

# ----------------------------------------------------
# RUN MODEL TWICE
# ----------------------------------------------------
from utils.gnn_surrogate import GNN
model = GNN(node_in_dim=F, hidden_dim=32, num_layers=3)
model.eval()

with torch.no_grad():
    pred1 = model(x.clone(), edge_index.clone(), batch.clone())
    pred2 = model(x.clone(), edge_index.clone(), batch.clone())



print("pred1:", pred1)
print("pred2:", pred2)
print("equal?", torch.equal(pred1, pred2))
print("max abs diff:", float((pred1 - pred2).abs().max()))
