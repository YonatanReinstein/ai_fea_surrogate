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
# DEFINE YOUR EXACT MODEL
# ----------------------------------------------------
class GNN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, num_layers=6):
        super().__init__()
        self.encoder = MLP([node_in_dim, hidden_dim, hidden_dim], norm=None)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EdgeConv(
                    MLP([2*hidden_dim, hidden_dim, hidden_dim], norm=None),
                    aggr="mean"      # same as your model
                )
            )
        self.head = MLP([hidden_dim, hidden_dim, 1], norm=None)

    def forward(self, x, edge_index, batch):
        h = self.encoder(x)
        for conv in self.convs:
            h = h + conv(h, edge_index)
        node_pred = self.head(h)
        graph_pred = global_max_pool(node_pred, batch)
        return graph_pred

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
model = GNN(node_in_dim=F, hidden_dim=32, num_layers=3)
model.eval()

with torch.no_grad():
    pred1 = model(x.clone(), edge_index.clone(), batch.clone())
    pred2 = model(x.clone(), edge_index.clone(), batch.clone())

print("pred1:", pred1)
print("pred2:", pred2)
print("equal?", torch.equal(pred1, pred2))
print("max abs diff:", float((pred1 - pred2).abs().max()))
