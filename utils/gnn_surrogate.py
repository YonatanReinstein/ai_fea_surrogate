import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, in_features: int, out_features_global: int, hidden: int = 128):
        super().__init__()
        self.g1 = GraphConv(in_features, hidden)
        self.g2 = GraphConv(hidden, hidden)

        self.global_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_features_global)
        )

    def forward(self, x, edge_index, batch):
        h = F.relu(self.g1(x, edge_index))
        h = F.relu(self.g2(h, edge_index))
        h_pool = global_mean_pool(h, batch)
        y_global = self.global_head(h_pool)
        return y_global
