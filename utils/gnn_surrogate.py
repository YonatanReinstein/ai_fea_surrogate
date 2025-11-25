import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_max_pool
from torch_geometric.nn.models import MLP
from torch_geometric.nn import BatchNorm
import torch


class GNN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=256, num_layers=6):
        super().__init__()
        self.encoder = MLP([node_in_dim, hidden_dim, hidden_dim],norm=None)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EdgeConv(MLP([2*hidden_dim, hidden_dim, hidden_dim], norm=None), aggr="mean"))
        self.head = MLP([hidden_dim, hidden_dim, 1], norm=None)

    def forward(self, x, edge_index, batch):
        h = self.encoder(x)
        for conv in self.convs:
            h = h + conv(h, edge_index)
        node_pred = self.head(h)
        graph_pred = global_max_pool(node_pred, batch)
        return graph_pred

