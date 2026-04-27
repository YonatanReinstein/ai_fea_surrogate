import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_geometric.nn.models import MLP


class EdgeAttrConv(MessagePassing):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr="mean")
        self.mlp = MLP([2 * hidden_dim + edge_dim, hidden_dim, hidden_dim], norm=None)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))


class GNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim=1, hidden_dim=128, num_layers=6):
        super().__init__()

        self.encoder = MLP([node_in_dim, hidden_dim, hidden_dim], norm=None)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EdgeAttrConv(hidden_dim, edge_in_dim))
        self.head = MLP([hidden_dim, hidden_dim, 1], norm=None)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x)
        for conv in self.convs:
            h = h + conv(h, edge_index, edge_attr)
        node_pred = self.head(h)
        graph_pred = global_max_pool(node_pred, batch)
        return graph_pred, node_pred
