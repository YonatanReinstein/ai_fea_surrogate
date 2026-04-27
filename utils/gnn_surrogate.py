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

        # Virtual node: one per graph, bidirectionally connected to every real
        # node, providing global context during message passing.
        self.virtual_node_emb = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.virtual_node_emb, std=0.02)
        self.virtual_edge_attr = nn.Parameter(torch.zeros(edge_in_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x)

        N = h.size(0)
        num_graphs = int(batch.max().item()) + 1 if N > 0 else 0
        device = h.device

        h_virtual = self.virtual_node_emb.unsqueeze(0).expand(num_graphs, -1)
        h_aug = torch.cat([h, h_virtual], dim=0)

        real_idx = torch.arange(N, device=device)
        virtual_idx = N + batch  # virtual node id for each real node
        v_to_r = torch.stack([virtual_idx, real_idx], dim=0)
        r_to_v = torch.stack([real_idx, virtual_idx], dim=0)
        virtual_edges = torch.cat([v_to_r, r_to_v], dim=1)
        edge_index_aug = torch.cat([edge_index, virtual_edges], dim=1)

        virtual_edge_attr = self.virtual_edge_attr.unsqueeze(0).expand(
            virtual_edges.size(1), -1
        )
        edge_attr_aug = torch.cat([edge_attr, virtual_edge_attr], dim=0)

        for conv in self.convs:
            h_aug = h_aug + conv(h_aug, edge_index_aug, edge_attr_aug)

        h = h_aug[:N]
        node_pred = self.head(h)
        graph_pred = global_max_pool(node_pred, batch)
        return graph_pred, node_pred
