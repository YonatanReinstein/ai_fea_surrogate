import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_max_pool
from torch_geometric.nn.models import MLP



class GNN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=64, num_layers=6):
        super().__init__()

        # Node encoder
        self.encoder = MLP([node_in_dim, hidden_dim, hidden_dim])

        # Build EdgeConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EdgeConv(
                    MLP([2*hidden_dim + 3, hidden_dim, hidden_dim])  # includes (x_i - x_j)
                )
            )

        # Output head
        self.head = MLP([hidden_dim, hidden_dim, 1])

    def forward(self, data):
        x, edge_index, pos, batch = data.x, data.edge_index, data.pos, data.batch

        # Append relative geometry into node feature space
        h = self.encoder(x)

        # 6-layer EdgeConv message passing
        for conv in self.convs:
            h = conv(h, edge_index)

        # Global pooling → predict max stress
        h_global = global_max_pool(h, batch)
        out = self.head(h_global)

        return out
