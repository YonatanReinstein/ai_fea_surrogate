import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FEMGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

if __name__ == "__main__":
    model = FEMGNN(in_channels=x.shape[1], hidden_channels=64, out_channels=3)  # 3 for dx,dy,dz
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
