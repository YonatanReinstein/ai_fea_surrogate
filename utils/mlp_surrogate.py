import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.net(x)
