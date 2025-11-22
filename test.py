import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils.gnn_surrogate import GNN


# Dummy node features
x_1 = torch.randn(4, 2)
x_2 = torch.randn(4, 2)
x_3 = torch.randn(4, 2)
x_4 = torch.randn(4, 2)

# Base edge index (must be LongTensor)
edge_index_base = torch.tensor([
    [0, 1, 2, 3, 0, 2],
    [1, 0, 3, 2, 2, 0],
], dtype=torch.long)

# Deep copies (tensors copy by value already)
edge_index_1 = edge_index_base.clone()
edge_index_2 = edge_index_base.clone()
edge_index_3 = edge_index_base.clone()
edge_index_4 = edge_index_base.clone()

# Targets
target_1 = torch.tensor([0.5], dtype=torch.float)
target_2 = torch.tensor([1.0], dtype=torch.float)
target_3 = torch.tensor([1.5], dtype=torch.float)
target_4 = torch.tensor([2.0], dtype=torch.float)

# Each single graph must have batch = zeros
batch_vec = torch.zeros(4, dtype=torch.long)

# Build Data objects
data_1 = Data(x=x_1, edge_index=edge_index_1, max_stress=target_1, batch=batch_vec)
data_2 = Data(x=x_2, edge_index=edge_index_2, max_stress=target_2, batch=batch_vec)
data_3 = Data(x=x_3, edge_index=edge_index_3, max_stress=target_3, batch=batch_vec)
data_4 = Data(x=x_4, edge_index=edge_index_4, max_stress=target_4, batch=batch_vec)

dataset = [data_1, data_2, data_3, data_4]



epochs = 1
num_samples = 4
lr= 1e-3
batch_size = 2



torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
targets = []

for data in dataset:
    targ = data.max_stress.float()
    targets.append(targ)

target_mat = torch.stack(targets)
glob_mean = target_mat.mean(dim=0)
glob_std  = target_mat.std(dim=0)
sample = dataset[0]
model = GNN(node_in_dim=2, num_layers=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

model.train()
epoch_losses = []

for epoch in range(1, epochs + 1):
    total_loss = 0.0

    for data in loader:
        optimizer.zero_grad()
        x, edge_index, batch = data.x.float().to(device), data.edge_index.to(device), data.batch.to(device)
        pred = model(x, edge_index, batch)
        targ = torch.cat([data.max_stress], dim=-1).to(device)
        targ.unsqueeze_(1)
        targ_norm = (targ - glob_mean.to(device)) / glob_std.to(device)
        loss = loss_fn(pred, targ_norm)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print("NaN in grad:", name)
                exit()
        optimizer.step()

        total_loss += loss.item()

    epoch_losses.append(total_loss)





