import torch
 
from torch_geometric.loader import DataLoader
from utils.gnn_surrogate import GNN

dataset_a_path = "data/arm/dataset/dataset_a.pt"
dataset_b_path = "data/arm/dataset/dataset_b.pt"
dataset_c_path = "data/arm/dataset/dataset_c.pt"


dataset_a = torch.load(dataset_a_path, weights_only=False)
dataset_b = torch.load(dataset_b_path, weights_only=False)
dataset_c = torch.load(dataset_c_path, weights_only=False)

dataset = dataset_a + dataset_b + dataset_c

dataset_20_path = "data/arm/dataset/dataset_20.pt"
dataset_20 = torch.load(dataset_20_path, weights_only=False)
sample = dataset_c[0]
#print("Sample data.x:", sample.dims)

#print(f"Total samples: {len(dataset)}")
#
x_a_mean = torch.mean(torch.cat([data.x[:, :3] for data in dataset_a], dim=0), dim=0)
print(f"Dataset A - Mean of first 3 features: {x_a_mean}")

x_b_mean = torch.mean(torch.cat([data.x[:, :3] for data in dataset_b], dim=0), dim=0)
print(f"Dataset B - Mean of first 3 features: {x_b_mean}")
x_c_mean = torch.mean(torch.cat([data.x[:, :3] for data in dataset_c], dim=0), dim=0)
print(f"Dataset C - Mean of first 3 features: {x_c_mean}")
x_20_mean = torch.mean(torch.cat([data.x[:, :3] for data in dataset_20], dim=0), dim=0)
print(f"Dataset 20 - Mean of first 3 features: {x_20_mean}")
x_mean = torch.mean(torch.cat([data.x[:, :3] for data in dataset], dim=0), dim=0)
print(f"Combined Dataset - Mean of first 3 features: {x_mean}")

x_a_std = torch.std(torch.cat([data.x[:, :3] for data in dataset_a], dim=0), dim=0)
print(f"Dataset A - Std of first 3 features: {x_a_std}")
x_b_std = torch.std(torch.cat([data.x[:, :3] for data in dataset_b], dim=0), dim=0)
print(f"Dataset B - Std of first 3 features: {x_b_std}")
x_c_std = torch.std(torch.cat([data.x[:, :3] for data in dataset_c], dim=0), dim=0)
print(f"Dataset C - Std of first 3 features: {x_c_std}")
x_20_std = torch.std(torch.cat([data.x[:, :3] for data in dataset_20], dim=0), dim=0)
print(f"Dataset 20 - Std of first 3 features: {x_20_std}")
x_std = torch.std(torch.cat([data.x[:, :3] for data in dataset], dim=0), dim=0)
print(f"Combined Dataset - Std of first 3 features: {x_std}")

target_a_mean = torch.mean(torch.cat([data.max_stress for data in dataset_a], dim=0), dim=0)
print(f"Dataset A - Mean of target max_stress: {target_a_mean}")
target_b_mean = torch.mean(torch.cat([data.max_stress for data in dataset_b], dim=0), dim=0)
print(f"Dataset B - Mean of target max_stress: {target_b_mean}")
target_c_mean = torch.mean(torch.cat([data.max_stress for data in dataset_c], dim=0), dim=0)
print(f"Dataset C - Mean of target max_stress: {target_c_mean}")
target_20_mean = torch.mean(torch.cat([data.max_stress for data in dataset_20], dim=0), dim=0)
print(f"Dataset 20 - Mean of target max_stress: {target_20_mean}")
target_mean = torch.mean(torch.cat([data.max_stress for data in dataset], dim=0), dim=0)
print(f"Combined Dataset - Mean of target max_stress: {target_mean}")

target_a_std = torch.std(torch.cat([data.max_stress for data in dataset_a], dim=0), dim=0)
print(f"Dataset A - Std of target max_stress: {target_a_std}")
target_b_std = torch.std(torch.cat([data.max_stress for data in dataset_b], dim=0), dim=0)
print(f"Dataset B - Std of target max_stress: {target_b_std}")
target_c_std = torch.std(torch.cat([data.max_stress for data in dataset_c], dim=0), dim=0)
print(f"Dataset C - Std of target max_stress: {target_c_std}")
target_20_std = torch.std(torch.cat([data.max_stress for data in dataset_20], dim=0), dim=0)
print(f"Dataset 20 - Std of target max_stress: {target_20_std}")
target_std = torch.std(torch.cat([data.max_stress for data in dataset], dim=0), dim=0)
print(f"Combined Dataset - Std of target max_stress: {target_std}")
