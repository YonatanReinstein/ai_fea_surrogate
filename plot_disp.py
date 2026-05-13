
import torch

dataset_path = "data/bistable/dataset/dataset.pt"
dataset = torch.load(dataset_path, weights_only=False)  
print(f"length: {len(dataset)}")
max_stress_values = []
for data in dataset:
    if data.max_stress <   10 * 1e20:
        max_stress_values.append(data.max_stress.item() / 1e+6)



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.hist(max_stress_values, bins=30, color='skyblue', edgecolor='black')
plt.title(f"Distribution of Max Stress {len(dataset)}")
plt.xlabel("Max Stress (MPa)")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75) 
plt.savefig("stress_distribution.png", dpi=150, bbox_inches="tight")
print("Saved stress_distribution.png")
        
