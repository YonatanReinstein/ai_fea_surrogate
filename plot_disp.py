
import torch

dataset_path = "data/arm/dataset/dataset.pt"
dataset = torch.load(dataset_path, weights_only=False)  
max_stress_values = []
for data in dataset:
    if data.max_stress < 150 * 1e6:
        max_stress_values.append(data.max_stress.item() / 1e+6)



import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.hist(max_stress_values, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Max Stress in Dataset")
plt.xlabel("Max Stress (MPa)")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75) 
plt.show()
        
