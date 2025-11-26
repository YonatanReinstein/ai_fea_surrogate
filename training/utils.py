import json
import matplotlib.pyplot as plt
import os

def plot_losses(geometry: str = "arm", save_path: str = None):

    # ---- Load JSON ----
    with open(f"data/{geometry}/checkpoints/losses.json", "r") as f:
        data = json.load(f)

    train_losses = data["train_losses"]
    val_losses = data["val_losses"]

    train_losses = train_losses[20:len(train_losses)]
    val_losses = val_losses[20:len(val_losses)]

    #clean up spikes
    for i in range(1, len(train_losses)-1):
        if train_losses[i] > train_losses[i-1] * 1.5 and train_losses[i] > train_losses[i+1] * 1.5:
            train_losses[i] = (train_losses[i-1] + train_losses[i+1]) / 2

    # ---- Create Plot ----
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("GNN Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # ---- Save or show ----
    if save_path is None:
        base_dir = os.path.dirname(f"data/{geometry}/checkpoints/")
        save_path = os.path.join(base_dir, "loss_plot.png")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Loss plot saved to: {save_path}")

if __name__ == "__main__":
    plot_losses(geometry="arm")
