import json
import matplotlib.pyplot as plt
import os

def plot_losses(run_dir: str = "training/runs/train", save_path: str = None):

    # ---- Load JSON ----
    with open(os.path.join(run_dir, "losses.json"), "r") as f:
        data = json.load(f)

    # ---- Create Plot ----
    plt.figure(figsize=(10, 5))
    for key, values in data.items():
        plt.plot(values, label=key.replace("_", " ").title())
    plt.title("GNN Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # ---- Save or show ----
    if save_path is None:
        save_path = os.path.join(run_dir, "loss_plot.png")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Loss plot saved to: {save_path}")

if __name__ == "__main__":
    plot_losses()
