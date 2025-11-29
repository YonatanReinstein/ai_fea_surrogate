import torch
import json
import os
from torch_geometric.loader import DataLoader
from utils.gnn_surrogate import GNN


# ======================================================
# Helper functions
# ======================================================
def gnn_input_fn(data):
    return data.x, data.edge_index, data.batch

def gnn_target_fn(data):
    # returns [batch_size, 1] tensor
    return torch.cat([data.max_stress], dim=-1)


# ======================================================
# Main Training Function
# ======================================================
def train_gnn_model(
    geometry: str,
    num_samples: int,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 4,
    hidden_dim: int = 128,
    conv_layers: int = 6
):

    torch.manual_seed(42)

    # ----------------------------------------------------
    # Paths
    # ----------------------------------------------------
    dataset_a_path = f"data/{geometry}/dataset/dataset_a.pt"
    dataset_b_path = f"data/{geometry}/dataset/dataset_b.pt"
    dataset_c_path = f"data/{geometry}/dataset/dataset_c.pt"
    save_dir       = f"data/{geometry}/checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------
    # Device
    # ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------------------------------
    # Load datasets
    # ----------------------------------------------------
    dataset_a = torch.load(dataset_a_path, weights_only=False)
    dataset_b = torch.load(dataset_b_path, weights_only=False)
    dataset_c = torch.load(dataset_c_path, weights_only=False)

    dataset = dataset_a + dataset_b  # + dataset_c if you want
    num_total = len(dataset)

    # ----------------------------------------------------
    # Pre-scale features
    # ----------------------------------------------------
    # Your x has 4 features: (x, y, z, force)  -> scale force
    # Your max_stress is in Pascals -> convert to MPa
    for sample in dataset:
        sample.x[:, 3] = sample.x[:, 3] / 1e6        # N -> MN
        sample.max_stress = sample.max_stress / 1e6  # Pa -> MPa

    # ----------------------------------------------------
    # Train/Val split
    # ----------------------------------------------------
    n_train = int(num_total * 0.8)
    train_set = dataset[:n_train]
    val_set   = dataset[n_train:num_total]

    # ----------------------------------------------------
    # Target (stress) normalization mean/std
    # ----------------------------------------------------
    all_targets = torch.cat([gnn_target_fn(d) for d in train_set], dim=0)
    target_mean = all_targets.mean(dim=0)
    target_std  = all_targets.std(dim=0) + 1e-8  # epsilon

    print("Target mean:", target_mean)
    print("Target std:", target_std)

    # ----------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # ----------------------------------------------------
    # Model setup
    # ----------------------------------------------------
    example = train_set[0]
    node_in_dim = example.x.shape[1]
    out_dim     = gnn_target_fn(example).shape[1]  # normally 1

    model = GNN(
        node_in_dim=node_in_dim,
        hidden_dim=hidden_dim,
        num_layers=conv_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn   = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.9
    )

    train_losses = []
    val_losses   = []

    # ======================================================
    # Training Loop
    # ======================================================
    for epoch in range(1, epochs + 1):

        # -----------------------------
        # TRAIN
        # -----------------------------
        model.train()
        total_train = 0.0

        for batch_data in train_loader:
            optimizer.zero_grad()

            x, edge_index, batch_idx = gnn_input_fn(batch_data)
            x         = x.float().to(device)
            edge_index = edge_index.to(device)
            batch_idx  = batch_idx.to(device)

            pred_norm = model(x, edge_index, batch_idx)

            targ      = gnn_target_fn(batch_data).float().to(device)
            targ_norm = (targ - target_mean.to(device)) / target_std.to(device)

            loss = loss_fn(pred_norm, targ_norm)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        total_val = 0.0

        with torch.no_grad():
            for batch_data in val_loader:
                x, edge_index, batch_idx = gnn_input_fn(batch_data)
                x         = x.float().to(device)
                edge_index = edge_index.to(device)
                batch_idx  = batch_idx.to(device)

                pred_norm = model(x, edge_index, batch_idx)

                targ    = gnn_target_fn(batch_data).float().to(device)
                targ_norm = (targ - target_mean.to(device)) / target_std.to(device)

                total_val += loss_fn(pred_norm, targ_norm).item()

        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)

        scheduler.step()

        # -----------------------------
        # Save checkpoint every 10 epochs
        # -----------------------------
        if epoch % 10 == 0:
            ckpt = {
                "model_state": model.state_dict(),
                "node_in_dim": node_in_dim,
                "out_dim": out_dim,
                "target_mean": target_mean.cpu(),
                "target_std": target_std.cpu(),
            }
            torch.save(ckpt, os.path.join(save_dir, f"{epoch}_epochs.pt"))

            with open(os.path.join(save_dir, "losses.json"), "w") as f:
                json.dump({
                    "train_losses": train_losses,
                    "val_losses": val_losses
                }, f, indent=4)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train: {avg_train:.6f} | "
            f"Val: {avg_val:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )


# ======================================================
# Command-Line
# ======================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default="arm", type=str)
    parser.add_argument("--num_samples", default=2000, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--conv_layers", default=6, type=int)
    args = parser.parse_args()

    train_gnn_model(
        geometry=args.geometry,
        num_samples=args.num_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        conv_layers=args.conv_layers
    )
