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
    # returns [batch_size, 1] tensor (graph-level)
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
    dataset_1_path = f"data/{geometry}/dataset/dataset_1.pt"
    dataset_2_path = f"data/{geometry}/dataset/dataset_2.pt"
    dataset_3_path = f"data/{geometry}/dataset/dataset_3.pt"
    save_dir       = f"data/{geometry}/checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------
    # Device
    # ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------------------------------
    # Load dataset
    # ----------------------------------------------------
    # If you want to use dataset_a + dataset_b instead:
    dataset_1 = torch.load(dataset_1_path, weights_only=False)
    dataset_2 = torch.load(dataset_2_path, weights_only=False)
    dataset_3 = torch.load(dataset_3_path, weights_only=False)

    dataset = dataset_1 + dataset_2 + dataset_3


    for sample in dataset:
        #convert force to MN
        sample.x[:, 3] = sample.x[:, 3] / 1e+6
        #convert max_stress to MPa
        sample.max_stress = sample.max_stress / 1e+6

    cleaned_dataset = []
    for data in dataset:
        if data.max_stress < 700:
            cleaned_dataset.append(data)
    dataset = cleaned_dataset

    num_samples = min(num_samples, len(dataset))
    print(f"Dataset size after cleaning: {len(dataset)}")

    total_samples = len(dataset)
    if num_samples is None or num_samples > total_samples:
        num_samples = total_samples


    # ----------------------------------------------------
    # Random train/val split (instead of slicing in order)
    # ----------------------------------------------------
    indices = torch.randperm(total_samples)[:num_samples]
    n_train = int(num_samples * 0.8)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]    

    train_set = [dataset[i] for i in train_idx]
    val_set   = [dataset[i] for i in val_idx]

    print(f"Total samples used: {num_samples}")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # ----------------------------------------------------
    # Compute normalization stats on TRAIN ONLY
    # ----------------------------------------------------
    # Target stats (e.g. max_stress)
    for set in [val_set, train_set]:
        all_targets = torch.cat([gnn_target_fn(d) for d in set], dim=0).float()
        target_mean = all_targets.mean(dim=0)
        target_std  = all_targets.std(dim=0) + 1e-8

        # Feature stats (node features)
        all_x = torch.cat([d.x for d in set], dim=0).float()
        x_mean = all_x.mean(dim=0)
        x_std  = all_x.std(dim=0) + 1e-8

        # move stats to device once
        target_mean = target_mean.to(device)
        target_std  = target_std.to(device)
        x_mean      = x_mean.to(device)
        x_std       = x_std.to(device)

        print("Target mean:", target_mean.detach().cpu().numpy())
        print("Target std:", target_std.detach().cpu().numpy())
        print("X mean:", x_mean.detach().cpu().numpy())
        print("X std:", x_std.detach().cpu().numpy())

    

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
    out_dim     = gnn_target_fn(example).shape[1]

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
            x = x.float().to(device)
            edge_index = edge_index.to(device)
            batch_idx  = batch_idx.to(device)

            # normalize features
            x_norm = (x - x_mean) / x_std

            pred = model(x_norm, edge_index, batch_idx)

            targ = gnn_target_fn(batch_data).float().to(device)
            targ_norm = (targ - target_mean) / target_std

            loss = loss_fn(pred, targ_norm)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train += loss.item()

        avg_train = total_train / max(len(train_loader), 1)
        train_losses.append(avg_train)

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        total_val = 0.0

        with torch.no_grad():
            for batch_data in val_loader:
                x, edge_index, batch_idx = gnn_input_fn(batch_data)
                x = x.float().to(device)
                edge_index = edge_index.to(device)
                batch_idx  = batch_idx.to(device)

                x_norm = (x - x_mean) / x_std

                pred = model(x_norm, edge_index, batch_idx)

                targ = gnn_target_fn(batch_data).float().to(device)
                targ_norm = (targ - target_mean) / target_std

                loss = loss_fn(pred, targ_norm)
                total_val += loss.item()

        avg_val = total_val / max(len(val_loader), 1)
        val_losses.append(avg_val)

        scheduler.step()

        # -----------------------------
        # Save checkpoint every 10 epochs
        # -----------------------------
        if epoch % 10 == 0 or epoch == epochs:
            ckpt = {
                "model_state": model.state_dict(),
                "node_in_dim": node_in_dim,
                "out_dim": out_dim,
                "target_mean": target_mean.detach().cpu(),
                "target_std": target_std.detach().cpu(),
                "x_mean": x_mean.detach().cpu(),
                "x_std": x_std.detach().cpu(),
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
    parser.add_argument("--num_samples", default=3000, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=5, type=int)
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
