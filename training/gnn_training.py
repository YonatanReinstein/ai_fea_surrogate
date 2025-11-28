import torch
from torch_geometric.loader import DataLoader
from utils.gnn_surrogate import GNN
import json
import os


def gnn_input_fn(data):
    return data.x, data.edge_index, data.batch

def gnn_target_fn(data): 
    return torch.cat([data.max_stress], dim=-1) 


def train_gnn_model(
    geometry: str,
    num_samples: int,
    epochs: int = 100,
    lr: float = 1e-5,
    batch_size: int = 4,
    hidden_dim: int = 128,
    conv_layers: int = 6
):

    torch.manual_seed(42)
    
    # ------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------
    dataset_a_path = f"data/{geometry}/dataset/dataset_a.pt"
    dataset_b_path = f"data/{geometry}/dataset/dataset_b.pt"
    dataset_c_path = f"data/{geometry}/dataset/dataset_c.pt"

    save_dir = f"data/{geometry}/checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    dataset_a = torch.load(dataset_a_path, weights_only=False)
    dataset_b = torch.load(dataset_b_path, weights_only=False)
    dataset_c = torch.load(dataset_c_path, weights_only=False)

    dataset = dataset_a + dataset_b  # + dataset_c   # if you want


    num_samples = len(dataset)

    n_train = int(num_samples * 0.8)
    train_set = dataset[:n_train]
    val_set = dataset[n_train:num_samples]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------
    # Compute NORMALIZATION of node features
    # ------------------------------------------------------------
    print("Computing node feature normalization...")

    # gather all features
    all_x = torch.cat([d.x for d in train_set], dim=0).float()  # [total_nodes, F]
    x_mean = all_x.mean(dim=0)         # shape [F]
    x_std  = all_x.std(dim=0) + 1e-8    # shape [F]

    x_mean = x_mean.to(device)
    x_std  = x_std.to(device)

    print("x_mean:", x_mean)
    print("x_std:", x_std)

    # ------------------------------------------------------------
    # Compute output normalization (stress)
    # ------------------------------------------------------------
    all_targets = torch.stack([gnn_target_fn(d) for d in train_set]).float()
    targets_mean = all_targets.mean(dim=0).to(device)
    targets_std = all_targets.std(dim=0).to(device) + 1e-8

    # ------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------
    sample = train_set[0]
    node_in_dim = sample.x.size(1)
    out_features_global = gnn_target_fn(sample).shape[1]

    model = GNN(
        node_in_dim=node_in_dim,
        hidden_dim=hidden_dim,
        num_layers=conv_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.9
    )

    train_losses = []
    val_losses = []

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------
    for epoch in range(1, epochs + 1):

        # ============================
        # TRAIN
        # ============================
        model.train()
        total_train_loss = 0.0

        for batch_data in train_loader:
            optimizer.zero_grad()

            x, edge_index, batch_inds = gnn_input_fn(batch_data)
            x = x.float().to(device)
            edge_index = edge_index.to(device)
            batch_inds = batch_inds.to(device)

            # ---- normalize input FEATURES ----
            x = (x - x_mean) / x_std

            pred = model(x, edge_index, batch_inds)

            targ = gnn_target_fn(batch_data).float().to(device)
            norm_t = (targ - targets_mean) / targets_std

            loss = loss_fn(pred, norm_t)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ============================
        # VALIDATION
        # ============================
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_data in val_loader:
                x, edge_index, batch_inds = gnn_input_fn(batch_data)
                x = x.float().to(device)
                edge_index = edge_index.to(device)
                batch_inds = batch_inds.to(device)

                # same normalization
                x = (x - x_mean) / x_std

                pred = model(x, edge_index, batch_inds)

                targ = gnn_target_fn(batch_data).float().to(device)
                norm_t = (targ - targets_mean) / targets_std

                loss = loss_fn(pred, norm_t)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # LR STEP
        scheduler.step()

        # ------------------------------------------------------------
        # SAVE CHECKPOINT EVERY 10 EPOCHS
        # ------------------------------------------------------------
        if epoch % 10 == 0:
            save_dict = {
                "model_state": model.state_dict(),
                "targets_mean": targets_mean.cpu(),
                "targets_std": targets_std.cpu(),
                "x_mean": x_mean.cpu(),
                "x_std": x_std.cpu(),
                "node_in_dim": node_in_dim,
                "out_global": out_features_global,
            }
            torch.save(save_dict, os.path.join(save_dir, f"{epoch}_epochs.pt"))

            with open(os.path.join(save_dir, "losses.json"), "w") as f:
                json.dump({
                    "train_losses": train_losses,
                    "val_losses": val_losses
                }, f, indent=4)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss = {avg_train_loss:.6f} | "
            f"Val Loss = {avg_val_loss:.6f} | "
            f"LR = {optimizer.param_groups[0]['lr']:.2e}"
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default="arm", type=str)
    parser.add_argument("--num_samples", default=2000, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
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
