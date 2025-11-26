import torch
from torch_geometric.loader import DataLoader
from utils.gnn_surrogate import GNN
import json
import os


def gnn_input_fn(data):
    x = data.x[:, :3] 
    return data.x, data.edge_index, data.batch

def gnn_target_fn(data): 
    return torch.cat([data.max_stress], dim=-1) 


def normalize(x, mean, std):
    return (x - mean) / std

def denormalize(x, mean, std):
    return x * std + mean

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
    

    dataset_a_path = f"data/{geometry}/dataset/dataset_a.pt"
    dataset_b_path = f"data/{geometry}/dataset/dataset_b.pt"
    dataset_c_path = f"data/{geometry}/dataset/dataset_c.pt"

    save_dir = f"data/{geometry}/checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # ---- Load dataset ----
    dataset_a = torch.load(dataset_a_path, weights_only=False)
    dataset_b = torch.load(dataset_b_path, weights_only=False)
    dataset_c = torch.load(dataset_c_path, weights_only=False)
    dataset = dataset_a + dataset_b# + dataset_c
    num_samples = len(dataset)
    import random

    random.shuffle(dataset)

    n_train = int(num_samples * 0.8)
    train_set = dataset[:n_train]
    val_set = dataset[n_train:num_samples]



    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # ---- Compute global normalization stats ----
    all_targets = torch.stack([gnn_target_fn(d) for d in train_set]).float()
    targets_mean = all_targets.mean(dim=0).to(device)
    targets_std = all_targets.std(dim=0).to(device) + 1e-8

    # ---- Model dims ----
    sample = train_set[0]
    node_in_dim = gnn_input_fn(sample)[0].shape[1]
    out_features_global = gnn_target_fn(sample).shape[1]

    model = GNN(node_in_dim=node_in_dim, hidden_dim=hidden_dim, num_layers=conv_layers).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    # DECAY LR every 20 epochs by 0.9
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.9
    )

    train_losses = []
    val_losses = []

    # ---- Training Loop ----
    for epoch in range(1, epochs + 1):

        # ---------------- TRAIN ----------------
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for batch_data in train_loader:
            optimizer.zero_grad()

            x, edge_index, batch_inds = gnn_input_fn(batch_data)
            x = x.to(device)
            edge_index = edge_index.to(device)
            batch_inds = batch_inds.to(device)

            pred = model(x, edge_index, batch_inds)

            targ = gnn_target_fn(batch_data).float().to(device)
            norm_t = (targ - targets_mean) / targets_std

            loss = loss_fn(pred, norm_t)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        # ---------------- VALIDATION ----------------
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                x, edge_index, batch_inds = gnn_input_fn(batch_data)
                x = x.to(device)
                edge_index = edge_index.to(device)
                batch_inds = batch_inds.to(device)

                pred = model(x, edge_index, batch_inds)

                targ = gnn_target_fn(batch_data).float().to(device)
                norm_t = (targ - targets_mean) / targets_std

                vloss = loss_fn(pred, norm_t)
                total_val_loss += vloss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        # ---- Update LR *once per epoch* ----
        scheduler.step()

        # ---- Save checkpoint every 10 epochs ----
        if epoch % 10 == 0:
            save_dict = {
                "model_state": model.state_dict(),
                "targets_mean": targets_mean.cpu(),
                "targets_std": targets_std.cpu(),
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

    hyper_params_path = "data/arm/checkpoints/hyper_parameters.json"
    hyper_params = json.load(open(hyper_params_path, "r"))

    train_gnn_model(
        geometry = "arm",
        num_samples = hyper_params["num_samples"],
        epochs=800,
        lr=hyper_params["lr"],
        batch_size=hyper_params["batch_size"],
        hidden_dim=hyper_params["hidden_dim"],
        conv_layers=hyper_params["conv_layers"]
    )

