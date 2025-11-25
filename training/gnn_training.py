import torch
from torch_geometric.loader import DataLoader
from utils.mlp_surrogate import MLP
from utils.gnn_surrogate import GNN
import json
import os


def mlp_input_fn(data):
    return torch.cat([data.dims], dim=-1)

def mlp_target_fn(data):
    return torch.cat([data.volume, data.max_stress, data.max_displacement], dim=-1)

def gnn_input_fn(data):
    x = data.x[:, :2] 
    return x, data.edge_index, data.batch

def gnn_target_fn(data): 
    return torch.cat([data.max_stress], dim=-1) 

def compute_dataset_stats(dataset):
    all_inputs = []
    all_targets = []

    for data in dataset:
        x = mlp_input_fn(data).float()
        y = mlp_target_fn(data).float()
        all_inputs.append(x)
        all_targets.append(y)

    X = torch.stack(all_inputs)
    Y = torch.stack(all_targets)

    return {
        "input_mean":  X.mean(dim=0),
        "input_std":   X.std(dim=0) + 1e-8,
        "target_mean": Y.mean(dim=0),
        "target_std":  Y.std(dim=0) + 1e-8,
    }


def normalize(x, mean, std):
    return (x - mean) / std

def denormalize(x, mean, std):
    return x * std + mean

def train_gnn_model(
    geometry: str,
    num_samples: int,
    epochs: int = 100,
    lr: float = 1e-5,
    batch_size: int = 4
):
    import os
    import json
    import torch
    from torch_geometric.loader import DataLoader
    import matplotlib.pyplot as plt

    torch.manual_seed(42)

    dataset_path = f"data/{geometry}/dataset/dataset.pt"
    save_dir = f"data/{geometry}/checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # ---- Load dataset ----
    dataset = torch.load(dataset_path, weights_only=False)

    # Split sets
    n_train = int(num_samples * 0.8)
    train_set = dataset[:n_train]
    val_set = dataset[n_train:num_samples]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # ---- Compute global normalization stats ----
    all_targets = torch.stack([gnn_target_fn(d) for d in dataset]).float()
    targets_mean = all_targets.mean(dim=0).to(device)
    targets_std = all_targets.std(dim=0).to(device) + 1e-8

    # ---- Model dims ----
    sample = train_set[0]
    node_in_dim = gnn_input_fn(sample)[0].shape[1]
    out_features_global = gnn_target_fn(sample).shape[1]

    model = GNN(node_in_dim=node_in_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

    ## ---- Plot ----
    #plt.figure(figsize=(8, 5))
    #plt.plot(train_losses, label="Train Loss")
    #plt.plot(val_losses, label="Validation Loss")
    #plt.title("GNN Loss Curve")
    #plt.xlabel("Epoch")
    #plt.ylabel("Loss")
    #plt.legend()
    #plt.grid(True)
    #plt.show()


if __name__ == "__main__":

    train_gnn_model(
        geometry = "arm",
        num_samples = 1000,
        epochs=800,
        lr=2e-4,
        batch_size=40
    )

    # ---- Load checkpoint if exists ----
    #if os.path.exists(save_path):
    #    checkpoint = torch.load(save_path, weights_only=False)
    #    model.load_state_dict(checkpoint["model_state"])
    #    targets_mean = checkpoint["targets_mean"].to(device)
    #    targets_std  = checkpoint["targets_std"].to(device)
#


            #for data in val_loader:


#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer,
#    mode="min",
#    factor=0.5,
#    patience=30,   # ~30 epochs without improvement
#    min_lr=1e-6,
#    verbose=True,
#)