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
    x = data.x[:, :3] 
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


def train_mlp_model(
    dataset_path: str,
    save_path: str,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 20
):
    torch.manual_seed(42)
    

    # ---------------------------
    # load dataset
    # ---------------------------
    dataset = torch.load(dataset_path, weights_only=False)

    # ---------------------------
    # compute normalization stats
    # ---------------------------
    stats = compute_dataset_stats(dataset)
    in_mean, in_std = stats["input_mean"], stats["input_std"]
    print( "Input Mean:", in_mean )
    print( "Input Std: ", in_std )
    out_mean, out_std = stats["target_mean"], stats["target_std"]
    print( "Output Mean:", out_mean )
    print( "Output Std: ", out_std )

    # ---------------------------
    # build model
    # ---------------------------
    sample = dataset[0]
    in_features  = mlp_input_fn(sample).numel()
    out_features = mlp_target_fn(sample).numel()
    model = MLP(in_features, out_features)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    epoch_losses = []

    # ---------------------------
    # training loop
    # ---------------------------
    for epoch in range(epochs):
        total_loss = 0.0

        for data in loader:

            optimizer.zero_grad()

            # raw input + target
            raw_x = mlp_input_fn(data).float()
            raw_y = mlp_target_fn(data).float()

            # normalize
            x = normalize(raw_x, in_mean, in_std)
            y = normalize(raw_y, out_mean, out_std)

            preds = model(x)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_losses.append(total_loss)

        print(f"Epoch {epoch+1:03d}/{epochs}  Loss={total_loss:.6f}")

    # ---------------------------
    # save model + normalization
    # ---------------------------
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_mean": in_mean,
            "input_std": in_std,
            "target_mean": out_mean,
            "target_std": out_std,
        },
        save_path
    )

    print(f"\nModel + normalization saved to: {save_path}")

    # ---------------------------
    # plot loss
    # ---------------------------
    import matplotlib.pyplot as plt
    plt.plot(epoch_losses)
    plt.title("MLP Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def train_gnn_model(
    geometry: str,
    num_samples: int,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 4
):
    torch.manual_seed(42)

    # ---------------------------
    # Load dataset
    # ---------------------------
    dataset_path = f"data/{geometry}/dataset/dataset_1000.pt"
    save_path = f"data/{geometry}/surrogates/gnn_surrogate_{num_samples}.pt"
    dataset = torch.load(dataset_path, weights_only=False)
    dataset = dataset[:num_samples]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---------------------------
    # Compute GLOBAL normalization stats only
    # ---------------------------
    all_global = []

    for data in dataset:
        targ = gnn_target_fn(data)
        all_global.append(targ) 

    global_mat = torch.stack(all_global).float()

    glob_mean = global_mat.mean(dim=0)
    glob_std  = global_mat.std(dim=0) + 1e-8

    print("Global mean:", glob_mean)
    print("Global std :", glob_std)

    # ---------------------------
    # Build GNN model
    # ---------------------------
    sample = dataset[0]
    node_in_dim = sample.x[:, :3].shape[1]
    out_features_global = gnn_target_fn(sample).shape[1]  # = 3
    print("GNN in_features:", node_in_dim)
    print("GNN out_features_global:", out_features_global)

    model = GNN(node_in_dim=node_in_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # ---------------------------
    # Training loop
    # ---------------------------
    model.train()
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for data in loader:
            optimizer.zero_grad()

            x, edge_index, batch = gnn_input_fn(data)
            pred = model(x[:, :3] , edge_index, batch)  # contains both heads
            print(pred * glob_std + glob_mean)

            targ = gnn_target_fn(data)
            glob_t = (targ - glob_mean) / glob_std  # normalized
            glob_t = targ
            loss = loss_fn(pred, glob_t)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_losses.append(total_loss)
        print(f"Epoch {epoch:03d}/{epochs} | Loss = {total_loss:.6f}")

    # ---------------------------
    # Save model + global normalization ONLY
    # ---------------------------
    print(glob_mean)
    print(glob_std)
    torch.save(
        {
            "model_state": model.state_dict(),
            "glob_mean": glob_mean,
            "glob_std": glob_std,
            "node_in_dim": node_in_dim,
            "out_global": out_features_global,
        },
        save_path
    )

    print(f"\nGNN (global-only) saved to: {save_path}")

    # ---------------------------
    # Plot Loss
    # ---------------------------
    import matplotlib.pyplot as plt
    plt.plot(epoch_losses)
    plt.title("GNN Global Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":


    #train_mlp_model(
    #    dataset_path="data/beam/dataset/dataset_100.pt",
    #    save_path="data/beam/surrogates/mlp_surrogate_100.pt",
    #    epochs=100,
    #    lr=1e-3,
    #    batch_size=20
    #)

    train_gnn_model(
        geometry = "arm",
        num_samples = 1000,
        epochs=300,
        lr=1e-5,
        batch_size=20
    )

