import torch

from utils.mlp_surrogate import MLP
from utils.gnn_surrogate import GNN
from utils.training import (
    train_model,
    mlp_input_fn, mlp_target_fn,
    gnn_input_fn, gnn_target_fn
)

def auto_train_surrogate(geometry: str = "beam", model_type: str = "mlp",
                         dataset_size: int = 100, epochs: int = 100, lr: float = 1e-3):

    dataset_path = f"data/{geometry}/dataset/dataset_{dataset_size}.pt"
    save_path    = f"data/{geometry}/surrogates/{model_type}_surrogate.pt"

    dataset = torch.load(dataset_path, weights_only=False)
    sample = dataset[0]

    if model_type == "mlp":
        in_features  = mlp_input_fn(sample).numel()
        out_features = mlp_target_fn(sample).numel()
        model = MLP(in_features=in_features, out_features=out_features)
        input_fn  = mlp_input_fn
        target_fn = mlp_target_fn

    elif model_type == "gnn":
        in_features  = sample.x.shape[-1]
        out_features_node = gnn_target_fn(sample)["node"].shape[-1] 
        out_features_global = gnn_target_fn(sample)["global"].numel()
        model = GNN(in_features=in_features, out_features_node = out_features_node, out_features_global=out_features_global)
        input_fn  = gnn_input_fn
        target_fn = gnn_target_fn

    else:
        raise ValueError("model_type must be 'mlp' or 'gnn'")

    train_model(
        model,
        dataset_path,
        save_path,
        input_fn=input_fn,
        target_fn=target_fn,
        epochs=epochs,
        lr=lr
    )


if __name__ == "__main__":
    auto_train_surrogate(geometry="arm", model_type="mlp", dataset_size=1, epochs=5, lr=1e-3)