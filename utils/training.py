import torch
from torch_geometric.loader import DataLoader





def mlp_input_fn(data):
    return torch.cat([data.dims, data.young, data.poisson], dim=-1)

def mlp_target_fn(data):
    return torch.cat([data.volume, data.max_stress, data.max_displacement], dim=-1)                  

def gnn_input_fn(data):
    return data.x, data.edge_index, data.batch

def gnn_target_fn(data):
    node_target = torch.cat([data.node_disp, data.node_stress], dim=-1)
    global_target = torch.cat([data.volume, data.max_stress, data.max_displacement], dim=-1)
    return {
        "node": node_target,
        "global": global_target
    }


def train_model(model, dataset_path: str, save_path: str, input_fn, target_fn, epochs: int = 100, lr: float = 1e-3, batch_size: int = 1):
    dataset = torch.load(dataset_path, weights_only=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for data in loader:
            optimizer.zero_grad()
            inputs = input_fn(data)
            preds = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            targets = target_fn(data)
            if isinstance(preds, dict):
                if not isinstance(targets, dict):
                    raise RuntimeError("train_model: model returned a dict but target_fn returned a non-dict.")
                loss = 0.0
                for key in preds.keys():
                    if key not in targets:
                        raise KeyError(f"Missing target key '{key}' in target_fn output.")
                    loss += criterion(preds[key], targets[key])
            else:
                if not torch.is_tensor(targets):
                    raise RuntimeError("train_model: model returned a tensor but target_fn returned a dict.")
                loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:03d}/{epochs}  Loss={total_loss/len(loader):.6f}")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
