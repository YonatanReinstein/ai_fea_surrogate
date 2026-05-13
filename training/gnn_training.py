import torch
import json
import os
import re
import glob
import time
from torch_geometric.loader import DataLoader
from utils.gnn_surrogate import GNN, HierarchicalGNN


# ======================================================
# Helper functions
# ======================================================
def gnn_input_fn(data):
    tile_idx  = getattr(data, 'tile_idx',  None)
    tile_NX   = getattr(data, 'tile_NX',   None)
    tile_NY   = getattr(data, 'tile_NY',   None)
    tile_NZ   = getattr(data, 'tile_NZ',   None)
    tile_x    = getattr(data, 'tile_x',    None)
    return data.x, data.edge_index, data.edge_attr, data.batch, tile_idx, tile_NX, tile_NY, tile_NZ, tile_x


def gnn_target_fn(data):
    # returns [batch_size, 1] tensor (graph-level)
    return torch.cat([data.max_stress], dim=-1)


def gnn_node_target_fn(data):
    # returns [N_total, 1] tensor (node-level von Mises stress)
    return data.node_stress.view(-1, 1)


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
    conv_layers: int = 6,
    node_loss_weight: float = 1.0,
    graph_loss_weight: float = 1.0,
    weight_decay: float = 1e-4,
    dataset_path: str = None,
):
    torch.manual_seed(42)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))

    # ----------------------------------------------------
    # Paths
    # ----------------------------------------------------
    dataset_path = dataset_path or f"data/{geometry}/dataset/dataset.pt"

    save_dir       = f"data/{geometry}/checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------
    # Device
    # ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"[gpu] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[gpu] device_name={torch.cuda.get_device_name(0)}")
        print(f"[gpu] capability={torch.cuda.get_device_capability(0)}")
        print(f"[gpu] torch.version.cuda={torch.version.cuda}")
        print(f"[gpu] arch_list={torch.cuda.get_arch_list()}")

    # ----------------------------------------------------
    # Load dataset
    # ----------------------------------------------------
    # If you want to use dataset_a + dataset_b instead:
    dataset = torch.load(dataset_path, weights_only=False)

 
    for sample in dataset:
        #convert force to MN
        sample.x[:, 3] = sample.x[:, 3] / 1e+6
        #convert max_stress to MPa
        sample.max_stress = sample.max_stress / 1e+6
        #convert node_stress to MPa
        sample.node_stress = sample.node_stress / 1e+6
        # inject per-node strut dimension: dims[3 + tile_idx] = d4..d103
        if hasattr(sample, 'tile_idx') and hasattr(sample, 'dims'):
            tile_struts = sample.dims[0, 3:].float()  # [K]
            strut_param = tile_struts[sample.tile_idx].unsqueeze(1)
            sample.x = torch.cat([sample.x, strut_param], dim=1)
            # per-tile feature: one strut value per tile virtual node, shape [K, 1]
            sample.tile_x = tile_struts.view(-1, 1)

    cleaned_dataset = []
    for data in dataset:
        if data.max_stress < 100000:
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
    all_targets = torch.cat([gnn_target_fn(d) for d in train_set], dim=0).float()
    target_mean = all_targets.mean(dim=0).to(device)
    target_std  = (all_targets.std(dim=0) + 1e-8).to(device)

    all_node_targets = torch.cat([gnn_node_target_fn(d) for d in train_set], dim=0).float()
    node_target_mean = all_node_targets.mean(dim=0).to(device)
    node_target_std  = (all_node_targets.std(dim=0) + 1e-8).to(device)

    all_x = torch.cat([d.x for d in train_set], dim=0).float()
    x_mean = all_x.mean(dim=0).to(device)
    x_std  = (all_x.std(dim=0) + 1e-8).to(device)

    all_edge_attr = torch.cat([d.edge_attr for d in train_set], dim=0).float()
    edge_mean = all_edge_attr.mean(dim=0).to(device)
    edge_std  = (all_edge_attr.std(dim=0) + 1e-8).to(device)

    print("Target mean:", target_mean.detach().cpu().numpy())
    print("Target std:", target_std.detach().cpu().numpy())
    print("Node target mean:", node_target_mean.detach().cpu().numpy())
    print("Node target std:", node_target_std.detach().cpu().numpy())
    print("X mean:", x_mean.detach().cpu().numpy())
    print("X std:", x_std.detach().cpu().numpy())
    print("Edge mean:", edge_mean.detach().cpu().numpy())
    print("Edge std:", edge_std.detach().cpu().numpy())

    

    # ----------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=8, persistent_workers=True, pin_memory=True,
                              prefetch_factor=4)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=4, persistent_workers=True, pin_memory=True,
                              prefetch_factor=4)

    # ----------------------------------------------------
    # Model setup
    # ----------------------------------------------------
    example = train_set[0]
    node_in_dim = example.x.shape[1]
    edge_in_dim = example.edge_attr.shape[1]
    out_dim     = gnn_target_fn(example).shape[1]

    hierarchical = hasattr(train_set[0], 'tile_idx')
    ModelClass = HierarchicalGNN if hierarchical else GNN
    model = ModelClass(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=hidden_dim,
        num_layers=conv_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = torch.nn.MSELoss()

    train_losses = []
    val_losses   = []
    train_graph_losses = []
    val_graph_losses   = []
    train_node_losses  = []
    val_node_losses    = []

    # ----------------------------------------------------
    # Resume from latest checkpoint if available
    # ----------------------------------------------------
    start_epoch = 0
    ckpt_files = glob.glob(os.path.join(save_dir, "*_epochs.pt"))
    epoch_re = re.compile(r"(\d+)_epochs\.pt$")
    parsed = [(int(m.group(1)), p) for p in ckpt_files for m in [epoch_re.search(p)] if m]
    if parsed:
        start_epoch, latest_ckpt = max(parsed, key=lambda t: t[0])
        print(f"Resuming from checkpoint: {latest_ckpt} (epoch {start_epoch})")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        losses_path = os.path.join(save_dir, "losses.json")
        if os.path.exists(losses_path):
            with open(losses_path, "r") as f:
                prev = json.load(f)
            train_losses = prev.get("train_losses", [])[:start_epoch]
            val_losses   = prev.get("val_losses", [])[:start_epoch]
            train_graph_losses = prev.get("train_graph_losses", [])[:start_epoch]
            val_graph_losses   = prev.get("val_graph_losses", [])[:start_epoch]
            train_node_losses  = prev.get("train_node_losses", [])[:start_epoch]
            val_node_losses    = prev.get("val_node_losses", [])[:start_epoch]

    if start_epoch > 0:
        for pg in optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.75,
        last_epoch=start_epoch - 1
    )

    # ======================================================
    # Training Loop
    # ======================================================
    for epoch in range(start_epoch + 1, epochs + 1):
        # re-seed per epoch so resumed runs don't replay the same batch order
        torch.manual_seed(42 + epoch)
        epoch_t0 = time.time()

        # -----------------------------
        # TRAIN
        # -----------------------------
        model.train()
        total_train = 0.0
        total_train_graph = 0.0
        total_train_node  = 0.0

        # Per-section timing (only collected on epoch 1 to keep overhead off the steady state).
        profile_this_epoch = (epoch == start_epoch + 1)
        t_data = t_h2d = t_fwd = t_bwd = t_step = 0.0
        n_steps = 0

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        loader_iter = iter(train_loader)
        if profile_this_epoch: _sync()
        t_mark = time.perf_counter()

        while True:
            try:
                batch_data = next(loader_iter)
            except StopIteration:
                break
            if profile_this_epoch:
                _sync(); t_now = time.perf_counter(); t_data += t_now - t_mark; t_mark = t_now

            optimizer.zero_grad()

            x, edge_index, edge_attr, batch_idx, tile_idx, tile_NX, tile_NY, tile_NZ, tile_x = gnn_input_fn(batch_data)
            num_graphs = batch_data.num_graphs
            x = x.float().to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            edge_attr  = edge_attr.float().to(device, non_blocking=True)
            batch_idx  = batch_idx.to(device, non_blocking=True)
            if tile_idx is not None:
                tile_idx = tile_idx.to(device, non_blocking=True)
                tile_NX  = tile_NX.to(device,  non_blocking=True)
                tile_NY  = tile_NY.to(device,  non_blocking=True)
                tile_NZ  = tile_NZ.to(device,  non_blocking=True)
            if tile_x is not None:
                tile_x = tile_x.float().to(device, non_blocking=True)

            # normalize features
            x_norm = (x - x_mean) / x_std
            edge_attr_norm = (edge_attr - edge_mean) / edge_std

            targ = gnn_target_fn(batch_data).float().to(device, non_blocking=True)
            targ_norm = (targ - node_target_mean) / node_target_std
            node_targ = gnn_node_target_fn(batch_data).float().to(device, non_blocking=True)
            node_targ_norm = (node_targ - node_target_mean) / node_target_std
            if profile_this_epoch:
                _sync(); t_now = time.perf_counter(); t_h2d += t_now - t_mark; t_mark = t_now

            graph_pred, node_pred = model(
                x_norm, edge_index, edge_attr_norm, batch_idx,
                tile_idx=tile_idx, tile_NX=tile_NX, tile_NY=tile_NY, tile_NZ=tile_NZ,
                tile_x=tile_x, num_graphs=num_graphs,
            )
            graph_loss = loss_fn(graph_pred, targ_norm)
            node_loss  = loss_fn(node_pred, node_targ_norm)
            loss = graph_loss_weight * graph_loss + node_loss_weight * node_loss
            if profile_this_epoch:
                _sync(); t_now = time.perf_counter(); t_fwd += t_now - t_mark; t_mark = t_now

            loss.backward()
            if profile_this_epoch:
                _sync(); t_now = time.perf_counter(); t_bwd += t_now - t_mark; t_mark = t_now

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if profile_this_epoch:
                _sync(); t_now = time.perf_counter(); t_step += t_now - t_mark; t_mark = t_now

            total_train       += loss.item()
            total_train_graph += graph_loss.item()
            total_train_node  += node_loss.item()
            n_steps += 1

        if profile_this_epoch and n_steps > 0:
            total = t_data + t_h2d + t_fwd + t_bwd + t_step
            print(
                f"[profile] steps={n_steps}  "
                f"data={t_data:6.1f}s ({100*t_data/total:4.1f}%)  "
                f"h2d={t_h2d:6.1f}s ({100*t_h2d/total:4.1f}%)  "
                f"fwd={t_fwd:6.1f}s ({100*t_fwd/total:4.1f}%)  "
                f"bwd={t_bwd:6.1f}s ({100*t_bwd/total:4.1f}%)  "
                f"opt={t_step:6.1f}s ({100*t_step/total:4.1f}%)"
            )

        n_train_batches = max(len(train_loader), 1)
        avg_train       = total_train       / n_train_batches
        avg_train_graph = total_train_graph / n_train_batches
        avg_train_node  = total_train_node  / n_train_batches
        train_losses.append(avg_train)
        train_graph_losses.append(avg_train_graph)
        train_node_losses.append(avg_train_node)

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[vram] peak={peak:.2f} GB | reserved={reserved:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        total_val = 0.0
        total_val_graph = 0.0
        total_val_node  = 0.0

        with torch.no_grad():
            for batch_data in val_loader:
                x, edge_index, edge_attr, batch_idx, tile_idx, tile_NX, tile_NY, tile_NZ, tile_x = gnn_input_fn(batch_data)
                num_graphs = batch_data.num_graphs
                x = x.float().to(device, non_blocking=True)
                edge_index = edge_index.to(device, non_blocking=True)
                edge_attr  = edge_attr.float().to(device, non_blocking=True)
                batch_idx  = batch_idx.to(device, non_blocking=True)
                if tile_idx is not None:
                    tile_idx = tile_idx.to(device, non_blocking=True)
                    tile_NX  = tile_NX.to(device,  non_blocking=True)
                    tile_NY  = tile_NY.to(device,  non_blocking=True)
                    tile_NZ  = tile_NZ.to(device,  non_blocking=True)
                if tile_x is not None:
                    tile_x = tile_x.float().to(device, non_blocking=True)

                x_norm = (x - x_mean) / x_std
                edge_attr_norm = (edge_attr - edge_mean) / edge_std

                targ = gnn_target_fn(batch_data).float().to(device, non_blocking=True)
                targ_norm = (targ - node_target_mean) / node_target_std

                node_targ = gnn_node_target_fn(batch_data).float().to(device, non_blocking=True)
                node_targ_norm = (node_targ - node_target_mean) / node_target_std

                graph_pred, node_pred = model(
                    x_norm, edge_index, edge_attr_norm, batch_idx,
                    tile_idx=tile_idx, tile_NX=tile_NX, tile_NY=tile_NY, tile_NZ=tile_NZ,
                    tile_x=tile_x, num_graphs=num_graphs,
                )
                graph_loss = loss_fn(graph_pred, targ_norm)
                node_loss  = loss_fn(node_pred, node_targ_norm)
                loss = graph_loss_weight * graph_loss + node_loss_weight * node_loss

                total_val       += loss.item()
                total_val_graph += graph_loss.item()
                total_val_node  += node_loss.item()

        n_val_batches = max(len(val_loader), 1)
        avg_val       = total_val       / n_val_batches
        avg_val_graph = total_val_graph / n_val_batches
        avg_val_node  = total_val_node  / n_val_batches
        val_losses.append(avg_val)
        val_graph_losses.append(avg_val_graph)
        val_node_losses.append(avg_val_node)

        scheduler.step()

        # -----------------------------
        # Save checkpoint every 10 epochs
        # -----------------------------
        if epoch % 10 == 0 or epoch == epochs:
            ckpt = {
                "model_state": model.state_dict(),
                "node_in_dim": node_in_dim,
                "edge_in_dim": edge_in_dim,
                "out_dim": out_dim,
                "hierarchical": hierarchical,
                "target_mean": target_mean.detach().cpu(),
                "target_std": target_std.detach().cpu(),
                "node_target_mean": node_target_mean.detach().cpu(),
                "node_target_std": node_target_std.detach().cpu(),
                "x_mean": x_mean.detach().cpu(),
                "x_std": x_std.detach().cpu(),
                "edge_mean": edge_mean.detach().cpu(),
                "edge_std": edge_std.detach().cpu(),
            }
            torch.save(ckpt, os.path.join(save_dir, f"{epoch}_epochs.pt"))

            with open(os.path.join("training/runs/train", "losses.json"), "w") as f:
                json.dump({
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_graph_losses": train_graph_losses,
                    "val_graph_losses": val_graph_losses,
                    "train_node_losses": train_node_losses,
                    "val_node_losses": val_node_losses,
                }, f, indent=4)

        epoch_secs = time.time() - epoch_t0
        print(
            f"Epoch {epoch:03d}/{epochs} | {epoch_secs:6.1f}s | "
            f"Train: {avg_train:.6f} (g {avg_train_graph:.6f} / n {avg_train_node:.6f}) | "
            f"Val: {avg_val:.6f} (g {avg_val_graph:.6f} / n {avg_val_node:.6f}) | "
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
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--conv_layers", default=6, type=int)
    parser.add_argument("--node_loss_weight", default=1.0, type=float)
    parser.add_argument("--graph_loss_weight", default=1.0, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--dataset", default=None, type=str)
    args = parser.parse_args()

    train_gnn_model(
        geometry=args.geometry,
        num_samples=args.num_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        conv_layers=args.conv_layers,
        node_loss_weight=args.node_loss_weight,
        graph_loss_weight=args.graph_loss_weight,
        weight_decay=args.weight_decay,
        dataset_path=args.dataset,
    )
