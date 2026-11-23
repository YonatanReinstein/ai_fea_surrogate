"""
Compares per-tile FD-sensitivity accuracy (d ks_stress / d tile thickness)
between two GNN surrogate checkpoints against real-CCX ground truth, on a
held-out set neither checkpoint's training ever saw. This is the actual
test of whether training/gnn_training.py's --grad_dataset_path loss fixed
the surrogate's broken sensitivities (see project_gradient_matching_loss.md,
project_surrogate_sensitivity_vs_value_gap.md): both checkpoints' *value*
predictions can look fine while their *gradients* -- what
optimization/topopt_optimizer.py's OC update actually consumes as dg --
are 10-280x off and sign-flipped, per the original probe.

Run: python -m evaluators.probe_gradloss_sensitivity \
    --geometry hollow_cube \
    --treatment_ckpt data/hollow_cube/gnn_surrogate.pt \
    --baseline_ckpt data/hollow_cube_ddptest/gnn_surrogate.pt \
    --holdout_dataset data/hollow_cube/dataset_grad_audit/dataset.pt
"""
import argparse
import importlib

import torch
from torch_geometric.data import Batch

from datasets.dataset_io import load_dataset
from training.gnn_training import gnn_input_fn, LOG_STRESS_EPS
from utils.gnn_surrogate import GNN, HierarchicalGNN
from utils.gnn_hierarchical_mp import HierarchicalGNNMP
from utils.gnn_hierarchical_cnn import HierarchicalGNNCNN
from core.tile_sensitivity import dims_key_order_permutation


def load_model(ckpt_path, geometry, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    node_target_mean = ckpt["node_target_mean"].to(device)
    node_target_std = ckpt["node_target_std"].to(device)
    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    edge_mean = ckpt["edge_mean"].to(device)
    edge_std = ckpt["edge_std"].to(device)
    stress_ks_rho_mpa = ckpt.get("stress_ks_rho_mpa", 0.15)

    hierarchical = ckpt.get("hierarchical", False)
    coarse_arch = ckpt.get("coarse_arch", "transformer")
    ModelClass = ({"transformer": HierarchicalGNN, "gnn": HierarchicalGNNMP,
                   "cnn": HierarchicalGNNCNN}[coarse_arch] if hierarchical else GNN)
    model_kwargs = dict(
        node_in_dim=ckpt["node_in_dim"],
        edge_in_dim=ckpt.get("edge_in_dim", 1),
        hidden_dim=ckpt.get("hidden_dim") or ckpt["model_state"]["encoder.lins.0.bias"].shape[0],
        num_layers=ckpt["conv_layers"],
        node_target_mean=node_target_mean,
        node_target_std=node_target_std,
        stress_ks_rho_mpa=stress_ks_rho_mpa,
    )
    if hierarchical:
        model_kwargs["num_pos_nodes"] = ckpt.get("num_pos_nodes", 0)
        model_kwargs["tile_in_dim"] = ckpt.get("tile_in_dim", 1)
        if coarse_arch == "transformer":
            model_kwargs["transformer_heads"] = ckpt["transformer_heads"]
            model_kwargs["transformer_ff_mult"] = ckpt["transformer_ff_mult"]
        elif coarse_arch == "gnn":
            model_kwargs["coarse_dim"] = ckpt.get("coarse_dim")
        else:
            model_kwargs["coarse_dim"] = ckpt.get("coarse_dim")
            model_kwargs["cnn_depth"] = ckpt.get("cnn_depth", 3)
            # Rebuild the per-node dKS/dX head when the checkpoint has one.
            # Checkpoints written before "predict_node_grad" was recorded
            # still carry grad_head.* weights, so fall back to sniffing the
            # state_dict -- otherwise the model is built without the head
            # and load_state_dict drops those tensors on the floor.
            model_kwargs["predict_node_grad"] = ckpt.get(
                "predict_node_grad",
                any(k.startswith("grad_head.") for k in ckpt["model_state"]))
    model = ModelClass(**model_kwargs).to(device)

    if hierarchical:
        # Must happen before load_state_dict: HierarchicalGNNCNN lazily builds
        # its U-Net submodules inside set_tile_grid (see evaluators/gnn_evaluator.py).
        module = importlib.import_module(f"data.{geometry}.boundary_conditions")
        get_tile_grid = getattr(module, "tile_grid")
        NX, NY, NZ = get_tile_grid(None)
        model.set_tile_grid(NX, NY, NZ)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # how the checkpoint expects its load (read by probe_dg_alignment)
    model.tile_load_input = bool(ckpt.get("tile_load_input", False))
    model.tile_load_scale_n = float(ckpt.get("tile_load_scale_n", 5e7))
    model.ks_exclude = ckpt.get("ks_exclude")
    return model, (x_mean, x_std, edge_mean, edge_std, node_target_mean, node_target_std)


def compute_sensitivities(model, stats, dataset, device, strut_offset):
    x_mean, x_std, edge_mean, edge_std, node_target_mean, node_target_std = stats
    preds, trues = [], []
    # dims-key order (d1..dK) is NOT tile_idx's own (IRIT-native) order
    # whenever NX != NY (see core/tile_sensitivity.py's
    # dims_key_order_permutation) -- fixed topology across the dataset, so
    # computed once from the first sample and reused.
    _NX0, _NY0, _NZ0 = int(dataset[0].tile_NX), int(dataset[0].tile_NY), int(dataset[0].tile_NZ)
    strut_perm = torch.tensor(dims_key_order_permutation(_NX0, _NY0, _NZ0), dtype=torch.long, device=device)
    strut_perm_inv = torch.argsort(strut_perm)
    for sample in dataset:
        batch = Batch.from_data_list([sample])
        x, edge_index, edge_attr, batch_idx, tile_idx, tile_NX, tile_NY, tile_NZ, tile_x, node_id = gnn_input_fn(batch)
        K = int(sample.tile_NX) * int(sample.tile_NY) * int(sample.tile_NZ)

        tile_struts_dims_key_order = sample.dims[0, strut_offset:strut_offset + K].float().to(device)
        tile_struts = tile_struts_dims_key_order[strut_perm]
        t_leaf = tile_struts.view(-1, 1).clone().requires_grad_(True)
        batch_idx = batch_idx.to(device)
        tile_idx = tile_idx.to(device)
        node_strut = t_leaf.view(1, K)[batch_idx, tile_idx].unsqueeze(1)

        x = x.float().to(device)
        x[:, 3:6] = x[:, 3:6] / 1e+6  # force vector -> MN, matches _preprocess in gnn_training.py
        x_grad = torch.cat([x, node_strut], dim=1)
        x_norm = (x_grad - x_mean) / x_std
        edge_attr_norm = (edge_attr.float().to(device) - edge_mean) / edge_std

        graph_pred, _ = model(
            x_norm, edge_index.to(device), edge_attr_norm, batch_idx,
            tile_idx=tile_idx, tile_NX=tile_NX, tile_NY=tile_NY, tile_NZ=tile_NZ,
            tile_x=t_leaf, num_graphs=1,
            node_id=node_id.to(device) if node_id is not None else None,
        )
        graph_pred_raw_mpa = (graph_pred * node_target_std + node_target_mean).clamp_min(LOG_STRESS_EPS)

        d_pred = torch.autograd.grad(graph_pred_raw_mpa.sum(), t_leaf)[0].view(-1)

        # grad_tile_idx is stored in dims-key order (dataset_builder.py's
        # _ccx_run_grad_sample builds it from "d{t+1}" keys); d_pred's
        # columns are tile_idx-native (t_leaf's own layout) -- convert
        # before indexing.
        grad_tile_idx = strut_perm_inv[sample.grad_tile_idx.to(device)]
        # Same unit trap as training/gnn_training.py's _preprocess: the two
        # --grad_mode builders store this label in DIFFERENT units.
        #   fd      -- dataset_builder._ccx_run_grad_sample finite-differences
        #              raw ccx ks_stress, giving Pa/unit, so it needs /1e6.
        #   adjoint -- dataset_builder._adjoint_run_grad_sample ALREADY divides
        #              by 1e6 when storing the label; dividing again makes the
        #              ground truth 1e6x too small and every model then looks
        #              ~1e6x "too large".
        # ks_stress_adjoint is written only by the adjoint builder, so it is
        # the reliable discriminator.
        grad_dks_dx_true = sample.grad_dks_dx.float()
        if not hasattr(sample, "ks_stress_adjoint"):
            grad_dks_dx_true = grad_dks_dx_true / 1e6  # Pa/unit -> MPa/unit

        preds.append(d_pred[grad_tile_idx].detach().cpu())
        trues.append(grad_dks_dx_true)

    return torch.cat(preds), torch.cat(trues)


def report(name, preds, trues):
    eps = 1e-8
    sign_match = (torch.sign(preds) == torch.sign(trues)).float().mean().item()
    # Ratio of magnitudes -- how many x too large/small the prediction is,
    # matching the "10-280x too large" framing used to describe the original bug.
    ratio = (preds.abs() + eps) / (trues.abs() + eps)
    log_ratio = ratio.log10()
    median_abs_log_ratio = log_ratio.abs().median().item()
    median_ratio = ratio.median().item()
    mae_mpa = (preds - trues).abs().mean().item()
    print(f"\n=== {name} (N={len(preds)} tile-sensitivities) ===")
    print(f"  sign agreement:            {sign_match * 100:.1f}%")
    print(f"  median |pred/true| ratio:  {median_ratio:.2f}x "
          f"(median magnitude error: 10^{median_abs_log_ratio:.2f} = {10**median_abs_log_ratio:.1f}x)")
    print(f"  mean |pred - true|:        {mae_mpa:.4f} MPa/unit")
    print(f"  mean |true|:               {trues.abs().mean().item():.4f} MPa/unit  "
          f"(scale reference)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default="hollow_cube")
    parser.add_argument("--treatment_ckpt", required=True,
                        help="Checkpoint trained WITH --grad_dataset_path.")
    parser.add_argument("--baseline_ckpt", required=True,
                        help="Checkpoint trained WITHOUT --grad_dataset_path (otherwise identical).")
    parser.add_argument("--old_production_ckpt", default=None,
                        help="Optional: the original full-scale production checkpoint, "
                             "for context only (confounded by dataset-size/epoch differences).")
    parser.add_argument("--holdout_dataset", required=True,
                        help="Path to a --grad_mode dataset neither checkpoint trained on.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = importlib.import_module(f"data.{args.geometry}.boundary_conditions")
    fixed_dims = getattr(module, "fixed_dims", lambda: {})()
    strut_offset = 0 if fixed_dims else 3

    dataset = load_dataset(args.holdout_dataset)
    assert dataset and hasattr(dataset[0], "grad_tile_idx"), \
        f"{args.holdout_dataset} has no grad_tile_idx/grad_dks_dx -- build with --grad_mode true"
    print(f"Held-out audit set: {len(dataset)} samples, "
          f"{len(dataset) * dataset[0].grad_tile_idx.shape[0]} tile-sensitivity pairs")

    ckpts = {"baseline (no grad-loss)": args.baseline_ckpt,
             "treatment (grad-loss)": args.treatment_ckpt}
    if args.old_production_ckpt:
        ckpts["old production (context only)"] = args.old_production_ckpt

    results = {}
    for name, ckpt_path in ckpts.items():
        model, stats = load_model(ckpt_path, args.geometry, device)
        preds, trues = compute_sensitivities(model, stats, dataset, device, strut_offset)
        results[name] = (preds, trues)
        report(name, preds, trues)

    print("\n=== Summary ===")
    for name, (preds, trues) in results.items():
        sign_match = (torch.sign(preds) == torch.sign(trues)).float().mean().item() * 100
        ratio = ((preds.abs() + 1e-8) / (trues.abs() + 1e-8)).log10().abs().median().item()
        print(f"  {name:32s} sign={sign_match:5.1f}%  median magnitude error=10^{ratio:.2f}={10**ratio:.1f}x")
