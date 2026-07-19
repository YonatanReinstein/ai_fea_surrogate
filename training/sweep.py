import argparse
import json
import os
import time

from training.gnn_training import train_gnn_model
from training.utils import plot_losses

# ======================================================
# Sweep grid — edit this to change what gets tried.
# Each entry overrides BASE_CONFIG for one combo.
#
# stress_weight_power=3.0 is deliberately excluded: already tested on the
# full dataset and it widened the train/val gap without lowering the
# achievable val graph loss (see training/runs/gnn_iteration_tracking.pptx).
# 0/0.1/0.5/1.0/2.0 covers uniform-weighting through moderately
# stress-concentrated, including the untested low end.
# ======================================================
SWEEP_GRID = [
    {"stress_weight_power": swp, "weight_decay": wd, "coarse_arch": arch}
    for swp in [0, 0.1, 0.5, 1.0, 2.0]
    for wd in [2e-4, 1e-3, 3e-3]
    for arch in ["transformer", "gnn"]
]

BASE_CONFIG = dict(
    geometry="hollow_cube",
    dataset_path="data/hollow_cube/dataset/dataset_200.pt",
    num_samples=100,
    epochs=200,
    lr=1e-3,
    batch_size=6,
    hidden_dim=128,
    conv_layers=6,
    node_loss_weight=1.0,
    graph_loss_weight=0.0,
    use_checkpoint=False,
    checkpoint_fine_only=False,
    use_node_emb=False,
)


def run_name_for(overrides):
    swp = overrides.get("stress_weight_power")
    wd = overrides.get("weight_decay")
    arch = overrides.get("coarse_arch")
    return f"sweep_swp{swp:g}_wd{wd:g}_{arch}"


def summarize(run_name, data):
    vg = data.get("val_graph_losses", [])
    if not vg:
        return {"run_name": run_name, "best_val_graph_loss": None, "best_epoch": None, "epochs": 0}
    best = min(vg)
    return {
        "run_name": run_name,
        "best_val_graph_loss": best,
        "best_epoch": vg.index(best) + 1,
        "final_val_graph_loss": vg[-1],
        "epochs": len(vg),
    }


def write_summary(results, sweep_root, final=False):
    ranked = sorted(
        (r for r in results if r["best_val_graph_loss"] is not None),
        key=lambda r: r["best_val_graph_loss"],
    )
    with open(os.path.join(sweep_root, "sweep_summary.json"), "w") as f:
        json.dump(ranked, f, indent=2)

    header = "FINAL" if final else "current"
    print(f"\n=== {header} sweep ranking (best val_graph_loss first) ===")
    for r in ranked:
        print(f"  {r['best_val_graph_loss']:.4f} @ epoch {r['best_epoch']:>4d}  "
              f"(final={r['final_val_graph_loss']:.4f}, {r['epochs']} epochs)  {r['run_name']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_root", default="training/runs", type=str,
                         help="Parent dir; each combo gets its own <sweep_root>/<run_name>/ folder.")
    parser.add_argument("--checkpoint_root", default="data/hollow_cube/checkpoints_sweep", type=str,
                         help="Parent dir for each combo's checkpoints, isolated from the "
                              "single-run checkpoint dir so combos never resume from each other's "
                              "weights and don't collide with a manually-submitted train_model.slurm run.")
    args = parser.parse_args()

    print(f"[sweep] {len(SWEEP_GRID)} combos queued")

    results = []
    for i, overrides in enumerate(SWEEP_GRID):
        run_name = run_name_for(overrides)
        run_dir = os.path.join(args.sweep_root, run_name)
        checkpoint_dir = os.path.join(args.checkpoint_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        target_epochs = overrides.get("epochs", BASE_CONFIG["epochs"])
        losses_path = os.path.join(run_dir, "losses.json")
        if os.path.exists(losses_path):
            with open(losses_path) as f:
                prev = json.load(f)
            if len(prev.get("val_graph_losses", [])) >= target_epochs:
                print(f"[sweep {i + 1}/{len(SWEEP_GRID)}] {run_name}: already complete, skipping")
                results.append(summarize(run_name, prev))
                write_summary(results, args.sweep_root)
                continue

        print(f"[sweep {i + 1}/{len(SWEEP_GRID)}] {run_name}: starting  overrides={overrides}")
        t0 = time.time()
        cfg = {**BASE_CONFIG, **overrides, "run_dir": run_dir, "checkpoint_dir": checkpoint_dir}
        train_gnn_model(**cfg)
        print(f"[sweep {i + 1}/{len(SWEEP_GRID)}] {run_name}: done in {time.time() - t0:.0f}s")

        plot_losses(run_dir=run_dir)
        with open(losses_path) as f:
            data = json.load(f)
        results.append(summarize(run_name, data))
        write_summary(results, args.sweep_root)

    write_summary(results, args.sweep_root, final=True)


if __name__ == "__main__":
    main()
