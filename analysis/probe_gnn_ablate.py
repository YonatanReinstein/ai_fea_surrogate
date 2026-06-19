"""Ablation: is the GNN's design-insensitivity caused by the per-node embedding?
Compare predicted max-stress spread across very different designs, with the
node embedding intact vs zeroed.
Run: python -m analysis.probe_gnn_ablate
"""
import numpy as np
import torch
from evaluators.gnn_evaluator import GNNEvaluator


def spread(ev, tag):
    names = [f"d{i + 1}" for i in range(8)]
    tests = [
        {n: 0.05 for n in names},
        {n: 0.45 for n in names},
        {n: (0.05 if i < 4 else 0.45) for i, n in enumerate(names)},
        {n: 0.25 for n in names},
    ]
    s = np.array(ev.evaluate(tests)["stress"], dtype=float)
    rng = (s.max() - s.min()) / s.mean()
    print(f"[{tag}] stress min={s.min():.4e} max={s.max():.4e} "
          f"rel_spread={rng:.3e}")
    return s


def main():
    ev = GNNEvaluator("hollow_cube", processes=4, batch_size=8)
    spread(ev, "node_emb INTACT")

    if getattr(ev.model, "node_emb", None) is not None:
        with torch.no_grad():
            ev.model.node_emb.weight.zero_()
        spread(ev, "node_emb ZEROED")
    else:
        print("model has no node_emb")


if __name__ == "__main__":
    main()
