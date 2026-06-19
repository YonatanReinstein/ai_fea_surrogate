"""Probe: does the GNN surrogate's max-stress prediction vary with dims?
Run: python -m analysis.probe_gnn_stress
"""
from evaluators.gnn_evaluator import GNNEvaluator


def main():
    ev = GNNEvaluator("hollow_cube", processes=4, batch_size=8)
    names = [f"d{i + 1}" for i in range(8)]
    tests = [
        {n: 0.05 for n in names},                                   # all thin
        {n: 0.45 for n in names},                                   # all thick
        {n: (0.05 if i < 4 else 0.45) for i, n in enumerate(names)},  # mixed
        {n: 0.25 for n in names},                                   # mid
    ]
    res = ev.evaluate(tests)
    print("=== PROBE RESULTS ===")
    for t, v, s in zip(tests, res["volume"], res["stress"]):
        print("dims", [round(t[n], 2) for n in names],
              "vol=%.4f" % v, "stress=%.6e" % s)


if __name__ == "__main__":
    main()
