"""One-off validation: take each generation's design from the CNN-surrogate
topopt run (optimization/artifacts/topopt_history/gen_*.npz), re-evaluate it
with a real CCX solve, and compare against the surrogate's stress prediction
for that same generation (already stored in the .npz as s0). Writes a JSON
of per-generation {gen, cnn_stress, ccx_stress} for later plotting.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluators.ccx_evaluator import CCXEvaluator

HIST_DIR = "optimization/artifacts/topopt_history"
OUT_PATH = "analysis_cnn_vs_ccx_validate_result.json"


def load_gen_designs(hist_dir):
    files = sorted(glob.glob(os.path.join(hist_dir, "gen_*.npz")))
    gens, dims_list, cnn_stress, cnn_vol = [], [], [], []
    for f in files:
        gen = int(os.path.basename(f)[4:-4])
        z = np.load(f, allow_pickle=True)
        names = z["free_names"]
        x = z["x"]
        dims_list.append({str(n): float(v) for n, v in zip(names, x)})
        gens.append(gen)
        cnn_stress.append(float(z["s0"]))
        cnn_vol.append(float(z["v0"]))
    return gens, dims_list, cnn_stress, cnn_vol


def main():
    gens, dims_list, cnn_stress, cnn_vol = load_gen_designs(HIST_DIR)
    print(f"[validate] {len(gens)} generations found: {gens[0]}..{gens[-1]}", flush=True)

    max_workers = int(os.environ.get("CCX_WORKERS", "40"))
    evaluator = CCXEvaluator("hollow_cube", max_workers=max_workers, nproc=1, mesh_threads=1,
                              stress_ks_rho_mpa=0.15)
    print(f"[validate] running {len(dims_list)} real CCX solves, max_workers={max_workers}", flush=True)
    result = evaluator.evaluate(dims_list)

    out = {
        "gen": gens,
        "cnn_stress": cnn_stress,
        "cnn_vol": cnn_vol,
        "ccx_stress": result["stress"],
        "ccx_vol": result["volume"],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    n_failed = sum(1 for s in result["stress"] if s is None)
    print(f"[validate] done, {n_failed}/{len(gens)} ccx solves failed. wrote {OUT_PATH}", flush=True)
    for g, cs, rs in zip(gens, cnn_stress, result["stress"]):
        if rs is None:
            print(f"  gen {g:3d} | cnn={cs:8.3f} | ccx=FAILED")
        else:
            print(f"  gen {g:3d} | cnn={cs:8.3f} | ccx={rs:8.3f} | err={cs - rs:+8.3f}")


if __name__ == "__main__":
    main()
