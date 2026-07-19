"""Validate the CalculiX (ccx) evaluator against Ansys MAPDL on real samples.

Runs the same set of randomly sampled dims through both MAPDLEvaluator and
CCXEvaluator for a given geometry, and reports per-sample + aggregate error
on stress (MPa), max displacement (m), and volume.

Usage:
    python -m scripts.compare_ccx_ansys --geometry hollow_cube --num_samples 5 --seed 42
"""
import argparse
import json
import random

import torch

from evaluators.ccx_evaluator import CCXEvaluator
from evaluators.mapdl_evaluator import MAPDLEvaluator


def make_dims_list(geometry, num_samples, seed):
    with open(f"data/{geometry}/CAD_model/dims.json") as f:
        dims_template = json.load(f)
    random.seed(seed)
    return [
        {k: random.uniform(v["min"], v["max"]) for k, v in dims_template.items()}
        for _ in range(num_samples)
    ]


def pct_err(a, b):
    if a is None or b is None:
        return None
    denom = abs(a) if abs(a) > 1e-12 else 1.0
    return 100.0 * (b - a) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="hollow_cube")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ccx_path", default="third_party/ccx/bin/ccx_run.sh")
    ap.add_argument("--ccx_workers", type=int, default=2)
    ap.add_argument("--ccx_nproc", type=int, default=4)
    ap.add_argument("--mapdl_pool_size", type=int, default=2)
    ap.add_argument("--mapdl_nproc", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dims_list = make_dims_list(args.geometry, args.num_samples, args.seed)

    print(f"Running {len(dims_list)} sample(s) through CCXEvaluator...", flush=True)
    ccx_ev = CCXEvaluator(
        args.geometry, max_workers=args.ccx_workers, nproc=args.ccx_nproc,
        ccx_path=args.ccx_path,
    )
    ccx_res = ccx_ev.evaluate(dims_list)
    ccx_ev.close()

    print(f"Running {len(dims_list)} sample(s) through MAPDLEvaluator...", flush=True)
    mapdl_ev = MAPDLEvaluator(
        args.geometry, pool_size=args.mapdl_pool_size, nproc=args.mapdl_nproc,
    )
    mapdl_res = mapdl_ev.evaluate(dims_list)
    mapdl_ev.close()

    rows = []
    for i in range(len(dims_list)):
        row = {
            "idx": i,
            "ansys_stress": mapdl_res["stress"][i],
            "ccx_stress": ccx_res["stress"][i],
            "stress_err_pct": pct_err(mapdl_res["stress"][i], ccx_res["stress"][i]),
            "ansys_disp": mapdl_res["disp"][i],
            "ccx_disp": ccx_res["disp"][i],
            "disp_err_pct": pct_err(mapdl_res["disp"][i], ccx_res["disp"][i]),
            "ansys_volume": mapdl_res["volume"][i],
            "ccx_volume": ccx_res["volume"][i],
            "volume_err_pct": pct_err(mapdl_res["volume"][i], ccx_res["volume"][i]),
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
