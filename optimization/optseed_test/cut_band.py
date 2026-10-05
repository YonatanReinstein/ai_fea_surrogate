"""Cut an existing solved dataset to a true-stress band. No FEA -- a filtered copy.

Unlike the screened build this uses the REAL ccx stress on every sample, so there
is no predictor bias: measured on hollow_cube_11x11, the same two-criterion weight
gave contrast 23.5 when applied to true stress against 7.3 when applied to a
predicted one. Cutting is strictly better data; its only limit is that it cannot
produce more samples than the source already contains.
"""
import os, sys, argparse, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training import dataset_io

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lo", type=float, default=40.0)
ap.add_argument("--hi", type=float, default=220.0)
ap.add_argument("--shard", type=int, default=100)
a = ap.parse_args()
os.makedirs(f"{a.out}/shards", exist_ok=True)
src = dataset_io.shard_paths(a.src)
print(f"[cut] {len(src)} source shards -> keeping {a.lo}-{a.hi} MPa", flush=True)

buf, start, kept, seen, t0 = [], 0, 0, 0, time.time()
ks_kept = []
for i, p in enumerate(src):
    for s in dataset_io.load_shard(p):
        seen += 1
        v = float(np.asarray(s.ks_stress).ravel()[0]) / 1e6
        if a.lo <= v <= a.hi:
            buf.append(s); kept += 1; ks_kept.append(v)
            if len(buf) == a.shard:
                dataset_io.save_shard(a.out, start, buf); start += a.shard; buf = []
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print(f"[cut] {i+1}/{len(src)} shards | seen {seen} kept {kept} "
              f"({100*kept/max(seen,1):.1f}%) | {el:.0f}s | ETA {(len(src)-i-1)*el/(i+1)/60:.0f} min", flush=True)
if buf:
    dataset_io.save_shard(a.out, start, buf)
k = np.array(ks_kept)
print(f"[done] {dataset_io.num_existing_samples(a.out)} samples in {a.out}/shards "
      f"from {seen} source ({100*kept/max(seen,1):.1f}% yield) in {time.time()-t0:.0f}s", flush=True)
print(f"[done] KS kept: min {k.min():.1f}  median {np.median(k):.1f}  max {k.max():.1f} MPa", flush=True)
