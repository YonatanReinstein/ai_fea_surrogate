"""Filter a sharded dataset to samples whose KS stress lies in [ks_min, ks_max] MPa.

Each input shard's survivors become one output shard (variable size, so the
kept ~60% is written once and never re-read to repack into 100-sample shards).
Output shards keep the standard shard_{start:06d}_{end:06d}.pt naming with
contiguous indices, in input order. gnn_training reads per-shard sizes from
every name, so variable sizes are fine. Shards are read in parallel because a
single NFS stream manages only ~13 MB/s (322 GB would take ~7 h).

KS is ks_stress / 1e6 (Pa -> MPa), the trainer's KS target. On
data/hollow_cube/dataset_gradadj it agrees with ks_stress_adjoint to 4e-5.

    python -m training.filter_dataset --src data/hollow_cube/dataset_gradadj \
        --dst data/hollow_cube/dataset_ks40_220 --ks_min 40 --ks_max 220 --workers 12
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch


def _filter_one(args):
    i, src, tmp_dir, ks_min, ks_max = args
    from training.dataset_io import load_shard
    samples = load_shard(src)
    keep = [s for s in samples if ks_min <= float(s.ks_stress.view(-1)[0]) / 1e6 <= ks_max]
    tmp = os.path.join(tmp_dir, f"part_{i:05d}.pt")
    if keep:
        torch.save(keep, tmp)
    return i, len(samples), len(keep), (tmp if keep else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ks_min", type=float, required=True)
    ap.add_argument("--ks_max", type=float, required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    from training.dataset_io import shard_paths
    srcs = shard_paths(a.src)
    assert srcs, f"no shards under {a.src}/shards"
    out_dir = os.path.join(a.dst, "shards")
    assert not os.path.exists(out_dir) or not os.listdir(out_dir), f"{out_dir} is not empty"
    tmp_dir = os.path.join(a.dst, "_tmp_parts")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    g = os.path.join(os.path.abspath(a.src), "graph.pt")
    if os.path.exists(g) and not os.path.exists(os.path.join(a.dst, "graph.pt")):
        os.symlink(g, os.path.join(a.dst, "graph.pt"))

    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_filter_one, (i, p, tmp_dir, a.ks_min, a.ks_max)) for i, p in enumerate(srcs)]
        for n_done, f in enumerate(as_completed(futs), 1):
            i, n_in, n_keep, tmp = f.result()
            res[i] = (n_in, n_keep, tmp)
            if n_done % 20 == 0 or n_done == len(srcs):
                tin = sum(r[0] for r in res.values()); tk = sum(r[1] for r in res.values())
                el = time.time() - t0
                print(f"[filter] {n_done}/{len(srcs)} shards, kept {tk}/{tin} ({100*tk/max(tin,1):.1f}%), "
                      f"{el/60:.1f} min, ETA {el/n_done*(len(srcs)-n_done)/60:.0f} min", flush=True)

    # rename the parts, in INPUT order, to contiguous shard names
    start = 0
    for i in range(len(srcs)):
        n_in, n_keep, tmp = res[i]
        if tmp is None:
            continue
        os.replace(tmp, os.path.join(out_dir, f"shard_{start:06d}_{start + n_keep - 1:06d}.pt"))
        start += n_keep
    os.rmdir(tmp_dir)
    total_in = sum(r[0] for r in res.values())
    meta = {"src": a.src, "ks_min_mpa": a.ks_min, "ks_max_mpa": a.ks_max,
            "n_in": total_in, "n_kept": start, "n_shards": len(os.listdir(out_dir)),
            "ks_field": "ks_stress / 1e6"}
    json.dump(meta, open(os.path.join(a.dst, "filter_meta.json"), "w"), indent=2)
    print(f"[filter] done: kept {start}/{total_in} ({100*start/total_in:.1f}%) in "
          f"{(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
