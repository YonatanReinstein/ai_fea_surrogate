"""Convert a build_cross4_dataset output directory to the slim sample format:
x (N,10) -> coords (N,3), with the static columns x[:, 3:] (forces, anchored,
tile position -- identical for every design at a load row) stored once per
row as graph_row{k}.pt's x_static.

Every sample's static columns are checked against its row's x_static before
they are dropped; a mismatch aborts that shard untouched. Shards are rewritten
atomically (tmp + rename) and samples already in the slim format are left
alone, so the script is safe to rerun and to run while the builder appends new
(already slim) shards.

    python -m training.repack_cross4_shards data/cross4_11x11/dataset_rows_30000 --workers 8
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import get_context

import torch


def add_static_to_graphs(out):
    """x_static for every graph_row file that lacks it, from one old-format sample."""
    from training import dataset_io
    missing = {int(os.path.basename(p)[9:-3]) for p in glob.glob(f"{out}/graph_row*.pt")
               if "x_static" not in torch.load(p, weights_only=False)}
    for f in sorted(glob.glob(f"{out}/shards/*.pt")):
        if not missing:
            break
        for s in dataset_io.load_shard(f):
            r = int(s.load_row)
            if r in missing and "x" in s:
                gp = f"{out}/graph_row{r}.pt"
                g = torch.load(gp, weights_only=False)
                g["x_static"] = s.x[:, 3:].clone()
                torch.save(g, gp + ".tmp")
                os.replace(gp + ".tmp", gp)
                missing.discard(r)
                print(f"graph_row{r}.pt: x_static added", flush=True)
    if missing:
        sys.exit(f"no old-format sample found for rows {sorted(missing)}")


def repack(job):
    path, statics = job
    from training import dataset_io
    torch.set_num_threads(1)
    samples = dataset_io.load_shard(path)
    if all("x" not in s for s in samples):
        return path, 0, "already slim"
    for s in samples:
        if "x" not in s:
            continue
        r = int(s.load_row)
        if not torch.equal(s.x[:, 3:], statics[r]):
            return path, -1, f"static columns differ from graph_row{r}.pt (sample geometry_idx {int(s.geometry_idx)})"
        s.coords = s.x[:, :3].clone()
        del s.x
    torch.save(samples, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path, len(samples), "ok"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("output_dir")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    out = args.output_dir
    add_static_to_graphs(out)
    statics = {int(os.path.basename(p)[9:-3]): torch.load(p, weights_only=False)["x_static"]
               for p in glob.glob(f"{out}/graph_row*.pt")}
    shards = sorted(glob.glob(f"{out}/shards/*.pt"))
    print(f"{len(shards)} shards", flush=True)
    t0, n_done, n_bad = time.time(), 0, 0
    with get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool:
        for k, (path, n, msg) in enumerate(pool.imap_unordered(repack, [(s, statics) for s in shards]), 1):
            if n < 0:
                n_bad += 1
                print(f"{os.path.basename(path)}: NOT repacked: {msg}", flush=True)
            n_done += max(n, 0)
            if k % 25 == 0 or k == len(shards):
                print(f"[{k}/{len(shards)}] {n_done} samples repacked, {n_bad} shards refused, "
                      f"{time.time() - t0:.0f}s", flush=True)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
