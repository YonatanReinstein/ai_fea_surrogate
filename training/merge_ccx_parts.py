"""Merge independently-built dataset_ccx part directories into one contiguous
shards/ directory a single training run can read.

Motivation: jobs/run_dataset_ccx.slurm now runs several dataset_builder.py
processes in parallel, each with its own smaller ProcessPoolExecutor pool
(pool_size split across parts rather than one big pool) -- see
project_dataset_pool_split memory. This bounds how many workers a single
BrokenProcessPool crash can take down, and how many workers a subsequent
retry has to re-fork under memory pressure, without reducing total
concurrency. Each part is a fully independent, independently-resumable
dataset_builder.py run (its own --output_dir, --seed, --num_samples), so it
never needs to know about the others -- this script is the one-time step
that stitches the parts into a single dataset for training.

Each part is internally contiguous (shard indices 0..N_i-1, since every
dataset_builder.py process numbers its own shards starting from its own
start_idx=0). Renumbering here just offsets each part's shard filenames by
the cumulative sample count of the parts merged before it -- shard CONTENTS
(the serialized Data objects) are copied byte-for-byte, unchanged.

Safe to re-run at any time (including while parts are still running/resuming):
it only ever reads from the part directories and fully rebuilds output_dir's
shards/ from their current state, so a re-run just picks up whatever
progress the parts have made since the last merge.
"""
import argparse
import os
import shutil

from training import dataset_io


def merge(part_dirs, output_dir):
    shards_dir = f"{output_dir}/shards"
    shutil.rmtree(shards_dir, ignore_errors=True)
    os.makedirs(shards_dir, exist_ok=True)

    offset = 0
    for part_dir in part_dirs:
        part_n = dataset_io.num_existing_samples(part_dir)
        for p in dataset_io.shard_paths(part_dir):
            base = os.path.basename(p)[len("shard_"):-len(".pt")]
            s, e = (int(x) for x in base.split("_"))
            new_path = f"{shards_dir}/shard_{s + offset:06d}_{e + offset:06d}.pt"
            shutil.copy2(p, new_path)
        print(f"  {part_dir}: {part_n} samples (offset {offset})", flush=True)
        offset += part_n

    print(f"Merged {offset} samples from {len(part_dirs)} parts into {shards_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge parallel dataset_ccx part directories into one dataset dir.")
    parser.add_argument("--geometry", type=str, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--parts_prefix", type=str, default=None,
                         help="Default: data/<geometry>/dataset_ccx_part (parts are <prefix>0, <prefix>1, ...)")
    parser.add_argument("--output_dir", type=str, default=None, help="Default: data/<geometry>/dataset_ccx")
    args = parser.parse_args()

    prefix = args.parts_prefix or f"data/{args.geometry}/dataset_ccx_part"
    output_dir = args.output_dir or f"data/{args.geometry}/dataset_ccx"
    part_dirs = [f"{prefix}{i}" for i in range(args.num_parts)]
    merge(part_dirs, output_dir)
