"""
Sharded on-disk format for CCX-generated datasets.

Motivation: dataset_builder's ccx path used to hold every completed sample in
one growing Python list and re-serialize the whole list to a single dataset.pt
every 100 samples. For hollow_cube (~20MB/sample once edge_index duplication
is removed, ~6.5MB/sample) a 50k-sample run would balloon past a job's memory
limit well before finishing, since the whole history had to stay resident in
RAM just to write it out again. Splitting into fixed-size shard files lets the
builder drop old samples from memory the moment they're durably saved.

Layout: <dataset_dir>/shards/shard_{start:06d}_{end:06d}.pt, each a plain list
of torch_geometric Data objects (same schema as before).
"""
import glob
import os
import torch


def shard_paths(dataset_dir: str) -> list:
    return sorted(glob.glob(os.path.join(dataset_dir, "shards", "shard_*.pt")))


def num_existing_samples(dataset_dir: str) -> int:
    total = 0
    for p in shard_paths(dataset_dir):
        base = os.path.basename(p)[len("shard_"):-len(".pt")]
        s, e = base.split("_")
        total += int(e) - int(s) + 1
    return total


def save_shard(dataset_dir: str, start_idx: int, samples: list) -> str:
    os.makedirs(os.path.join(dataset_dir, "shards"), exist_ok=True)
    end_idx = start_idx + len(samples) - 1
    path = os.path.join(dataset_dir, "shards", f"shard_{start_idx:06d}_{end_idx:06d}.pt")
    torch.save(samples, path)
    return path


def load_sharded_dataset(dataset_dir: str) -> list:
    dataset = []
    for p in shard_paths(dataset_dir):
        dataset.extend(torch.load(p, weights_only=False))
    return dataset


def load_dataset(dataset_path: str) -> list:
    """Back-compat loader: if dataset_path's directory has a shards/
    subdirectory, load+concatenate those; otherwise torch.load the file
    directly (legacy single-file datasets, e.g. the ansys/MAPDL pipeline)."""
    d = os.path.dirname(dataset_path)
    if os.path.isdir(os.path.join(d, "shards")):
        return load_sharded_dataset(d)
    return torch.load(dataset_path, weights_only=False)


def migrate_legacy_dataset(dataset_dir: str, shard_size: int = 100) -> int:
    """One-time migration of a legacy single dataset.pt into shards/,
    canonicalizing edge_index (identical across samples for fixed-topology
    geometries) to a single shared tensor so it's stored once per shard
    instead of once per sample. Returns the number of migrated samples.
    Leaves the original file renamed to dataset.pt.bak."""
    legacy_path = os.path.join(dataset_dir, "dataset.pt")
    if not os.path.exists(legacy_path):
        return 0
    legacy = torch.load(legacy_path, weights_only=False)
    canonical_edge_index = legacy[0].edge_index if legacy else None
    for sample in legacy:
        sample.edge_index = canonical_edge_index
    for start in range(0, len(legacy), shard_size):
        save_shard(dataset_dir, start, legacy[start:start + shard_size])
    n = len(legacy)
    del legacy
    os.rename(legacy_path, legacy_path + ".bak")
    return n
