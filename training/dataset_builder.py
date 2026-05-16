import json, random, torch, os, importlib, signal, sys, time
from datetime import datetime, timedelta
print("Starting dataset_builder.py", flush=True)
import socket
print(f"Running on hostname: {socket.gethostname()}", flush=True)
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
# PyMAPDL defaults to UDS transport on Linux, but MAPDL 2025 R1 only serves
# gRPC over plain TCP — force insecure (TCP) transport so the client connects.
os.environ["PYMAPDL_GRPC_TRANSPORT"] = "insecure"

_pool_ref = None

def _cleanup_and_exit(signum, frame):
    print(f"Signal {signum} received, shutting down pool...", flush=True)
    if _pool_ref is not None:
        try:
            _pool_ref.exit()
        except Exception:
            pass
    sys.exit(1)

signal.signal(signal.SIGTERM, _cleanup_and_exit)
signal.signal(signal.SIGINT, _cleanup_and_exit)

import ansys.mapdl.core.pool as _mapdl_pool
_orig_version_from_path = _mapdl_pool.version_from_path
def _patched_version_from_path(product, path, *args, **kwargs):
    if path and "ansys" in path.lower():
        return 251
    return _orig_version_from_path(product, path, *args, **kwargs)
_mapdl_pool.version_from_path = _patched_version_from_path

from core.component import Component
from core.IritModel import IritModel, IritCModel
from ansys.mapdl.core import MapdlPool
from ansys.mapdl.core.errors import MapdlRuntimeError


def build_dataset(
    geometry: str,
    num_samples: int = 10,
    seed: int = 42,
    pool_size: int = 4,
    nproc: int = 3,
    output_dir: str = None,
    run_location: str = None,
):
    random.seed(seed)
    torch.manual_seed(seed)

    base_path = f"data/{geometry}"
    model_path = f"{base_path}/CAD_model"
    dims_json_path = f"{base_path}/CAD_model/dims.json"
    material_props_path = f"{base_path}/CAD_model/material_properties.json"

    dataset_dir = output_dir if output_dir is not None else f"{base_path}/dataset"
    screenshots_dir = f"{dataset_dir}/screenshots"
    if run_location is None:
        run_location = base_path

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)

    dataset, metadata = [], []

    # if dataset exists, load and continue
    if os.path.exists(f"{dataset_dir}/dataset.pt"):
        print(f"Loading existing dataset from {dataset_dir}/dataset.pt")
        dataset = torch.load(f"{dataset_dir}/dataset.pt", weights_only=False)
        with open(f"{dataset_dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        start_idx = len(dataset)
        print(f"Continuing from sample {start_idx}...")
    else:
        start_idx = 0

    with open(material_props_path, "r") as f:
        material_props = json.load(f)
        young = material_props["young_modulus"]
        poisson = material_props["poisson_ratio"]

    with open(dims_json_path, "r") as f:
        dims_template = json.load(f)

    module = importlib.import_module(f"data.{geometry}.boundary_conditions")
    anchor_condition = module.anchor_condition
    force_pattern = module.force_pattern
    mesh_resolution = module.mesh_resolution
    get_tile_grid = getattr(module, "tile_grid", None)  # optional, geometry-specific

    # Generate ALL dims upfront in the main process from the seeded RNG.
    # When resuming, start at dim index start_idx*2 so resumed runs never
    # overlap with dims (including buffer) used by previous runs.
    dim_offset = start_idx * 2
    samples_needed = num_samples - start_idx
    all_dims = [
        {k: random.uniform(v["min"], v["max"]) for k, v in dims_template.items()}
        for _ in range(dim_offset + samples_needed)
    ]
    print(f"Generated dims [{dim_offset}..{dim_offset + samples_needed - 1}] (offset={dim_offset})", flush=True)

    def run_single_sample(mapdl, dim_idx):
        dims = all_dims[dim_idx]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                exe_path = os.path.join(model_path, "model")
                exe_win_path = os.path.join(model_path, "model.exe")

                if os.path.isfile(exe_path):
                    CAD_model = IritCModel(exe_path, dims_dict=dims)
                elif os.path.isfile(exe_win_path):
                    CAD_model = IritCModel(exe_win_path, dims_dict=dims)
                else:
                    CAD_model = IritModel(os.path.join(model_path, "model.irt"), dims_dict=dims)

                comp = Component(CAD_model, young=young, poisson=poisson)
                U, V, W = mesh_resolution()
                comp.generate_mesh(U=U, V=V, W=W)
                comp.mesh.anchor_nodes_by_condition(anchor_condition)
                comp.mesh.apply_force_by_pattern(force_pattern)
                comp.ansys_sim(mapdl=mapdl, screenshot_path=None)

                tg = get_tile_grid(dims) if get_tile_grid is not None else None
                data = comp.to_graph_with_labels(tile_grid=tg)
                return dim_idx, data, {
                    "id": dim_idx,
                    **dims,
                    "volume": comp.get_volume(),
                    "max_stress": comp.mesh.get_max_stress(),
                }
            except MapdlRuntimeError as e:
                print(f"dim_idx={dim_idx} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying...", flush=True)
                try:
                    mapdl.clear()
                except Exception as clear_err:
                    # Instance is permanently dead. Kill it so the pool monitor
                    # respawns it, then return a failure marker so the worker
                    # thread stays alive and the sample can be re-queued.
                    print(f"dim_idx={dim_idx} MAPDL instance unrecoverable: {clear_err}", flush=True)
                    try:
                        mapdl.exit()
                    except Exception:
                        pass
                    return dim_idx, None, None
            except Exception as e:
                print(f"dim_idx={dim_idx} unexpected error (attempt {attempt+1}/{max_retries}): {e}. Retrying...", flush=True)
                try:
                    mapdl.clear()
                except Exception as clear_err:
                    print(f"dim_idx={dim_idx} MAPDL instance unrecoverable: {clear_err}", flush=True)
                    try:
                        mapdl.exit()
                    except Exception:
                        pass
                    return dim_idx, None, None
        print(f"dim_idx={dim_idx} failed after {max_retries} attempts, skipping.", flush=True)
        return dim_idx, None, None

    print(f"Creating MapdlPool with pool_size={pool_size}, nproc={nproc}, run_location={base_path}", flush=True)
    try:
        pool = MapdlPool(n_instances=pool_size, nproc=nproc, run_location=run_location, license_server_check=False, start_timeout=120)
        global _pool_ref
        _pool_ref = pool
        print("MapdlPool created successfully, starting pool.map()...", flush=True)
    except Exception as e:
        print(f"ERROR creating MapdlPool: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    try:
        pending = list(range(dim_offset, dim_offset + samples_needed))
        completed = start_idx
        t_start = time.time()
        max_passes = 3

        for pass_idx in range(max_passes):
            if not pending:
                break
            print(f"Pass {pass_idx+1}/{max_passes}: processing {len(pending)} samples...", flush=True)
            failed_this_pass = []

            for submitted_idx, result in zip(pending, pool.map(run_single_sample, pending)):
                # run_single_sample always returns (dim_idx, data_or_None, meta_or_None);
                # None-tuple indicates failure/skip (either out-of-retries or dead instance).
                if result is None or result[1] is None:
                    failed_this_pass.append(submitted_idx)
                    print(f"dim_idx={submitted_idx} skipped, will retry in next pass.", flush=True)
                    continue
                dim_idx, data, meta = result
                dataset.append(data)
                metadata.append(meta)
                completed += 1

                elapsed = time.time() - t_start
                done_so_far = completed - start_idx
                remaining_count = num_samples - completed
                eta_str = ""
                if done_so_far > 0:
                    eta_sec = elapsed / done_so_far * remaining_count
                    eta_str = f"  ETA {timedelta(seconds=int(eta_sec))}"
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] [{completed:04d}/{num_samples}] {geometry}: σmax={meta['max_stress']:.2e}{eta_str}", flush=True)

                if completed % 100 == 0:
                    torch.save(dataset, f"{dataset_dir}/dataset.pt")
                    with open(f"{dataset_dir}/metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)
                    print(f"--- Checkpoint saved at sample {completed} ---", flush=True)

            pending = failed_this_pass

        if pending:
            print(f"WARNING: {len(pending)} samples failed after {max_passes} passes: {pending}", flush=True)

    finally:
        pool.exit()

    torch.save(dataset, f"{dataset_dir}/dataset.pt")
    with open(f"{dataset_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Dataset saved to {dataset_dir}/dataset.pt")

    return dataset, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FEA dataset for given geometry.")
    parser.add_argument("--geometry", type=str, default="arm", help="Geometry name (e.g., 'beam', 'arm').")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--pool_size", type=int, default=4, help="Number of parallel MAPDL instances.")
    parser.add_argument("--nproc", type=int, default=1, help="CPUs per MAPDL instance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save dataset (default: data/<geometry>/dataset).")
    parser.add_argument("--run_location", type=str, default=None, help="Working directory for MAPDL instances (default: data/<geometry>).")
    args = parser.parse_args()
    build_dataset(args.geometry, num_samples=args.num_samples, pool_size=args.pool_size, nproc=args.nproc, seed=args.seed, output_dir=args.output_dir, run_location=args.run_location)
