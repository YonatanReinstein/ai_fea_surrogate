import json, random, torch, os, importlib, tempfile, shutil
from multiprocessing import Pool

from core.component import Component
from core.IritModel import IIritModel
from ansys.mapdl.core import launch_mapdl
from ansys.mapdl.core.errors import MapdlRuntimeError
import random

def get_free_port():
    return random.randint(10000, 59999)




# ================================================================
# ===============  WORKER FUNCTION (RUNS 1 SAMPLE)  ==============
# ================================================================

def _run_sample_worker(args):
    (
        geometry,
        dims_json_path,
        model_path,
        young,
        poisson
    ) = args

    # Unique temp directory for MAPDL instance
    workdir = tempfile.mkdtemp(prefix="mapdl_tmp_")

    # Load dimension template once per worker call
    with open(dims_json_path, "r") as f:
        dims_template = json.load(f)

    # Retry loop (MAPDL sometimes crashes randomly)
    while True:
        try:
            # Random dimension sampling
            dims = {
                k: random.uniform(v["min"], v["max"])
                for k, v in dims_template.items()
            }

            # Create model & component
            CAD_model = IIritModel(model_path, dims_dict=dims)
            comp = Component(CAD_model, young=young, poisson=poisson)

            # Load problem-specific boundary conditions
            module = importlib.import_module(f"data.{geometry}.boundary_conditions")
            anchor_condition = module.anchor_condition
            force_pattern = module.force_pattern
            U, V, W = module.mesh_resolution()

            # Build mesh and apply BCs
            comp.generate_mesh(U=U, V=V, W=W)
            comp.mesh.anchor_nodes_by_condition(anchor_condition)
            comp.mesh.apply_force_by_pattern(force_pattern)

            mapdl = launch_mapdl(
                run_location=workdir,
                jobname="job",
                port=get_free_port(),
                nproc=1,
                override=True,
                loglevel="ERROR"
            )


            # Solve using this MAPDL instance
            comp.mesh.solve(
                young=young,
                poisson=poisson,
                mapdl=mapdl           # <--- ONLY REQUIRED CHANGE IN SOLVE()
            )

            # Convert to graph
            data = comp.to_graph_with_labels()

            # Metadata
            meta = {
                **dims,
                "volume": comp.get_volume(),
                "max_stress": comp.mesh.get_max_stress(),
            }

            mapdl.exit()
            break  # solved successfully

        except MapdlRuntimeError as e:
            print(f"[Worker] MAPDL crashed, retrying... {e}")
            continue

        except Exception as e:
            print(f"[Worker] Unexpected error in worker: {e}")
            raise e

        finally:
            # Always clean the working directory
            if os.path.exists(workdir):
                shutil.rmtree(workdir)

    return data, meta



# ================================================================
# ================  PARALLEL DATASET BUILDER  ====================
# ================================================================

def build_dataset(
    geometry: str,
    num_samples: int = 10,
    young: float = 2e11,
    poisson: float = 0.3,
    seed: int = 42,
    workers: int = 4
):
    random.seed(seed)

    base_path = f"data/{geometry}"
    model_path = f"{base_path}/CAD_model/model.irt"
    dims_json_path = f"{base_path}/CAD_model/dims.json"
    dataset_dir = f"{base_path}/dataset"

    os.makedirs(dataset_dir, exist_ok=True)

    # Prepare worker arguments
    worker_args = [
        (geometry, dims_json_path, model_path, young, poisson)
        for _ in range(num_samples)
    ]

    print(f"\n🚀 Starting {num_samples} samples with {workers} parallel workers...\n")

    dataset = []
    metadata = []

    # Parallel execution
    with Pool(processes=workers) as pool:
        for idx, (data, meta) in enumerate(
            pool.imap_unordered(_run_sample_worker, worker_args),
            start=1
        ):
            meta["id"] = idx
            dataset.append(data)
            metadata.append(meta)

            print(f"[{idx:03d}/{num_samples}] σmax={meta['max_stress']:.2e}")

            if idx == num_samples:
                break

    # Save dataset
    out_pt = f"{dataset_dir}/dataset_{num_samples}.pt"
    out_json = f"{dataset_dir}/metadata_{num_samples}.json"

    torch.save(dataset, out_pt)
    with open(out_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Dataset saved to: {out_pt}")
    print(f"📄 Metadata saved to: {out_json}\n")

    return dataset, metadata



# ================================================================
# ========================  MAIN ENTRY  ==========================
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FEA dataset in parallel.")
    parser.add_argument("--geometry", type=str, required=True, help="Geometry name")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    build_dataset(
        args.geometry,
        num_samples=args.num_samples,
        workers=args.workers
    )
