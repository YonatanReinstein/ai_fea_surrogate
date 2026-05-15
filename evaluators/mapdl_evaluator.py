import importlib
import os
import time

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
os.environ["ANSYS251_DIR"] = "/ansys_inc/v251/ansys/bin"

import ansys.tools.path
_original_version_from_path = ansys.tools.path.version_from_path
def _patched_version_from_path(product, path):
    if path and "ansys" in path.lower():
        return 251
    return _original_version_from_path(product, path)
ansys.tools.path.version_from_path = _patched_version_from_path
ansys.tools.path.path.version_from_path = _patched_version_from_path

import ansys.mapdl.core.pool as _mapdl_pool
_orig_launch_mapdl = _mapdl_pool.launch_mapdl
def _patched_launch_mapdl(*args, **kwargs):
    run_location = kwargs.get('run_location')
    max_retries = 3
    for attempt in range(max_retries):
        if run_location:
            os.makedirs(run_location, exist_ok=True)
        try:
            return _orig_launch_mapdl(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"MAPDL launch attempt {attempt+1}/{max_retries} failed: {e}. Retrying in 15s...", flush=True)
                time.sleep(15)
            else:
                raise
_mapdl_pool.launch_mapdl = _patched_launch_mapdl

from ansys.mapdl.core import MapdlPool
from ansys.mapdl.core.errors import MapdlRuntimeError

from .base_evaluator import BaseEvaluator
from core.component import Component
from core.IritModel import IritModel, IritCModel
import json


class MAPDLEvaluator(BaseEvaluator):
    def __init__(self, geometry_name, pool_size=4, nproc=3, run_location=None):
        super().__init__(geometry_name)
        self.geometry = geometry_name
        self.model_path = f"data/{geometry_name}/CAD_model"
        module = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
        self.anchor_condition = module.anchor_condition
        self.force_pattern = module.force_pattern
        self.mesh_resolution = module.mesh_resolution
        self.U, self.V, self.W = self.mesh_resolution()
        with open(f"data/{geometry_name}/CAD_model/material_properties.json", "r") as f:
            self.material_properties = json.load(f)

        if run_location is None:
            run_location = f"data/{geometry_name}"

        print(f"Creating MapdlPool with pool_size={pool_size}, nproc={nproc}, run_location={run_location}", flush=True)
        self.pool = MapdlPool(
            n_instances=pool_size,
            nproc=nproc,
            run_location=run_location,
            license_server_check=False,
            start_timeout=120,
        )
        print("MapdlPool created successfully.", flush=True)

    def _run_single(self, mapdl, dims: dict):
        exe_path = os.path.join(self.model_path, "model")
        exe_win_path = os.path.join(self.model_path, "model.exe")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if os.path.isfile(exe_path):
                    CAD_model = IritCModel(exe_path, dims_dict=dims)
                elif os.path.isfile(exe_win_path):
                    CAD_model = IritCModel(exe_win_path, dims_dict=dims)
                else:
                    CAD_model = IritModel(os.path.join(self.model_path, "model.irt"), dims_dict=dims)

                comp = Component(CAD_model=CAD_model, young=self.material_properties["young_modulus"], poisson=self.material_properties["poisson_ratio"])
                comp.generate_mesh(U=self.U, V=self.V, W=self.W)
                comp.mesh.anchor_nodes_by_condition(self.anchor_condition)
                comp.mesh.apply_force_by_pattern(self.force_pattern)
                comp.ansys_sim(mapdl=mapdl, screenshot_path=None)

                return {
                    "volume": comp.get_volume(),
                    "stress": comp.mesh.get_max_stress(),
                    "disp":   comp.mesh.get_max_displacement(),
                }
            except MapdlRuntimeError as e:
                print(f"MAPDL sim failed (attempt {attempt+1}/{max_retries}): {e}. Retrying...", flush=True)
                try:
                    mapdl.clear()
                except Exception as clear_err:
                    print(f"MAPDL instance unrecoverable: {clear_err}", flush=True)
                    try:
                        mapdl.exit()
                    except Exception:
                        pass
                    return None
            except Exception as e:
                print(f"Unexpected error (attempt {attempt+1}/{max_retries}): {e}. Retrying...", flush=True)
                try:
                    mapdl.clear()
                except Exception as clear_err:
                    print(f"MAPDL instance unrecoverable: {clear_err}", flush=True)
                    try:
                        mapdl.exit()
                    except Exception:
                        pass
                    return None
        print(f"Sample failed after {max_retries} attempts, skipping.", flush=True)
        return None

    def evaluate(self, dims_list: list[dict]):
        pending = list(range(len(dims_list)))
        results = [None] * len(dims_list)
        max_passes = 3

        for pass_idx in range(max_passes):
            if not pending:
                break
            pending_dims = [dims_list[i] for i in pending]
            print(f"Pass {pass_idx+1}/{max_passes}: processing {len(pending)} samples...", flush=True)
            still_pending = []
            for orig_idx, result in zip(pending, self.pool.map(self._run_single, pending_dims)):
                if result is None:
                    still_pending.append(orig_idx)
                else:
                    results[orig_idx] = result
            pending = still_pending

        if pending:
            print(f"WARNING: {len(pending)} samples failed after {max_passes} passes.", flush=True)

        return {
            "stress": [r["stress"] if r is not None else None for r in results],
            "volume": [r["volume"] if r is not None else None for r in results],
            "disp":   [r["disp"]   if r is not None else None for r in results],
        }

    def close(self):
        self.pool.exit()

    def __del__(self):
        try:
            self.pool.exit()
        except Exception:
            pass
