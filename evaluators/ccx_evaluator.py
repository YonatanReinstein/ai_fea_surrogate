import importlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from .base_evaluator import BaseEvaluator
from core.component import Component
from core.IritModel import IritModel, IritCModel


def _run_single(model_path, mesh_threads, fixed_dims, anchor_condition, force_pattern,
                 U, V, W, young, poisson, ccx_path, nproc, dims):
    exe_path = os.path.join(model_path, "model")
    exe_win_path = os.path.join(model_path, "model.exe")
    try:
        if os.path.isfile(exe_path):
            CAD_model = IritCModel(exe_path, dims_dict=dims, mesh_threads=mesh_threads, fixed_dims=fixed_dims)
        elif os.path.isfile(exe_win_path):
            CAD_model = IritCModel(exe_win_path, dims_dict=dims, mesh_threads=mesh_threads, fixed_dims=fixed_dims)
        else:
            CAD_model = IritModel(os.path.join(model_path, "model.irt"), dims_dict=dims, mesh_threads=mesh_threads)

        comp = Component(CAD_model=CAD_model, young=young, poisson=poisson)
        comp.generate_mesh(U=U, V=V, W=W)
        comp.mesh.anchor_nodes_by_condition(anchor_condition)
        comp.mesh.apply_force_by_pattern(force_pattern)
        comp.ccx_sim(ccx_path=ccx_path, nproc=nproc)

        return {
            "volume": comp.get_volume(),
            # Consistent with MAPDLEvaluator/GNNEvaluator: stress in MPa.
            "stress": comp.mesh.get_max_stress() / 1e6,
            "disp": comp.mesh.get_max_displacement(),
        }
    except Exception as e:
        print(f"CalculiX sim failed: {e}", flush=True)
        return None


class CCXEvaluator(BaseEvaluator):
    """Open-source drop-in for MAPDLEvaluator, backed by CalculiX (ccx)
    instead of Ansys MAPDL. No license/pool machinery is needed: each
    sample is just a subprocess, so parallelism is a plain process pool.

    Requires the `ccx` binary on PATH (e.g. `apt install calculix-ccx`,
    or point ccx_path at a built binary).
    """

    def __init__(self, geometry_name, max_workers=8, nproc=1, mesh_threads=4, ccx_path="ccx"):
        super().__init__(geometry_name)
        self.geometry = geometry_name
        self.mesh_threads = mesh_threads
        self.nproc = nproc
        self.ccx_path = ccx_path
        self.max_workers = max_workers
        self.model_path = f"data/{geometry_name}/CAD_model"
        module = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
        self.anchor_condition = module.anchor_condition
        self.force_pattern = module.force_pattern
        self.mesh_resolution = module.mesh_resolution
        self.U, self.V, self.W = self.mesh_resolution()
        self.fixed_dims = getattr(module, "fixed_dims", lambda: {})()
        with open(f"data/{geometry_name}/CAD_model/material_properties.json", "r") as f:
            self.material_properties = json.load(f)

    def evaluate(self, dims_list: list[dict]):
        results = [None] * len(dims_list)
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(
                    _run_single, self.model_path, self.mesh_threads, self.fixed_dims,
                    self.anchor_condition, self.force_pattern, self.U, self.V, self.W,
                    self.material_properties["young_modulus"], self.material_properties["poisson_ratio"],
                    self.ccx_path, self.nproc, dims,
                ): i
                for i, dims in enumerate(dims_list)
            }
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

        return {
            "stress": [r["stress"] if r is not None else None for r in results],
            "volume": [r["volume"] if r is not None else None for r in results],
            "disp": [r["disp"] if r is not None else None for r in results],
        }

    def close(self):
        pass
