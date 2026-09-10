import importlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from .base_evaluator import BaseEvaluator
from core.component import Component
from core.IritModel import IritModel, IritCModel

DEFAULT_CCX_AXIS_PATH = os.path.expanduser("~/opt/calculix/root/usr/bin/ccx_axis")


def _run_single(model_path, mesh_threads, fixed_dims, anchor_condition, force_pattern,
                 U, V, W, young, poisson, ccx_path, nproc, dims, stress_ks_rho_mpa):
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

        # KS aggregation instead of a hard max: smooth and non-zero for
        # every node near the peak, not just whichever single node happens
        # to hold it this iteration -- see core/mesh.py:get_aggregated_stress.
        # rho is tuned in 1/MPa; get_aggregated_stress works in Pa (node.stress's
        # native unit), hence the 1e-6 conversion.
        return {
            "volume": comp.get_volume(),
            # Consistent with MAPDLEvaluator/GNNEvaluator: stress in MPa.
            "stress": comp.mesh.get_aggregated_stress(rho=stress_ks_rho_mpa * 1e-6) / 1e6,
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

    def __init__(self, geometry_name, max_workers=8, nproc=1, mesh_threads=4, ccx_path="ccx",
                 stress_ks_rho_mpa=0.15, ccx_axis_path=None, gradient_nproc=None,
                 gradient_source="adjoint"):
        super().__init__(geometry_name)
        self.geometry = geometry_name
        self.mesh_threads = mesh_threads
        self.nproc = nproc
        self.ccx_path = ccx_path
        self.max_workers = max_workers
        # KS aggregation sharpness (1/MPa) for the "stress" reported to
        # optimizers -- see get_aggregated_stress(). Larger = closer to the
        # true hard max (noisier per-tile sensitivities); smaller = smoother
        # sensitivities but more conservative (overestimates the true peak).
        self.stress_ks_rho_mpa = stress_ks_rho_mpa
        # Patched ccx for compute_topopt_gradient (tools/build_ccx_axis.sh).
        # Only that method needs it; plain evaluate() still uses self.ccx_path.
        self.ccx_axis_path = ccx_axis_path or DEFAULT_CCX_AXIS_PATH
        # The gradient path is 4 solves on ONE mesh, not max_workers meshes in
        # parallel, so it should be given the whole machine rather than the
        # single core an evaluate() worker gets.
        self.gradient_nproc = gradient_nproc or max(1, max_workers)
        # "adjoint" (default): core.fem_adjoint's exact shape adjoint -- one
        # factorization of our own assembly, two back-substitutions, every
        # tile. "axis": the patched ccx *SENSITIVITY route. On hollow_cube's
        # 840 tiles at (3,3,3), scored against the full 1680-solve remesh FD
        # map, sign / spearman / ratio-median / sum-rule are
        #     axis     56/840  -0.4716  -19.00  x-2.79   433 s
        #     adjoint 824/840  +0.9997   0.9985  x0.9953   46 s
        # -- the axis route's error is its finite-differenced d(K_e)/d(x),
        # which the adjoint does not use. See core/fem_adjoint.py.
        self.gradient_source = gradient_source
        self.model_path = f"data/{geometry_name}/CAD_model"
        module = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
        self.anchor_condition = module.anchor_condition
        self.force_pattern = module.force_pattern
        self.mesh_resolution = module.mesh_resolution
        self.U, self.V, self.W = self.mesh_resolution()
        # (NX, NY, NZ) tile counts; only compute_topopt_gradient needs it, to
        # map elem_to_tile's IRIT-native tile index onto dims.json keys.
        self.tile_grid = getattr(module, "tile_grid", None)
        self.fixed_dims = getattr(module, "fixed_dims", lambda: {})()
        with open(f"data/{geometry_name}/CAD_model/material_properties.json", "r") as f:
            self.material_properties = json.load(f)
        # dims.json's own key order, which is what tile_index_to_dims_key
        # indexes into. TopologyOptimizer._vec_to_dims emits free dims before
        # fixed ones, so the dict it hands compute_topopt_gradient is NOT
        # necessarily in this order -- take the order from the file.
        with open(f"data/{geometry_name}/CAD_model/dims.json", "r") as f:
            self.dims_keys = list(json.load(f).keys())

    def evaluate(self, dims_list: list[dict]):
        results = [None] * len(dims_list)
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(
                    _run_single, self.model_path, self.mesh_threads, self.fixed_dims,
                    self.anchor_condition, self.force_pattern, self.U, self.V, self.W,
                    self.material_properties["young_modulus"], self.material_properties["poisson_ratio"],
                    self.ccx_path, self.nproc, dims, self.stress_ks_rho_mpa,
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

    def compute_topopt_gradient(self, dims: dict):
        """(volume, stress_mpa, yield_strength_mpa, dV_dx, dG_dx) in the exact
        shape optimization.TopologyOptimizer's gradient_func expects -- a
        constant-cost drop-in for its default N+1-evaluator-call
        finite-difference sweep. On hollow_cube's 840-tile grid that sweep is
        841 full CalculiX solves per generation; this is four, whatever the
        tile count:

            1 CAD remesh   the geometric Jacobian dX/dt bumps every tile's
                           thickness at once (tiles never share a
                           thickness-sensitive node, so one extra mesh
                           recovers all of them -- see
                           core.geometric_jacobian)
            3 ccx solves   the global-axis nodal gradient dKS/dX, from the
                           patched binary (Mesh.solve_ccx_axis_gradient)
            1 ccx solve    the plain C3D8I static solve for the REPORTED
                           stress (see the element-type note below)

        contracted per tile by core.tile_sensitivity's
        assemble_tile_thickness_sensitivity_axis.

        Two element types, on purpose. The rest of this codebase -- evaluate(),
        the training dataset, the screenshots -- solves with C3D8I, whose
        incompatible modes close the gap to MAPDL's SOLID185 to 1-2% where
        plain C3D8 under-predicts peak stress by 6-9%. But ccx's
        semi-analytical shape sensitivity differentiates those statically
        condensed modes badly: measured against brute-force nodal FD on
        hollow_cube_one, the direction cosine is -0.68 median with C3D8I and
        +0.90 with C3D8. So the GRADIENT is taken on C3D8 and the reported
        STRESS on C3D8I, which keeps the constraint value g0 = s0/yield - 1 on
        exactly the same metric the fitness function, the screenshots and the
        final design validation use. Mixing the two is deliberate: OC needs
        the gradient's direction and the constraint's absolute level, and each
        element type is the better instrument for one of them.

        Accuracy is set by elements-per-wall, not tile count -- see
        Mesh.solve_ccx_axis_gradient. Give the geometry a mesh_resolution that
        puts at least 5 elements through its thinnest wall or the per-tile
        numbers are wrong by several times, silently.
        """
        from core.fem_adjoint import ShapeAdjoint
        from core.geometric_jacobian import compute_thickness_jacobian
        from core.mesh import Mesh
        from core.node import Node
        from core.element import Element
        from core.tile_sensitivity import (
            assemble_tile_thickness_sensitivity_axis,
            hollow_tube_volume_sensitivity,
        )

        if self.tile_grid is None:
            raise RuntimeError(
                f"data/{self.geometry}/boundary_conditions.py defines no "
                "tile_grid(); compute_topopt_gradient needs it to map tile "
                "indices onto dims.json keys."
            )
        if self.gradient_source == "axis" and not os.path.isfile(self.ccx_axis_path):
            raise FileNotFoundError(
                f"patched CalculiX not found at {self.ccx_axis_path}. "
                "compute_topopt_gradient needs the CCX_DESI_AXIS build -- run "
                "tools/build_ccx_axis.sh (or pass ccx_axis_path). Falling back "
                "to the finite-difference sweep is the caller's decision, not "
                "this method's: it costs one solve per tile."
            )

        # Same CAD-model resolution order _run_single uses.
        model_cls, exe = IritCModel, os.path.join(self.model_path, "model")
        if not os.path.isfile(exe):
            exe_win = os.path.join(self.model_path, "model.exe")
            if os.path.isfile(exe_win):
                exe = exe_win
            else:
                model_cls, exe = IritModel, os.path.join(self.model_path, "model.irt")
        dX_dt, nodes0, elements0, elem_to_tile, cad = compute_thickness_jacobian(
            exe, dims, self.U, self.V, self.W, fixed_dims=self.fixed_dims,
            mesh_threads=self.mesh_threads, model_cls=model_cls, return_model=True,
        )

        def build_mesh():
            nodes = {nid: Node(nid, list(xyz)) for nid, xyz in nodes0.items()}
            elements = {eid: Element(eid, [nodes[nid] for nid in nlist])
                        for eid, nlist in elements0.items()}
            mesh = Mesh(nodes, elements)
            mesh.anchor_nodes_by_condition(self.anchor_condition)
            mesh.apply_force_by_pattern(self.force_pattern)
            return mesh

        young = self.material_properties["young_modulus"]
        poisson = self.material_properties["poisson_ratio"]

        # Every node is offered as a design variable, interior included --
        # dropping them is the single largest error term in the contraction
        # (see Mesh.solve_ccx_axis_gradient).
        # All three axes, every node. Skipping the axes and nodes whose
        # dX/dt is zero would be exact (they contribute nothing to the
        # contraction) and on hollow_cube's one-tile-deep 40x21x1 grid it
        # measured 482 s -> 291 s per generation, agreeing with the full
        # gradient to 3.3e-05 over all 840 tiles. It is deliberately NOT done:
        # which axes and nodes are live is a property of the particular
        # outline and tile design, and this path has to behave identically for
        # every geometry it is handed, not just the ones that happen to be
        # flat in x.
        def _gradient():
            return build_mesh().solve_ccx_axis_gradient(
                young, poisson, sorted(nodes0),
                stress_ks_rho_mpa=self.stress_ks_rho_mpa,
                ccx_path=self.ccx_axis_path, nproc=self.gradient_nproc,
                interior=True, element_type="C3D8",
            )

        # Reported stress on the project's own C3D8I convention, same call
        # evaluate() makes, so g0 is comparable across the whole pipeline.
        def _value():
            value_mesh = build_mesh()
            value_mesh.solve_ccx(young, poisson, ccx_path=self.ccx_path,
                                 nproc=self.gradient_nproc)
            return value_mesh.get_aggregated_stress(
                rho=self.stress_ks_rho_mpa * 1e-6) / 1e6

        if self.gradient_source == "adjoint":
            # One assembly + factorization gives BOTH the reported stress and
            # every tile's sensitivity, so there is no separate value solve to
            # run alongside -- and no C3D8I/C3D8 split either: g0 and dG come
            # from the same discretization for the first time. This element
            # reproduces ccx's own C3D8 to 5 significant figures (KS 39.1027
            # vs 39.1023 MPa on hollow_cube_one, |u|max identical), so the
            # constraint stays on the same scale the rest of the pipeline uses.
            sa = ShapeAdjoint(nodes0, elements0,
                              [n.id for n in build_mesh().all_nodes() if n.anchored],
                              [n.id for n in build_mesh().all_nodes() if any(n.forces)],
                              [sum(n.forces[j] for n in build_mesh().all_nodes())
                               for j in range(3)],
                              young, poisson)
            sa.assert_tie_is_fixed(dX_dt)
            sa.solve()
            stress_mpa = sa.ks(self.stress_ks_rho_mpa) / 1e6
            NX, NY, NZ = self.tile_grid()
            dKS_dt = sa.tile_gradient(dX_dt, elem_to_tile, self.dims_keys,
                                      NX, NY, NZ,
                                      rho_mpa=self.stress_ks_rho_mpa)
            yld = self.material_properties["yield_strength"]
            dG_dx = {k: (v / 1e6) / yld for k, v in dKS_dt.items()}
            volume = cad.get_volume()
            dV_dt = hollow_tube_volume_sensitivity(
                [dims[k] for k in self.dims_keys], volume)
            return volume, stress_mpa, yld, dict(zip(self.dims_keys, dV_dt)), dG_dx

        # The value solve shares nothing with the gradient solves but the
        # (read-only) nodes0/elements0 it builds its own Mesh from, so it runs
        # alongside them instead of after: it is a fourth independent ccx
        # process, and waiting for it serially added its full wall time to
        # every generation. Peak thread demand is therefore 4 x gradient_nproc,
        # which is what --cpus-per-task has to cover.
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_value = ex.submit(_value)
            _, grad = _gradient()
            stress_mpa = fut_value.result()

        NX, NY, NZ = self.tile_grid()
        dims_keys = self.dims_keys
        dKS_dt = assemble_tile_thickness_sensitivity_axis(
            dX_dt, grad, elements0, elem_to_tile, NX, NY, NZ, dims_keys)

        # dKS_dt is Pa per unit thickness; stress/yield are both MPa.
        yld = self.material_properties["yield_strength"]
        dG_dx = {k: (v / 1e6) / yld for k, v in dKS_dt.items()}

        volume = cad.get_volume()
        dV_dt = hollow_tube_volume_sensitivity(
            [dims[k] for k in dims_keys], volume)
        dV_dx = dict(zip(dims_keys, dV_dt))

        # Both IritCModels compute_thickness_jacobian built drop their own
        # tmp dir in IritModelBase.__del__; nothing to clean up here.
        return volume, stress_mpa, yld, dV_dx, dG_dx

    def close(self):
        pass
