import importlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from .base_evaluator import BaseEvaluator
from core.component import Component
from core.IritModel import IritModel, IritCModel

DEFAULT_CCX_AXIS_PATH = os.path.expanduser("~/opt/calculix/root/usr/bin/ccx_axis")


def _ks_excluded_node_ids(model_path, elements, elem_to_tile, nodes=None):
    """Node ids left out of KS, or an empty set. TOPOPT_KS_EXCLUDE is a
    comma-separated list:

      load_tile  every node of the loaded tile's elements -- the free-end tile
                 in the geometry's load row (boundary_conditions.load_row(),
                 middle by default), as in datasets/build_tile_load_dataset.py:
                 the stress there is the load introduction, not the
                 structure's, and the design cannot reduce it.
      load_nbhd  the loaded tile and every tile touching it, 3x3 block (use with
                 TOPOPT_FIX_LOAD_NBHD, which fixes their thickness).
      pad       every node of cross4's pad elements: the anchored pad
                 (CROSS4_PAD_THICKNESS, z in [0, pad]) and the load pad
                 (CROSS4_LOAD_PAD_THICKNESS, z in [L - pad, L] in the load
                 row). Both must be set explicitly (0 = none), so this mask
                 and the CAD model cannot disagree. Needs `nodes` ({id: xyz}).
    """
    modes = {m.strip() for m in os.environ.get("TOPOPT_KS_EXCLUDE", "").split(",") if m.strip()}
    unknown = modes - {"load_tile", "load_nbhd", "pad"}
    if unknown:
        raise ValueError(f"TOPOPT_KS_EXCLUDE: unknown mode(s) {sorted(unknown)}")
    out = set()
    if "load_tile" in modes:
        geom = os.path.basename(os.path.dirname(os.path.abspath(model_path)))
        bc = importlib.import_module(f"data.{geom}.boundary_conditions")
        NX, NY, NZ = bc.tile_grid(None)
        row = bc.load_row() if hasattr(bc, "load_row") else NY // 2
        load_tile = (NX - 1) + NX * row
        for eid, nl in elements.items():
            if elem_to_tile[eid] == load_tile:
                out.update(nl)
    if "load_nbhd" in modes:
        # the loaded tile and every in-grid tile touching it: 3x3 block,
        # diagonals included (planar grids) -- optimize_dims' _load_tile_block
        geom = os.path.basename(os.path.dirname(os.path.abspath(model_path)))
        bc = importlib.import_module(f"data.{geom}.boundary_conditions")
        NX, NY, NZ = bc.tile_grid(None)
        row = bc.load_row() if hasattr(bc, "load_row") else NY // 2
        ix = NX - 1
        tiles = {(ix + a) + NX * (row + b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                 if 0 <= ix + a < NX and 0 <= row + b < NY}
        for eid, nl in elements.items():
            if elem_to_tile[eid] in tiles:
                out.update(nl)
    if "pad" in modes:
        # anchored pad z in [0, pad]; load pad z in [L - load_pad, L] within
        # the load row. Both thicknesses must be set explicitly (0 = none).
        vals = {k: os.environ.get(k) for k in ("CROSS4_PAD_THICKNESS", "CROSS4_LOAD_PAD_THICKNESS")}
        missing = [k for k, v in vals.items() if v in (None, "")]
        if missing:
            raise ValueError(f"TOPOPT_KS_EXCLUDE=pad needs {missing} set explicitly")
        if nodes is None:
            raise ValueError("TOPOPT_KS_EXCLUDE=pad needs node coordinates")
        pad, lpad = (float(vals[k]) for k in ("CROSS4_PAD_THICKNESS", "CROSS4_LOAD_PAD_THICKNESS"))
        geom = os.path.basename(os.path.dirname(os.path.abspath(model_path)))
        bc = importlib.import_module(f"data.{geom}.boundary_conditions")
        rows = bc.load_rows() if hasattr(bc, "load_rows") else [bc.load_row()]
        # pad thicknesses are in TILE units (model_lin.c builds in the unit
        # cell) and rows are tile rows: convert with the physical tile size,
        # which is 1 on cross4_11x11 but 11/21 on cross4_21x21
        NX, NY, NZ = bc.tile_grid(None)
        L = max(xyz[2] for xyz in nodes.values())
        H = max(xyz[1] for xyz in nodes.values())
        tz, ty = L / NX, H / NY
        pad, lpad = pad * tz, lpad * tz
        y0, y1 = rows[0] * ty, (rows[-1] + 1) * ty
        for nl in elements.values():
            P = [nodes[n] for n in nl]
            if pad > 0 and all(p[2] <= pad + 1e-6 for p in P):
                out.update(nl)
            elif lpad > 0 and all(p[2] >= L - lpad - 1e-6 and y0 - 1e-6 <= p[1] <= y1 + 1e-6 for p in P):
                out.update(nl)
    return out


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
            "stress": _masked_ks_mpa(comp, model_path, stress_ks_rho_mpa),
            "disp": comp.mesh.get_max_displacement(),
        }
    except Exception as e:
        print(f"CalculiX sim failed: {e}", flush=True)
        return None


def _masked_ks_mpa(comp, model_path, stress_ks_rho_mpa):
    excl = _ks_excluded_node_ids(
        model_path, {e.id: [n.id for n in e.nodes] for e in comp.mesh.all_elements()}, comp.elem_to_tile,
        {n.id: n.coords for n in comp.mesh.all_nodes()})
    if not excl:
        return comp.mesh.get_aggregated_stress(rho=stress_ks_rho_mpa * 1e-6) / 1e6
    import numpy as np
    vm = np.array([n.stress for n in comp.mesh.all_nodes() if n.id not in excl])
    rho = stress_ks_rho_mpa * 1e-6
    return float(vm.max() + np.log(np.exp(rho * (vm - vm.max())).sum()) / rho) / 1e6


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

    def _colored_tile_gradient(self, sa, D_tile, elements0, cad, stress_mpa):
        """dG/dt and dV/dt from per-tile node motion D_tile (see
        core.geometric_jacobian.compute_tile_jacobian_colored): each tile's
        derivative is the nodal gradient contracted with that tile's own node
        motion. dV/dt uses the mesh's own volume (sum of element |det J|),
        differentiated w.r.t. node positions the same way, so it holds for any
        tile shape -- unlike hollow_tube_volume_sensitivity."""
        import numpy as np
        import torch
        from core.fem_adjoint import _geom

        gX = sa.gradient(self.stress_ks_rho_mpa).detach().numpy()        # (N,3), sa.ids order
        row = {nid: i for i, nid in enumerate(sa.ids)}
        X = sa.X.detach().clone().requires_grad_(True)
        conn = torch.as_tensor(sa.conn)
        _, detJ = _geom(X[conn])
        vol_mesh = detJ.abs().sum()                                     # 2x2x2 Gauss, unit weights
        vol_mesh.backward()
        gV = X.grad.numpy()
        yld = self.material_properties["yield_strength"]
        dG_dx, dV_dx = {}, {}
        for k in self.dims_keys:
            d = D_tile.get(k, {})
            dG_dx[k] = sum(float(gX[row[n]] @ np.asarray(v)) for n, v in d.items()) / 1e6 / yld
            dV_dx[k] = sum(float(gV[row[n]] @ np.asarray(v)) for n, v in d.items())
        # Report the MESH volume: dV_dx above is its exact derivative. With
        # curved (rounded) tiles the hex mesh's volume is not the CAD volume
        # (0.35% apart at 3,3,3) and the two move differently with t, so the
        # CAD volume paired with this gradient is an inconsistent objective.
        volume = float(vol_mesh)
        if not getattr(self, "_colored_vol_checked", False):
            self._colored_vol_checked = True
            print(f"[ccx colored jacobian] mesh volume {volume:.4f} (optimized) vs CAD volume {cad.get_volume():.4f}", flush=True)
        return volume, stress_mpa, yld, dV_dx, dG_dx

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
        # TOPOPT_TILE_JACOBIAN=colored: per-tile node motion for tiles whose
        # parameter also moves their neighbours' nodes (cross4_11x11's min-rule
        # leg ends); the default single bump assumes no node moves with two
        # tiles, which is exact for hollow_cube. Adjoint gradient source only.
        colored = os.environ.get("TOPOPT_TILE_JACOBIAN") == "colored"
        if colored:
            from core.geometric_jacobian import compute_tile_jacobian_colored
            if self.gradient_source != "adjoint":
                raise ValueError("TOPOPT_TILE_JACOBIAN=colored needs gradient_source='adjoint'")
            D_tile, nodes0, elements0, elem_to_tile, cad = compute_tile_jacobian_colored(
                exe, dims, self.U, self.V, self.W, self.tile_grid(), fixed_dims=self.fixed_dims,
                mesh_threads=self.mesh_threads, model_cls=model_cls)
            dX_dt = {}
            for _d in D_tile.values():
                for _n, _v in _d.items():
                    _o = dX_dt.get(_n, (0.0, 0.0, 0.0))
                    dX_dt[_n] = tuple(a + b for a, b in zip(_o, _v))
        else:
            dX_dt, nodes0, elements0, elem_to_tile, cad = compute_thickness_jacobian(
                exe, dims, self.U, self.V, self.W, fixed_dims=self.fixed_dims,
                mesh_threads=self.mesh_threads, model_cls=model_cls, return_model=True,
            )

        # TOPOPT_STIFF_PENALTY: per-element stiffness scale s(t) of its tile
        from core.mesh import (element_stiffness_scales, stiffness_penalty_ts, stiffness_scale_dt,
                               element_stress_weights, stress_relax_params, stress_weight_dt)
        e_scales = element_stiffness_scales(dims, elem_to_tile, self.fixed_dims, self.dims_keys)
        # TOPOPT_STRESS_RELAX: per-element weight w(t) on the REAL material's stress
        s_weights = element_stress_weights(dims, elem_to_tile, self.fixed_dims, self.dims_keys)

        def build_mesh():
            nodes = {nid: Node(nid, list(xyz)) for nid, xyz in nodes0.items()}
            elements = {eid: Element(eid, [nodes[nid] for nid in nlist])
                        for eid, nlist in elements0.items()}
            mesh = Mesh(nodes, elements)
            if e_scales:
                mesh.elem_scale = e_scales
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
            _mesh = build_mesh()
            _nodes = _mesh.all_nodes()          # one build, read three times
            if getattr(_mesh, "rigid_load", True):
                sa = ShapeAdjoint(nodes0, elements0,
                                  [n.id for n in _nodes if n.anchored],
                                  [n.id for n in _nodes if any(n.forces)],
                                  [sum(n.forces[j] for n in _nodes) for j in range(3)],
                                  young, poisson,
                                  e_scale=[e_scales[e] for e in sorted(elements0)] if e_scales else None,
                                  stress_scale=[s_weights[e] for e in sorted(elements0)] if s_weights else None)
                sa.assert_tie_is_fixed(dX_dt)
            else:
                # Distributed surface load (e.g. FEA_LOAD_MODE=void_traction):
                # each node carries its own force, no rigid tie. The forces
                # are held fixed through the derivative, which is exact only
                # while the loaded surface does not move -- freeze the loaded
                # tile (TOPOPT_FIX_LOAD_TILE) so its void never changes.
                sa = ShapeAdjoint(nodes0, elements0,
                                  [n.id for n in _nodes if n.anchored],
                                  [], (0.0, 0.0, 0.0), young, poisson, rigid_tie=False,
                                  nodal_forces={n.id: list(n.forces) for n in _nodes if any(n.forces)},
                                  e_scale=[e_scales[e] for e in sorted(elements0)] if e_scales else None,
                                  stress_scale=[s_weights[e] for e in sorted(elements0)] if s_weights else None)
            _excl = _ks_excluded_node_ids(self.model_path, elements0, elem_to_tile, nodes0)
            if _excl:
                sa.ks_mask = [nid not in _excl for nid in sa.ids]
            sa.solve()
            stress_mpa = sa.ks(self.stress_ks_rho_mpa) / 1e6
            NX, NY, NZ = self.tile_grid()
            # The constraint is the raw KS (rho per MPa) -- an upper bound on the
            # peak (sits ~17 MPa above it at rho 0.25 on cross4) with exactly the
            # gradient computed below. The STRESS_KS_NORMALIZE=max/smooth
            # rescaling c = peak/KS was removed 2026-10-22: the constraint
            # then hinged on the nodal peak, which the surrogates predict far
            # worse off-trajectory than KS. The peak is still logged.
            import numpy as _np
            _, _vm = sa.nodal_stress()
            _vm = _vm.detach().numpy()
            if sa.ks_mask is not None:
                _vm = _vm[_np.asarray(sa.ks_mask, dtype=bool)]
            self.last_peak_mpa = float(_vm.max()) / 1e6
            print(f"[ccx] KS {stress_mpa:.2f} MPa, peak {self.last_peak_mpa:.2f} MPa", flush=True)
            def _stiffness_term(dG):
                """Add each tile's stiffness-penalty term, sum over its elements
                of d KS / d s_e * ds/dt (sa.gE from the gradient pass), in dG's
                units (MPa / yield)."""
                if not e_scales and not s_weights:
                    return dG
                from core.tile_sensitivity import tile_index_to_dims_key
                ts = stiffness_penalty_ts()
                rp = stress_relax_params()
                yld_ = self.material_properties["yield_strength"]
                accE, accW = {}, {}
                for k_e, eid in enumerate(sa.eids):
                    key = tile_index_to_dims_key(elem_to_tile[eid], NX, NY, NZ, self.dims_keys)
                    if e_scales:
                        accE[key] = accE.get(key, 0.0) + float(sa.gE[k_e])
                    if s_weights:
                        accW[key] = accW.get(key, 0.0) + float(sa.gW[k_e])
                out = {}
                for k, g in dG.items():
                    t = float(dims[k])
                    extra = 0.0
                    if e_scales:
                        extra += accE.get(k, 0.0) * stiffness_scale_dt(t, ts)
                    if s_weights:
                        extra += accW.get(k, 0.0) * stress_weight_dt(t, *rp)
                    out[k] = g + extra / 1e6 / yld_
                return out

            if colored:
                v, s_, y, dV, dG = self._colored_tile_gradient(sa, D_tile, elements0, cad, stress_mpa)
                dG = _stiffness_term(dG)
                return v, s_, y, dV, dG
            dKS_dt = sa.tile_gradient(dX_dt, elem_to_tile, self.dims_keys,
                                      NX, NY, NZ,
                                      rho_mpa=self.stress_ks_rho_mpa)
            yld = self.material_properties["yield_strength"]
            dG_dx = {k: (v / 1e6) / yld for k, v in dKS_dt.items()}
            dG_dx = _stiffness_term(dG_dx)
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
