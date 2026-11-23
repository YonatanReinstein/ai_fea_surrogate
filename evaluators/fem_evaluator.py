import importlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from .base_evaluator import BaseEvaluator
from core.component import Component
from core.IritModel import IritCModel

def _ks_excluded_node_ids(model_path, elements, elem_to_tile, nodes=None):
    """Node ids left out of KS, or an empty set. TOPOPT_KS_EXCLUDE is a
    comma-separated list:

      load_tile  every node of the loaded tile's elements -- the free-end tile
                 in the geometry's load row (boundary_conditions.load_row(),
                 middle by default):
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
                 U, V, W, young, poisson, nproc, dims, stress_ks_rho_mpa):
    import torch
    torch.set_num_threads(max(1, nproc))
    try:
        CAD_model = IritCModel(os.path.join(model_path, "model"), dims_dict=dims,
                               mesh_threads=mesh_threads, fixed_dims=fixed_dims)

        comp = Component(CAD_model=CAD_model, young=young, poisson=poisson)
        comp.generate_mesh(U=U, V=V, W=W)
        comp.mesh.anchor_nodes_by_condition(anchor_condition)
        comp.mesh.apply_force_by_pattern(force_pattern)
        comp.fem_sim()

        # KS aggregation instead of a hard max: smooth and non-zero for
        # every node near the peak, not just whichever single node happens
        # to hold it this iteration -- see core/mesh.py:get_aggregated_stress.
        # rho is tuned in 1/MPa; get_aggregated_stress works in Pa (node.stress's
        # native unit), hence the 1e-6 conversion.
        return {
            "volume": comp.get_volume(),
            # stress in MPa, like GNNDsigEvaluator
            "stress": _masked_ks_mpa(comp, model_path, stress_ks_rho_mpa),
            "disp": comp.mesh.get_max_displacement(),
        }
    except Exception as e:
        print(f"FEM solve failed: {e}", flush=True)
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


class FEMEvaluator(BaseEvaluator):
    """The exact-physics evaluator (--arch fem): IRIT mesh + core.fem_adjoint's
    own C3D8I solve (validated against CalculiX to <4e-5 of peak per node).
    evaluate() is a plain process pool, one design per worker. The old
    CalculiX-backed version was CCXEvaluator in git fed57d9.
    """

    def __init__(self, geometry_name, max_workers=8, nproc=1, mesh_threads=4,
                 stress_ks_rho_mpa=0.15):
        super().__init__(geometry_name)
        self.geometry = geometry_name
        self.mesh_threads = mesh_threads
        self.nproc = nproc
        self.max_workers = max_workers
        # KS aggregation sharpness (1/MPa) for the "stress" reported to
        # optimizers -- see get_aggregated_stress(). Larger = closer to the
        # true hard max (noisier per-tile sensitivities); smaller = smoother
        # sensitivities but more conservative (overestimates the true peak).
        self.stress_ks_rho_mpa = stress_ks_rho_mpa
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
                    self.nproc, dims, self.stress_ks_rho_mpa,
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
            print(f"[fem colored jacobian] mesh volume {volume:.4f} (optimized) vs CAD volume {cad.get_volume():.4f}", flush=True)
        return volume, stress_mpa, yld, dV_dx, dG_dx

    def compute_topopt_gradient(self, dims: dict):
        """(volume, stress_mpa, yield_strength_mpa, dV_dx, dG_dx) in the exact
        shape optimization.TopologyOptimizer's gradient_func expects: one CAD
        remesh for the geometric Jacobian dX/dt, then core.fem_adjoint's exact
        shape adjoint -- one factorization gives the reported KS and every
        tile's sensitivity. On hollow_cube's 840 tiles it matched the full
        1680-solve remesh FD map (sign 824/840, spearman +0.9997, ratio median
        0.9985) in 46 s; the older patched-CalculiX 'axis' route it replaced scored
        56/840 in 433 s (git fed57d9).
        """
        from core.fem_adjoint import ShapeAdjoint
        from core.geometric_jacobian import compute_tile_jacobian_colored
        from core.mesh import Mesh
        from core.node import Node
        from core.element import Element

        if self.tile_grid is None:
            raise RuntimeError(
                f"data/{self.geometry}/boundary_conditions.py defines no "
                "tile_grid(); compute_topopt_gradient needs it to map tile "
                "indices onto dims.json keys."
            )
        # Per-tile node motion, exact even where one tile's parameter also moves
        # its neighbours' nodes (cross4's leg ends). The single all-tiles bump
        # that was exact only for hollow_cube is in git fed57d9.
        D_tile, nodes0, elements0, elem_to_tile, cad = compute_tile_jacobian_colored(
            os.path.join(self.model_path, "model"), dims, self.U, self.V, self.W, self.tile_grid(),
            fixed_dims=self.fixed_dims, mesh_threads=self.mesh_threads, model_cls=IritCModel)
        dX_dt = {}
        for _d in D_tile.values():
            for _n, _v in _d.items():
                _o = dX_dt.get(_n, (0.0, 0.0, 0.0))
                dX_dt[_n] = tuple(a + b for a, b in zip(_o, _v))

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

        # One assembly + factorization gives BOTH the reported stress and
        # every tile's sensitivity, so there is no separate value solve to
        # run alongside -- and no C3D8I/C3D8 split either: g0 and dG come
        # from the same discretization for the first time. This element
        # reproduces CalculiX's own element to 5 significant figures (KS 39.1027
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
        print(f"[fem] KS {stress_mpa:.2f} MPa, peak {self.last_peak_mpa:.2f} MPa", flush=True)
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

        v, s_, y, dV, dG = self._colored_tile_gradient(sa, D_tile, elements0, cad, stress_mpa)
        return v, s_, y, dV, _stiffness_term(dG)

    def close(self):
        pass
