"""Optimizer evaluator for the multi-load stress-tensor surrogate
(training/train_multiload.py checkpoints, HierarchicalGNNCNN node_out_dim 6).

compute_topopt_gradient(dims) -> (volume, stress_MPa, yield, dV/dt, dG/dt), the
same contract as CCXEvaluator, for ONE load case: the free-end tile in
FEA_LOAD_ROW, direction MULTILOAD_DIR (0 = down -y, 1 = toward the anchor -z),
total FEA_LOAD_N (the model was trained at the dataset's load_N; stress is
scaled linearly, which is exact for linear elasticity).

Value: von Mises from the predicted tensor -> KS (rho, nodes of the loaded
tile excluded, as in the training filter); STRESS_KS_NORMALIZE=max/smooth
scale value and gradient by c ~ peak/KS exactly as the ccx evaluator does.

Gradient: everything that depends on the design is rebuilt inside autograd --
node coordinates X, edge vectors/lengths, the per-node load weights (they
depend on the loaded tile's node positions), the node and tile thickness
features -- so one backward pass gives d(KS)/dX and the direct d(KS)/dt; the
geometry part is chained with the exact colored Jacobian dX/dt
(core.geometric_jacobian.compute_tile_jacobian_colored, the same one the ccx
adjoint path uses):
    dKS/dt_i = sum_n <dKS/dX_n, dX_n/dt_i> + dKS/dt_i|direct
Volume is the mesh volume (sum of hex Gauss-point detJ), chained the same way.

Needs the CAD env the dataset was built with (CROSS4_LEG_RULE=avg,
FEA_LOAD_MODE=tile_nodes, CROSS4_PAD_THICKNESS=0.1,
CROSS4_LOAD_PAD_THICKNESS=0, FEA_MESH_RES=6,6,4); the mesh connectivity is
checked against the dataset's mesh.npz on the first call.
"""
import importlib
import json
import os

import numpy as np
import torch

from .base_evaluator import BaseEvaluator


class GNNMultiLoadEvaluator(BaseEvaluator):
    def __init__(self, geometry_name, ckpt_path=None, dataset_dir=None, stress_ks_rho_mpa=0.25,
                 mesh_threads=8, jacobian_workers=None):
        super().__init__(geometry_name)
        from training.multiload_data import MultiLoadStatic
        from utils.gnn_hierarchical_cnn import HierarchicalGNNCNN
        self.geometry = geometry_name
        self.ckpt_path = ckpt_path or os.environ.get("MULTILOAD_CKPT")
        self.dataset_dir = dataset_dir or os.environ.get(
            "MULTILOAD_DATA", "data/cross4_11x11/dataset_multiload_5000")
        self.rho = stress_ks_rho_mpa * 1e-6
        self.mesh_threads = mesh_threads
        self.jacobian_workers = jacobian_workers or int(os.environ.get("MULTILOAD_JAC_WORKERS") or 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bc = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
        self.bc = bc
        self.NX, self.NY, self.NZ = bc.tile_grid()
        self.fixed_dims = bc.fixed_dims()
        self.U, self.V, self.W = bc.mesh_resolution()
        self.dims_keys = list(json.load(open(f"data/{geometry_name}/CAD_model/dims.json")))
        self.material = json.load(open(f"data/{geometry_name}/CAD_model/material_properties.json"))

        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.model = HierarchicalGNNCNN(**ck["model_kwargs"]).to(self.device)
        self.model.set_tile_grid(self.NX, self.NY, self.NZ)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        for p_ in self.model.parameters():
            p_.requires_grad_(False)
        self.st = {k: v.to(self.device).float() for k, v in ck["stats"].items()}
        self.out_dim = int(ck["model_kwargs"]["node_out_dim"])
        assert self.out_dim in (6, 9), f"unexpected node_out_dim {self.out_dim}"

        self.static = MultiLoadStatic(self.dataset_dir, self.NX, self.NY, bc.LENGTH, bc.HEIGHT)
        cfg = self.static.cfg
        self.row = bc.load_row()
        self.direction = int(os.environ.get("MULTILOAD_DIR", "0"))
        self.F_scale = float(os.environ.get("FEA_LOAD_N") or cfg["load_N"]) / float(cfg["load_N"])
        dev = self.device
        self.edge_index = self.static.edge_index.to(dev)
        self.node_tile = torch.as_tensor(self.static.node_tile, device=dev)
        self.anch = torch.as_tensor(self.static.anchored, dtype=torch.float32, device=dev)[:, None]
        self.load_nodes = torch.as_tensor(self.static.load_nodes[self.row], device=dev)
        keep = torch.ones(self.static.N, dtype=torch.bool, device=dev)
        keep[self.load_nodes] = False
        self.ks_keep = keep
        ix, iy = np.meshgrid(np.arange(self.NX), np.arange(self.NY), indexing="ij")
        # dims order (ix*NY + iy) -> IRIT tile order (ix + NX*iy)
        self.irit_of_dims = torch.as_tensor((ix + self.NX * iy).reshape(-1), device=dev)
        self.load_tile = (self.NX - 1) + self.NX * self.row
        self._mesh_checked = False
        # MULTILOAD_CCX_CHECK_EVERY=N: every Nth design is also solved with the
        # exact ccx adjoint path, and its true peak (loaded tile excluded) and
        # gradient cosine are printed next to the surrogate's -- the honest
        # check the optimizer's own log never gives for a surrogate run.
        self.ccx_every = int(os.environ.get("MULTILOAD_CCX_CHECK_EVERY") or 0)
        self._calls = 0
        self._ccx = None
        self._ccx_last = None
        # MULTILOAD_FT=1: fine-tune the surrogate on exact labels of the designs
        # this run visits (training/multiload_online.py; schedule via MULTILOAD_FT_*).
        self.ft = None
        if os.environ.get("MULTILOAD_FT") == "1":
            from training.multiload_online import MultiLoadOnlineFT
            self.ft = MultiLoadOnlineFT(self)
        print(f"[GNNMultiLoad] ckpt {self.ckpt_path} (out {self.out_dim}) on {dev}; load row {self.row}, "
              f"dir {'down' if self.direction == 0 else 'anchor'}, F x{self.F_scale:g}", flush=True)

    # ------------------------------------------------------------------
    def _features(self, X, t_dims):
        """Model inputs from coords X (N,3) and thickness t_dims (K, dims order),
        both inside autograd."""
        st, NX = self.st, self.NX
        t_irit = torch.empty_like(t_dims)
        t_irit[self.irit_of_dims] = t_dims
        src, dst = self.edge_index
        d = X[dst] - X[src]
        edge_attr = torch.cat([d, d.norm(dim=1, keepdim=True)], dim=1)
        # per-node load / F: weights 1 - r/r_max about the loaded tile's centre (y,z)
        ids = self.load_nodes
        c = torch.tensor([(self.row + 0.5) * self.static.ty, self.static.length - 0.5 * self.static.tz],
                         device=X.device, dtype=X.dtype)
        r = (X[ids][:, 1:3] - c).norm(dim=1)
        w = 1.0 - r / r.max()
        w = w / w.sum()
        dirv = torch.tensor([0.0, -1.0, 0.0] if self.direction == 0 else [0.0, 0.0, -1.0],
                            device=X.device, dtype=X.dtype)
        load = torch.zeros_like(X).index_put((ids,), w[:, None] * dirv[None, :])
        x = torch.cat([X, load, self.anch.to(X.dtype), t_irit[self.node_tile][:, None]], dim=1)
        tile_x = torch.zeros(NX * self.NY, 4, device=X.device, dtype=X.dtype)
        tile_x[:, 0] = t_irit
        tile_x[self.load_tile, 1] = 1.0
        tile_x[self.load_tile, 2:] = dirv[1:]
        tile_x = torch.cat([t_irit[:, None], tile_x[:, 1:]], dim=1)
        x = (x - st["x_mean"]) / st["x_std"]
        edge_attr = (edge_attr - st["edge_mean"]) / st["edge_std"]
        tile_x = (tile_x - st["tile_mean"]) / st["tile_std"]
        return x, edge_attr, tile_x

    def _predict_stress(self, X, t_dims):
        x, ea, tx = self._features(X, t_dims)
        N = X.shape[0]
        batch = torch.zeros(N, dtype=torch.long, device=X.device)
        _, pred = self.model(x, self.edge_index, ea, batch, tile_idx=self.node_tile,
                             tile_NX=self.NX, tile_NY=self.NY, tile_NZ=self.NZ, tile_x=tx,
                             num_graphs=1, node_id=torch.arange(N, device=X.device))
        P = pred * self.st["y_std"] + self.st["y_mean"]
        S = P[:, 3:] if self.out_dim == 9 else P
        return S * self.F_scale                                       # Pa

    @staticmethod
    def _von_mises(S):
        sxx, syy, szz, sxy, sxz, syz = (S[:, i] for i in range(6))
        return torch.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                          + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2) + 1e-12)

    def _mesh(self, dims):
        from core.IritModel import IritCModel
        from core.geometric_jacobian import compute_tile_jacobian_colored
        exe = f"data/{self.geometry}/CAD_model/model"
        D_tile, nodes0, elements0, elem_to_tile, cad = compute_tile_jacobian_colored(
            exe, dims, self.U, self.V, self.W, (self.NX, self.NY, self.NZ), fixed_dims=self.fixed_dims,
            mesh_threads=self.mesh_threads, model_cls=IritCModel, workers=self.jacobian_workers)
        ids = sorted(nodes0)
        if not self._mesh_checked:
            m = np.load(os.path.join(self.dataset_dir, "mesh.npz"))
            assert list(m["node_ids"]) == ids, "mesh node ids differ from the dataset's mesh.npz"
            index = {n: i for i, n in enumerate(ids)}
            conn = np.array([[index[n] for n in elements0[e][:8]] for e in sorted(elements0)])
            assert np.array_equal(conn, m["elements"]), "mesh connectivity differs from the dataset's"
            self._mesh_checked = True
        X = np.array([nodes0[i] for i in ids], dtype=np.float64)
        return X, ids, D_tile

    # ------------------------------------------------------------------
    def compute_topopt_gradient(self, dims: dict):
        from core.fem_adjoint import _geom
        dims = {k: float(dims[k]) for k in self.dims_keys}
        if self.ft is not None:
            self.ft.maybe_update(dims)
        Xnp, ids, D_tile = self._mesh(dims)
        dev = self.device
        X = torch.tensor(Xnp, dtype=torch.float32, device=dev, requires_grad=True)
        t = torch.tensor([dims[k] for k in self.dims_keys], dtype=torch.float32, device=dev,
                         requires_grad=True)
        S = self._predict_stress(X, t)
        vm = self._von_mises(S)[self.ks_keep].double()
        m = vm.max().detach()
        ks = m + torch.log(torch.exp(self.rho * (vm - m)).sum()) / self.rho
        ks.backward()
        gX = X.grad.detach().cpu().double().numpy()
        gt = t.grad.detach().cpu().double().numpy()
        ks_mpa = float(ks) / 1e6
        # STRESS_KS_NORMALIZE as in CCXEvaluator: "max" reports the peak and
        # scales the KS gradient by c = peak/KS; "smooth" lets c follow
        # peak/KS at STRESS_KS_C_RATE per call and reports c * KS.
        scale = 1.0
        mode = os.environ.get("STRESS_KS_NORMALIZE")
        if mode == "max":
            scale = float(m) / float(ks)
        elif mode == "smooth":
            ratio = float(m) / float(ks)
            c_prev = getattr(self, "_ks_c", None)
            rate = float(os.environ.get("STRESS_KS_C_RATE", "0.2"))
            self._ks_c = ratio if c_prev is None else c_prev + rate * (ratio - c_prev)
            scale = self._ks_c
        value_mpa = scale * ks_mpa
        # volume = sum of hex Gauss-point detJ (unit weights), same mesh
        Xv = torch.tensor(Xnp, dtype=torch.float64, requires_grad=True)
        conn = torch.as_tensor(self.static.elements)
        _, detJ = _geom(Xv[conn])
        vol = detJ.abs().sum()
        vol.backward()
        gV = Xv.grad.numpy()
        row = {n: i for i, n in enumerate(ids)}
        yld = float(self.material["yield_strength"])
        dG, dV = {}, {}
        for i, k in enumerate(self.dims_keys):
            d = D_tile.get(k, {})
            geo_g = sum(float(gX[row[n]] @ np.asarray(v)) for n, v in d.items())
            geo_v = sum(float(gV[row[n]] @ np.asarray(v)) for n, v in d.items())
            dG[k] = scale * (geo_g + gt[i]) / 1e6 / yld
            dV[k] = geo_v
        print(f"[GNNMultiLoad] KS {ks_mpa:.2f} MPa, peak {float(m) / 1e6:.2f} MPa, c {scale:.4f} -> "
              f"constraint {value_mpa:.2f} MPa, volume {float(vol):.3f}", flush=True)
        self._calls += 1
        if self.ccx_every and (self._calls - 1) % self.ccx_every == 0:
            self._ccx_check(dims, float(m) / 1e6, dG)
        return float(vol), value_mpa, yld, dV, dG

    def _ccx_solve(self, dims):
        """Exact ccx adjoint at dims -> (true peak MPa, dG dict, same units as
        ours); the last result is cached so a screenshot right after a ccx check
        on the same design reuses it."""
        key = tuple(float(dims[k]) for k in self.dims_keys)
        if self._ccx_last is not None and self._ccx_last[0] == key:
            return self._ccx_last[1]
        if self._ccx is None:
            from .ccx_evaluator import CCXEvaluator
            self._ccx = CCXEvaluator(self.geometry, max_workers=4, mesh_threads=8,
                                     stress_ks_rho_mpa=self.rho * 1e6, gradient_source="adjoint")
        prev = os.environ.get("STRESS_KS_NORMALIZE")
        os.environ["STRESS_KS_NORMALIZE"] = "max"            # ccx then returns its true peak
        try:
            _, peak_ccx, _, _, g = self._ccx.compute_topopt_gradient(dict(zip(self.dims_keys, key)))
        finally:
            if prev is None:
                os.environ.pop("STRESS_KS_NORMALIZE", None)
            else:
                os.environ["STRESS_KS_NORMALIZE"] = prev
        self._ccx_last = (key, (peak_ccx, g))
        return peak_ccx, g

    def reference_gradient(self, dims):
        """The solver's (ccx adjoint) dG at dims, for the topopt gradient
        heatmap's side-by-side panel."""
        return self._ccx_solve(dims)[1]

    def _ccx_check(self, dims, peak_surr, dG):
        try:
            peak_ccx, g = self._ccx_solve(dims)
            a = np.array([dG[k] for k in self.dims_keys]); b = np.array([g[k] for k in self.dims_keys])
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
            print(f"[GNNMultiLoad ccx check] call {self._calls}: peak surrogate {peak_surr:.2f} MPa, "
                  f"ccx {peak_ccx:.2f} MPa ({peak_surr / peak_ccx - 1:+.1%}); dG cosine {cos:+.3f}, "
                  f"sign agreement {np.mean(np.sign(a) == np.sign(b)):.2f}", flush=True)
        except Exception as e:
            print(f"[GNNMultiLoad ccx check] failed: {e}", flush=True)

    def evaluate(self, dims_list):
        out = []
        for d in dims_list:
            v, s, *_ = self.compute_topopt_gradient(d)
            out.append((v, s))
        return out
