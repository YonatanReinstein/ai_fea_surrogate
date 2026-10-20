"""Optimizer evaluator for the superposing stress surrogate
(training/train_dsig.py checkpoints): any combination of the 11 free-end tile
loads of cross4_11x11.

compute_topopt_gradient(dims) -> (volume, stress_MPa, yield, dV/dt, dG/dt), the
same contract as CCXEvaluator.

The model predicts, per load row k, the nodal stress tensor S_k(t) as a
function of the tile thicknesses t only. Rows with a nonzero weight run as one
batched forward pass, and stress is linear in the load, so:
    DSIG_COMBINE=superpose (default)  one load case, sum_k w_k * F_k applied
        together: S = sum_k w_k S_k, G = KS(von Mises(S))
    DSIG_COMBINE=envelope             each row is its own load case, scaled by
        w_k: G = KS over the per-case KS values (same rho), i.e. the worst case
DSIG_LOAD_WEIGHTS=w0,...,w10 sets the weights; without it FEA_LOAD_ROW must name
a single row (one-hot), which makes this directly comparable with a ccx run at
that row. Stress is then scaled by FEA_LOAD_N / the dataset's load_N.
KS runs over every node except the anchored pad, rho = stress_ks_rho_mpa,
exactly as the training labels.

Gradient: one backward pass to t gives the total dG/dt -- the model never sees
the node coordinates, so there is no geometry term to chain. The CAD mesh and
its tile Jacobian are still built every call, for the exact volume and dV/dt
(and the pyramid load weights, which the labels hold fixed).

DSIG_EXACT_CHECK_EVERY=N: every Nth design is also solved exactly
(training.build_cross4_dsig_dataset.solve_design: S and dS/dt of every used
row, then the same combination), and the true G and gradient cosine are
printed next to the surrogate's. Needs the dataset's env (FEA_LOAD_MODE=
tile_nodes, FEA_LOAD_WEIGHT=pyramid, CROSS4_LOAD_PAD_THICKNESS=0).
"""
import importlib
import json
import os

import numpy as np
import torch

from training.train_multiload import von_mises

from .base_evaluator import BaseEvaluator


class GNNDsigEvaluator(BaseEvaluator):
    def __init__(self, geometry_name, ckpt_path=None, dataset_dir=None, stress_ks_rho_mpa=0.25,
                 mesh_threads=8, jacobian_workers=None):
        super().__init__(geometry_name)
        from training.dsig_data import DsigStatic
        from training.train_dsig import Batcher
        from utils.gnn_hierarchical_cnn import HierarchicalGNNCNN
        self.geometry = geometry_name
        self.ckpt_path = ckpt_path or os.environ.get("DSIG_CKPT")
        self.dataset_dir = dataset_dir or os.environ.get("DSIG_DATA", "data/cross4_11x11/dataset_dsig_2500_tmin003")
        self.rho = float(stress_ks_rho_mpa)
        self.mesh_threads = mesh_threads
        self.jacobian_workers = jacobian_workers or int(os.environ.get("DSIG_JAC_WORKERS") or 10)
        self.device = dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bc = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
        self.bc = bc
        self.NX, self.NY, self.NZ = bc.tile_grid()
        self.fixed_dims = bc.fixed_dims()
        self.U, self.V, self.W = bc.mesh_resolution()
        self.dims_keys = list(json.load(open(f"data/{geometry_name}/CAD_model/dims.json")))

        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.model = HierarchicalGNNCNN(**ck["model_kwargs"]).to(dev)
        self.model.set_tile_grid(self.NX, self.NY, self.NZ)
        self.model.float()            # importing core.fem_adjoint makes float64 the default dtype
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        for p_ in self.model.parameters():
            p_.requires_grad_(False)
        self.static = DsigStatic(self.dataset_dir, self.NX, self.NY, bc.LENGTH, bc.HEIGHT)
        self.static.ref_coords = ck["ref_coords"]                         # the run's own reference;
        src, dst = self.static.edge_index                                 # edge_attr follows it
        d = torch.as_tensor(self.static.ref_coords[dst] - self.static.ref_coords[src], dtype=torch.float32)
        self.static.edge_attr = torch.cat([d, d.norm(dim=1, keepdim=True)], dim=1)
        self.st = {k: v.to(dev).float() for k, v in ck["stats"].items()}
        self.bt = Batcher(self.static, self.st, dev)
        assert len(self.dims_keys) == self.static.K

        cfg = self.static.cfg
        C = len(self.static.rows)
        wenv = os.environ.get("DSIG_LOAD_WEIGHTS")
        if wenv:
            w = np.array([float(x) for x in wenv.split(",")])
            assert w.size == C, f"DSIG_LOAD_WEIGHTS needs {C} values, got {w.size}"
        else:
            rows = bc.load_rows()
            assert len(rows) == 1, (f"FEA_LOAD_ROW={rows}: several rows are ONE pyramid over the combined "
                                    "region in ccx, not a superposition -- set DSIG_LOAD_WEIGHTS instead")
            w = np.zeros(C); w[rows[0]] = 1.0
        self.weights = w
        self.used = [k for k in range(C) if w[k] != 0.0]
        self.combine = os.environ.get("DSIG_COMBINE", "superpose")
        assert self.combine in ("superpose", "envelope"), f"DSIG_COMBINE={self.combine}"
        self.F_scale = float(os.environ.get("FEA_LOAD_N") or cfg["load_N"]) / float(cfg["load_N"])
        self.yld = float(self.material_properties["yield_strength"])
        self._mesh_checked = False
        self.exact_every = int(os.environ.get("DSIG_EXACT_CHECK_EVERY") or 0)
        self._calls = 0
        self._exact_last = None
        # DSIG_FT=1: fine-tune the surrogate on exact labels of the designs this
        # run visits (training/dsig_online.py; schedule via DSIG_FT_*).
        self.ft = None
        if os.environ.get("DSIG_FT") == "1":
            from training.dsig_online import DsigOnlineFT
            self.ft = DsigOnlineFT(self)
        print(f"[GNNDsig] ckpt {self.ckpt_path} on {dev}; {self.combine} of rows "
              + ", ".join(f"{self.static.rows[k]}:{w[k]:g}" for k in self.used)
              + f"; F x{self.F_scale:g}, rho {self.rho} /MPa", flush=True)

    # ------------------------------------------------------------------
    def _mesh(self, dims):
        from core.IritModel import IritCModel
        from core.geometric_jacobian import compute_tile_jacobian_colored
        exe = f"data/{self.geometry}/CAD_model/model"
        D_tile, nodes0, elements0, _, _ = compute_tile_jacobian_colored(
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
        return np.array([nodes0[i] for i in ids], dtype=np.float64), ids, D_tile

    def _combine(self, S_rows, w):
        """S_rows (B,N,6) MPa of the used rows, w (B,) -> (G, peak), MPa."""
        bt = self.bt
        if self.combine == "superpose":
            S = torch.einsum("b,bnk->nk", w, S_rows)
            vm = von_mises(S)
            return bt.ks(S, self.rho), vm[bt.keep].max()
        ks = bt.ks(w[:, None, None] * S_rows, self.rho)                     # (B,) per-case KS
        m = ks.max().detach()
        peak = max(von_mises(w[b] * S_rows[b])[bt.keep].max() for b in range(len(w)))
        return m + torch.logsumexp(self.rho * (ks - m), 0) / self.rho, peak

    # ------------------------------------------------------------------
    def predict(self, dims, X=None):
        """(G MPa, peak MPa, dG/dt (K,) MPa per unit t in dims order)."""
        dims = {k: float(dims[k]) for k in self.dims_keys}
        if X is None:
            X = self._mesh(dims)[0]
        dev, s = self.device, self.static
        rows = [s.rows[k] for k in self.used]
        lw = torch.as_tensor(np.stack([s.load_weights(X, r) for r in rows]), dtype=torch.float32, device=dev)
        t = torch.tensor([dims[k] for k in self.dims_keys], dtype=torch.float32, device=dev, requires_grad=True)
        S = self.bt.forward(self.model, t.expand(len(rows), -1), lw, rows).double() * self.F_scale
        w = torch.as_tensor(self.weights[self.used], dtype=torch.float64, device=dev)
        G, peak = self._combine(S, w)
        g, = torch.autograd.grad(G, t)
        return float(G), float(peak), g.double().cpu().numpy()

    def compute_topopt_gradient(self, dims: dict):
        from core.fem_adjoint import _geom
        dims = {k: float(dims[k]) for k in self.dims_keys}
        if self.ft is not None:
            self.ft.maybe_update(dims)
        Xnp, ids, D_tile = self._mesh(dims)
        G, peak, gt = self.predict(dims, Xnp)
        # STRESS_KS_NORMALIZE as in CCXEvaluator: "max" reports the peak and
        # scales the KS gradient by c = peak/KS; "smooth" lets c follow
        # peak/KS at STRESS_KS_C_RATE per call and reports c * KS.
        scale = 1.0
        mode = os.environ.get("STRESS_KS_NORMALIZE")
        if mode == "max":
            scale = peak / G
        elif mode == "smooth":
            ratio = peak / G
            c_prev = getattr(self, "_ks_c", None)
            rate = float(os.environ.get("STRESS_KS_C_RATE", "0.2"))
            self._ks_c = ratio if c_prev is None else c_prev + rate * (ratio - c_prev)
            scale = self._ks_c
        value_mpa = scale * G
        # volume = sum of hex Gauss-point detJ (unit weights), chained through dX/dt
        Xv = torch.tensor(Xnp, dtype=torch.float64, requires_grad=True)
        _, detJ = _geom(Xv[torch.as_tensor(self.static.elements)])
        vol = detJ.abs().sum()
        vol.backward()
        gV = Xv.grad.numpy()
        row = {n: i for i, n in enumerate(ids)}
        dG = {k: scale * float(gt[i]) / self.yld for i, k in enumerate(self.dims_keys)}
        dV = {k: sum(float(gV[row[n]] @ np.asarray(v)) for n, v in D_tile.get(k, {}).items())
              for k in self.dims_keys}
        print(f"[GNNDsig] KS {G:.2f} MPa, peak {peak:.2f} MPa, c {scale:.4f} -> constraint {value_mpa:.2f} MPa, "
              f"volume {float(vol):.3f}", flush=True)
        self._calls += 1
        if self.exact_every and (self._calls - 1) % self.exact_every == 0:
            self._exact_check(dims, G, gt)
        return float(vol), value_mpa, self.yld, dV, dG

    # ------------------------------------------------------------------
    def exact(self, dims):
        """Exact (G MPa, peak MPa, dG/dt (K,)) for this design and combination,
        from the dataset builder's direct sensitivities. Cached for one design."""
        key = tuple(float(dims[k]) for k in self.dims_keys)
        if self._exact_last is not None and self._exact_last[0] == key:
            return self._exact_last[1]
        from training.build_cross4_dsig_dataset import solve_design
        rows = [self.static.rows[k] for k in self.used]
        out, _ = solve_design(dict(zip(self.dims_keys, key)), self.bc, rows, float(self.static.cfg["load_N"]),
                              self.rho, workers=self.jacobian_workers)
        S = torch.as_tensor(out["S"], device=self.device) * self.F_scale                 # (B,N,6) MPa
        dS = torch.as_tensor(out["dS"], device=self.device) * self.F_scale               # (B,K,N,6)
        w = torch.as_tensor(self.weights[self.used], dtype=torch.float64, device=self.device)
        S.requires_grad_(True)
        G, peak = self._combine(S, w)
        gS, = torch.autograd.grad(G, S)
        g = torch.einsum("bnc,bknc->k", gS, dS).cpu().numpy()
        res = (float(G), float(peak), g)
        self._exact_last = (key, res)
        return res

    def reference_gradient(self, dims):
        """Exact dG at dims, same units as compute_topopt_gradient's (no KS
        normalization), for the topopt gradient heatmap's side-by-side panel."""
        g = self.exact(dims)[2]
        return {k: float(g[i]) / self.yld for i, k in enumerate(self.dims_keys)}

    def _exact_check(self, dims, G_surr, g_surr):
        try:
            G, peak, g = self.exact(dims)
            cos = float(g_surr @ g / (np.linalg.norm(g_surr) * np.linalg.norm(g) + 1e-30))
            wrong = np.sign(g_surr) != np.sign(g)
            print(f"[GNNDsig exact check] call {self._calls}: KS surrogate {G_surr:.2f} MPa, exact {G:.2f} MPa "
                  f"({G_surr / G - 1:+.1%}), exact peak {peak:.2f}; dG cosine {cos:+.3f}, wrong-way mass "
                  f"{np.abs(g)[wrong].sum() / np.abs(g).sum():.1%}", flush=True)
        except Exception as e:
            print(f"[GNNDsig exact check] failed: {e}", flush=True)

    def close(self):
        if self.ft is not None:
            self.ft.close()

    def evaluate(self, dims_list):
        return [tuple(self.compute_topopt_gradient(d)[:2]) for d in dims_list]
