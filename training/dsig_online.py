"""Synchronous on-trajectory fine-tuning of the superposing stress surrogate
(training/train_dsig.py checkpoints) inside a topopt run, the dsig analogue of
training/multiload_online.py.

Every DSIG_FT_EVERY-th design the optimizer asks GNNDsigEvaluator about, before
the surrogate evaluates it:
  1. LABEL -- the design plus DSIG_FT_SAMPLES-1 jittered copies (Gaussian, sd
     DSIG_FT_JITTER in thickness, clipped to the run's bounds) are solved
     exactly with the dataset builder's own solver
     (build_cross4_dsig_dataset.solve_design): S and dS/dt of every load row the
     evaluator uses, at the dataset's load_N.
  2. TRAIN -- DSIG_FT_STEPS AdamW steps on the pretraining loss (standardized
     stress MSE + sob_weight x saturated relative error of the directional
     derivative J v against sum_j v_j dS_j), DSIG_FT_BATCH cases per step, each
     batch mixing the trajectory buffer (FIFO of the last DSIG_FT_BUFFER labelled
     cases) with REPLAY cases of training designs (DSIG_FT_REPLAY_FRAC), so the
     update cannot specialise onto the narrow stream of trajectory designs.
  3. GATE -- the new weights are scored on a fixed HOLD-OUT set (cases of the
     pretraining validation designs); the update is discarded if the median KS
     relative error grows by more than DSIG_FT_GATE (relative) or the median
     dKS/dt cosine drops by more than DSIG_FT_GATE_COS (absolute).
The design is then evaluated with the (possibly) updated weights.

Logged per update: the surrogate's KS and dG/dt against the exact labels of the
current design for the run's own load combination BEFORE training (the honest
on-trajectory accuracy) and after, the hold-out scores and the gate decision.
Fine-tuned weights go to $TOPOPT_ARTIFACT_DIR/surrogate_ft.pt after every
accepted update.
"""
import copy
import importlib
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from training.dsig_data import design_path
from training.train_dsig import directional, rel_err


def _env(name, default, cast=float):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


def _label_one(args):
    """Worker: exact S, dS for one design and the used rows. Fresh process."""
    dims, rows, F, rho, threads, jac_workers = args
    torch.set_num_threads(threads)
    from training.build_cross4_dsig_dataset import solve_design
    bc = importlib.import_module("data.cross4_11x11.boundary_conditions")
    t0 = time.time()
    s, _ = solve_design(dims, bc, rows, F, rho, workers=jac_workers)
    return dict(coords=s["coords"].astype(np.float64), S=s["S"].astype(np.float32),
                dS=s["dS"].astype(np.float16), secs=time.time() - t0)


class DsigOnlineFT:
    def __init__(self, ev):
        """ev: the GNNDsigEvaluator (its model is trained in place)."""
        self.ev = ev
        self.every = _env("DSIG_FT_EVERY", 1, int)
        self.samples = _env("DSIG_FT_SAMPLES", 4, int)
        self.jitter = _env("DSIG_FT_JITTER", 0.03)
        self.steps = _env("DSIG_FT_STEPS", 20, int)
        self.batch = _env("DSIG_FT_BATCH", 4, int)
        self.lr = _env("DSIG_FT_LR", 5e-5)
        self.replay_frac = _env("DSIG_FT_REPLAY_FRAC", 0.5)
        self.buffer_n = _env("DSIG_FT_BUFFER", 40, int)
        self.gate = _env("DSIG_FT_GATE", 0.10)
        # DSIG_FT_GATE_KS_ABS: allow the hold-out median KS error to rise by this
        # many absolute points (0.02 = 2 percentage points) instead of DSIG_FT_GATE
        # relative -- 10% of a ~5% base is only 0.5 points and blocked most updates
        self.gate_abs = _env("DSIG_FT_GATE_KS_ABS", None)
        self.gate_cos = _env("DSIG_FT_GATE_COS", 0.05)
        self.label_threads = _env("DSIG_FT_LABEL_THREADS", 2, int)
        self.jac_workers = _env("DSIG_FT_JAC_WORKERS", 6, int)
        n_replay = _env("DSIG_FT_REPLAY_DESIGNS", 48, int)
        n_hold = _env("DSIG_FT_HOLDOUT_DESIGNS", 12, int)
        seed = _env("DSIG_FT_SEED", 0, int)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.art = os.environ.get("TOPOPT_ARTIFACT_DIR", ".")
        s = ev.static
        self.rows = [s.rows[k] for k in ev.used]
        self.F_data = float(s.cfg["load_N"])
        dj = json.load(open(f"data/{ev.geometry}/CAD_model/dims.json"))
        self.lo = np.array([dj[k]["min"] for k in ev.dims_keys])
        self.hi = np.array([dj[k]["max"] for k in ev.dims_keys])
        if os.environ.get("TOPOPT_T_MIN"):
            self.lo = np.maximum(self.lo, float(os.environ["TOPOPT_T_MIN"]))
        if os.environ.get("TOPOPT_T_MAX"):
            self.hi = np.minimum(self.hi, float(os.environ["TOPOPT_T_MAX"]))
        ck = torch.load(ev.ckpt_path, map_location="cpu", weights_only=False)
        self.sob_weight = float(ck["args"]["sob_weight"])
        self.sob_dirs = int(ck["args"]["sob_dirs"])
        self.sob_power = float(ck["args"].get("sob_node_power", 1.0))
        split = json.load(open(os.path.join(ck["args"]["run_dir"], "split.json")))
        t0 = time.time()
        # replay: random rows of training designs; hold-out: the run's own rows
        # of validation designs the base never trained on
        self.replay = self._load_cases(split["train"], n_replay, per_design=2)
        self.holdout = self._load_cases(split["val"], n_hold, per_design=1, prefer=self.rows)
        print(f"[DsigFT] every {self.every} designs: {self.samples} labels (jitter {self.jitter}) of rows {self.rows}, "
              f"{self.steps} steps x {self.batch} cases, lr {self.lr:g}, sob weight {self.sob_weight:g}, "
              f"replay {self.replay_frac:.0%}, buffer {self.buffer_n}, gate KS "
              f"{f'+{self.gate_abs:g} abs' if self.gate_abs is not None else f'+{self.gate:.0%}'} / cos -{self.gate_cos}; "
              f"replay pool {len(self.replay)} cases, hold-out {len(self.holdout)} ({time.time() - t0:.0f}s to load)",
              flush=True)
        self.buffer = []
        self.calls = self.updates = self.accepted = 0
        self.opt = torch.optim.AdamW(list(ev.model.parameters()), lr=self.lr, weight_decay=0.0)
        # spawn, not fork: the parent already holds a CUDA context
        self.pool = ProcessPoolExecutor(max_workers=self.samples,
                                        mp_context=__import__("multiprocessing").get_context("spawn"))
        self.base_hold = self._holdout()
        print(f"[DsigFT] base hold-out: KS rel {self.base_hold[0]:.4f}, dKS cos {self.base_hold[1]:+.3f}, "
              f"|dKS| ratio {self.base_hold[2]:.3f} (medians)",
              flush=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _case(t, load_w, S, dS, row, dks=None, ks=None):
        return dict(t=torch.as_tensor(t, dtype=torch.float32), load_w=torch.as_tensor(load_w, dtype=torch.float32),
                    S=torch.as_tensor(S, dtype=torch.float32), dS=torch.as_tensor(dS, dtype=torch.float16),
                    row=row, dks=None if dks is None else torch.as_tensor(dks, dtype=torch.float32),
                    ks=None if ks is None else float(ks))

    def _load_cases(self, ids, n, per_design, prefer=()):
        s, out = self.ev.static, []
        for i in self.rng.sample(ids, min(n, len(ids))):
            with open(design_path(s.root, i), "rb") as f:
                z = dict(np.load(f))
            coords = z["coords"].astype(np.float64)
            pick = [r for r in prefer][:per_design]
            rest = [r for r in s.rows if r not in pick]
            pick += self.rng.sample(rest, per_design - len(pick))
            for r in pick:
                c = s.rows.index(r)
                out.append(self._case(z["t"], s.load_weights(coords, r), z["S"][c], z["dS"][c], r,
                                      z["dks"][c], z["ks"][c]))
        return out

    def _stack(self, cases, dev):
        t = torch.stack([c["t"] for c in cases]).to(dev)
        lw = torch.stack([c["load_w"] for c in cases]).to(dev)
        return t, lw, torch.stack([c["S"] for c in cases]).to(dev), [c["row"] for c in cases]

    def _holdout(self):
        """(median KS rel error, median dKS/dt cosine, median |dKS/dt| ratio
        surrogate / exact) on the hold-out cases."""
        ev, dev = self.ev, self.ev.device
        ev.model.eval()
        kr, cs, nr = [], [], []
        for c in self.holdout:
            t, lw, _, rows = self._stack([c], dev)
            t.requires_grad_(True)
            S = ev.bt.forward(ev.model, t, lw, rows)
            ks = ev.bt.ks(S, ev.rho)
            g, = torch.autograd.grad(ks.sum(), t)
            g, gt = g[0].double().cpu(), c["dks"].double()
            kr.append(abs(float(ks) / c["ks"] - 1))
            cs.append(float(g @ gt / (g.norm() * gt.norm() + 1e-30)))
            nr.append(float(g.norm() / (gt.norm() + 1e-30)))
        return float(np.median(kr)), float(np.median(cs)), float(np.median(nr))

    def _traj_err(self, x, lab):
        """Surrogate vs exact, for the run's own combination of the labelled rows."""
        ev = self.ev
        G, _, g = ev.predict({k: float(v) for k, v in zip(ev.dims_keys, x)}, lab["coords"])
        S = torch.as_tensor(lab["S"], dtype=torch.float64, device=ev.device) * ev.F_scale
        dS = torch.as_tensor(lab["dS"], device=ev.device).double() * ev.F_scale
        w = torch.as_tensor(ev.weights[ev.used], dtype=torch.float64, device=ev.device)
        S.requires_grad_(True)
        Gt, peak = ev._combine(S, w)
        gS, = torch.autograd.grad(Gt, S)
        gt = torch.einsum("bnc,bknc->k", gS, dS).cpu().numpy()
        cos = float(g @ gt / (np.linalg.norm(g) * np.linalg.norm(gt) + 1e-30))
        wrong = np.abs(gt)[np.sign(g) != np.sign(gt)].sum() / np.abs(gt).sum()
        ratio = float(np.linalg.norm(g) / (np.linalg.norm(gt) + 1e-30))
        return G, float(Gt), float(peak), cos, float(wrong), ratio

    def _loss(self, cases):
        ev, dev, st = self.ev, self.ev.device, self.ev.st
        t, lw, S_t, rows = self._stack(cases, dev)
        t.requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            S_p = ev.bt.forward(ev.model, t, lw, rows)
        value = (((S_p - S_t) / st["y_std"]) ** 2).mean()
        dS = torch.stack([c["dS"] for c in cases]).to(dev).float()
        dirs = [torch.randn(len(cases), ev.static.K, device=dev, dtype=torch.float32) for _ in range(self.sob_dirs)]
        sob = 0.0
        for v, Jv in zip(dirs, directional(S_p, t, dirs)):
            r = rel_err(Jv, torch.einsum("bk,bknc->bnc", v, dS), self.sob_power)
            sob = sob + (r / (1 + r / 2)).mean() / self.sob_dirs
        return value + self.sob_weight * sob

    # ------------------------------------------------------------------
    def maybe_update(self, dims):
        """Called by the evaluator with the design it is about to evaluate."""
        self.calls += 1
        if (self.calls - 1) % self.every:
            return
        ev, t0 = self.ev, time.time()
        x0 = np.array([dims[k] for k in ev.dims_keys])
        designs = [x0] + [np.clip(x0 + self.jitter * self.np_rng.standard_normal(x0.size), self.lo, self.hi)
                          for _ in range(self.samples - 1)]
        jobs = [({k: float(v) for k, v in zip(ev.dims_keys, x)}, self.rows, self.F_data, ev.rho,
                 self.label_threads, self.jac_workers) for x in designs]
        labels = list(self.pool.map(_label_one, jobs))
        t_lab = time.time() - t0
        pre = self._traj_err(x0, labels[0])            # on-trajectory accuracy before training
        for x, lab in zip(designs, labels):
            for b, r in enumerate(self.rows):
                self.buffer.append(self._case(x, ev.static.load_weights(lab["coords"], r), lab["S"][b],
                                              lab["dS"][b], r))
        self.buffer = self.buffer[-self.buffer_n:]

        snap = copy.deepcopy(ev.model.state_dict())
        opt_snap = copy.deepcopy(self.opt.state_dict())
        for p_ in ev.model.parameters():
            p_.requires_grad_(True)
        ev.model.train()
        n_traj = max(1, round(self.batch * (1 - self.replay_frac)))
        losses = []
        for _ in range(self.steps):
            cases = ([self.rng.choice(self.buffer) for _ in range(n_traj)]
                     + [self.rng.choice(self.replay) for _ in range(self.batch - n_traj)])
            loss = self._loss(cases)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ev.model.parameters(), 1.0)
            self.opt.step()
            losses.append(float(loss))
        ev.model.eval()
        for p_ in ev.model.parameters():
            p_.requires_grad_(False)
        self.opt.zero_grad(set_to_none=True)

        hold = self._holdout()
        post = self._traj_err(x0, labels[0])
        self.updates += 1
        ks_max = (self.base_hold[0] + self.gate_abs if self.gate_abs is not None
                  else self.base_hold[0] * (1 + self.gate))
        ok = hold[0] <= ks_max and hold[1] >= self.base_hold[1] - self.gate_cos
        if ok:
            self.accepted += 1
            path = os.path.join(self.art, "surrogate_ft.pt")
            torch.save({**torch.load(ev.ckpt_path, map_location="cpu", weights_only=False),
                        "model": {k: v.cpu() for k, v in ev.model.state_dict().items()},
                        "ft_updates": self.accepted}, path + ".tmp")
            os.replace(path + ".tmp", path)
        else:
            ev.model.load_state_dict(snap)
            self.opt.load_state_dict(opt_snap)
        print(f"[DsigFT] update {self.updates} (call {self.calls}): exact KS {pre[1]:.2f} MPa (peak {pre[2]:.2f}); "
              f"before: KS {pre[0]:.2f} ({pre[0] / pre[1] - 1:+.1%}), dG cos {pre[3]:+.3f}, wrong-way {pre[4]:.1%}; "
              f"after: KS {post[0]:.2f} ({post[0] / post[1] - 1:+.1%}), cos {post[3]:+.3f}; loss {np.mean(losses):.4f}; "
              f"|dG| ratio before {pre[5]:.3f} after {post[5]:.3f}; "
              f"hold-out KS rel {hold[0]:.4f} cos {hold[1]:+.3f} |dKS| ratio {hold[2]:.3f} "
              f"(base {self.base_hold[0]:.4f} / {self.base_hold[1]:+.3f} / {self.base_hold[2]:.3f})"
              f" -> {'ACCEPTED' if ok else 'DISCARDED'}; {self.accepted}/{self.updates} kept; label {t_lab:.0f}s "
              f"(slowest solve {max(l['secs'] for l in labels):.0f}s), total {time.time() - t0:.0f}s", flush=True)

    def close(self):
        """Kill the label workers: an idle spawn pool whose workers have
        started their own process pools kept the topopt process from exiting."""
        for p_ in list(getattr(self.pool, "_processes", {}).values()):
            p_.kill()
        self.pool.shutdown(wait=False, cancel_futures=True)
