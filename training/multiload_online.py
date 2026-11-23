"""Synchronous on-trajectory fine-tuning of the multi-load stress surrogate
(training/train_multiload.py checkpoints) inside a topopt run.

Every MULTILOAD_FT_EVERY-th design the optimizer asks GNNMultiLoadEvaluator
about, before the surrogate evaluates it:
  1. LABEL -- the design plus MULTILOAD_FT_SAMPLES-1 jittered copies (Gaussian,
     sd MULTILOAD_FT_JITTER in thickness, clipped to dims.json bounds) are
     solved exactly for the run's single load case, in parallel, with the
     dataset builder's own solver (build_cross4_multiload_dataset.solve_design:
     same mesh, load rule, C3D8I stress recovery), at the dataset's load_N.
  2. TRAIN -- MULTILOAD_FT_STEPS AdamW steps of MULTILOAD_FT_BATCH graphs on the
     pretraining loss (MSE of the standardized stress tensor, bf16 autocast).
     Each batch mixes the trajectory buffer (FIFO of the last
     MULTILOAD_FT_BUFFER labels) with REPLAY cases from training designs of the
     dataset (MULTILOAD_FT_REPLAY_FRAC of every batch), so the update cannot
     specialise onto the narrow stream of trajectory designs.
  3. GATE -- the new weights are scored on a fixed HOLD-OUT set (cases of the
     pretraining validation designs, which the base never trained on); if the
     median von Mises relative error there grows by more than
     MULTILOAD_FT_GATE (relative to the base's), the update is discarded and
     the previous weights are kept.
The design is then evaluated with the (possibly) updated weights, so its
gradient comes from a model that has just seen it. Nothing is asynchronous:
a run is reproducible given the seed.

Logged per update: the surrogate's error on the fresh labels BEFORE training
(the honest on-trajectory accuracy), the ccx peak of the current design, the
hold-out error and the gate decision. Fine-tuned weights are saved to
$TOPOPT_ARTIFACT_DIR/surrogate_ft.pt after every accepted update.
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
from torch_geometric.data import Batch

from training.multiload_data import DesignDataset, design_cases


def _env(name, default, cast=float):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


def _label_one(args):
    """Worker: exact stress for one design, one case. Runs in a fresh process."""
    dims, row, direction, F, threads = args
    torch.set_num_threads(threads)
    from datasets.build_cross4_multiload_dataset import solve_design
    bc = importlib.import_module("data.cross4_11x11.boundary_conditions")
    t0 = time.time()
    s, _ = solve_design(dims, bc, [(row, direction)], F, 0.25)
    return dict(coords=s["coords"], S=s["S"][0], u=s["u"][0], max_vm=float(s["max_vm"][0]),
                max_vm_excl=float(s["max_vm_excl"][0]), secs=time.time() - t0)


def _von_mises(S):
    sxx, syy, szz, sxy, sxz, syz = (S[..., i] for i in range(6))
    return torch.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                      + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2) + 1e-12)


class MultiLoadOnlineFT:
    def __init__(self, ev):
        """ev: the GNNMultiLoadEvaluator (its model is trained in place)."""
        self.ev = ev
        self.every = _env("MULTILOAD_FT_EVERY", 1, int)
        self.samples = _env("MULTILOAD_FT_SAMPLES", 4, int)
        self.jitter = _env("MULTILOAD_FT_JITTER", 0.05)
        self.steps = _env("MULTILOAD_FT_STEPS", 20, int)
        self.batch = _env("MULTILOAD_FT_BATCH", 4, int)
        self.lr = _env("MULTILOAD_FT_LR", 1e-4)
        self.replay_frac = _env("MULTILOAD_FT_REPLAY_FRAC", 0.5)
        self.buffer_n = _env("MULTILOAD_FT_BUFFER", 40, int)
        self.gate = _env("MULTILOAD_FT_GATE", 0.10)
        self.label_threads = _env("MULTILOAD_FT_LABEL_THREADS", 2, int)
        n_replay = _env("MULTILOAD_FT_REPLAY_DESIGNS", 48, int)
        n_hold = _env("MULTILOAD_FT_HOLDOUT_DESIGNS", 12, int)
        seed = _env("MULTILOAD_FT_SEED", 0, int)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.art = os.environ.get("TOPOPT_ARTIFACT_DIR", ".")
        st = ev.static
        self.case_k = st.cases.index((ev.row, ev.direction))
        self.F_data = float(st.cfg["load_N"])
        dj = json.load(open(f"data/{ev.geometry}/CAD_model/dims.json"))
        self.lo = np.array([dj[k]["min"] for k in ev.dims_keys])
        self.hi = np.array([dj[k]["max"] for k in ev.dims_keys])

        # Train/val split exactly as train_multiload (random.Random(seed).shuffle,
        # first val_designs are validation): replay from train designs, hold-out
        # from validation designs the base never saw.
        ck_args = torch.load(ev.ckpt_path, map_location="cpu", weights_only=False).get("args", {})
        n_designs = int(st.cfg["num_designs"])
        ids = list(range(n_designs))
        random.Random(int(ck_args.get("seed", 0))).shuffle(ids)
        nv = int(ck_args.get("val_designs", 250))
        val_ids, train_ids = ids[:nv], ids[nv:]
        peak_max = float(ck_args.get("peak_max_mpa", 300.0)) * 1e6
        t0 = time.time()
        self.replay = self._load_cases(train_ids, n_replay, peak_max, per_design=3)
        # hold-out: the run's own load case where it passes the filter, else any kept case
        self.holdout = self._load_cases(val_ids, n_hold, peak_max, per_design=2, prefer=self.case_k)
        print(f"[MultiLoadFT] every {self.every} designs: {self.samples} labels (jitter {self.jitter}), "
              f"{self.steps} steps x {self.batch} graphs, lr {self.lr:g}, replay {self.replay_frac:.0%}, "
              f"buffer {self.buffer_n}, gate +{self.gate:.0%}; replay pool {len(self.replay)} cases, "
              f"hold-out {len(self.holdout)} cases ({time.time() - t0:.0f}s to load)", flush=True)

        self.buffer = []
        self.calls = 0
        self.updates = self.accepted = 0
        self.opt = torch.optim.AdamW([p for p in ev.model.parameters()], lr=self.lr, weight_decay=0.0)
        # spawn, not fork: the parent already holds a CUDA context
        self.pool = ProcessPoolExecutor(max_workers=self.samples,
                                        mp_context=__import__("multiprocessing").get_context("spawn"))
        self.base_hold = self._holdout_err()
        self.cur_hold = self.base_hold
        print(f"[MultiLoadFT] base hold-out vm_rel {self.base_hold:.4f} (median); gate at "
              f"{self.base_hold * (1 + self.gate):.4f}", flush=True)

    # ------------------------------------------------------------------
    def _load_cases(self, design_ids, n, peak_max, per_design, prefer=None):
        ds = DesignDataset(self.ev.dataset_dir, [], self.ev.static, peak_max)
        out = []
        for i in self.rng.sample(design_ids, min(n, len(design_ids))):
            path = os.path.join(ds.root, "designs", f"{i:05d}.npz")
            with open(path, "rb") as f:
                z = dict(np.load(f))
            keep = [k for k in range(len(ds.static.cases)) if z["max_vm_excl"][k] <= peak_max]
            if not keep:
                continue
            pick = []
            if prefer is not None and prefer in keep:
                pick.append(prefer)
            rest = [k for k in keep if k not in pick]
            pick += self.rng.sample(rest, min(per_design - len(pick), len(rest)))
            out += design_cases(ds.static, z, pick)
        return out

    def _to_batch(self, cases):
        st, dev = self.ev.st, self.ev.device
        b = Batch.from_data_list(cases).to(dev, non_blocking=True)
        b.y = b.y[:, 3:9] if self.ev.out_dim == 6 else b.y
        b.x = (b.x - st["x_mean"]) / st["x_std"]
        b.edge_attr = (b.edge_attr - st["edge_mean"]) / st["edge_std"]
        b.tile_x = (b.tile_x - st["tile_mean"]) / st["tile_std"]
        return b

    def _forward(self, b):
        ev = self.ev
        _, pred = ev.model(b.x, b.edge_index, b.edge_attr, b.batch, tile_idx=b.tile_idx,
                           tile_NX=ev.NX, tile_NY=ev.NY, tile_NZ=ev.NZ, tile_x=b.tile_x,
                           num_graphs=b.num_graphs, node_id=b.node_id)
        return pred

    @torch.no_grad()
    def _errors(self, cases):
        """Per case: (vm_rel L2, peak rel error with the case's loaded tile excluded), fp32."""
        st, out = self.ev.st, []
        self.ev.model.eval()
        for j in range(0, len(cases), 2):
            b = self._to_batch(cases[j:j + 2])
            P = self._forward(b).float() * st["y_std"] + st["y_mean"]
            for g in range(b.num_graphs):
                m = b.batch == g
                vp, vt = _von_mises(P[m]), _von_mises(b.y[m])
                row = self.ev.static.cases[int(b.case[g])][0]
                keep = torch.ones(int(m.sum()), dtype=torch.bool, device=vp.device)
                keep[torch.as_tensor(self.ev.static.load_nodes[row], device=vp.device)] = False
                out.append((float((vp - vt).norm() / vt.norm().clamp_min(1e-30)),
                            float(vp[keep].max() / vt[keep].max() - 1)))
        return out

    def _holdout_err(self):
        return float(np.median([e[0] for e in self._errors(self.holdout)]))

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
        jobs = [({k: float(v) for k, v in zip(ev.dims_keys, x)}, ev.row, ev.direction, self.F_data,
                 self.label_threads) for x in designs]
        new, secs = [], []
        for x, lab in zip(designs, self.pool.map(_label_one, jobs)):
            z = {"coords": lab["coords"], "t": x.astype(np.float32),
                 "u": {self.case_k: lab["u"]}, "S": {self.case_k: lab["S"]},
                 "max_vm": {self.case_k: lab["max_vm"]}, "max_vm_excl": {self.case_k: lab["max_vm_excl"]}}
            new += design_cases(ev.static, z, [self.case_k])
            secs.append(lab["secs"])
        t_lab = time.time() - t0
        pre = self._errors(new)                               # on-trajectory accuracy before training
        peak_ccx = new[0].max_vm_excl.item() / 1e6 * ev.F_scale
        self.buffer = (self.buffer + new)[-self.buffer_n:]

        snap = copy.deepcopy(ev.model.state_dict())
        opt_snap = copy.deepcopy(self.opt.state_dict())
        st = ev.st
        for p_ in ev.model.parameters():
            p_.requires_grad_(True)
        ev.model.train()
        n_traj = max(1, round(self.batch * (1 - self.replay_frac)))
        losses = []
        for _ in range(self.steps):
            cases = ([self.rng.choice(self.buffer) for _ in range(n_traj)]
                     + [self.rng.choice(self.replay) for _ in range(self.batch - n_traj)])
            b = self._to_batch(cases)
            yn = (b.y - st["y_mean"]) / st["y_std"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ev.device.type == "cuda"):
                pred = self._forward(b)
            loss = ((pred.float() - yn) ** 2).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ev.model.parameters(), 1.0)
            self.opt.step()
            losses.append(float(loss))
        ev.model.eval()
        for p_ in ev.model.parameters():
            p_.requires_grad_(False)
        self.opt.zero_grad(set_to_none=True)

        hold = self._holdout_err()
        post = self._errors(new[:1])[0]
        self.updates += 1
        ok = hold <= self.base_hold * (1 + self.gate)
        if ok:
            self.accepted += 1
            self.cur_hold = hold
            torch.save({**torch.load(ev.ckpt_path, map_location="cpu", weights_only=False),
                        "model": {k: v.cpu() for k, v in ev.model.state_dict().items()},
                        "ft_updates": self.accepted}, os.path.join(self.art, "surrogate_ft.pt.tmp"))
            os.replace(os.path.join(self.art, "surrogate_ft.pt.tmp"), os.path.join(self.art, "surrogate_ft.pt"))
        else:
            ev.model.load_state_dict(snap)
            self.opt.load_state_dict(opt_snap)
        pre_vm = np.median([e[0] for e in pre]); pre_pk = np.median([e[1] for e in pre])
        print(f"[MultiLoadFT] update {self.updates} (call {self.calls}): ccx peak {peak_ccx:.2f} MPa; "
              f"before: vm_rel {pre_vm:.4f} peak {pre_pk:+.1%} (this design {pre[0][1]:+.1%}); "
              f"after: this design vm_rel {post[0]:.4f} peak {post[1]:+.1%}; loss {np.mean(losses):.4f}; "
              f"hold-out {hold:.4f} (base {self.base_hold:.4f}) -> {'ACCEPTED' if ok else 'DISCARDED'}; "
              f"{self.accepted}/{self.updates} kept; label {t_lab:.0f}s "
              f"(slowest solve {max(secs):.0f}s), total {time.time() - t0:.0f}s", flush=True)
