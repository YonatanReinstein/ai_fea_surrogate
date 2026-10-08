import importlib
import json
import os
import pickle
from time import perf_counter

import numpy as np

from .screenshot import screenshot


class TopologyOptimizer:
    """Optimality-criteria (OC) optimizer over per-tile thicknesses.

    Problem: minimize volume V(x) subject to G(x) = stress/yield - 1 <= 0 and
    lo <= x <= hi, with exact sensitivities from `gradient_func` (the ccx
    shape adjoint or a surrogate's autodiff):

        gradient_func(dims) -> (volume, stress, yield, dV_dx: dict, dG_dx: dict)

    Each generation is one gradient call and one OC step:

        B_e(lam) = lam * (-dG/dx_e) / (dV/dx_e)
        x_new_e  = clip(x_e * B_e(lam) ** eta, x_e -+ move, [lo_e, hi_e])

    with lam found by bisection so that the linearized constraint
    g0 + dG . (x_new - x) lands at -g_backoff (a small margin: stress is
    convex in thinning, so the linear model under-predicts the rise).

    The only safeguard kept is an adaptive move limit. Stress goes like 1/t,
    so a step that is mild on thick tiles overshoots on thin ones: when the
    measured g exceeds last generation's prediction by more than tr_tol the
    move scale halves (and a step that made an infeasible design worse is
    rejected and retried from the previous design); when the prediction was
    accurate it grows back by 1.3x.

    This replaces a version (commit 7bf07d9) built when the gradients were
    finite differences or an unreliable surrogate: trust sets that throttled
    "noisy" tiles to 10% of a step, rolling ccx calibration, gradient
    magnitude calibration, surrogate-reload re-anchoring, the online-learning
    label spool, the FD sweep, a density/projection mode and three sharpening
    modes. With exact gradients those mostly froze tiles that had real signal.

    Optional spatial filtering over the tile grid (filter_target):
      "none"     -- none
      "gradient" -- Sigmund (1997) sensitivity filter on dV and dG
      "design"   -- smooth the design after each OC step (x <- H x)
    """

    def __init__(self, fitness_func, dims_dict, gradient_func, generations=200,
                 geometry="hollow_cube", move_limit=0.2, oc_eta=0.5,
                 filter_target="none", filter_radius=1.5,
                 tr_tol=0.05, tr_min_scale=0.02, g_backoff=0.02,
                 stagnation_gens=0, init_thickness=None, stop_at_gen=None,
                 screenshot_heatmap=True, screenshot_every=1):
        if gradient_func is None:
            raise ValueError("TopologyOptimizer needs a gradient_func (ccx adjoint/axis or a "
                             "surrogate); the finite-difference sweep was removed (commit 7bf07d9 has it)")
        if filter_target not in ("none", "gradient", "design"):
            raise ValueError(f"filter_target {filter_target!r}: use none | gradient | design "
                             "(density/projection was removed; commit 7bf07d9 has it)")
        self.fitness_func = fitness_func          # kept for interface parity; unused here
        self.gradient_func = gradient_func
        # Optional dims -> dims map applied to every design handed out (gradient
        # calls, screenshots, best/final designs), e.g. the loaded tile tied to
        # its neighbours' mean (optimize_dims, TOPOPT_LOAD_TILE_MEAN). The tied
        # dim must be fixed (min == max) and gradient_func must chain-rule it.
        self.dims_transform = None
        self.dims_dict = dims_dict
        self.geometry = geometry
        self.generations = generations
        self.stop_at_gen = None if stop_at_gen is None else int(stop_at_gen)
        self.oc_eta = oc_eta
        self.filter_target = filter_target
        self.filter_radius = filter_radius
        self.tr_tol = tr_tol
        self.tr_min_scale = tr_min_scale
        self.g_backoff = g_backoff
        self.stagnation_gens = int(stagnation_gens)
        self.init_thickness = init_thickness
        # Each screenshot re-solves the design with ccx (heatmap) and usually
        # costs more than the gradient call; render every Nth generation.
        self.screenshot_heatmap = screenshot_heatmap
        self.screenshot_every = max(1, screenshot_every)

        self.dim_names = list(dims_dict)
        self.free_names = [n for n in self.dim_names if dims_dict[n]["max"] > dims_dict[n]["min"]]
        self.fixed = {n: float(dims_dict[n]["default"]) for n in self.dim_names if n not in self.free_names}
        self.lo = np.array([dims_dict[n]["min"] for n in self.free_names], dtype=float)
        self.hi = np.array([dims_dict[n]["max"] for n in self.free_names], dtype=float)
        self.span = self.hi - self.lo
        self.move = move_limit * self.span           # absolute per-tile step at move_scale 1
        self.N = len(self.free_names)
        self.filt = self._build_filter(filter_radius) if filter_target != "none" else None

        self._move_scale = 1.0
        self._prev = None           # last step: x, df, dg, g0, pred_g (for the move-scale check)
        self._stagnant = 0
        self._screenshot_owed = False

        # Per-run artifact/screenshot dirs, so concurrent runs never share a
        # checkpoint (a shared one silently resumes the other run's design).
        self.artifact_dir = os.environ.get("TOPOPT_ARTIFACT_DIR", "optimization/artifacts")
        self.history_dir = f"{self.artifact_dir}/topopt_history"
        self.screenshot_dir = os.environ.get("TOPOPT_SCREENSHOT_DIR", "optimization/screenshots")
        for d in (self.artifact_dir, self.history_dir, self.screenshot_dir):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # tile grid and spatial filter
    # ------------------------------------------------------------------
    def _tile_grid(self):
        try:
            fn = getattr(importlib.import_module(f"data.{self.geometry}.boundary_conditions"), "tile_grid", None)
        except ImportError:
            return None
        if fn is None:
            return None
        try:
            return fn()
        except TypeError:
            return fn(self.dims_dict)

    def _build_filter(self, r_min):
        """Cone weights W_ij = max(0, r_min - dist_ij) over the tile grid, or
        None when the dims are not a tile grid or r_min <= 1 (identity)."""
        grid = self._tile_grid()
        if grid is None or r_min <= 1.0:
            print(f"[TOPOPT] no spatial filter (grid={grid}, r_min={r_min})", flush=True)
            return None
        NX, NY, NZ = grid
        coords = []
        for name in self.free_names:      # dims.json order is X-outermost: k = (ix*NY + iy)*NZ + iz
            try:
                k = int(name[1:]) - 1
            except ValueError:
                return None
            if not 0 <= k < NX * NY * NZ:
                return None
            coords.append((k // (NY * NZ), (k // NZ) % NY, k % NZ))
        c = np.array(coords, dtype=float)
        dist = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        W = np.clip(r_min - dist, 0.0, None)
        print(f"[TOPOPT] {self.filter_target} filter: grid={NX}x{NY}x{NZ} r_min={r_min}", flush=True)
        return W, W.sum(axis=1)

    def _filter_sensitivity(self, s, x):
        """Sigmund (1997): s_e <- sum_i H_ei x_i s_i / (max(x_e, 1e-3) sum_i H_ei)."""
        if self.filt is None or self.filter_target != "gradient":
            return s
        W, row_sum = self.filt
        return (W @ (x * s)) / (np.maximum(x, 1e-3) * row_sum)

    def _filter_design(self, x):
        if self.filt is None or self.filter_target != "design":
            return x
        W, row_sum = self.filt
        return np.clip((W @ x) / row_sum, self.lo, self.hi)

    def _vec_to_dims(self, x):
        d = {n: float(x[i]) for i, n in enumerate(self.free_names)}
        d.update(self.fixed)
        return self.dims_transform(d) if self.dims_transform is not None else d

    # ------------------------------------------------------------------
    # OC step: bisection on lambda against the linearized constraint
    # ------------------------------------------------------------------
    def _oc_update(self, x, df, dg, g0, bisect_iters=60):
        move = self.move * self._move_scale
        df_safe = np.maximum(df, 1e-9)
        neg_dg = np.maximum(-dg, 1e-9)             # tiles with dG >= 0 only ever thin

        def x_new_for(lam):
            B = np.clip(lam * neg_dg / df_safe, 1e-12, 1e12)
            xn = np.clip(x * B ** self.oc_eta, x - move, x + move)
            return np.clip(xn, self.lo, self.hi)

        def g_lin(lam):
            return g0 + float(dg @ (x_new_for(lam) - x))

        # g_lin falls as lam grows (more material, less stress)
        tgt, l1, l2 = -self.g_backoff, 1e-9, 1e9
        if g_lin(l1) <= tgt:
            lam = l1                                # feasible even at minimum material
        elif g_lin(l2) > tgt:
            lam = l2                                # infeasible even at maximum material
        else:
            for _ in range(bisect_iters):
                lm = np.sqrt(l1 * l2)
                l1, l2 = (lm, l2) if g_lin(lm) > tgt else (l1, lm)
            lam = np.sqrt(l1 * l2)
        return x_new_for(lam), lam

    # ------------------------------------------------------------------
    # per-tile gradient heatmap (saved next to each screenshot)
    # ------------------------------------------------------------------
    def _save_gradient_heatmap(self, gen, dg, x, save_path):
        """dG/dx and thickness on the physical tile grid. dg arrives in dims-key
        order, elem_to_tile numbers tiles X-fastest: permute before reshaping
        (the transpose bug class this project has hit twice). The dG colour
        scale is this generation's 98th percentile -- a scale fixed at gen 0
        clipped most tiles once the design thinned and sensitivities grew 100x."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from core.tile_sensitivity import dims_key_order_permutation
        grid = self._tile_grid()
        if grid is None:
            return
        NX, NY, NZ = grid
        perm = np.asarray(dims_key_order_permutation(NX, NY, NZ))
        img = lambda v: np.asarray(v, dtype=float)[perm].reshape(NZ, NY, NX)[NZ // 2]
        lim = float(np.percentile(np.abs(dg), 98)) or float(np.abs(dg).max()) or 1.0
        fig, axes = plt.subplots(2, 1, figsize=(13, 6.4))
        im = axes[0].imshow(img(dg), cmap="seismic", origin="lower", aspect="auto", vmin=-lim, vmax=lim)
        axes[0].set_title(f"dG/dx used by OC  [+/-{lim:.3g} (98th pct this gen), "
                          f"{100 * float((np.abs(dg) > lim).mean()):.0f}% beyond; "
                          f"{100 * float((dg > 0).mean()):.0f}% positive = thins]", fontsize=9)
        fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.01)
        im = axes[1].imshow(img(x), cmap="viridis", origin="lower", aspect="auto",
                            vmin=float(self.lo.min()), vmax=float(self.hi.max()))
        axes[1].set_title(f"thickness  [this gen {x.min():.3f}..{x.max():.3f}]", fontsize=9)
        fig.colorbar(im, ax=axes[1], fraction=0.025, pad=0.01)
        for ax in axes:
            ax.set_xlabel("tile X", fontsize=8)
            ax.set_ylabel("tile Y", fontsize=8)
        fig.suptitle(f"topopt gen {gen:03d}  --  per-tile gradient field", fontsize=11)
        fig.tight_layout()
        fig.savefig(save_path, dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def _checkpoint_path(self):
        return f"{self.artifact_dir}/topopt_checkpoint.pkl"

    def _save_checkpoint(self, x, best_dims, best_vol, next_gen):
        with open(self._checkpoint_path(), "wb") as f:
            pickle.dump({"x": x, "gen": next_gen, "best_dims": best_dims, "best_vol": best_vol,
                         "move_scale": self._move_scale, "prev": self._prev,
                         "stagnant": self._stagnant, "screenshot_owed": self._screenshot_owed}, f)

    def _load_checkpoint(self):
        if not os.path.exists(self._checkpoint_path()):
            return None, 0, None, float("inf")
        with open(self._checkpoint_path(), "rb") as f:
            s = pickle.load(f)
        self._move_scale = s.get("move_scale", 1.0)
        self._prev = s.get("prev")
        self._stagnant = s.get("stagnant", 0)
        self._screenshot_owed = bool(s.get("screenshot_owed", False))
        print(f"[TOPOPT] Resuming from checkpoint gen {s['gen']} (move_scale={self._move_scale:.4f})", flush=True)
        return s["x"], s["gen"], s["best_dims"], s["best_vol"]

    def _write_params(self, start_gen):
        rec = {"start_gen": int(start_gen)}
        for k in ("generations", "stop_at_gen", "oc_eta", "filter_target", "filter_radius",
                  "tr_tol", "tr_min_scale", "g_backoff", "stagnation_gens", "init_thickness"):
            rec[k] = getattr(self, k)
        rec["move"] = self.move.tolist()
        rec["env"] = {k: v for k, v in os.environ.items()
                      if k.startswith(("TOPOPT_", "STRESS_KS", "FEA_", "CROSS4_"))}
        with open(f"{self.artifact_dir}/topopt_params.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------
    def run(self):
        x, start_gen, best_dims, best_vol = self._load_checkpoint()
        if x is None:
            x = (np.full(self.N, float(self.init_thickness)) if self.init_thickness is not None
                 else np.array([self.dims_dict[n]["default"] for n in self.free_names], dtype=float))
            x = np.clip(x, self.lo, self.hi)
            print(f"[TOPOPT] New run: free_dim={self.N} filter={self.filter_target} "
                  f"r={self.filter_radius} oc_eta={self.oc_eta}", flush=True)
        self._write_params(start_gen)
        end_gen = self.stop_at_gen if self.stop_at_gen is not None else start_gen + self.generations

        for gen in range(start_gen, end_gen):
            t_gen = perf_counter()
            dims = self._vec_to_dims(x)
            t = perf_counter()
            result = self.gradient_func(dims)
            t_grad = perf_counter() - t
            if result is None:
                print(f"[TOPOPT] Gen {gen + 1}: evaluation failed, skipping", flush=True)
                continue
            v0, s0, yld, dV, dG = result
            v0, s0 = float(v0), float(s0)
            g0 = s0 / yld - 1.0
            df_raw = np.array([dV[n] for n in self.free_names], dtype=float)
            dg_raw = np.array([dG[n] for n in self.free_names], dtype=float)
            df = self._filter_sensitivity(df_raw, x)
            dg = self._filter_sensitivity(dg_raw, x)

            # ---- adaptive move limit -----------------------------------------
            if self._prev is not None:
                over = g0 - self._prev["pred_g"]
                if over > self.tr_tol:
                    self._move_scale = max(self.tr_min_scale, 0.5 * self._move_scale)
                    # a step that made an already-violated constraint worse is
                    # rejected: retry from the previous design (its gradient is
                    # still exact there) with the smaller move
                    if g0 > 0.0 and g0 > self._prev["g0"] and self._move_scale > self.tr_min_scale:
                        p = self._prev
                        x = p["x"]
                        x_new, lam = self._oc_update(x, p["df"], p["dg"], p["g0"])
                        x_new = self._filter_design(x_new)
                        p["pred_g"] = p["g0"] + float(p["dg"] @ (x_new - x))
                        print(f"[TOPOPT] Gen {gen + 1} | REJECTED: g={g0:+.3e} vs predicted "
                              f"{g0 - over:+.3e} (over by {over:+.3e}); retrying from gen {gen} "
                              f"at move_scale={self._move_scale:.4f}", flush=True)
                        self._screenshot_owed |= gen % self.screenshot_every == 0
                        self._save_checkpoint(x_new, best_dims, best_vol, gen + 1)
                        x = x_new
                        continue
                elif over < 0.25 * self.tr_tol:
                    self._move_scale = min(1.0, 1.3 * self._move_scale)

            t = perf_counter()
            x_new, lam = self._oc_update(x, df, dg, g0)
            x_new = self._filter_design(x_new)
            self._prev = dict(x=x.copy(), df=df, dg=dg, g0=g0, pred_g=g0 + float(dg @ (x_new - x)))
            t_oc = perf_counter() - t

            feasible = s0 <= yld
            if feasible and v0 < best_vol:
                best_vol, best_dims, self._stagnant = v0, dims, 0
            else:
                self._stagnant += 1
            change = float(np.max(np.abs(x_new - x))) if self.N else 0.0
            best_s = f"{best_vol:.4e}" if np.isfinite(best_vol) else "none"
            print(f"[TOPOPT] Gen {gen + 1} | vol={v0:.4e} stress={s0:.4e} g={g0:+.3e} lambda={lam:.3e} "
                  f"max|dx|={change:.3e} mscale={self._move_scale:.4f} feasible={feasible} | "
                  f"best_feas_vol={best_s}", flush=True)
            np.savez(f"{self.history_dir}/gen_{gen:03d}.npz", free_names=np.array(self.free_names),
                     x=x, x_new=x_new, df_raw=df_raw, dg_raw=dg_raw, df_filt=df, dg_filt=dg,
                     v0=v0, s0=s0, g0=g0, lam=lam, change=change, feasible=feasible,
                     move_scale=self._move_scale)

            t_shot = 0.0
            if gen % self.screenshot_every == 0 or self._screenshot_owed:
                self._screenshot_owed = False
                t = perf_counter()
                path = f"{self.screenshot_dir}/topopt_gen_{gen:03d}"
                try:
                    screenshot(geometry=self.geometry, dims=dims, save_path=f"{path}.png",
                               banner=f"[TOPOPT gen {gen:03d}] vol: {v0:.4e}, stress: {s0:.4e}",
                               heatmap=self.screenshot_heatmap)
                except Exception as e:
                    print(f"[TOPOPT] screenshot failed: {e}", flush=True)
                try:
                    self._save_gradient_heatmap(gen, dg, x, f"{path}_grad.png")
                except Exception as e:
                    print(f"[TOPOPT] gradient heatmap failed: {e}", flush=True)
                t_shot = perf_counter() - t

            self._save_checkpoint(x_new, best_dims, best_vol, gen + 1)
            print(f"[TOPOPT] Gen {gen + 1} timing | total={perf_counter() - t_gen:.2f}s "
                  f"gradient={t_grad:.2f}s oc={t_oc:.3f}s screenshot={t_shot:.2f}s", flush=True)

            if self.stagnation_gens and self._stagnant >= self.stagnation_gens:
                print(f"[TOPOPT] Gen {gen + 1} | STOPPING: no improvement on best_feas_vol={best_s} "
                      f"in {self._stagnant} generations", flush=True)
                break
            if change < 1e-6:
                print("[TOPOPT] Stopping early: design stopped moving", flush=True)
                break
            x = x_new

        if best_dims is None:
            print("[TOPOPT] WARNING: no feasible design found; returning the current iterate", flush=True)
            return self._vec_to_dims(x)
        return best_dims
