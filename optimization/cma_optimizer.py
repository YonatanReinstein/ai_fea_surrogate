import os
import pickle
import numpy as np
import cma
from cma.constraints_handler import AugmentedLagrangian

from .screenshot import screenshot


class CMAOptimizer:


    def __init__(self, fitness_func, dims_dict,
                 pop_size=None, generations=200,
                 seed=0, geometry="bistable",
                 sigma0=None, penalty_weight=10.0, fail_penalty=1.0,
                 constraint_mode="auglag", search_space="logit",
                 reset_on_stagnation=False, max_unproductive_resets=2):
        self.fitness_func = fitness_func
        self.dims_dict = dims_dict
        self.geometry = geometry
        self.generations = generations
        self.seed = seed
        self.penalty_weight = penalty_weight
        # fail_penalty: objective assigned to samples where MAPDL failed.
        # Must be worse than any feasible solution (v/V_ref <= 1) but not
        # so large that it drowns out the constraint penalty signal.
        # For quadratic mode with penalty_weight=100, a 2× yield overshoot
        # (g=1) gives penalty=100, so we set fail_penalty = 1 + penalty_weight.
        if fail_penalty == 1.0 and constraint_mode == "quadratic":
            fail_penalty = 1.0 + penalty_weight
        self.fail_penalty = fail_penalty
        if constraint_mode not in ("linear", "auglag", "quadratic"):
            raise ValueError(f"constraint_mode must be 'linear', 'auglag', or 'quadratic', got {constraint_mode!r}")
        self.constraint_mode = constraint_mode
        # search_space:
        #   "logit":   unbounded R^N search; each dim mapped to (lo, hi) via a
        #              sigmoid. Sigma is in logit units (~2.0 covers the range).
        #   "box":     normalized [0, 1]^N search; dim = lo + p * span linearly.
        #              cma.BoundPenalty enforces the box (no transform inside the
        #              box -> no near-edge stiffness). Sigma is in [0, 1] units
        #              (~0.25 covers most of the range).
        #   "reflect": unbounded R^N folded into [lo, hi] by a triangle wave of
        #              period 2 (reflecting boundaries). Like "box" it is
        #              locally linear (no interior compression) and every sample
        #              lands in [lo, hi], but it needs no bounds/penalty: CMA can
        #              wander past a bound and just bounces back in. Sigma is in
        #              the same normalized units as box (~0.25).
        # Bounded search composes cleanly with the augmented Lagrangian: the
        # AL only sees the stress constraint, the box handler only sees dim
        # bounds, so the two are independent.
        if search_space not in ("logit", "box", "reflect"):
            raise ValueError(
                f"search_space must be 'logit', 'box', or 'reflect', got {search_space!r}")
        self.search_space = search_space
        if sigma0 is None:
            sigma0 = 2.0 if search_space == "logit" else 0.25
        self.sigma0 = sigma0
        # Augmented Lagrangian state (built lazily once search-space dim is known).
        # Replaces the static linear penalty with adaptive multipliers (lam, mu)
        # so the search can settle smoothly *on* the stress constraint boundary
        # instead of bouncing across the kink of a linear penalty.
        self.al = None

        # Volume normalizer; set lazily to the first finite volume seen
        # (a near-default design from generation 0).
        self.V_ref = None

        self.dim_names = list(dims_dict.keys())
        self.free_names = [n for n in self.dim_names
                           if dims_dict[n]["max"] > dims_dict[n]["min"]]
        self.fixed = {n: float(dims_dict[n]["default"])
                      for n in self.dim_names if n not in self.free_names}

        self.lo = np.array([dims_dict[n]["min"] for n in self.free_names], dtype=float)
        self.hi = np.array([dims_dict[n]["max"] for n in self.free_names], dtype=float)
        self.span = self.hi - self.lo

        # cma's default popsize (4 + 3*ln(N)) is used when pop_size is None.
        self.pop_size = pop_size

        # Stagnation handling. Two tiers:
        #   - "soft" stagnation: cma flags tolstagnation/tolflatfitness.
        #     Covariance + sigma are reset (mean kept) and the run continues.
        #     This re-isotropizes exploration after CMA has collapsed one
        #     direction (the d8-wall collapse pattern), without abandoning
        #     the position found so far. AL multipliers stay -- still
        #     calibrated to the same constraint.
        #   - "hard" stagnation: max_unproductive_resets consecutive resets
        #     produced no improvement in best feasible volume -> stop.
        #   - real convergence (tolfun/tolfunhist/tolx) -> stop regardless,
        #     since resetting can't unstick a true minimum.
        # reset_on_stagnation=False reverts to the original "stop on any
        # cma stop reason" behavior.
        self.reset_on_stagnation = reset_on_stagnation
        if max_unproductive_resets < 1:
            raise ValueError(
                f"max_unproductive_resets must be >= 1, got {max_unproductive_resets!r}")
        self.max_unproductive_resets = max_unproductive_resets

        self.artifact_dir = "optimization/artifacts"
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs("optimization/screenshots", exist_ok=True)

    # ------------------------------------------------------------------
    # unbounded search space  <->  physical dims, via a logistic transform
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(u):
        # numerically stable logistic: no overflow for large-magnitude u
        u = np.asarray(u, dtype=float)
        return np.where(u >= 0,
                        1.0 / (1.0 + np.exp(-u)),
                        np.exp(u) / (1.0 + np.exp(u)))

    def _denorm(self, u):
        # search vector -> physical dims.
        if self.search_space == "logit":
            return self.lo + self._sigmoid(u) * self.span
        if self.search_space == "reflect":
            # Triangle wave, period 2: u=0 -> lo, u=1 -> hi, u=2 -> lo, ...
            # 1 - |((u mod 2) - 1)| maps any real u into [0, 1] continuously,
            # reflecting at each bound (slope +-1, so no interior compression).
            u = np.asarray(u, dtype=float)
            p = 1.0 - np.abs(np.mod(u, 2.0) - 1.0)
            return self.lo + p * self.span
        # box: u is in [0, 1]; cma.BoundPenalty mostly keeps it there but can
        # let small violations through during sampling -- clip defensively.
        p = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        return self.lo + p * self.span

    def _norm(self, x):
        # physical dims -> search vector (inverse of _denorm).
        p = (np.asarray(x, dtype=float) - self.lo) / self.span
        if self.search_space == "box":
            return np.clip(p, 0.0, 1.0)
        if self.search_space == "reflect":
            # Principal rising branch of the triangle wave: u = p in [0, 1]
            # maps straight back to x through _denorm.
            return np.clip(p, 0.0, 1.0)
        # logit: keep finite if x sits on a bound
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    def _vec_to_dims(self, z):
        x = self._denorm(z)
        d = {n: float(x[i]) for i, n in enumerate(self.free_names)}
        d.update(self.fixed)
        return d

    def _objective(self, volumes, stresses, yield_strength):
        objs = []
        for v, s in zip(volumes, stresses):
            if v is None or s is None:
                objs.append(self.fail_penalty)
                continue
            violation = max(float(s) / yield_strength - 1.0, 0.0)
            if self.constraint_mode == "quadratic":
                objs.append(float(v) / self.V_ref + self.penalty_weight * violation ** 2)
            else:
                objs.append(float(v) / self.V_ref + self.penalty_weight * violation)
        return np.array(objs, dtype=float)

    def _build_fg(self, volumes, stresses, yield_strength):
        # F: per-sample raw objective (dimensionless volume).
        # G: per-sample constraint vector. Convention: g_i <= 0 is feasible.
        # Failed samples (None) are treated as severely infeasible so AL still
        # gets clean numeric inputs and the sample is penalized.
        F, G, failed = [], [], []
        for v, s in zip(volumes, stresses):
            if v is None or s is None:
                F.append(self.fail_penalty)
                G.append([1.0])
                failed.append(True)
            else:
                F.append(float(v) / self.V_ref)
                G.append([float(s) / yield_strength - 1.0])
                failed.append(False)
        return F, G, failed

    def _auglag_objective(self, volumes, stresses, yield_strength,
                          mean_vol, mean_stress):
        F, G, failed = self._build_fg(volumes, stresses, yield_strength)

        # set_coefficients is a no-op once lam/mu are set; on early gens it
        # waits until the population straddles the constraint, then initializes.
        self.al.set_coefficients(F, G)

        # AL.update expects f and g at the *distribution mean* (Atamna 2017).
        # If the mean point failed in MAPDL we just skip the update this gen.
        if mean_vol is not None and mean_stress is not None:
            f_mean = float(mean_vol) / self.V_ref
            g_mean = [float(mean_stress) / yield_strength - 1.0]
            self.al.update(f_mean, g_mean)

        objs = []
        for f, g, fl in zip(F, G, failed):
            if fl:
                objs.append(self.fail_penalty)
            else:
                objs.append(f + sum(self.al(g)))
        return np.array(objs, dtype=float)

    # ------------------------------------------------------------------
    # checkpointing — pickle the whole CMAEvolutionStrategy each generation
    # ------------------------------------------------------------------
    def _checkpoint_path(self):
        return f"{self.artifact_dir}/cma_checkpoint.pkl"

    def _load_checkpoint(self):
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return None, 0
        with open(path, "rb") as f:
            data = pickle.load(f)
        gen = data["gen"]
        es = data["es"]
        self.V_ref = data.get("V_ref")
        self.al = data.get("al")  # may be None for old checkpoints / linear mode
        self._best_feas_vol = data.get("best_feas_vol", float("inf"))
        self._best_feas_dims = data.get("best_feas_dims")
        self._reset_count = data.get("reset_count", 0)
        self._unproductive_resets = data.get("unproductive_resets", 0)
        self._best_feas_vol_at_last_reset = data.get(
            "best_feas_vol_at_last_reset", float("inf"))
        # The CMAEvolutionStrategy state (mean, sigma, C) is expressed in the
        # search-space coordinates active when it was saved. Resuming after
        # switching search_space would silently corrupt the run.
        saved_space = data.get("search_space", "logit")
        if saved_space != self.search_space:
            raise RuntimeError(
                f"Checkpoint was written with search_space={saved_space!r} "
                f"but current run uses {self.search_space!r}. Delete "
                f"{path} to start a fresh run."
            )
        print(f"[CMA] Resuming from checkpoint gen {gen} "
              f"(V_ref={self.V_ref}, al={'set' if self.al else 'none'}, "
              f"search_space={self.search_space})", flush=True)
        return es, gen

    def _new_es(self, x0=None, seed_offset=0):
        # x0: optional initial mean in search-space coordinates. Used by the
        # stagnation-reset path to spawn a fresh CMA at the current mean
        # rather than at the default. seed_offset lets each reset use a
        # different RNG stream so we don't replay the same trajectory.
        if x0 is None:
            x0 = self._norm([self.dims_dict[n]["default"]
                             for n in self.free_names])
        opts = {
            "seed": self.seed + seed_offset,
            "verbose": -9,
            # cma's default tolerances (~1e-11) are unreachable for a volume
            # objective of order ~1, so the run never detects convergence and
            # never stops. These are scaled to this problem.
            "tolfun": 1e-4,
            "tolfunhist": 1e-5,
            "tolx": 1e-4,
            "tolstagnation": 100,
            "tolflatfitness": 20,
        }
        if self.pop_size is not None:
            opts["popsize"] = self.pop_size
        if self.search_space == "box":
            # Box search in normalized [0, 1]^N. BoundPenalty keeps the
            # interior strictly linear (unlike BoundTransform, which would
            # re-introduce a sigmoid-like stiffness near the bounds): samples
            # are drawn from an unbounded Gaussian, evaluated at the clipped
            # design, and only the *out-of-bounds excursion* is penalized.
            # At N=1, cma's BoundPenalty (and even setting bounds alone) triggers
            # an uninitialized sigma_vec bug in _stds_into_limits. For N=1 skip
            # both — _denorm's defensive clip already keeps samples in [lo, hi].
            n = len(self.free_names)
            if n > 1:
                opts["bounds"] = [[0.0] * n, [1.0] * n]
                opts["BoundaryHandler"] = cma.BoundPenalty
        # In "logit"/"reflect" mode no bounds are set: _denorm already keeps
        # every physical sample inside [lo, hi] (sigmoid / triangle-wave fold).
        es = cma.CMAEvolutionStrategy(list(x0), self.sigma0, opts)
        print(f"[CMA] New run: free_dim={len(self.free_names)} "
              f"popsize={es.popsize} sigma0={self.sigma0} "
              f"search_space={self.search_space}", flush=True)
        return es

    # ------------------------------------------------------------------
    def run(self):
        # Default the best-feasible state to "nothing seen yet" -- the
        # checkpoint loader may overwrite these from a saved run.
        self._best_feas_vol = float("inf")
        self._best_feas_dims = None
        # Counter for stagnation resets (also varies the RNG seed each time).
        self._reset_count = 0
        # Hard-stagnation tracking: how many consecutive resets produced no
        # improvement, and the best-feas-vol baseline at the previous reset.
        self._unproductive_resets = 0
        self._best_feas_vol_at_last_reset = float("inf")

        es, start_gen = self._load_checkpoint()
        if es is None:
            es = self._new_es()
            start_gen = 0

        if self.constraint_mode == "auglag" and self.al is None:
            self.al = AugmentedLagrangian(es.N)
            # Newer cma docs recommend this over the dimension-derived default.
            self.al.chi_domega = 2.0
            print(f"[CMA] AugmentedLagrangian initialized (dim={es.N})",
                  flush=True)

        for gen in range(start_gen, start_gen + self.generations):
            solutions = es.ask()
            dims_dicts = [self._vec_to_dims(z) for z in solutions]

            if self.constraint_mode == "auglag":
                mean_dims = self._vec_to_dims(es.mean)
                eval_dims = dims_dicts + [mean_dims]
            else:
                eval_dims = dims_dicts

            res = self.fitness_func(eval_dims)
            yld = res["yield_strength"]

            if self.constraint_mode == "auglag":
                pop_vols = list(res["volume"][:-1])
                pop_stress = list(res["stress"][:-1])
                mean_vol = res["volume"][-1]
                mean_stress = res["stress"][-1]
            else:
                pop_vols = list(res["volume"])
                pop_stress = list(res["stress"])
                mean_vol = None
                mean_stress = None

            # Lock the volume normalizer to the first finite volume seen.
            if self.V_ref is None:
                all_vols = pop_vols + ([mean_vol] if mean_vol is not None else [])
                finite = [float(v) for v in all_vols if v is not None]
                self.V_ref = finite[0] if finite else 1.0
                print(f"[CMA] volume reference V_ref={self.V_ref:.4e}", flush=True)

            if self.constraint_mode == "auglag":
                objs = self._auglag_objective(pop_vols, pop_stress, yld,
                                              mean_vol, mean_stress)
            else:  # linear or quadratic
                objs = self._objective(pop_vols, pop_stress, yld)

            es.tell(solutions, objs.tolist())

            # Use pop_vols / pop_stress (mean point excluded) so indices line
            # up with `solutions`, `dims_dicts`, and `objs`.
            vol = np.array([np.nan if v is None else float(v)
                            for v in pop_vols], dtype=float)
            strs = np.array([np.nan if s is None else float(s)
                             for s in pop_stress], dtype=float)

            gen_best_idx = int(np.argmin(objs))

            # Global-best tracking by *raw* feasible volume, not by the
            # AL-penalized objective. The AL penalty for a very-feasible
            # sample can dip to -lam^2/(2*mu), so a transient lam spike makes
            # the penalized objective fake-negative and pollutes any history
            # that records "smallest objective ever seen." Instead, take the
            # lightest sample whose stress is at or below yield.
            feas_mask = strs <= yld  # NaN <= yld is False, so failed samples excluded
            if np.any(feas_mask):
                feas_vols = np.where(feas_mask, vol, np.inf)
                feas_best_idx = int(np.argmin(feas_vols))
                feas_best_vol = float(feas_vols[feas_best_idx])
                if feas_best_vol < self._best_feas_vol:
                    self._best_feas_vol = feas_best_vol
                    self._best_feas_dims = dims_dicts[feas_best_idx]

            n_feasible = int(np.sum(feas_mask))
            al_tag = ""
            if self.constraint_mode == "auglag" and self.al is not None:
                lam = getattr(self.al, "lam", None)
                mu = getattr(self.al, "mu", None)
                lam_s = f"{float(lam[0]):.2e}" if lam is not None and len(lam) else "unset"
                mu_s = f"{float(mu[0]):.2e}" if mu is not None and len(mu) else "unset"
                al_tag = f" lam={lam_s} mu={mu_s}"
            best_feas_s = (f"{self._best_feas_vol:.4e}"
                           if np.isfinite(self._best_feas_vol) else "none")
            print(
                f"[CMA] Gen {gen + 1} | best_obj={objs[gen_best_idx]:.4e} "
                f"vol={vol[gen_best_idx]:.4e} stress={strs[gen_best_idx]:.4e} "
                f"feasible={n_feasible}/{len(objs)} sigma={es.sigma:.3e}{al_tag} "
                f"| best_feas_vol={best_feas_s}",
                flush=True,
            )

            banner = (
                f"[CMA gen {gen:03d}] vol: {vol[gen_best_idx]:.4e}, "
                f"stress: {strs[gen_best_idx]:.4e} => obj: {objs[gen_best_idx]:.4e}"
            )
            try:
                screenshot(
                    geometry=self.geometry,
                    dims=dims_dicts[gen_best_idx],
                    save_path=f"optimization/screenshots/cma_gen_{gen:03d}.png",
                    banner=banner,
                )
            except Exception as e:
                print(f"[CMA] screenshot failed: {e}", flush=True)

            with open(self._checkpoint_path(), "wb") as f:
                pickle.dump({"es": es, "gen": gen + 1, "V_ref": self.V_ref,
                             "al": self.al,
                             "search_space": self.search_space,
                             "best_feas_vol": self._best_feas_vol,
                             "best_feas_dims": self._best_feas_dims,
                             "reset_count": self._reset_count,
                             "unproductive_resets": self._unproductive_resets,
                             "best_feas_vol_at_last_reset":
                                 self._best_feas_vol_at_last_reset}, f)

            if es.stop():
                stop_reason = es.stop()

                # If stagnation handling is disabled, preserve the original
                # behavior: any stop reason ends the run.
                if not self.reset_on_stagnation:
                    print(f"[CMA] Stopping early: {stop_reason}", flush=True)
                    break

                # Classify: stagnation-flavored stops vs real-convergence
                # stops. Convergence stops mean CMA has actually flattened
                # the objective / shrunk the step below tolerance, so a
                # covariance reset can't unstick anything -- exit.
                stagnation_keys = {"tolstagnation", "tolflatfitness"}
                is_soft_stagnation = bool(
                    stagnation_keys.intersection(stop_reason))
                if not is_soft_stagnation:
                    print(f"[CMA] Stopping early (real convergence): "
                          f"{stop_reason}", flush=True)
                    break

                # Hard stagnation = N consecutive resets that produced no
                # improvement in best feasible volume. Compares current
                # tracked best to the baseline saved at the prior reset.
                if self._best_feas_vol < self._best_feas_vol_at_last_reset:
                    self._unproductive_resets = 0
                else:
                    self._unproductive_resets += 1
                self._best_feas_vol_at_last_reset = self._best_feas_vol

                if self._unproductive_resets >= self.max_unproductive_resets:
                    print(f"[CMA] Hard stagnation: "
                          f"{self._unproductive_resets} consecutive resets "
                          f"without improvement (last={stop_reason}). "
                          f"Stopping.", flush=True)
                    break

                # Soft stagnation reset: keep the current mean, throw away
                # the collapsed covariance and shrunken sigma, vary the RNG
                # seed so the next batch isn't an identical replay. AL
                # multipliers stay -- they still describe the same
                # constraint correctly.
                self._reset_count += 1
                current_mean = np.array(es.mean, dtype=float)
                print(f"[CMA] Soft stagnation reset #{self._reset_count} "
                      f"(unproductive={self._unproductive_resets}/"
                      f"{self.max_unproductive_resets}): "
                      f"stop={stop_reason} keeping mean, sigma "
                      f"{es.sigma:.3e} -> {self.sigma0}",
                      flush=True)
                es = self._new_es(x0=current_mean,
                                  seed_offset=self._reset_count)

        if self._best_feas_dims is None:
            print("[CMA] WARNING: no feasible sample was ever found; "
                  "returning current distribution mean as a fallback.",
                  flush=True)
            return self._vec_to_dims(es.mean)
        return self._best_feas_dims
