import os
import pickle
import numpy as np

from .screenshot import screenshot


class GradientOptimizer:
    """Finite-difference gradient-descent optimizer over CAD dimensions.

    Minimizes a *scale-normalized* objective with a smooth, ramping penalty:

        objective = volume / V_ref + w_gen * max(stress / yield - 1, 0)**2

    where V_ref is the volume of the starting (default) design, so the volume
    term is O(1), and the penalty term is dimensionless. The penalty weight
    ramps geometrically with the generation index,

        w_gen = penalty_w0 * penalty_growth ** gen

    so early generations explore loosely and later ones are pushed hard onto
    the feasible region (stress <= yield). The penalty is *quadratic* rather
    than the linear max(.,0): it is C1-smooth, so a finite-difference gradient
    stays well-behaved right at the constraint boundary where the optimum sits.

    Each generation estimates the gradient by perturbing every free dim, one
    at a time, by a single step (forward finite difference), then takes one
    normalized-gradient-descent step. With N free dims this costs N + 1
    evaluations per generation, submitted in a single batched call.

    Search variables are the *free* dims (min < max), normalized to [0, 1].
    Fixed dims (min == max) are held at their default. The descent step is
    taken along the *unit* gradient, so `step_size` is a displacement in
    normalized [0, 1] coordinates, independent of the objective's scale.
    """

    def __init__(self, fitness_func, dims_dict,
                 pop_size=None, generations=200,
                 seed=0, geometry="bistable",
                 fd_step=0.02, step_size=0.05, step_decay=1.0,
                 penalty_w0=1.0, penalty_growth=1.1, fail_penalty=1e12):
        self.fitness_func = fitness_func
        self.dims_dict = dims_dict
        self.geometry = geometry
        self.generations = generations
        self.seed = seed                       # unused; kept for interface parity
        self.fd_step = fd_step                 # finite-difference step (normalized)
        self.step_size = step_size             # descent step length (normalized)
        self.step_decay = step_decay           # multiplied into step_size each gen
        self.penalty_w0 = penalty_w0           # penalty weight at generation 0
        self.penalty_growth = penalty_growth   # geometric ramp factor per generation
        self.fail_penalty = fail_penalty

        self.dim_names = list(dims_dict.keys())
        self.free_names = [n for n in self.dim_names
                           if dims_dict[n]["max"] > dims_dict[n]["min"]]
        self.fixed = {n: float(dims_dict[n]["default"])
                      for n in self.dim_names if n not in self.free_names}

        self.lo = np.array([dims_dict[n]["min"] for n in self.free_names], dtype=float)
        self.hi = np.array([dims_dict[n]["max"] for n in self.free_names], dtype=float)
        self.span = self.hi - self.lo

        # Volume normalizer; set lazily to the first finite volume seen
        # (the starting/default design on generation 0).
        self.V_ref = None

        self.artifact_dir = "optimization/artifacts"
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs("optimization/screenshots", exist_ok=True)

    # ------------------------------------------------------------------
    # normalized [0, 1] <-> physical dims
    # ------------------------------------------------------------------
    def _denorm(self, z):
        return self.lo + np.clip(np.asarray(z, dtype=float), 0.0, 1.0) * self.span

    def _norm(self, x):
        return (np.asarray(x, dtype=float) - self.lo) / self.span

    def _vec_to_dims(self, z):
        x = self._denorm(z)
        d = {n: float(x[i]) for i, n in enumerate(self.free_names)}
        d.update(self.fixed)
        return d

    def _objective(self, volumes, stresses, yield_strength, weight):
        objs = []
        for v, s in zip(volumes, stresses):
            if v is None or s is None:
                objs.append(self.fail_penalty)
                continue
            violation = max(float(s) / yield_strength - 1.0, 0.0)
            objs.append(float(v) / self.V_ref + weight * violation ** 2)
        return np.array(objs, dtype=float)

    # ------------------------------------------------------------------
    # checkpointing — pickle the iterate + running best each generation
    # ------------------------------------------------------------------
    def _checkpoint_path(self):
        return f"{self.artifact_dir}/grad_checkpoint.pkl"

    def _load_checkpoint(self):
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return None, 0, None, float("inf")
        with open(path, "rb") as f:
            state = pickle.load(f)
        gen = state["gen"]
        self.V_ref = state["V_ref"]
        print(f"[GRAD] Resuming from checkpoint gen {gen} "
              f"(V_ref={self.V_ref:.4e})", flush=True)
        return state["z"], gen, state["best_dims"], state["best_obj"]

    # ------------------------------------------------------------------
    def run(self):
        z, start_gen, best_dims, best_obj = self._load_checkpoint()
        if z is None:
            z = np.clip(self._norm([self.dims_dict[n]["default"]
                                    for n in self.free_names]), 0.0, 1.0)
            start_gen = 0
            best_dims, best_obj = None, float("inf")
            print(f"[GRAD] New run: free_dim={len(self.free_names)} "
                  f"fd_step={self.fd_step} step_size={self.step_size} "
                  f"penalty_w0={self.penalty_w0} growth={self.penalty_growth}",
                  flush=True)

        N = len(self.free_names)

        for gen in range(start_gen, start_gen + self.generations):
            weight = self.penalty_w0 * (self.penalty_growth ** gen)

            # Forward differences; flip to a backward step near the upper
            # bound so every perturbed point stays inside [0, 1].
            deltas = np.where(z + self.fd_step <= 1.0, self.fd_step, -self.fd_step)

            # Base point first, then one perturbed point per free dim.
            pts = [z.copy()]
            for i in range(N):
                zp = z.copy()
                zp[i] = z[i] + deltas[i]
                pts.append(zp)

            dims_dicts = [self._vec_to_dims(p) for p in pts]
            res = self.fitness_func(dims_dicts)
            yld = res["yield_strength"]

            # Lock the volume normalizer to the starting design's volume.
            if self.V_ref is None:
                finite = [float(v) for v in res["volume"] if v is not None]
                self.V_ref = finite[0] if finite else 1.0
                print(f"[GRAD] volume reference V_ref={self.V_ref:.4e}", flush=True)

            objs = self._objective(res["volume"], res["stress"], yld, weight)

            base_obj = objs[0]
            grad = (objs[1:] - base_obj) / deltas      # finite-difference gradient
            gnorm = float(np.linalg.norm(grad))

            step = self.step_size * (self.step_decay ** gen)
            if gnorm > 1e-12:
                z_new = np.clip(z - step * grad / gnorm, 0.0, 1.0)
            else:
                z_new = z

            # Running best across every point evaluated this generation.
            gen_best_idx = int(np.argmin(objs))
            if objs[gen_best_idx] < best_obj:
                best_obj = float(objs[gen_best_idx])
                best_dims = dims_dicts[gen_best_idx]

            vol = np.array([np.nan if v is None else float(v)
                            for v in res["volume"]], dtype=float)
            strs = np.array([np.nan if s is None else float(s)
                             for s in res["stress"]], dtype=float)
            n_feasible = int(np.sum(strs <= yld))      # NaN <= yld is False

            print(
                f"[GRAD] Gen {gen + 1} | base_obj={base_obj:.4e} "
                f"vol={vol[0]:.4e} stress={strs[0]:.4e} w={weight:.3e} "
                f"|grad|={gnorm:.4e} step={step:.3e} "
                f"feasible={n_feasible}/{len(objs)} | global_best={best_obj:.4e}",
                flush=True,
            )

            banner = (
                f"[GRAD gen {gen:03d}] vol: {vol[0]:.4e}, "
                f"stress: {strs[0]:.4e} => obj: {base_obj:.4e}"
            )
            try:
                screenshot(
                    geometry=self.geometry,
                    dims=dims_dicts[0],
                    save_path=f"optimization/screenshots/grad_gen_{gen:03d}.png",
                    banner=banner,
                )
            except Exception as e:
                print(f"[GRAD] screenshot failed: {e}", flush=True)

            with open(self._checkpoint_path(), "wb") as f:
                pickle.dump({"z": z_new, "best_dims": best_dims,
                             "best_obj": best_obj, "V_ref": self.V_ref,
                             "gen": gen + 1}, f)

            if gnorm <= 1e-12 or np.allclose(z_new, z):
                print("[GRAD] Stopping early: gradient vanished / no movement",
                      flush=True)
                break
            z = z_new

        return best_dims
