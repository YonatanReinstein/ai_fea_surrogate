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
    lo <= x <= hi, with exact sensitivities from `gradient_func` (the fem
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
    "noisy" tiles to 10% of a step, rolling solver calibration, gradient
    magnitude calibration, surrogate-reload re-anchoring, the online-learning
    label spool, the FD sweep, a density/projection mode and three sharpening
    modes. With exact gradients those mostly froze tiles that had real signal.

    Optional spatial filtering over the tile grid (filter_target):
      "none"     -- none
      "gradient" -- Sigmund (1997) sensitivity filter on dV and dG
      "design"   -- smooth the design after each OC step (x <- H x), radius
                    filter_radius, optionally decaying linearly to
                    filter_radius_end over filter_decay_gens generations
                    starting at filter_decay_start; then optionally sharpen
                    (sharpen="skeleton": the grey-level skeleton flow of the
                    old optimizer, weight ramping 0 -> sharpen_w_max over
                    sharpen_start..sharpen_end).
      "density"  -- Bruns/Bourdin density filter: the optimizer's variables y
                    are never smoothed; the design is x = H y (same radius
                    schedule), recomputed from y every generation, and the
                    gradients are chained through it (dG/dy = H^T dG/dx).
                    Sharpening, when on, acts on y.
    "design" re-smooths an already smoothed design every generation, i.e. a
    diffusion: run cross4_mid_simple_filter sat 20-35% over yield for 20
    generations with lambda at its cap, because the thick spot the loaded tile
    needs was spread to its neighbours faster than OC could build it. "density"
    has no such accumulation.
    The smoothing and sharpening are part of the step that lambda is chosen
    for: the bisection evaluates the linearized constraint on the
    post-processed design. The old optimizer chose lambda on the raw OC step
    and smoothed afterwards, which erased exactly the isolated thick tile the
    stress needed (the loaded tile among thinner neighbours) while keeping the
    thinning elsewhere -- run cross4_mid_legacy_filter went 104 -> 458 MPa over
    20 generations with the loaded tile's OC ratio > 1 the whole time.
    """

    def __init__(self, dims_dict, gradient_func, generations=200,
                 geometry="hollow_cube", move_limit=0.2, oc_eta=0.5,
                 filter_target="none", filter_radius=1.5,
                 filter_radius_end=None, filter_decay_start=0, filter_decay_gens=0,
                 sharpen="none", sharpen_start=0, sharpen_end=0, sharpen_w_max=0.05,
                 skel_tau=0.03, skel_tau_end=None,
                 tr_tol=0.05, tr_min_scale=0.02, g_backoff=0.02,
                 adapt_move=True, tile_move_adapt=False, tile_move_shrink=0.7, tile_move_grow=1.2,
                 tile_move_min=0.05, update="oc", mma_asy_init=0.5, mma_asy_incr=1.2,
                 mma_asy_decr=0.7, mma_low_min=None,
                 stagnation_gens=0, init_thickness=None, stop_at_gen=None,
                 screenshot_heatmap=True, screenshot_every=1, reference_gradient=None,
                 screenshot_stress=None, screenshot_grad=True):
        if sharpen not in ("none", "skeleton"):
            raise ValueError(f"sharpen {sharpen!r}: use none | skeleton")
        if sharpen != "none" and filter_target not in ("design", "density"):
            raise ValueError("sharpening needs filter_target='design' or 'density'")
        if filter_target not in ("none", "gradient", "design", "density"):
            raise ValueError(f"filter_target {filter_target!r}: use none | gradient | design | density")
        self.gradient_func = gradient_func
        # Optional dims -> {dim: dG} from an exact solver (surrogate runs): drawn
        # next to the OC gradient in each screenshot's gradient heatmap.
        self.reference_gradient = reference_gradient
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
        self.filter_radius_end = filter_radius_end
        self.filter_decay_start = int(filter_decay_start)
        self.filter_decay_gens = int(filter_decay_gens)
        self.sharpen = sharpen
        self.sharpen_start, self.sharpen_end = int(sharpen_start), int(sharpen_end)
        self.sharpen_w_max = float(sharpen_w_max)
        self.skel_tau = float(skel_tau)
        self.skel_tau_end = None if skel_tau_end is None else float(skel_tau_end)
        self._filt_cache = {}
        self._skel_cache = None
        self.tr_tol = tr_tol
        # adapt_move=False: every step uses the full move limit -- no
        # move-scale halving/growth and no rejected generations.
        self.adapt_move = bool(adapt_move)
        # Per-tile step limits (MMA-style oscillation damping): a tile whose
        # step reverses sign between consecutive accepted steps has its own
        # limit shrunk, one that keeps its direction has it grown back toward
        # the global limit. Only oscillation is judged, never the gradient.
        # Run cross4_mid_density_r15 sat in a 2-generation cycle for 180
        # generations: one tile bounced by the full step while the global
        # move-scale check, seeing a steady 0.045 miss inside its dead band,
        # never reacted.
        self.tile_move_adapt = bool(tile_move_adapt)
        self.tile_move_shrink = float(tile_move_shrink)
        self.tile_move_grow = float(tile_move_grow)
        self.tile_move_min = float(tile_move_min)
        # update="mma": Svanberg's Method of Moving Asymptotes (1987) instead
        # of the OC fixed-point update; see _mma_update.
        if update not in ("oc", "mma"):
            raise ValueError(f"update {update!r}: use oc | mma")
        # MMA works on the optimizer's own variables y; with the density
        # filter the evaluated design is x = H y and df, dg are chained back
        # through H^T (_chain), so MMA's model is in y and needs nothing more.
        # The "design" filter re-smooths every step (_post), which MMA's
        # closed-form step cannot include -- not supported, nor sharpening.
        if update == "mma" and (filter_target == "design" or sharpen != "none"):
            raise ValueError("update='mma' supports filter_target none | gradient | density, no sharpening")
        self.update = update
        self.mma_asy_init = float(mma_asy_init)
        self.mma_asy_incr = float(mma_asy_incr)
        self.mma_asy_decr = float(mma_asy_decr)
        # mma_low_min: floor for every tile's LOWER asymptote (e.g. 0). Stress
        # in a leg blows up as t -> 0, so t = 0 is the physical asymptote;
        # standard MMA lets the asymptote drift far below it while a tile keeps
        # thinning (widths 0.09 -> 0.40 in cross4_row5_F5e6, lower asymptote
        # at about -0.1 for a 0.1 tile), which makes the model nearly linear.
        self.mma_low_min = None if mma_low_min is None else float(mma_low_min)
        self._mma = None            # MMA state: xold1, xold2, low, upp (accepted steps only)
        self.tr_min_scale = tr_min_scale
        self.g_backoff = g_backoff
        self.stagnation_gens = int(stagnation_gens)
        self.init_thickness = init_thickness
        # Each screenshot re-solves the design with fem_adjoint (heatmap) and usually
        # costs more than the gradient call; render every Nth generation.
        self.screenshot_heatmap = screenshot_heatmap
        self.screenshot_every = max(1, screenshot_every)
        # Optional (dims, node_ids, coords) -> nodal von Mises MPa: colours the
        # screenshot from the surrogate instead of a fem re-solve. screenshot_grad
        # =False skips the _grad.png panel (its solver column is an exact solve).
        self.screenshot_stress = screenshot_stress
        self.screenshot_grad = screenshot_grad

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
        self._tile_scale = np.ones(self.N)
        self._last_dx = None
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
        print(f"[TOPOPT] {self.filter_target} filter: grid={NX}x{NY}x{NZ} r_min={r_min:.3f}", flush=True)
        return W, W.sum(axis=1)

    def _filter_sensitivity(self, s, x):
        """Sigmund (1997): s_e <- sum_i H_ei x_i s_i / (max(x_e, 1e-3) sum_i H_ei)."""
        if self.filt is None or self.filter_target != "gradient":
            return s
        W, row_sum = self.filt
        return (W @ (x * s)) / (np.maximum(x, 1e-3) * row_sum)

    def _design_radius(self, gen):
        if self.filter_radius_end is None or self.filter_decay_gens <= 0:
            return self.filter_radius
        t = min(1.0, max(0.0, (gen - self.filter_decay_start) / float(self.filter_decay_gens)))
        return self.filter_radius + (self.filter_radius_end - self.filter_radius) * t

    def _sharpen_w(self, gen):
        if self.sharpen == "none" or gen <= self.sharpen_start:
            return 0.0
        if gen >= self.sharpen_end or self.sharpen_end <= self.sharpen_start:
            return self.sharpen_w_max
        return self.sharpen_w_max * (gen - self.sharpen_start) / float(self.sharpen_end - self.sharpen_start)

    def _H(self, gen):
        r = round(self._design_radius(gen), 3)
        if r not in self._filt_cache:
            self._filt_cache[r] = self._build_filter(r) if r > 1.0 else None
        return self._filt_cache[r]

    def _phys(self, y, gen):
        """Physical design: x = H y for the density filter, else y itself."""
        if self.filter_target != "density":
            return y
        H = self._H(gen)
        return y if H is None else np.clip((H[0] @ y) / H[1], self.lo, self.hi)

    def _chain(self, s, gen):
        """dF/dy = H^T dF/dx for x = (W y) / rowsum (W symmetric)."""
        if self.filter_target != "density":
            return s
        H = self._H(gen)
        return s if H is None else H[0] @ (s / H[1])

    def _post(self, x, gen):
        """Design post-processing applied to every OC trial step (inside the
        lambda bisection): smooth (scheduled radius; "design" only), then
        sharpen."""
        if self.filter_target not in ("design", "density"):
            return x
        H = self._H(gen) if self.filter_target == "design" else None
        if H is not None:
            x = np.clip((H[0] @ x) / H[1], self.lo, self.hi)
        w = self._sharpen_w(gen)
        if w > 0.0:
            t = 0.0 if self.sharpen_w_max <= 0 else w / self.sharpen_w_max
            tau = self.skel_tau if self.skel_tau_end is None else self.skel_tau + (self.skel_tau_end - self.skel_tau) * t
            x = self._skeleton_flow(x, w, tau)
        return x

    # Grey-level skeleton flow (from the old optimizer, commit 7bf07d9): score
    # each tile by the sum over thickness levels of its capped distance to the
    # edge of the level set {x >= level}; each tile above tau sends the
    # fraction w of its material above the floor, split equally, to face
    # neighbours with a strictly higher score. Bands narrow into thicker
    # centrelines; material is conserved except for ceiling clipping.
    SKEL_LEVELS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
    SKEL_DCAP = 3

    def _skeleton_grid(self):
        if self._skel_cache is None:
            NX, NY, NZ = self._tile_grid()
            coords = []
            for name in self.free_names:
                k = int(name[1:]) - 1
                coords.append((k // (NY * NZ), (k // NZ) % NY, k % NZ))
            grid = -np.ones((NX, NY, NZ), int)
            for i, c in enumerate(coords):
                grid[c] = i
            nbrs = []
            for ix, iy, iz in coords:
                nb = []
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    j = (ix + d[0], iy + d[1], iz + d[2])
                    if all(0 <= j[a] < (NX, NY, NZ)[a] for a in range(3)) and grid[j] >= 0:
                        nb.append(grid[j])
                nbrs.append(nb)
            self._skel_cache = ((NX, NY, NZ), coords, nbrs)
        return self._skel_cache

    def _skeleton_flow(self, x, w, tau):
        from scipy import ndimage
        (NX, NY, NZ), coords, nbrs = self._skeleton_grid()
        score = np.zeros(self.N)
        for lvl in self.SKEL_LEVELS:
            sup = np.zeros((NX, NY, NZ), bool)
            for i, c in enumerate(coords):
                sup[c] = x[i] >= lvl
            if not sup.any():
                continue
            # outside the grid counts as support: the part's boundary is not a band edge
            d = ndimage.distance_transform_cdt(np.pad(sup, 1, constant_values=True), metric="taxicab")
            d = np.minimum(d[1:-1, 1:-1, 1:-1], self.SKEL_DCAP)
            score += np.array([d[c] for c in coords], float)
        send, recv = np.zeros(self.N), np.zeros(self.N)
        for i in range(self.N):
            if x[i] <= tau:
                continue
            ups = [j for j in nbrs[i] if score[j] > score[i]]
            if ups:
                send[i] = w * (x[i] - self.lo[i])
                for j in ups:
                    recv[j] += send[i] / len(ups)
        return np.clip(x - send + recv, self.lo, self.hi)

    def _vec_to_dims(self, x):
        # Key ORDER matters, not just the keys: IritModel writes the dims as
        # an IRIT list in dict order and model_lin.c reads them by position.
        # Free-then-fixed order put every fixed tile's value at the END of the
        # list, i.e. on the wrong tiles (TOPOPT_FIX_LOAD_NBHD's 3x3 block came
        # out as the last 6 tiles d116..d121; TOPOPT_LOAD_TILE_MEAN's tied
        # d116 shifted d117..d121 by one).
        free = {n: float(x[i]) for i, n in enumerate(self.free_names)}
        d = {n: free[n] if n in free else self.fixed[n] for n in self.dim_names}
        return self.dims_transform(d) if self.dims_transform is not None else d

    # ------------------------------------------------------------------
    # OC step: bisection on lambda against the linearized constraint
    # ------------------------------------------------------------------
    def _pred_g(self, p, x_new, gen):
        """Linearized constraint after a step, in physical-design space (for
        "density": the step's x = H y_new against the evaluated x)."""
        return p["g0"] + float(p["dg_x"] @ (self._phys(x_new, gen + 1) - p["x_phys"]))

    def _oc_update(self, x, df, dg, g0, gen, dg_x=None, x_phys=None, bisect_iters=60):
        p = dict(g0=g0, dg_x=dg if dg_x is None else dg_x, x_phys=x if x_phys is None else x_phys)
        move = self.move * self._move_scale * self._tile_scale
        df_safe = np.maximum(df, 1e-9)
        neg_dg = np.maximum(-dg, 1e-9)             # tiles with dG >= 0 only ever thin

        def x_new_for(lam):
            B = np.clip(lam * neg_dg / df_safe, 1e-12, 1e12)
            xn = np.clip(x * B ** self.oc_eta, x - move, x + move)
            return self._post(np.clip(xn, self.lo, self.hi), gen)

        def g_lin(lam):
            return self._pred_g(p, x_new_for(lam), gen)

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
    # MMA step (Svanberg 1987, as in his mmasub; one constraint)
    # ------------------------------------------------------------------
    def _mma_update(self, x, df, dg, g0, state, bisect_iters=80):
        """One MMA step from x. Returns (x_new, lam, pred_g, new_state).

        Each function is approximated per tile by convex terms
            p / (U - x) + q / (x - L)
        around the asymptotes L < x < U, which every tile moves on its own:
        closer when its last two steps reversed direction (oscillation ->
        a more curved, more cautious model), further when they agreed. The
        constraint's approximation is exact at x (value g0, gradient dg), so
        pred_g is MMA's own prediction, curved, not the linearization.
        With one constraint the subproblem is solved exactly through its dual:
        for a multiplier lam each tile's minimizer is closed-form, and lam is
        bisected so the approximated constraint meets -g_backoff (the same
        target as OC). The step is also capped at the move limit, so the
        move-scale check and rejection work as with OC.
        """
        span = self.span
        if state is None or state.get("xold2") is None:
            # first two steps: asymptotes at +-asy_init of the bound span
            low = x - self.mma_asy_init * span
            upp = x + self.mma_asy_init * span
        else:
            z = (x - state["xold1"]) * (state["xold1"] - state["xold2"])
            gam = np.where(z < 0, self.mma_asy_decr, np.where(z > 0, self.mma_asy_incr, 1.0))
            low = x - gam * (state["xold1"] - state["low"])
            upp = x + gam * (state["upp"] - state["xold1"])
            low = np.clip(low, x - 10 * span, x - 0.01 * span)
            upp = np.clip(upp, x + 0.01 * span, x + 10 * span)
        if self.mma_low_min is not None:
            low = np.minimum(np.maximum(low, self.mma_low_min), x - 1e-6)
        move = self.move * self._move_scale * self._tile_scale
        alpha = np.maximum.reduce([self.lo, low + 0.1 * (x - low), x - move])
        beta = np.minimum.reduce([self.hi, upp - 0.1 * (upp - x), x + move])

        ux2, xl2 = (upp - x) ** 2, (x - low) ** 2
        eps = 1e-5 / np.maximum(span, 1e-12)
        def pq(d):
            return (ux2 * (1.001 * np.maximum(d, 0) + 0.001 * np.maximum(-d, 0) + eps),
                    xl2 * (0.001 * np.maximum(d, 0) + 1.001 * np.maximum(-d, 0) + eps))
        p0, q0 = pq(df)
        p1, q1 = pq(dg)
        r1 = g0 - float(np.sum(p1 / (upp - x) + q1 / (x - low)))

        def x_of(lam):
            P, Q = np.sqrt(p0 + lam * p1), np.sqrt(q0 + lam * q1)
            return np.clip((P * low + Q * upp) / (P + Q), alpha, beta)

        def g_of(xn):
            return r1 + float(np.sum(p1 / (upp - xn) + q1 / (xn - low)))

        tgt, l1, l2 = -self.g_backoff, 0.0, 1e9
        if g_of(x_of(l1)) <= tgt:
            lam = l1
        elif g_of(x_of(l2)) > tgt:
            lam = l2
        else:
            l1 = 1e-12
            for _ in range(bisect_iters):
                lm = np.sqrt(l1 * l2)
                l1, l2 = (lm, l2) if g_of(x_of(lm)) > tgt else (l1, lm)
            lam = np.sqrt(l1 * l2)
        x_new = x_of(lam)
        new_state = dict(xold1=x.copy(), xold2=None if state is None else state["xold1"],
                         low=low, upp=upp)
        return x_new, lam, g_of(x_new), new_state

    # ------------------------------------------------------------------
    # per-tile gradient heatmap (saved next to each screenshot)
    # ------------------------------------------------------------------
    def _save_gradient_heatmap(self, gen, dg, x, save_path, dims=None):
        """dG/dx and thickness on the physical tile grid. dg arrives in dims-key
        order, elem_to_tile numbers tiles X-fastest: permute before reshaping
        (the transpose bug class this project has hit twice). The dG colour
        scale is this generation's 98th percentile -- a scale fixed at gen 0
        clipped most tiles once the design thinned and sensitivities grew 100x.
        With a reference_gradient (surrogate runs) the solver's dG at the same
        design is drawn under it, on its own 98th-percentile scale."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from core.tile_sensitivity import dims_key_order_permutation
        grid = self._tile_grid()
        if grid is None:
            return
        NX, NY, NZ = grid
        perm = np.asarray(dims_key_order_permutation(NX, NY, NZ))
        # dg and x cover the FREE tiles only: scatter them onto every dims key
        # (fixed tiles show their fixed thickness and a zero gradient)
        slot = {n: i for i, n in enumerate(self.dim_names)}
        idx = [slot[n] for n in self.free_names]
        dg_full = np.zeros(len(self.dim_names))
        x_full = np.array([self.fixed.get(n, 0.0) for n in self.dim_names])
        dg_full[idx], x_full[idx] = np.asarray(dg, dtype=float), np.asarray(x, dtype=float)
        dg, x = dg_full, x_full
        ref = None
        if self.reference_gradient is not None and dims is not None:
            try:
                g = self.reference_gradient(dims)
                ref = np.zeros(len(self.dim_names))
                ref[idx] = [float(g[n]) for n in self.free_names]
            except Exception as e:
                print(f"[TOPOPT] reference gradient failed: {e}", flush=True)
        img = lambda v: np.asarray(v, dtype=float)[perm].reshape(NZ, NY, NX)[NZ // 2]
        fields = [("surrogate dG/dx used by OC" if ref is not None else "dG/dx used by OC", dg)]
        if ref is not None:
            fields.append(("solver (fem adjoint) dG/dx", ref))
        fig, axes = plt.subplots(len(fields) + 1, 1, figsize=(13, 3.2 * (len(fields) + 1)))
        for ax, (name, v) in zip(axes, fields):
            lim = float(np.percentile(np.abs(v), 98)) or float(np.abs(v).max()) or 1.0
            im = ax.imshow(img(v), cmap="seismic", origin="lower", aspect="auto", vmin=-lim, vmax=lim)
            ax.set_title(f"{name}  [+/-{lim:.3g} (98th pct this gen), "
                         f"{100 * float((np.abs(v) > lim).mean()):.0f}% beyond; "
                         f"{100 * float((v > 0).mean()):.0f}% positive = thins]", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        im = axes[-1].imshow(img(x), cmap="viridis", origin="lower", aspect="auto",
                             vmin=float(self.lo.min()), vmax=float(self.hi.max()))
        axes[-1].set_title(f"thickness  [this gen {x.min():.3f}..{x.max():.3f}]", fontsize=9)
        fig.colorbar(im, ax=axes[-1], fraction=0.025, pad=0.01)
        for ax in axes:
            ax.set_xlabel("tile X", fontsize=8)
            ax.set_ylabel("tile Y", fontsize=8)
        sup = f"topopt gen {gen:03d}  --  per-tile gradient field"
        if ref is not None:
            a, b = dg[idx], ref[idx]
            sup += (f"   |   surrogate vs solver: cosine {a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30):+.3f}, "
                    f"sign agreement {np.mean(np.sign(a) == np.sign(b)):.2f}, "
                    f"norm ratio {np.linalg.norm(a) / (np.linalg.norm(b) + 1e-30):.2f}")
        fig.suptitle(sup, fontsize=11)
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
                         "tile_scale": self._tile_scale, "last_dx": self._last_dx,
                         "stagnant": self._stagnant, "screenshot_owed": self._screenshot_owed,
                         "mma": self._mma}, f)

    def _load_checkpoint(self):
        if not os.path.exists(self._checkpoint_path()):
            return None, 0, None, float("inf")
        with open(self._checkpoint_path(), "rb") as f:
            s = pickle.load(f)
        self._move_scale = s.get("move_scale", 1.0)
        self._prev = s.get("prev")
        self._tile_scale = s.get("tile_scale", np.ones(self.N))
        self._last_dx = s.get("last_dx")
        self._stagnant = s.get("stagnant", 0)
        self._screenshot_owed = bool(s.get("screenshot_owed", False))
        self._mma = s.get("mma")
        print(f"[TOPOPT] Resuming from checkpoint gen {s['gen']} (move_scale={self._move_scale:.4f})", flush=True)
        return s["x"], s["gen"], s["best_dims"], s["best_vol"]

    def _write_params(self, start_gen):
        rec = {"start_gen": int(start_gen)}
        for k in ("generations", "stop_at_gen", "oc_eta", "filter_target", "filter_radius",
                  "filter_radius_end", "filter_decay_start", "filter_decay_gens", "sharpen",
                  "sharpen_start", "sharpen_end", "sharpen_w_max", "skel_tau", "skel_tau_end",
                  "tr_tol", "tr_min_scale", "g_backoff", "stagnation_gens", "init_thickness",
                  "adapt_move", "tile_move_adapt", "tile_move_shrink", "tile_move_grow", "tile_move_min",
                  "update", "mma_asy_init", "mma_asy_incr", "mma_asy_decr", "mma_low_min"):
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
            print(f"[TOPOPT] New run: free_dim={self.N} update={self.update} filter={self.filter_target} "
                  f"r={self.filter_radius} oc_eta={self.oc_eta}", flush=True)
        else:
            # bounds may have changed since the checkpoint (TOPOPT_T_MIN)
            n_out = int(np.sum((x < self.lo) | (x > self.hi)))
            if n_out:
                print(f"[TOPOPT] {n_out} tile(s) outside the current bounds, clipped back", flush=True)
            x = np.clip(x, self.lo, self.hi)
        self._write_params(start_gen)
        end_gen = self.stop_at_gen if self.stop_at_gen is not None else start_gen + self.generations

        for gen in range(start_gen, end_gen):
            t_gen = perf_counter()
            x_phys = self._phys(x, gen)         # == x unless filter_target == "density"
            dims = self._vec_to_dims(x_phys)
            t = perf_counter()
            # TOPOPT_PENALTY_SCHEDULE="ts0,ts1,g0,g1": stiffness-penalty
            # continuation -- t_s goes linearly from ts0 (before gen g0) to
            # ts1 (from gen g1), set for this generation's evaluation; the
            # stress relaxation width follows at TOPOPT_RELAX_RATIO * t_s
            # (default 0.25). Raising t_s makes tiles below ~t_s too soft to
            # be worth their volume, so they either thicken into real members
            # or empty to the floor (no thin tiles carrying load). Small
            # per-generation changes keep the move-limit check from reading
            # the schedule as a model miss.
            sched = os.environ.get("TOPOPT_PENALTY_SCHEDULE")
            if sched:
                ts0, ts1, g0, g1 = (float(v) for v in sched.split(","))
                f = min(1.0, max(0.0, (gen - g0) / max(g1 - g0, 1.0)))
                ts = ts0 + (ts1 - ts0) * f
                os.environ["TOPOPT_STIFF_PENALTY"] = f"{ts:.6g}"
                os.environ["TOPOPT_STRESS_RELAX"] = f"{ts * float(os.environ.get('TOPOPT_RELAX_RATIO') or 0.25):.6g}"
                print(f"[TOPOPT] Gen {gen + 1} penalty t_s={ts:.4f} relax t_w={float(os.environ['TOPOPT_STRESS_RELAX']):.4f}",
                      flush=True)
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
            df = self._chain(self._filter_sensitivity(df_raw, x), gen)
            dg = self._chain(self._filter_sensitivity(dg_raw, x), gen)

            # ---- adaptive move limit -----------------------------------------
            if self._prev is not None and self.adapt_move:
                over = g0 - self._prev["pred_g"]
                if over > self.tr_tol:
                    self._move_scale = max(self.tr_min_scale, 0.5 * self._move_scale)
                    # a step that made an already-violated constraint worse is
                    # rejected: retry from the previous design (its gradient is
                    # still exact there) with the smaller move
                    if g0 > 0.0 and g0 > self._prev["g0"] and self._move_scale > self.tr_min_scale:
                        p = self._prev
                        x = p["x"]
                        if self.update == "mma":
                            # retry from the asymptotes the rejected step started from
                            x_new, lam, p["pred_g"], self._mma = self._mma_update(
                                x, p["df"], p["dg"], p["g0"], p["mma_before"])
                        else:
                            x_new, lam = self._oc_update(x, p["df"], p["dg"], p["g0"], gen,
                                                         p.get("dg_x"), p.get("x_phys"))
                            p["pred_g"] = self._pred_g(dict(g0=p["g0"], dg_x=p.get("dg_x", p["dg"]),
                                                            x_phys=p.get("x_phys", x)), x_new, gen)
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
            dg_x = dg_raw if self.filter_target == "density" else dg
            mma_before = self._mma
            if self.update == "mma":
                x_new, lam, mma_pred, self._mma = self._mma_update(x, df, dg, g0, mma_before)
            else:
                x_new, lam = self._oc_update(x, df, dg, g0, gen, dg_x, x_phys)
            if self.tile_move_adapt:
                dx = x_new - x
                if self._last_dx is not None:
                    flip = dx * self._last_dx < 0.0
                    keep = dx * self._last_dx > 0.0
                    self._tile_scale[flip] = np.maximum(self.tile_move_min,
                                                        self._tile_scale[flip] * self.tile_move_shrink)
                    self._tile_scale[keep] = np.minimum(1.0, self._tile_scale[keep] * self.tile_move_grow)
                self._last_dx = dx
            self._prev = dict(x=x.copy(), df=df, dg=dg, g0=g0, dg_x=dg_x, x_phys=x_phys.copy(),
                              mma_before=mma_before,
                              pred_g=mma_pred if self.update == "mma" else
                              self._pred_g(dict(g0=g0, dg_x=dg_x, x_phys=x_phys), x_new, gen))
            t_oc = perf_counter() - t

            feasible = s0 <= yld
            if feasible and v0 < best_vol:
                best_vol, best_dims, self._stagnant = v0, dims, 0
            else:
                self._stagnant += 1
            change = float(np.max(np.abs(self._phys(x_new, gen + 1) - x_phys))) if self.N else 0.0
            best_s = f"{best_vol:.4e}" if np.isfinite(best_vol) else "none"
            print(f"[TOPOPT] Gen {gen + 1} | vol={v0:.4e} stress={s0:.4e} g={g0:+.3e} lambda={lam:.3e} "
                  f"max|dx|={change:.3e} mscale={self._move_scale:.4f} "
                  + (f"tile_scale min/med={self._tile_scale.min():.2f}/{np.median(self._tile_scale):.2f} "
                     if self.tile_move_adapt else "")
                  + (f"asy_width med={np.median(self._mma['upp'] - self._mma['low']):.3f} "
                     if self.update == "mma" and self._mma is not None else "")
                  + f"feasible={feasible} | "
                  f"best_feas_vol={best_s}", flush=True)
            np.savez(f"{self.history_dir}/gen_{gen:03d}.npz", free_names=np.array(self.free_names),
                     x=x_phys, x_new=self._phys(x_new, gen + 1), y=x, y_new=x_new, df_raw=df_raw, dg_raw=dg_raw, df_filt=df, dg_filt=dg,
                     v0=v0, s0=s0, g0=g0, lam=lam, change=change, feasible=feasible,
                     move_scale=self._move_scale, tile_scale=self._tile_scale)

            t_shot = 0.0
            if gen % self.screenshot_every == 0 or self._screenshot_owed:
                self._screenshot_owed = False
                t = perf_counter()
                path = f"{self.screenshot_dir}/topopt_gen_{gen:03d}"
                try:
                    screenshot(geometry=self.geometry, dims=dims, save_path=f"{path}.png",
                               banner=f"[TOPOPT gen {gen:03d}] vol: {v0:.4e}, stress: {s0:.4e}"
                                      + (" (surrogate stress field)" if self.screenshot_stress else ""),
                               heatmap=self.screenshot_heatmap, stress_fn=self.screenshot_stress)
                except Exception as e:
                    print(f"[TOPOPT] screenshot failed: {e}", flush=True)
                try:
                    if self.screenshot_grad:
                        self._save_gradient_heatmap(gen, dg_x, x_phys, f"{path}_grad.png", dims)
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
            return self._vec_to_dims(self._phys(x, end_gen))
        return best_dims
