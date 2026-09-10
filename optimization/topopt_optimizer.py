import importlib
import os
import pickle
from time import perf_counter
import numpy as np

from .screenshot import screenshot


class TopologyOptimizer:
    """SIMP-style optimality-criteria (OC) topology optimizer over per-tile
    dims (e.g. hollow_cube's 840 per-tile FrameThickness values on its
    NX*NY*NZ grid).

    Problem: minimize volume V(x) subject to stress(x) <= yield, lo <= x <= hi
    -- the same problem GeneticAlgorithm/CMAOptimizer/GradientOptimizer solve,
    but here x plays the role of a SIMP pseudo-density and is updated with
    the classic OC fixed point instead of a population or plain descent step.

    Classic OC (Bendsoe & Sigmund) minimizes a cheap/explicit quantity
    (volume) subject to an expensive/nonlinear one (compliance), with the
    per-element compliance sensitivity (which varies a lot between
    load-critical and irrelevant elements) in the numerator and the
    ~uniform element-volume sensitivity in the denominator. Our problem has
    objective and constraint swapped -- volume is cheap/explicit (and its
    per-tile sensitivity ~uniform), stress is the expensive nonlinear one
    (and its per-tile sensitivity is what varies a lot, strongly negative
    near the load path, ~zero far from it) -- so the *varying* sensitivity
    now belongs to the constraint, and the fixed point is used with objective
    and constraint's roles exchanged accordingly:

        B_e(lambda) = lambda * (-dG/dx_e) / (dV/dx_e)
        x_new_e     = clip(x_e * B_e(lambda) ** eta, x_e - move, x_e + move, [lo_e, hi_e])

    where G = stress / yield - 1 <= 0. The bisection on lambda (log-scale,
    since it spans many decades) targets a first-order (finite difference)
    linearization of G around the current iterate, driving it to zero --
    cheap, since it only touches precomputed sensitivities, not the
    evaluator. One outer iteration still costs N + 1 evaluator calls (a
    forward-difference sweep over every free dim, batched in one call), same
    as GradientOptimizer.

    A Sigmund (1997) sensitivity filter is applied over the tile grid
    (`data.{geometry}.boundary_conditions.tile_grid()`) before the OC step,
    to suppress checkerboarding / mesh-scale noise in the finite-difference
    sensitivities and keep neighboring tiles coupled. If the geometry has no
    tile_grid() (e.g. a handful of unrelated CAD dims, not a spatial grid),
    filtering is skipped.
    """

    def __init__(self, fitness_func, dims_dict,
                 tr_tol=0.05, tr_min_scale=0.02, g_backoff=0.02,
                 calib_every=5, calib_smooth=0.5, stagnation_gens=0,
                 pop_size=None, generations=200,
                 seed=0, geometry="hollow_cube",
                 fd_step=0.02, move_limit=0.2, filter_radius=1.5,
                 oc_eta=0.5, trust_rel_tol=0.5, min_trust_frac=0.05,
                 init_thickness=None,
                 untrusted_move_frac=0.1,
                 fail_penalty=1e12, screenshot_heatmap=True, screenshot_every=1,
                 gradient_func=None):
        # gradient_func(dims_dict) -> (volume, stress, yield_strength,
        # dV_dx_dict, dG_dx_dict), e.g. GNNEvaluator.compute_topopt_gradient
        # -- when set, run() uses it instead of fitness_func's N+1-call
        # finite-difference sweep for the per-generation sensitivities
        # (exact w.r.t. the surrogate, via its own autodiff, one call
        # instead of N+1). fitness_func is still used for screenshots/best-
        # design bookkeeping elsewhere and must remain compatible with it.
        self.fitness_func = fitness_func
        self.gradient_func = gradient_func
        self.dims_dict = dims_dict
        self.geometry = geometry
        self.generations = generations
        self.seed = seed               # unused; kept for interface parity
        self.pop_size = pop_size       # unused; kept for interface parity
        self.fd_step = fd_step
        self.filter_radius = filter_radius
        self.oc_eta = oc_eta
        self.init_thickness = init_thickness
        self.trust_rel_tol = trust_rel_tol
        self.min_trust_frac = min_trust_frac
        self.untrusted_move_frac = untrusted_move_frac
        # Screenshots re-solve the design independently of the fitness_func
        # evaluator (same pattern as CMAOptimizer/GradientOptimizer), so this
        # runs one extra real CCX solve per generation to color the mesh by
        # nodal von Mises stress. Requires `ccx` on PATH; set False if the
        # evaluator isn't ccx-backed (e.g. GNN surrogate runs), or the
        # already-try/excepted screenshot call will just fail every generation.
        self.screenshot_heatmap = screenshot_heatmap
        # Each screenshot re-solves the design independently (real CCX solve
        # for screenshot_heatmap=True, regardless of --arch -- see above),
        # measured at ~14s/gen vs ~3s for the gradient_func fitness call
        # itself, i.e. it dominates per-generation wall time. Gate it to
        # every screenshot_every-th generation (gen 0 always included) when
        # only the OC trajectory/final design matters, not a smooth video.
        self.screenshot_every = max(1, screenshot_every)
        self.fail_penalty = fail_penalty

        self.dim_names = list(dims_dict.keys())
        self.free_names = [n for n in self.dim_names
                           if dims_dict[n]["max"] > dims_dict[n]["min"]]
        self.fixed = {n: float(dims_dict[n]["default"])
                      for n in self.dim_names if n not in self.free_names}

        self.lo = np.array([dims_dict[n]["min"] for n in self.free_names], dtype=float)
        self.hi = np.array([dims_dict[n]["max"] for n in self.free_names], dtype=float)
        self.span = self.hi - self.lo
        # move_limit is a fraction of each dim's span, converted to an
        # absolute per-tile step (classic SIMP move=0.2 in [0, 1] space).
        self.move = move_limit * self.span
        # Trust region on the CONSTRAINT LINEARIZATION (see run()). _oc_update
        # always predicts a step landing exactly on g = 0; tr_tol is how much
        # worse than that prediction the measured g may come out before the
        # step size is cut. _move_scale multiplies self.move and lives in
        # [tr_min_scale, 1].
        self.tr_tol = tr_tol
        self.tr_min_scale = tr_min_scale
        self._move_scale = 1.0
        self._prev = None
        # Constraint back-off: _oc_update aims at g_lin = -g_backoff. See there.
        self.g_backoff = g_backoff
        self.stagnation_gens = int(stagnation_gens)
        self._stagnant = 0
        # Rolling calibration of the gradient_func's stress against the real
        # solver. See run(). None until the first calibration solve lands.
        self.calib_every = calib_every
        self.calib_smooth = calib_smooth
        self._calib = None

        self.N = len(self.free_names)
        self.filt = self._build_filter(filter_radius)

        self.V_ref = None

        # Fixed path by default, but overridable so two topopt runs can go at
        # once without fighting over topopt_checkpoint.pkl -- whoever writes
        # last wins otherwise, and the loser silently RESUMES the winner's
        # design on its next start (that trap cost a run today: a stale
        # checkpoint at gen 308 was picked up as gen 1 of a fresh job).
        self.artifact_dir = os.environ.get("TOPOPT_ARTIFACT_DIR", "optimization/artifacts")
        # Per-generation dump of the full design vector + both raw and
        # filtered sensitivities (optimization/artifacts/topopt_history/),
        # for debugging trajectory behavior -- the checkpoint only ever
        # holds the *current* iterate, so without this there's no way to
        # inspect e.g. per-tile sensitivity signs/magnitudes from a past
        # generation after the fact.
        self.history_dir = f"{self.artifact_dir}/topopt_history"
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs("optimization/screenshots", exist_ok=True)

    # ------------------------------------------------------------------
    # spatial sensitivity filter over the tile grid
    # ------------------------------------------------------------------
    def _build_filter(self, r_min):
        if r_min <= 0:
            return None
        try:
            module = importlib.import_module(f"data.{self.geometry}.boundary_conditions")
            tile_grid_fn = getattr(module, "tile_grid", None)
        except ImportError:
            tile_grid_fn = None
        if tile_grid_fn is None:
            print(f"[TOPOPT] {self.geometry!r} has no tile_grid(); "
                  f"running without a spatial sensitivity filter", flush=True)
            return None
        try:
            NX, NY, NZ = tile_grid_fn()
        except TypeError:
            NX, NY, NZ = tile_grid_fn(self.dims_dict)

        coords = []
        for name in self.free_names:
            try:
                k = int(name[1:]) - 1
            except ValueError:
                coords = None
                break
            if not (0 <= k < NX * NY * NZ):
                coords = None
                break
            iz = k % NZ
            rem = k // NZ
            iy = rem % NY
            ix = rem // NY
            coords.append((ix, iy, iz))

        if coords is None:
            print(f"[TOPOPT] free dims don't map onto {self.geometry!r}'s "
                  f"{NX}x{NY}x{NZ} tile grid; running without a spatial "
                  f"sensitivity filter", flush=True)
            return None

        coords = np.array(coords, dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        W = np.clip(r_min - dist, 0.0, None)
        row_sum = W.sum(axis=1)
        row_sum[row_sum == 0] = 1.0
        print(f"[TOPOPT] spatial filter built: grid={NX}x{NY}x{NZ} "
              f"r_min={r_min} free_dim={self.N}", flush=True)
        return W, row_sum

    def _filter_sensitivity(self, s, x):
        # Sigmund (1997) sensitivity filter:
        #   s_tilde_e = 1/max(x_e,eps) * sum_i(H_ei x_i s_i) / sum_i(H_ei)
        if self.filt is None:
            return s
        W, row_sum = self.filt
        weighted = W @ (x * s)
        denom = np.maximum(x, 1e-3) * row_sum
        return weighted / denom

    # ------------------------------------------------------------------
    # dims dict <-> free-dim vector
    # ------------------------------------------------------------------
    def _vec_to_dims(self, x):
        d = {n: float(x[i]) for i, n in enumerate(self.free_names)}
        d.update(self.fixed)
        return d

    # ------------------------------------------------------------------
    # OC fixed-point update with a bisection on the linearized constraint
    # ------------------------------------------------------------------
    def _oc_update(self, x, df, dg, g0, bisect_iters=60):
        # df = dV/dx_e (objective sensitivity, expected > 0: more material,
        #      more volume). dg = dG/dx_e (constraint sensitivity, expected
        #      < 0: more material, less stress).
        #
        # KKT stationarity at an interior optimum: df_e + lambda*dg_e = 0,
        # i.e. -df_e/(lambda*dg_e) = 1. That ratio and its reciprocal
        # -lambda*dg_e/df_e both equal 1 at the exact optimum, but the OC
        # *iteration* x_new = x * B_e**eta is not invariant to which
        # orientation is used away from it: whichever sensitivity sits in
        # the numerator is what differentiates elements from each other.
        # Classic SIMP puts dc_e (compliance sensitivity, varies a lot
        # between load-critical and irrelevant elements) over dv_e (element
        # volume, ~constant) -- the varying term goes on top. Here dg_e is
        # the varying one (strongly negative for tiles that matter to the
        # global peak stress, near zero for tiles that don't) and df_e is
        # the ~uniform one (volume cost is roughly the same per tile) -- the
        # mirror image of the classic setup, since objective and constraint
        # are swapped. The correct orientation is therefore
        #
        #   B_e(lambda) = lambda * (-dg_e) / df_e
        #
        # not lambda's reciprocal form. Getting this backwards (df_e over
        # dg_e) made *small*-|dg_e| tiles -- the low-stress, structurally
        # irrelevant ones -- get the *largest* B_e, so they grew fastest,
        # while high-sensitivity tiles near the stress peak got the
        # smallest B_e and shrank: exactly inverted from the intended
        # SIMP behavior.
        #
        # Tiles with dg_e >= 0 (no measured stress benefit from more
        # material) bypass B_e entirely: removing material there is only a
        # "pure" volume win while the design is currently feasible (g0 <=
        # 0), since dg_e * (-move) <= 0 can't push a feasible g0 positive.
        # While infeasible (g0 > 0), a max-stress constraint typically
        # leaves *most* tiles "untrusted" (only the handful near the
        # current stress peak show a nonzero sensitivity to the global
        # max), so unconditionally shrinking all of them every generation
        # overwhelms whatever growth lambda drives into the trusted set --
        # observed as volume collapsing for ~10 generations after going
        # infeasible instead of recovering (topopt_optimizer 840-tile
        # hollow_cube run, gens 6-13: vol 442->107 while g0 climbed
        # 0.16->4.6). Freezing untrusted tiles (no forced move either way)
        # while infeasible lets the trusted-tile growth actually move the
        # aggregate back toward the boundary instead of being outvoted.
        #
        # A bare `dg < -eps` cutoff (eps ~ 1e-9) takes that "only a
        # handful" premise on faith: measured on a real 840-tile hollow_cube
        # run (gen 59, post sensitivity-filter dg), 56% of tiles came out
        # "trusted" with 17.5% of face-adjacent tile pairs disagreeing on
        # trusted/untrusted -- dg's per-tile FD/mesh noise (std ~8x its
        # mean there) swamps eps, so the sign is close to a coin flip for
        # most tiles rather than a real stress-benefit signal. Since
        # trusted vs. untrusted picks between two very different moves
        # (OC growth vs. freeze/shrink), that coin flip lands neighboring
        # tiles on opposite moves generation after generation -- visible in
        # screenshots as checkerboarding (isolated max-thickness tiles next
        # to min-thickness ones) and in the log as a limit cycle (max|dx|
        # pinned at the move limit for dozens of generations without
        # reaching feasible). Scaling the cutoff to this generation's own
        # dg spread (trust_rel_tol * std(dg)) makes "trusted" require a
        # sensitivity clearly above that noise floor, matching the "only a
        # handful" tiles the comment above already assumes.
        eps = 1e-9
        threshold = max(eps, self.trust_rel_tol * float(np.std(dg)))
        trusted = dg < -threshold

        # The threshold above can legitimately shrink the trusted set to
        # zero -- observed on a real 840-tile hollow_cube run (fd_step=0.04,
        # gens 100-114): as the design got thin/porous, std(dg) grew ~20x
        # and its mean flipped sign, so no tile cleared -threshold at gen
        # 114. With zero trusted tiles, x_new_for() has nothing left to move
        # (untrusted tiles freeze whenever g0 > 0, see above), lambda's
        # bisection pins uselessly at l2, and the run deadlocks permanently
        # while still infeasible -- max|dx| goes to exactly 0 and stays
        # there for the rest of the run. A floor on the trusted count (the
        # min_trust_frac most-negative-dg tiles, taken unconditionally) keeps
        # the design able to add material even when the noise floor swamps
        # the relative threshold, so an infeasible run always has some path
        # back toward feasibility.
        min_trusted = max(1, int(round(self.min_trust_frac * len(dg))))
        if int(trusted.sum()) < min_trusted:
            floor_idx = np.argsort(dg)[:min_trusted]
            trusted = np.zeros_like(trusted)
            trusted[floor_idx] = True

        df_safe = np.maximum(df, eps)
        neg_dg_safe = np.maximum(-dg, eps)   # = -dg_e, floored away from 0

        # Earlier versions gave untrusted tiles a binary target (freeze at x
        # while infeasible, snap to x - move the instant g0 <= 0 / -prune_margin
        # while feasible). That caused two real failure modes on the same
        # 840-tile hollow_cube run: full freeze -> permanent deadlock (job
        # 45778, trusted set collapsed to 0, max|dx| pinned at exactly 0 for
        # 15+ gens) and, after flooring the trusted count, a full-design snap
        # on every noise-level feasible touch (job 47985, gen 140->141: g0 =
        # -6.997e-05 triggered a one-shot prune of 723/840 tiles, volume
        # 403.0 -> 357.5, straight back to deeply infeasible) -- a limit
        # cycle that never beat gen 94's best_feas_vol across 55 generations.
        # Gating the snap on a feasibility margin (prune_margin) stopped the
        # snap but produced a *third* failure mode (job 47989, gens 94-189):
        # once the design got thin/porous, only the ~5% floor ever cleared
        # the trust threshold, so untrusted tiles just sat frozen instead --
        # 800/840 tiles stopped moving entirely for 90+ generations while the
        # floor alone crept the design to a worse local point (394.4 -> 401).
        #
        # Both binary branches assumed "untrusted" means "no real signal, so
        # don't trust it with a full move" -- true, but on the actual gen-188
        # data 762/840 tiles have dg < 0 (a real, if weak, stress-reducing
        # direction), only 78 have dg >= 0; freezing all 762 throws away
        # signal that's directionally correct, just noisy in *magnitude*.
        # Giving every tile the same OC step but capping *untrusted* tiles to
        # a much smaller move (untrusted_move_frac * self.move, e.g. 10%)
        # keeps the low-confidence tiles responsive without letting FD/mesh
        # noise move them far. Replayed against gens 160-188 of job 47989:
        # 751-802 tiles move (vs 38-42 frozen-only), net volume direction is
        # consistent with feasibility (positive dvol while genuinely
        # infeasible, negative once near the boundary -- unlike the ~0 net
        # change the frozen version was stuck at), and face-adjacent tiles
        # disagree on grow/shrink direction 2-13% of the time, well under
        # the ~17.5% seen in the original (much larger-amplitude) checkerboard
        # incident this trust mechanism was built to prevent. This also
        # removes the g0-conditional branch entirely -- every tile responds
        # to the same lambda every generation, just with a smaller leash.
        move = self.move * self._move_scale
        step_cap = np.where(trusted, move, self.untrusted_move_frac * move)

        def x_new_for(lam):
            B = lam * neg_dg_safe / df_safe
            B = np.clip(B, 1e-12, 1e12)
            xn = x * (B ** self.oc_eta)
            xn = np.clip(xn, x - step_cap, x + step_cap)
            return np.clip(xn, self.lo, self.hi)

        def g_lin_for(lam):
            xn = x_new_for(lam)
            return g0 + float(np.dot(dg, xn - x))

        # lambda = l1 (~0) pushes every trusted x_e to its lower move limit
        # (min material); lambda = l2 (large) pushes every trusted x_e to
        # its upper move limit (max material) -- B_e now scales *with*
        # lambda instead of inversely, so this orientation is the mirror
        # image of the pre-fix version. g_lin is therefore DECREASING in
        # lambda (more material -> less stress -> smaller g): l1 is the
        # least-feasible extreme, l2 the most-feasible (largest-volume) one.
        # Aim at g_lin = -g_backoff, not 0. Stress is convex in thinning, so a
        # linear model systematically UNDER-predicts the rise: aiming exactly
        # at the boundary lands just outside it nearly every generation. Job
        # 51500 rode g in +0.008..+0.054 for 140 straight generations and
        # never once tripped `feasible`, so best_feas_vol stayed frozen at
        # generation 3's 623.9 and run() would have returned that design
        # instead of the vol~270 one it actually found. The back-off is the
        # standard constraint margin that makes the iterate land on or just
        # inside the boundary; it costs a little volume and buys designs that
        # are actually admissible.
        tgt = -self.g_backoff
        l1, l2 = 1e-9, 1e9
        g_l1, g_l2 = g_lin_for(l1), g_lin_for(l2)
        if g_l1 > tgt and g_l2 > tgt:
            lam = l2           # infeasible even at max material added; do the best we can
        elif g_l1 <= tgt and g_l2 <= tgt:
            lam = l1           # feasible even at min material; shrink volume maximally
        else:
            for _ in range(bisect_iters):
                lmid = np.sqrt(l1 * l2)   # geometric bisection: lambda spans many decades
                if g_lin_for(lmid) > tgt:
                    l1 = lmid              # infeasible -> need more material -> larger lambda
                else:
                    l2 = lmid              # feasible -> can afford less material -> smaller lambda
            lam = np.sqrt(l1 * l2)
        return x_new_for(lam), lam

    def _save_tile_gradient_heatmap(self, gen, dg_raw, dg_filt, x, save_path, dims=None):
        """Per-tile dG/dx laid out on the physical tile grid, saved next to each
        screenshot so the gradient field driving OC is visible, not just the
        geometry it produced.

        Layout: elem_to_tile numbers tiles X-fastest / Z-slowest (see
        core.tile_sensitivity.tile_idx_to_dims_idx), while dg arrives in
        dims-key order -- indexing one by the other silently transposes the
        map whenever NX != NY, which is exactly the bug class that has cost
        this project two separate hunts. Permute first, then reshape
        (NZ, NY, NX) so rows are Y and columns are X.

        Diverging colormap centred on zero: the SIGN is what OC consumes
        (dG<0 -> thicken, dG>0 -> thin), so red/blue must split at 0 and the
        limits must be symmetric. Scaled by a robust percentile because the
        field spans ~6 decades and a few tiles would otherwise flatten it.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from core.tile_sensitivity import dims_key_order_permutation

        # Resolve the grid the same way _build_filter does -- there is no
        # self.tile_grid attribute; it is a local in that method.
        _mod = importlib.import_module(f"data.{self.geometry}.boundary_conditions")
        _fn = getattr(_mod, "tile_grid")
        try:
            NX, NY, NZ = _fn()
        except TypeError:
            NX, NY, NZ = _fn(self.dims_dict)
        perm = np.asarray(dims_key_order_permutation(NX, NY, NZ))
        def grid(v):
            return np.asarray(v, dtype=float)[perm].reshape(NZ, NY, NX)[NZ // 2]

        # Ground-truth comparison. TOPOPT_CCX_REF=1 runs core/fem_adjoint.py's
        # exact per-tile adjoint on THIS design and adds a difference panel.
        # This is the only measurement that tests the surrogate gradient in the
        # regime that matters: every held-out probe so far scored it on
        # dataset-distribution designs, never on a partially-optimised one, and
        # the topopt result stopped tracking probe quality (epoch 4 and epoch 8
        # gave the same design and the same ~10.7% stress error despite clearly
        # different dg metrics). ~46s per call, so it only fires on screenshot
        # generations.
        ref = None
        if dims is not None and os.environ.get("TOPOPT_CCX_REF") == "1":
            try:
                if getattr(self, "_ccx_ref", None) is None:
                    from evaluators.ccx_evaluator import CCXEvaluator
                    self._ccx_ref = CCXEvaluator(
                        self.geometry, max_workers=4, mesh_threads=4, nproc=2,
                        # TopologyOptimizer never receives rho (it goes to the
                        # evaluator, not here), so read it explicitly -- it MUST
                        # match the value the run reports stress at, or the
                        # reference gradient belongs to a different functional.
                        stress_ks_rho_mpa=float(os.environ.get("STRESS_KS_RHO_MPA", "0.15")),
                        gradient_source="adjoint")
                _v, _s, _y, _dV, ref_dG = self._ccx_ref.compute_topopt_gradient(dims)
                ref = np.array([ref_dG[n] for n in self.free_names], dtype=float)
            except Exception as e:
                print(f"[TOPOPT] ccx reference gradient failed: {e}", flush=True)
                ref = None

        panels = [("dG/dx  (surrogate, post-bias)", grid(dg_raw), "seismic"),
                  ("dG/dx  (filtered, used by OC)", grid(dg_filt), "seismic")]
        if ref is not None:
            m, r = np.asarray(dg_raw, dtype=float), ref
            cos = float(m @ r / (np.linalg.norm(m) * np.linalg.norm(r) + 1e-30))
            ok = np.sign(m) == np.sign(r)
            wrong = float(np.abs(r)[~ok].sum() / (np.abs(r).sum() + 1e-30))
            panels.append((f"dG/dx  (CCX ADJOINT, truth)", grid(r), "seismic"))
            panels.append((f"DIFF surrogate - ccx   [cos {cos:+.3f}, "
                           f"wrong-way mass {100*wrong:.1f}%]", grid(m - r), "seismic"))
            print(f"[TOPOPT] gen {gen:03d} vs ccx adjoint: cosine {cos:+.4f}  "
                  f"wrong-way mass {100*wrong:.1f}%  "
                  f"|surrogate|/|ccx| median {np.median(np.abs(m))/ (np.median(np.abs(r))+1e-30):.3f}",
                  flush=True)
        panels.append(("thickness", grid(x), "viridis"))
        fig, axes = plt.subplots(len(panels), 1, figsize=(13, 3.1 * len(panels)))
        for ax, (title, img, cmap) in zip(np.atleast_1d(axes), panels):
            if cmap == "seismic":
                lim = np.percentile(np.abs(img), 98) or (np.abs(img).max() or 1.0)
                im = ax.imshow(img, cmap=cmap, origin="lower", aspect="auto",
                               vmin=-lim, vmax=lim)
                frac_pos = float((img > 0).mean())
                ax.set_title(f"{title}   [+/-{lim:.3g}]   {100*frac_pos:.0f}% positive (OC thins)",
                             fontsize=9)
            else:
                im = ax.imshow(img, cmap=cmap, origin="lower", aspect="auto")
                ax.set_title(f"{title}   [{img.min():.3f}, {img.max():.3f}]", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
            ax.set_xlabel("tile X", fontsize=8); ax.set_ylabel("tile Y", fontsize=8)
        fig.suptitle(f"topopt gen {gen:03d}  --  per-tile gradient field", fontsize=11)
        fig.tight_layout()
        fig.savefig(save_path, dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def _checkpoint_path(self):
        return f"{self.artifact_dir}/topopt_checkpoint.pkl"

    def _load_checkpoint(self):
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return None, 0, None, float("inf")
        with open(path, "rb") as f:
            state = pickle.load(f)
        gen = state["gen"]
        self.V_ref = state["V_ref"]
        print(f"[TOPOPT] Resuming from checkpoint gen {gen} "
              f"(V_ref={self.V_ref:.4e})", flush=True)
        return state["x"], gen, state["best_dims"], state["best_vol"]

    # ------------------------------------------------------------------
    def run(self):
        x, start_gen, best_dims, best_vol = self._load_checkpoint()
        if x is None:
            # init_thickness overrides dims.json's per-dim "default" as the
            # starting design. Worth setting when the mesh resolution has
            # changed: the reported KS rises steeply with elements-through-wall
            # (94.3 MPa at 3,3,3, 132.4 at 5,3,3 for the same t=0.3), so a
            # start that was comfortably feasible on a coarse mesh begins
            # infeasible on a finer one, and OC has to claw material back
            # before it can do anything useful.
            if self.init_thickness is not None:
                x = np.full(self.N, float(self.init_thickness), dtype=float)
                print(f"[TOPOPT] initial thickness override: {self.init_thickness}",
                      flush=True)
            else:
                x = np.array([self.dims_dict[n]["default"] for n in self.free_names],
                             dtype=float)
            x = np.clip(x, self.lo, self.hi)
            start_gen = 0
            best_dims, best_vol = None, float("inf")
            print(f"[TOPOPT] New run: free_dim={self.N} fd_step={self.fd_step} "
                  f"filter_radius={self.filter_radius} oc_eta={self.oc_eta}",
                  flush=True)

        N = self.N

        for gen in range(start_gen, start_gen + self.generations):
            t_gen_start = perf_counter()

            if self.gradient_func is not None:
                # O(1) path: one call gets the base point plus its exact
                # (w.r.t. the surrogate) per-tile sensitivities directly,
                # replacing the N+1-evaluator-call FD sweep below.
                dims_dicts = [self._vec_to_dims(x)]
                t = perf_counter()
                result = self.gradient_func(dims_dicts[0])
                t_fitness = perf_counter() - t
                if result is None:
                    print(f"[TOPOPT] Gen {gen + 1}: base point failed evaluation, "
                          f"skipping (no state change)", flush=True)
                    continue
                v0, s0, yld, dV_dx, dG_dx = result
                v0, s0 = float(v0), float(s0)
                df_raw = np.array([dV_dx[n] for n in self.free_names], dtype=float)
                dg_raw = np.array([dG_dx[n] for n in self.free_names], dtype=float)

                # ---- rolling calibration against the real solver ----------
                # gradient_func's stress comes from core/fem_adjoint.py's own
                # C3D8; the rest of the pipeline (and every design validation)
                # uses ccx's C3D8I. They agree to +0.29% at the uniform
                # t=0.30 start -- which is the only place the adjoint's
                # forward solve had ever been checked -- and diverge as the
                # design thins, because C3D8 shear-locks and locking is worst
                # in slender members. Measured on job 51500's own designs:
                #     design                        adjoint      ccx     diff
                #     uniform t=0.30                 76.958   76.739   +0.29%
                #     thinned gen  8 (t .105-.367)  101.100  109.648   -7.80%
                #     thinned gen 16 (t .016-.434)  103.276  108.677   -4.97%
                #     thinned gen 24 (t .010-.470)  103.352  117.055  -11.71%
                # The adjoint reads LOW, i.e. the constraint is optimistic by
                # roughly the margin the optimizer is trying to hold: at gen
                # 24 it believed 103.4 MPa (on the boundary) while ccx said
                # 117.1 (17% over yield). So job 51500 converged to a design
                # that is genuinely over yield.
                #
                # One ccx solve every calib_every generations gives the ratio;
                # it is smoothed (the bias drifts with the design, it does not
                # jump) and applied to BOTH s0 and dg_raw, so the constraint
                # and its gradient stay the same functional -- scaling KS by c
                # scales dKS/dt by c.
                #
                # As of the C3D8I fix in core.fem_adjoint (ShapeAdjoint now
                # reproduces ccx's own C3D8I to 1.7e-5%) this is no longer a
                # crutch holding the constraint together -- it is a monitor.
                # Expect calib to sit at 1.00 and stay there; a run where it
                # drifts the way 53474's did (0.997 at the uniform start,
                # 1.32 by gen 60) means the element agreement has broken
                # again and the constraint is moving under the optimizer.
                if self.calib_every and gen % self.calib_every == 0:
                    try:
                        cal = self.fitness_func([dims_dicts[0]])
                        s_ref = cal["stress"][0]
                        if s_ref is not None and s0 > 0:
                            f = float(s_ref) / s0
                            self._calib = (f if self._calib is None else
                                           (1.0 - self.calib_smooth) * self._calib
                                           + self.calib_smooth * f)
                            print(f"[TOPOPT] Gen {gen + 1} calibration | "
                                  f"adjoint={s0:.3f} ref={float(s_ref):.3f} "
                                  f"ratio={f:.4f} -> calib={self._calib:.4f}",
                                  flush=True)
                    except Exception as e:
                        print(f"[TOPOPT] Gen {gen + 1} calibration failed: {e}",
                              flush=True)
                if self._calib is not None:
                    s0 *= self._calib
                    dg_raw = dg_raw * self._calib

                g0 = s0 / yld - 1.0
            else:
                # Forward differences; flip to backward near the upper bound
                # so every perturbed point stays inside [lo, hi].
                step = self.fd_step * self.span
                deltas = np.where(x + step <= self.hi, step, -step)

                pts = [x.copy()]
                for i in range(N):
                    xp = x.copy()
                    xp[i] = x[i] + deltas[i]
                    pts.append(xp)

                dims_dicts = [self._vec_to_dims(p) for p in pts]
                t = perf_counter()
                res = self.fitness_func(dims_dicts)
                t_fitness = perf_counter() - t
                yld = res["yield_strength"]
                vols = res["volume"]
                strs = res["stress"]

                if vols[0] is None or strs[0] is None:
                    print(f"[TOPOPT] Gen {gen + 1}: base point failed evaluation, "
                          f"skipping (no state change)", flush=True)
                    continue

                v0 = float(vols[0])
                s0 = float(strs[0])
                g0 = s0 / yld - 1.0

                df_raw = np.empty(N, dtype=float)
                dg_raw = np.empty(N, dtype=float)
                for i in range(N):
                    vi, si = vols[i + 1], strs[i + 1]
                    if vi is None or si is None:
                        df_raw[i] = 0.0
                        dg_raw[i] = 0.0
                        continue
                    gi = float(si) / yld - 1.0
                    df_raw[i] = (float(vi) - v0) / deltas[i]
                    dg_raw[i] = (gi - g0) / deltas[i]

            if self.V_ref is None:
                self.V_ref = v0
                print(f"[TOPOPT] volume reference V_ref={self.V_ref:.4e}", flush=True)

            t = perf_counter()
            df = self._filter_sensitivity(df_raw, x)
            dg = self._filter_sensitivity(dg_raw, x)

            # ---- trust region on the constraint linearization ------------
            # _oc_update's lambda bisection solves g0 + dg.(x_new - x) = 0
            # exactly, so it ALWAYS reports a step that lands precisely on the
            # constraint boundary -- and nothing used to check whether that
            # prediction came true. Measured on job 50796, gens 2-24: g_pred
            # was 0.0000 every single generation while the actual g0 went
            # +0.058, +0.111, +0.498, +1.09, ... +16.4. The prediction error
            # WAS the entire constraint value. Because dg is huge on the few
            # tiles at the stress peak, the bisection can always claim some
            # step wipes out the violation, so it picks the smallest lambda
            # that does -- i.e. the one that removes the most material -- and
            # volume falls monotonically while stress rises monotonically,
            # forever. That is the collapse; it is independent of
            # trust_rel_tol and of the KS sharpness, and it sank job 49780 too.
            #
            # The linearization itself is fine when the step is small: gens
            # 0-1 of the same run predicted to within 0.014 and 0.022 at the
            # SAME 0.0368 absolute step. What changes is the design -- stress
            # goes like 1/t, so the curvature blows up as tiles thin, and a
            # step that is mild at t~0.3 is wild at t~0.05. A fixed absolute
            # move limit cannot be right at both ends, so make it adaptive:
            # cut it when the model over-promises, restore it when the model
            # is accurate. Standard SLP/trust-region practice, and it costs no
            # extra solves -- rejection reuses the previous generation's
            # cached gradient, which is still exact at the design we return to.
            if self._prev is not None:
                # DO NOT credit g_backoff back here. That was tried and
                # measured worse, so the plain form below is deliberate.
                #
                # The reasoning for crediting it looked sound: the bisection
                # drives pred_g to exactly -g_backoff every step (dg.dx came
                # out as -(g0 + 0.02) to five decimals across all of job
                # 54664), so a step landing exactly on the boundary still
                # reports over ~ g_backoff as though the model had failed.
                # That dead band opens whenever 0.25*tr_tol < g_backoff <
                # tr_tol, and the defaults sit in it (0.0125 < 0.02 < 0.05).
                # A replay over 54664's 575 recorded generations showed the
                # credited form cutting frozen generations 519 -> 72.
                #
                # But the A/B on the real objective went the other way. Job
                # 65489 ran the credited form against job 54664's plain form,
                # same config otherwise, and tracked WORSE from generation 75
                # onward, finishing 2% heavier:
                #
                #     gen      54664 (plain)   65489 (credited)
                #     100          254.4            260.8
                #     150          250.0            257.8
                #     best    250.02 @ g145    255.14 @ g167
                #
                # Un-freezing the trust region let more tiles move per step,
                # which made the steps noisier rather than better-directed --
                # the replay measured activity, not progress. The dead band
                # was acting as an accidental but useful damper.
                #
                # Both runs plateau by generation ~150, so what actually
                # needed fixing was the wasted tail, and --stagnation_gens
                # (below) handles that directly.
                over = g0 - self._prev["pred_g"]
                if over > self.tr_tol:
                    self._move_scale = max(self.tr_min_scale,
                                           0.5 * self._move_scale)
                    # Reject only a step that made an ALREADY-violated
                    # constraint worse: that is unambiguously a bad step, and
                    # the design we came from is known-better. A step that
                    # merely overshot into mild infeasibility is kept (the
                    # smaller move_scale handles it next generation), so the
                    # optimizer can still ride the constraint boundary.
                    if (g0 > 0.0 and g0 > self._prev["g0"]
                            and self._move_scale > self.tr_min_scale):
                        p = self._prev
                        x = p["x"].copy()
                        x_new, lam = self._oc_update(x, p["df"], p["dg"], p["g0"])
                        self._prev = dict(
                            x=x.copy(), df=p["df"], dg=p["dg"], g0=p["g0"],
                            pred_g=p["g0"] + float(np.dot(p["dg"], x_new - x)))
                        print(f"[TOPOPT] Gen {gen + 1} | REJECTED: g={g0:+.3e} vs "
                              f"predicted {p['pred_g']:+.3e} (over by {over:+.3e}); "
                              f"reverting to gen-{gen} design and retrying at "
                              f"move_scale={self._move_scale:.4f}", flush=True)
                        with open(self._checkpoint_path(), "wb") as f:
                            pickle.dump({"x": x_new, "best_dims": best_dims,
                                         "best_vol": best_vol,
                                         "V_ref": self.V_ref, "gen": gen + 1}, f)
                        x = x_new
                        continue
                elif over < 0.25 * self.tr_tol:
                    self._move_scale = min(1.0, 1.3 * self._move_scale)

            x_new, lam = self._oc_update(x, df, dg, g0)
            self._prev = dict(x=x.copy(), df=df, dg=dg, g0=g0,
                              pred_g=g0 + float(np.dot(dg, x_new - x)))
            t_oc = perf_counter() - t

            feasible = s0 <= yld
            if feasible and v0 < best_vol:
                best_vol = v0
                best_dims = dims_dicts[0]
                self._stagnant = 0
            else:
                self._stagnant += 1

            change = float(np.max(np.abs(x_new - x))) if N else 0.0
            best_vol_s = f"{best_vol:.4e}" if np.isfinite(best_vol) else "none"
            print(
                f"[TOPOPT] Gen {gen + 1} | vol={v0:.4e} stress={s0:.4e} "
                f"g={g0:+.3e} lambda={lam:.3e} max|dx|={change:.3e} "
                f"mscale={self._move_scale:.4f} "
                f"calib={(self._calib if self._calib is not None else float('nan')):.4f} "
                f"feasible={feasible} | best_feas_vol={best_vol_s}",
                flush=True,
            )

            # STAGNATION STOP. Job 54664 found its best design at generation
            # 145 and spent the remaining 455 -- about 13 hours -- drifting
            # 12% heavier at constant stress without ever improving on it.
            # The best design is tracked, so stopping costs nothing but the
            # chance of a late improvement, and 54664 shows what that chance
            # is worth. 0 disables (a run that must cover a fixed generation
            # count for comparability).
            if self.stagnation_gens and self._stagnant >= self.stagnation_gens:
                print(f"[TOPOPT] Gen {gen + 1} | STOPPING: no improvement on "
                      f"best_feas_vol={best_vol_s} in {self._stagnant} "
                      f"generations (--stagnation_gens "
                      f"{self.stagnation_gens}).", flush=True)
                np.savez(
                    f"{self.history_dir}/gen_{gen:03d}.npz",
                    free_names=np.array(self.free_names),
                    x=x, x_new=x_new, df_raw=df_raw, dg_raw=dg_raw,
                    df_filt=df, dg_filt=dg, v0=v0, s0=s0, g0=g0, lam=lam,
                    change=change, feasible=feasible)
                break

            np.savez(
                f"{self.history_dir}/gen_{gen:03d}.npz",
                free_names=np.array(self.free_names),
                x=x, x_new=x_new,
                df_raw=df_raw, dg_raw=dg_raw,
                df_filt=df, dg_filt=dg,
                v0=v0, s0=s0, g0=g0, lam=lam, change=change,
                feasible=feasible,
            )

            t_screenshot = 0.0
            if gen % self.screenshot_every == 0:
                banner = f"[TOPOPT gen {gen:03d}] vol: {v0:.4e}, stress: {s0:.4e}"
                save_path = f"optimization/screenshots/topopt_gen_{gen:03d}.png"
                t = perf_counter()
                # base_heatmap rides along on fitness_func's own evaluation when
                # available; the gradient_func fast path doesn't produce one, so
                # screenshot() below re-solves independently in that case.
                base_heatmap = None if self.gradient_func is not None else res.get("base_heatmap")
                try:
                    if base_heatmap is not None:
                        # GNN's own node-stress prediction for the base design --
                        # already computed as part of this generation's fitness
                        # call, no extra solve needed.
                        base_heatmap["mesh"].plot_stress_heatmap(
                            base_heatmap["stress"], save_path=save_path, banner=banner,
                            resolution=(5120, 2880), aa_type="ssaa", clim=(0.0, yld),
                        )
                    else:
                        screenshot(
                            geometry=self.geometry,
                            dims=dims_dicts[0],
                            save_path=save_path,
                            banner=banner,
                            heatmap=self.screenshot_heatmap,
                        )
                except Exception as e:
                    print(f"[TOPOPT] screenshot failed: {e}", flush=True)
                try:
                    self._save_tile_gradient_heatmap(
                        gen, dg_raw, dg, x,
                        f"optimization/screenshots/topopt_gen_{gen:03d}_grad.png",
                        dims=dims_dicts[0])
                except Exception as e:
                    print(f"[TOPOPT] gradient heatmap failed: {e}", flush=True)
                t_screenshot = perf_counter() - t

            t = perf_counter()
            with open(self._checkpoint_path(), "wb") as f:
                pickle.dump({"x": x_new, "best_dims": best_dims,
                             "best_vol": best_vol, "V_ref": self.V_ref,
                             "gen": gen + 1}, f)
            t_checkpoint = perf_counter() - t

            t_gen_total = perf_counter() - t_gen_start
            print(
                f"[TOPOPT] Gen {gen + 1} timing | total={t_gen_total:.2f}s "
                f"fitness={t_fitness:.2f}s oc={t_oc:.3f}s "
                f"screenshot={t_screenshot:.2f}s checkpoint={t_checkpoint:.3f}s",
                flush=True,
            )

            if change < 1e-6:
                print("[TOPOPT] Stopping early: design stopped moving", flush=True)
                break
            x = x_new

        if best_dims is None:
            print("[TOPOPT] WARNING: no feasible sample was ever found; "
                  "returning current iterate as a fallback.", flush=True)
            return self._vec_to_dims(x)
        return best_dims
