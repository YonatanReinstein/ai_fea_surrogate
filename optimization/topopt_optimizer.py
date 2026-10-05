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
                 calib_every=5, calib_smooth=0.5, calib_max_dev=2.0, stagnation_gens=0,
                 pop_size=None, generations=200,
                 seed=0, geometry="hollow_cube",
                 fd_step=0.02, move_limit=0.2, filter_radius=1.5, filter_target="gradient",
                 filter_radius_end=None, filter_decay_gens=0, filter_decay_start=0,
                 proj_eta=None, proj_beta_end=16.0, proj_start_gen=100, proj_end_gen=300, proj_w_max=1.0,
                 sharpen_mode="gate", unsharp_radius=None, skel_tau=0.03, skel_tau_end=None, stop_at_gen=None,
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
        # What the spatial filter is applied to. "gradient" (default) is the
        # Sigmund sensitivity filter on df/dg; "design" leaves the gradients
        # untouched and instead smooths the DESIGN after each OC step
        # (density filter, x <- H x / sum H); "none" disables both.
        self.filter_target = filter_target
        # Continuation on the DESIGN filter: radius goes linearly from
        # filter_radius at gen 0 to filter_radius_end at gen filter_decay_gens
        # and stays there. A radius <= 1 is the identity on this grid (only
        # dist 0 has positive weight), i.e. no smoothing. Lets a run start
        # smooth (fast, stable descent) and finish crisp (voids can commit).
        self.filter_radius_end = filter_radius_end
        self.filter_decay_gens = int(filter_decay_gens)
        # Decay starts at this gen (radius constant before it). Lets the
        # topology form under the full radius before the blur is relaxed.
        self.filter_decay_start = int(filter_decay_start)
        self._design_filt_r = None
        self._design_filt = None
        # filter_target="density": the proper density filter + gated
        # threshold projection, with the chain rule on both sensitivities.
        #   x (design var, what OC moves)
        #   x_tilde = H x                    (H = row-normalised cone filter)
        #   x_phys  = lo + (x_tilde - lo) * S_beta(t),  t = (x_tilde-lo)/span
        #   S_beta(t) = sigmoid(beta (t - eta_t)) / sigmoid(beta (1 - eta_t))
        # Everything that touches the solver/surrogate (evaluation, spool,
        # screenshots, best_dims) sees x_phys; OC sees dV/dx = H^T (dV/dx_phys
        # * dx_phys/dx_tilde), same for dG. beta = 0 is the identity gate;
        # ramped linearly from 0 at proj_start_gen to proj_beta_end at
        # proj_end_gen and held. As beta grows, tiles whose FILTERED thickness
        # sits below proj_eta are driven to the floor while thicker ones keep
        # their (continuous) thickness -- the "defined geometry" the plain
        # radius decay could not give: the moment the design filter reached
        # identity in runs G/H the constraint reading jumped, lambda pinned,
        # and the run oscillated 260-285 for 200 gens; runs that kept a grey
        # web (G: 599/840 tiles at 0.01-0.10 holding 46% of the material)
        # never got a crisp topology. Radius must stay >= ~1.5 in this mode;
        # sharpening comes from the gate, not from removing the blur.
        # proj_eta is an ABSOLUTE thickness (same units as dims.json), chosen
        # in the valley between the web and the members (G: 0.10-0.15).
        self.proj_eta = None if proj_eta is None else float(proj_eta)
        self.proj_beta_end = float(proj_beta_end)
        self.proj_start_gen = int(proj_start_gen)
        self.proj_end_gen = int(proj_end_gen)
        # Ceiling of the gate blend weight. 1.0 in density mode (the gate is
        # a fixed map of a persistent design variable, so full strength is
        # the sharp limit). In filter_target="design" the gate is applied to
        # the design ITSELF after the smoothing, every generation, so it
        # compounds: a sub-threshold tile loses the fraction w*(1-S) of its
        # thickness per generation, i.e. w is an erosion RATE there, and OC
        # re-grows what the stress needs. Keep it small (0.02-0.05).
        self.proj_w_max = float(proj_w_max)
        # sharpen_mode (filter_target="design" only):
        #   "gate"    -- threshold gate at proj_eta (see _project). Applied to
        #                the design every generation it is an EROSION: every
        #                sub-threshold tile loses w*(1-S) of its thickness per
        #                gen regardless of what the stress gradient says. Run L
        #                (w<=0.05) did nothing; run M (w=0.25) ran away -- the
        #                blur dragged member edges under eta, erosion ate them,
        #                stress rose, the trust region shrank OC's step below the
        #                erosion rate, vol 272 -> 34.
        #   "unsharp" -- derivative subtraction / unsharp mask:
        #                x <- x + w * (x - H_s x), H_s the cone blur at
        #                unsharp_radius (default: the current design radius).
        #                x - H_s x is minus the discrete Laplacian, so this is
        #                x - w*lap(x): contrast is boosted only where the design
        #                differs from its neighbourhood (member edges), bulk
        #                regions -- web interior included -- are untouched. It
        #                removes blur, not material. Per-gen composite
        #                (I + w(I - H_s)) H has every mode gain <= 1 for w <= 1
        #                with H_s = H (checked numerically), so it does not run
        #                away on its own.
        #   "skeleton" -- grey-level skeleton flow (mass-conserving concentration):
        #                score every tile by the sum over thickness levels
        #                (0.04..0.20) of its capped chamfer distance to the edge
        #                of the level set {x >= level}; a band of 0.08 tiles in
        #                a 0.03 background scores highest along its centreline,
        #                and the cap (3 tiles) keeps plate-scale geometry out of
        #                it. Each tile above skel_tau then sends the fraction w
        #                of its material above the floor, split equally, to
        #                face-neighbours with a strictly higher score. Wide thin
        #                bands narrow into thicker centrelines; total material
        #                is conserved except for ceiling clipping. Tested
        #                post-hoc on run K gen 320: one round at w=0.2 gave a
        #                visibly more defined truss for +5% ccx KS with the
        #                load path intact; iterating it (10 rounds) fragments
        #                the web into isolated tiles, so inside the optimiser
        #                it relies on OC re-connecting between applications.
        self.sharpen_mode = sharpen_mode
        self.unsharp_radius = None if unsharp_radius is None else float(unsharp_radius)
        self.skel_tau = float(skel_tau)
        # Optional schedule: tau goes linearly from skel_tau to skel_tau_end
        # over the same proj_start_gen..proj_end_gen window as the blend
        # weight, so the donor cut-off lowers as the sharpening strengthens.
        self.skel_tau_end = None if skel_tau_end is None else float(skel_tau_end)
        self._skel_cache = None
        # Absolute generation to stop at (exclusive). `generations` counts
        # from the (possibly resumed) start gen, which made "GENS=600" on a
        # resume from 300 mean 900; stop_at_gen says where to end outright.
        self.stop_at_gen = None if stop_at_gen is None else int(stop_at_gen)
        self._unsharp_filt = None
        self._unsharp_r = None
        self._proj_log_beta = None
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
        # Outlier guard on the calibration ratio. The reference solve is a real
        # ccx run and it occasionally returns a wildly wrong stress for a
        # perfectly ordinary design -- job 48454 read 758.57 MPa for the
        # uniform t=0.30 design that measures 93.93 and could not be
        # reproduced in four attempts, and job 183931 read 5186.26 MPa for the
        # one that measures 76.74 (ratio 67.6) while every calibration after it
        # sat at ~1.04. Without a guard that single sample enters the EMA and
        # multiplies BOTH s0 and dg_raw by it, which pins lambda at its 1e9 cap
        # and thickens the whole design for the ~30 generations the EMA needs
        # to decay back. The adjoint tracks ccx's C3D8I to 1.7e-5% and the
        # physical bias this correction exists for is a few per cent, so any
        # ratio outside [1/calib_max_dev, calib_max_dev] is a bad solve, not a
        # bias: drop it and keep the previous calib. 0 disables the guard.
        self.calib_max_dev = float(calib_max_dev or 0.0)
        self._calib = None
        # Gradient-MAGNITUDE calibration (TOPOPT_DG_CALIB=1). The surrogate's
        # dG direction is right (cosine +0.9-0.999 vs ccx) but its magnitude
        # runs 2.7x ccx at gen 0 and 4-13x on the trajectory (rho-0.25 run
        # 1399058). OC's bisection then believes every thinning step costs
        # several times the stress it really does and takes a fraction of the
        # move limit; the ccx-driven run removed 290 volume in 10 gens, the
        # surrogate runs ~55. The ccx reference field is computed every
        # screenshot generation anyway (TOPOPT_CCX_REF=1); this multiplies
        # dg_raw by the smoothed median |ccx|/|surrogate| from it -- the same
        # mechanism as the stress calibration above, applied to the gradient.
        self._dg_calib = None

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
        # Screenshots are keyed by generation only, so two concurrent runs
        # overwrite each other's; TOPOPT_SCREENSHOT_DIR gives a run its own.
        self.screenshot_dir = os.environ.get("TOPOPT_SCREENSHOT_DIR",
                                             "optimization/screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # spatial sensitivity filter over the tile grid
    # ------------------------------------------------------------------
    def _build_filter(self, r_min, quiet=False):
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
        if not quiet:
            print(f"[TOPOPT] spatial filter built: grid={NX}x{NY}x{NZ} "
                  f"r_min={r_min} free_dim={self.N}", flush=True)
        return W, row_sum

    def _design_radius(self, gen):
        if self.filter_radius_end is None or self.filter_decay_gens <= 0:
            return self.filter_radius
        t = min(1.0, max(0.0, (gen - self.filter_decay_start) / float(self.filter_decay_gens)))
        return self.filter_radius + (self.filter_radius_end - self.filter_radius) * t

    # ------------------------------------------------------------------
    # density filter + gated projection (filter_target="density")
    # ------------------------------------------------------------------
    def _density_H(self, gen):
        """(W, row_sum) for the current radius, rebuilt only when the rounded
        radius changes; None when r <= 1 (identity on this grid)."""
        r = round(self._design_radius(gen), 3)
        if r != self._design_filt_r:
            self._design_filt = None if r <= 1.0 else self._build_filter(r, quiet=True)
            self._design_filt_r = r
        return self._design_filt

    def _proj_w(self, gen):
        """Blend weight of the gate, 0 (identity) -> 1 (full gate) linearly
        over proj_start_gen..proj_end_gen. The gate itself keeps a FIXED
        sharpness proj_beta_end throughout; only how much of it is applied
        ramps. Ramping the sharpness instead (run I, job 1403130) scaled
        every intermediate tile down at low beta -- a 0.15 tile to 60% at
        beta 3.2 -- so the whole design shrank every generation regardless
        of the OC step; the trust region rejected five steps in a row,
        pinned move_scale at its floor, and the design collapsed (vol 292 ->
        190, stress 157 MPa within 20 gens of the ramp start). With a fixed
        sharp gate, tiles above eta see S ~ 1 at every w, so the blend only
        ever moves the sub-threshold web, and by a fraction 1/(end-start) of
        its thickness per generation."""
        if (self.proj_eta is None and self.sharpen_mode not in ("unsharp", "skeleton")) or gen <= self.proj_start_gen:
            return 0.0
        if gen >= self.proj_end_gen or self.proj_end_gen <= self.proj_start_gen:
            return self.proj_w_max
        return self.proj_w_max * (gen - self.proj_start_gen) / float(self.proj_end_gen - self.proj_start_gen)

    def _skel_tau_at(self, gen):
        if self.skel_tau_end is None or self.proj_w_max <= 0:
            return self.skel_tau
        t = self._proj_w(gen) / self.proj_w_max
        return self.skel_tau + (self.skel_tau_end - self.skel_tau) * t

    def _beta(self, gen):
        # kept for logging: effective sharpness in use (0 while the gate is off)
        return self.proj_beta_end if self._proj_w(gen) > 0.0 else 0.0

    def _project(self, x_tilde, gen):
        """Gated projection of the filtered design. Returns (x_phys,
        dx_phys/dx_tilde). w=0 -> identity, derivative 1."""
        w = self._proj_w(gen) if self.proj_eta is not None else 0.0
        if w <= 0.0:
            return x_tilde, np.ones_like(x_tilde)
        beta = self.proj_beta_end
        t = (x_tilde - self.lo) / self.span
        eta = (self.proj_eta - self.lo) / self.span      # per-tile, in [0, 1]
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))
        norm = sig(beta * (1.0 - eta))
        S = sig(beta * (t - eta)) / norm
        dS = beta * S * (1.0 - S * norm) / self.span     # d/dx_tilde of S
        G = (1.0 - w) + w * S
        dG = w * dS
        x_phys = self.lo + (x_tilde - self.lo) * G
        dphys = G + (x_tilde - self.lo) * dG
        return x_phys, dphys

    def _pred_g(self, g0, dg, dg_phys, x, x_new, gen):
        """Linear prediction of next gen's g. In density mode it is made in
        PHYSICAL space and includes the schedule drift (radius / gate weight
        change between gen and gen+1), which the OC step does not control
        but which moves x_phys all the same."""
        if self.filter_target != "density":
            return g0 + float(np.dot(dg, x_new - x))
        return g0 + float(np.dot(dg_phys, self._phys(x_new, gen + 1)[0] - self._phys(x, gen)[0]))

    def _phys(self, x, gen):
        """x -> (x_phys, dx_phys/dx_tilde). Identity outside density mode."""
        if self.filter_target != "density":
            return x, None
        H = self._density_H(gen)
        x_tilde = x if H is None else (H[0] @ x) / H[1]
        x_phys, dphys = self._project(x_tilde, gen)
        return np.clip(x_phys, self.lo, self.hi), dphys

    def _chain(self, v_phys, dphys, gen):
        """dF/dx from dF/dx_phys: H^T (v_phys * dx_phys/dx_tilde). W is
        symmetric, so H^T v = W (v / row_sum)."""
        if self.filter_target != "density":
            return v_phys
        v = v_phys * dphys
        H = self._density_H(gen)
        return v if H is None else H[0] @ (v / H[1])

    def _log_density_state(self, gen):
        if self.filter_target != "density":
            return
        beta = self._beta(gen)
        if gen % 10 == 0 or self._proj_log_beta is None or (beta > 0) != (self._proj_log_beta > 0):
            print(f"[TOPOPT] gen {gen}: density filter r={self._design_radius(gen):.3f} "
                  f"gate w={self._proj_w(gen):.3f} beta={beta:.1f} eta={self.proj_eta}", flush=True)
        self._proj_log_beta = beta

    def _filter_design(self, x_new, gen=0):
        """Density filter on the design itself (filter_target='design'):
        x <- H x / sum_i H, plain neighbourhood average, gradients untouched.
        Radius follows _design_radius(gen); the matrix is rebuilt only when
        the (rounded) radius changes."""
        if self.filt is None or self.filter_target != "design":
            return x_new
        r = round(self._design_radius(gen), 3)
        if r <= 1.0:
            if self._design_filt_r != r:
                print(f"[TOPOPT] gen {gen}: design filter radius {r:.3f} -> identity (no smoothing)", flush=True)
                self._design_filt_r = r
            return self._sharpen_design(x_new, gen)
        if self._design_filt_r != r:
            self._design_filt = self._build_filter(r, quiet=True)
            self._design_filt_r = r
            if gen % 10 == 0 or r == self.filter_radius:
                print(f"[TOPOPT] gen {gen}: design filter radius {r:.3f}", flush=True)
        W, row_sum = self._design_filt
        return self._sharpen_design(np.clip((W @ x_new) / row_sum, self.lo, self.hi), gen)

    SKEL_LEVELS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
    SKEL_DCAP = 3

    def _skeleton_grid(self):
        """Tile-grid coords / face-neighbour lists for the free dims (cached).
        Same index convention as _build_filter. None if the dims don't map
        onto the geometry's tile grid."""
        if self._skel_cache is not None:
            return self._skel_cache
        try:
            module = importlib.import_module(f"data.{self.geometry}.boundary_conditions")
            NX, NY, NZ = module.tile_grid()
        except Exception:
            self._skel_cache = False
            return False
        coords = []
        for name in self.free_names:
            try:
                k = int(name[1:]) - 1
            except ValueError:
                self._skel_cache = False
                return False
            coords.append((k // (NZ * NY), (k // NZ) % NY, k % NZ))
        grid = -np.ones((NX, NY, NZ), int)
        for i, c in enumerate(coords):
            grid[c] = i
        nbrs = []
        for (ix, iy, iz) in coords:
            nb = []
            for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                jx, jy, jz = ix+dx, iy+dy, iz+dz
                if 0 <= jx < NX and 0 <= jy < NY and 0 <= jz < NZ and grid[jx, jy, jz] >= 0:
                    nb.append(grid[jx, jy, jz])
            nbrs.append(nb)
        self._skel_cache = ((NX, NY, NZ), coords, nbrs)
        return self._skel_cache

    def _skeleton_flow(self, x, w, tau=None):
        """One round of the grey-level skeleton flow (see sharpen_mode)."""
        from scipy import ndimage
        tau = self.skel_tau if tau is None else tau
        g = self._skeleton_grid()
        if not g or w <= 0.0:
            return x, 0, 0.0
        (NX, NY, NZ), coords, nbrs = g
        score = np.zeros(self.N)
        for lvl in self.SKEL_LEVELS:
            sup = np.zeros((NX, NY, NZ), bool)
            for i, c in enumerate(coords):
                sup[c] = x[i] >= lvl
            if not sup.any():
                continue
            # outside the grid counts as support: boundary chords are real
            # edges of the part, not band edges to be eaten
            padded = np.pad(sup, 1, constant_values=True)
            d = np.minimum(ndimage.distance_transform_cdt(padded, metric="taxicab")[1:-1, 1:-1, 1:-1],
                           self.SKEL_DCAP)
            score += np.array([d[c] for c in coords], float)
        send = np.zeros(self.N); recv = np.zeros(self.N); n_don = 0
        for i in range(self.N):
            if x[i] <= tau:
                continue
            ups = [j for j in nbrs[i] if score[j] > score[i]]
            if not ups:
                continue
            amt = w * (x[i] - self.lo[i]); send[i] = amt; n_don += 1
            for j in ups:
                recv[j] += amt / len(ups)
        return np.clip(x - send + recv, self.lo, self.hi), n_don, float(send.sum())

    def _sharpen_design(self, x_sm, gen):
        """filter_target='design' + proj_eta: gate the SMOOTHED design in
        place (OC -> smooth -> sharpen). Post-hoc like the smoothing itself,
        so no chain rule; the trust region tests the step actually taken.
        Because it is re-applied to the design every generation the effect
        compounds -- see proj_w_max."""
        w = self._proj_w(gen)
        if w <= 0.0:
            return x_sm
        if self.sharpen_mode == "skeleton":
            tau = self._skel_tau_at(gen)
            x_sh, n_don, moved = self._skeleton_flow(x_sm, w, tau)
            if gen % 10 == 0:
                print(f"[TOPOPT] gen {gen}: design sharpen skeleton w={w:.4f} tau={tau:.4f} "
                      f"donors {n_don}, moved {moved:.3f} (sum {x_sm.sum():.2f} -> {x_sh.sum():.2f})", flush=True)
            return x_sh
        if self.sharpen_mode == "unsharp":
            r = round(self.unsharp_radius if self.unsharp_radius is not None else self._design_radius(gen), 3)
            if r != self._unsharp_r:
                self._unsharp_filt = None if r <= 1.0 else self._build_filter(r, quiet=True)
                self._unsharp_r = r
            if self._unsharp_filt is None:
                return x_sm
            W, row_sum = self._unsharp_filt
            hp = x_sm - (W @ x_sm) / row_sum          # -Laplacian (high-pass)
            x_sh = np.clip(x_sm + w * hp, self.lo, self.hi)
            if gen % 10 == 0:
                d = x_sh - x_sm
                print(f"[TOPOPT] gen {gen}: design sharpen unsharp w={w:.3f} r={r:.3f} "
                      f"up {int(np.sum(d > 1e-4))} / down {int(np.sum(d < -1e-4))} tiles, "
                      f"max |d|={float(np.max(np.abs(d))):.4f}", flush=True)
            return x_sh
        if self.proj_eta is None:
            return x_sm
        x_sh, _ = self._project(x_sm, gen)
        if gen % 10 == 0:
            n_er = int(np.sum(x_sm - x_sh > 1e-4))
            print(f"[TOPOPT] gen {gen}: design sharpen w={w:.4f} beta={self.proj_beta_end:.0f} "
                  f"eta={self.proj_eta} eroded {n_er} tiles, max d={float(np.max(x_sm - x_sh)):.4f}", flush=True)
        return np.clip(x_sh, self.lo, self.hi)

    def _filter_sensitivity(self, s, x):
        # Sigmund (1997) sensitivity filter:
        #   s_tilde_e = 1/max(x_e,eps) * sum_i(H_ei x_i s_i) / sum_i(H_ei)
        if self.filt is None or self.filter_target != "gradient":
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
        # Density mode: a tile whose whole filtered neighbourhood is gated off
        # has df ~ dg ~ 0, and B = lam*eps/eps = lam would walk it up the
        # move cap on every lambda-capped generation (~15% of them) until it
        # popped back through the gate. Freeze those instead; they can still
        # return once a neighbour thickens and H carries signal to them.
        dead = (df < 1e-3 * float(np.median(df))) if self.filter_target == "density" else np.zeros_like(df, bool)
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
            xn = np.where(dead, x, xn)
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

    def _save_tile_gradient_heatmap(self, gen, dg_raw, dg_filt, x, save_path, dims=None, s_sur=None):
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
                # TOPOPT_CCX_REF_DIR: keep the reference solve, in the SAME
                # layout as training/trajectory_labeller.py's per-gen labels,
                # so optimization/plot_trajectory_accuracy.py can score a run
                # with no labellers (a frozen-surrogate run) exactly the way it
                # scores an online one.
                _rd = os.environ.get("TOPOPT_CCX_REF_DIR")
                if _rd and s_sur is not None:
                    os.makedirs(_rd, exist_ok=True)
                    np.savez(os.path.join(_rd, f"gen_{gen:06d}.npz"),
                             free_names=np.asarray(self.free_names), x=np.asarray(x, dtype=float),
                             dg_ccx=ref, surrogate_dg=np.asarray(dg_raw, dtype=float),
                             ccx_stress=float(_s), ccx_volume=float(_v), yield_strength=float(_y),
                             surrogate_stress=float(s_sur), gen=int(gen))
            except Exception as e:
                print(f"[TOPOPT] ccx reference gradient failed: {e}", flush=True)
                ref = None

        panels = [("dG/dx  (surrogate, post-bias)", grid(dg_raw), "seismic"),
                  (f"dG/dx  (used by OC: {'filtered' if self.filter_target == 'gradient' else 'RAW, no filter'})", grid(dg_filt), "seismic")]
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
            if os.environ.get("TOPOPT_DG_CALIB") == "1":
                # m is the field OC used this generation (already scaled by the
                # current factor), so the update is multiplicative on the factor.
                # Least-squares scale k = argmin ||k*m - r|| = (m.r)/(m.m): the
                # scalar that makes the surrogate's LINEARISED CONSTRAINT best
                # match ccx's, dominated by the mass-carrying tiles OC actually
                # steps on. The first version used the median |ratio|, which is
                # a small-gradient tile: the surrogate compresses dynamic range
                # (small tiles ~2x too big, big tiles ~right), so the median
                # said "shrink 2x" while the important tiles were already right
                # -- factor drifted to 0.16 by gen 60 (job 1399247), OC then
                # overshot every step (15 rejections) and stalled at 611.
                live = np.asarray(x, dtype=float) > 0.02
                mm, rr = m[live], r[live]
                ratio = float(mm @ rr / (mm @ mm + 1e-30))
                cur = 1.0 if self._dg_calib is None else self._dg_calib
                new = float(np.clip(cur * ratio, 0.2, 5.0))
                self._dg_calib = new if self._dg_calib is None else \
                    (1.0 - self.calib_smooth) * cur + self.calib_smooth * new
                print(f"[TOPOPT] gen {gen:03d} dG magnitude calibration | measured |ccx|/|sur| {ratio:.3f} "
                      f"-> factor {cur:.3f} -> {self._dg_calib:.3f}", flush=True)
        panels.append(("thickness", grid(x), "viridis"))
        # ONE colour scale for every dG panel, in this picture and every later
        # one: fixed at the first screenshot from the 98th percentile of the ccx
        # reference (or the surrogate if there is none), or TOPOPT_HEATMAP_DG_LIM.
        # Per-panel autoscaling made a 2x-too-large surrogate look identical to
        # ccx and made the diff panel unreadable across generations.
        if getattr(self, "_heat_lim", None) is None:
            env_lim = os.environ.get("TOPOPT_HEATMAP_DG_LIM")
            base = ref if ref is not None else np.asarray(dg_raw, dtype=float)
            self._heat_lim = float(env_lim) if env_lim else \
                float(np.percentile(np.abs(base), 98) or (np.abs(base).max() or 1.0))
            print(f"[TOPOPT] heatmap dG colour scale fixed at +/-{self._heat_lim:.4g} for the whole run", flush=True)
        lim = self._heat_lim
        fig, axes = plt.subplots(len(panels), 1, figsize=(13, 3.1 * len(panels)))
        for ax, (title, img, cmap) in zip(np.atleast_1d(axes), panels):
            if cmap == "seismic":
                im = ax.imshow(img, cmap=cmap, origin="lower", aspect="auto",
                               vmin=-lim, vmax=lim)
                frac_pos = float((img > 0).mean()); clipped = float((np.abs(img) > lim).mean())
                ax.set_title(f"{title}   [fixed +/-{lim:.3g}, {100*clipped:.0f}% clipped]   "
                             f"{100*frac_pos:.0f}% positive (OC thins)", fontsize=9)
            else:
                im = ax.imshow(img, cmap=cmap, origin="lower", aspect="auto",
                               vmin=float(self.lo.min()), vmax=float(self.hi.max()))
                ax.set_title(f"{title}   [fixed {float(self.lo.min()):.2f}..{float(self.hi.max()):.2f}; "
                             f"this gen {img.min():.3f}..{img.max():.3f}]", fontsize=9)
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
        # Trust-region / calibration state. Without it a resume restarted at
        # move_scale 1.0 with no baseline and a freshly-read g: run G's resume
        # went lambda-capped for 5 gens and vol 275 -> 340 before recovering.
        # _prev's prediction was made under the weights the old job had; it
        # is re-anchored against the current weights at the top of run().
        self._move_scale = state.get("move_scale", 1.0)
        self._prev = state.get("prev")
        self._calib = state.get("calib")
        self._dg_calib = state.get("dg_calib")
        self._stagnant = state.get("stagnant", 0)
        self._spooled_gens = set(state.get("spooled_gens", ()))
        self._screenshot_owed = bool(state.get("screenshot_owed", False))
        print(f"[TOPOPT] Resuming from checkpoint gen {gen} "
              f"(V_ref={self.V_ref:.4e}, move_scale={self._move_scale:.4f}, "
              f"prev={'yes' if self._prev is not None else 'no'})", flush=True)
        return state["x"], gen, state["best_dims"], state["best_vol"]

    def _checkpoint_state(self, x_new, best_dims, best_vol, gen):
        return {"x": x_new, "best_dims": best_dims, "best_vol": best_vol,
                "V_ref": self.V_ref, "gen": gen + 1,
                "move_scale": self._move_scale, "prev": self._prev,
                "calib": self._calib, "dg_calib": self._dg_calib,
                "stagnant": self._stagnant,
                "spooled_gens": sorted(getattr(self, "_spooled_gens", set())),
                "screenshot_owed": bool(getattr(self, "_screenshot_owed", False))}

    def _write_params(self, start_gen):
        """Append this (segment of the) run's schedule parameters to
        <artifact_dir>/topopt_params.jsonl so a resumed run records what each
        stretch of generations was run with."""
        import json
        keys = ("generations", "stop_at_gen", "move", "filter_radius", "filter_target",
                "filter_radius_end", "filter_decay_gens", "filter_decay_start",
                "proj_eta", "proj_beta_end", "proj_start_gen", "proj_end_gen", "proj_w_max",
                "sharpen_mode", "unsharp_radius", "skel_tau", "skel_tau_end", "oc_eta", "tr_tol",
                "tr_min_scale", "g_backoff", "trust_rel_tol", "min_trust_frac",
                "untrusted_move_frac", "calib_every", "calib_smooth", "calib_max_dev",
                "stagnation_gens")
        rec = {"start_gen": int(start_gen)}
        for k in keys:
            v = getattr(self, k, None)
            rec[k] = v.tolist() if isinstance(v, np.ndarray) else v
        rec["env"] = {k: os.environ.get(k) for k in os.environ if k.startswith(("TOPOPT_", "GNN_LIVE", "STRESS_KS"))}
        try:
            with open(f"{self.artifact_dir}/topopt_params.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"[TOPOPT] params record failed: {e}", flush=True)

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
                  f"filter_radius={self.filter_radius} filter_target={self.filter_target} "
                  f"filter_radius_end={self.filter_radius_end} filter_decay_gens={self.filter_decay_gens} "
                  f"filter_decay_start={self.filter_decay_start} proj_eta={self.proj_eta} "
                  f"proj_beta_end={self.proj_beta_end} proj_gens={self.proj_start_gen}-{self.proj_end_gen} "
                  f"oc_eta={self.oc_eta}",
                  flush=True)
        # A restored _prev was predicted under another process's weights;
        # force the re-anchor branch below on the first generation.
        self._seen_reload = -1 if (start_gen > 0 and self._prev is not None) else 0
        self._write_params(start_gen)

        N = self.N

        end_gen = start_gen + self.generations
        if self.stop_at_gen is not None:
            end_gen = self.stop_at_gen
            print(f"[TOPOPT] running gens {start_gen}..{end_gen - 1} (stop_at_gen)", flush=True)
        for gen in range(start_gen, end_gen):
            t_gen_start = perf_counter()

            if self.gradient_func is not None:
                # O(1) path: one call gets the base point plus its exact
                # (w.r.t. the surrogate) per-tile sensitivities directly,
                # replacing the N+1-evaluator-call FD sweep below.
                self._log_density_state(gen)
                x_phys, dphys = self._phys(x, gen)
                dims_dicts = [self._vec_to_dims(x_phys)]
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
                dg_phys = dg_raw

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
                            lo = (1.0 / self.calib_max_dev) if self.calib_max_dev else 0.0
                            hi = self.calib_max_dev if self.calib_max_dev else float("inf")
                            if not (lo <= f <= hi):
                                print(f"[TOPOPT] Gen {gen + 1} calibration "
                                      f"REJECTED (bad reference solve) | "
                                      f"adjoint={s0:.3f} ref={float(s_ref):.3f} "
                                      f"ratio={f:.4f} outside [{lo:.3f}, {hi:.3f}]"
                                      f" -> calib held at "
                                      f"{'none' if self._calib is None else f'{self._calib:.4f}'}",
                                      flush=True)
                            else:
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
                if self._dg_calib is not None:
                    dg_raw = dg_raw * self._dg_calib
                dg_phys = dg_raw

                g0 = s0 / yld - 1.0
                # Chain rule through filter + projection (identity unless
                # filter_target="density").
                df_raw = self._chain(df_raw, dphys, gen)
                dg_raw = self._chain(dg_raw, dphys, gen)
            else:
                if self.filter_target == "density":
                    raise NotImplementedError("filter_target='density' needs a gradient_func (adjoint/surrogate) path")
                x_phys = x
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
                dg_phys = dg_raw

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
            # Online fine-tuning hot-reloads the surrogate mid-run. The trust
            # region compares this generation's g0 against LAST generation's
            # linear prediction -- both from the surrogate -- so a weight swap
            # between them shows up as a bogus 'over by' equal to the shift in
            # stress calibration. Seen live: update 1 moved stress bias
            # -12.97% -> +6.95% and gen 7 was REJECTED 'over by +0.2196', i.e.
            # exactly that 20-point swing. When the model changed, the previous
            # prediction is simply not comparable; drop it rather than punish
            # the optimiser for it.
            #
            # Dropping the baseline was the right call when reloads were rare
            # and the swing was 20 points. With an update EVERY generation
            # (the lag-4 online runs) it disabled the trust region outright:
            # _prev was reset before every comparison, mscale sat at 1.0000
            # for all 361 generations, and the run plateaued at vol ~420 in a
            # limit cycle (lambda log10 std 2.56, voids sloshing 13-26%).
            # The ccx-driven run 54664 reached the same ~410 at gen 14, cut
            # mscale to 0.1-0.2 and rode the boundary with 0.004-0.007 steps
            # down to 250. The surrogate's gradient was not the problem (99.5%
            # sign agreement with ccx on the cheapest-to-thin tiles at gen 300).
            #
            # So instead of dropping the baseline, RE-ANCHOR it: evaluate the
            # previous design under the reloaded weights and rebuild the
            # linear prediction from that. One extra surrogate call (~5s,
            # mostly the mesh rebuild) on generations where a reload happened;
            # the cached dg from the old weights is still the right slope to
            # use, only the intercept moved.
            _ev = getattr(self.gradient_func, "__self__", None)
            _rc = getattr(_ev, "reload_count", 0) if _ev is not None else 0
            if _rc != getattr(self, "_seen_reload", 0):
                self._seen_reload = _rc
                if self._prev is not None:
                    p = self._prev
                    try:
                        _pg = p.get("gen", gen - 1)
                        r = self.gradient_func(self._vec_to_dims(self._phys(p["x"], _pg)[0]))
                        if r is None:
                            raise RuntimeError("previous design failed evaluation")
                        s_prev = float(r[1]) * (self._calib if self._calib is not None else 1.0)
                        g_prev = s_prev / yld - 1.0
                        old_g, old_pred = p["g0"], p["pred_g"]
                        p["g0"] = g_prev
                        p["pred_g"] = self._pred_g(g_prev, p["dg"], p.get("dg_phys"), p["x"], x, _pg)
                        print(f"[TOPOPT] Gen {gen + 1} | weights reloaded -- trust-region "
                              f"baseline re-anchored: prev g {old_g:+.3e} -> {g_prev:+.3e}, "
                              f"pred {old_pred:+.3e} -> {p['pred_g']:+.3e}", flush=True)
                    except Exception as e:
                        print(f"[TOPOPT] Gen {gen + 1} | weights reloaded -- re-anchor "
                              f"failed ({e}); trust-region baseline reset", flush=True)
                        self._prev = None
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
                        x_new = self._filter_design(x_new, gen)
                        self._prev = dict(
                            x=x.copy(), df=p["df"], dg=p["dg"], dg_phys=p.get("dg_phys"), g0=p["g0"],
                            gen=gen,
                            pred_g=self._pred_g(p["g0"], p["dg"], p.get("dg_phys"), x, x_new, gen))
                        print(f"[TOPOPT] Gen {gen + 1} | REJECTED: g={g0:+.3e} vs "
                              f"predicted {p['pred_g']:+.3e} (over by {over:+.3e}); "
                              f"reverting to gen-{gen} design and retrying at "
                              f"move_scale={self._move_scale:.4f}", flush=True)
                        with open(self._checkpoint_path(), "wb") as f:
                            pickle.dump(self._checkpoint_state(x_new, best_dims, best_vol, gen), f)
                        # A rejected step skips the rest of the iteration, so a
                        # screenshot generation that lands here used to be lost
                        # (D: gens 90/170/220). Carry the flag to the next gen.
                        if gen % self.screenshot_every == 0:
                            self._screenshot_owed = True
                        x = x_new
                        continue
                elif over < 0.25 * self.tr_tol:
                    self._move_scale = min(1.0, 1.3 * self._move_scale)

            x_new, lam = self._oc_update(x, df, dg, g0)
            x_new = self._filter_design(x_new, gen)
            self._prev = dict(x=x.copy(), df=df, dg=dg, dg_phys=dg_phys, g0=g0, gen=gen,
                              pred_g=self._pred_g(g0, dg, dg_phys, x, x_new, gen))
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
                    x=x, x_phys=x_phys, x_new=x_new, df_raw=df_raw, dg_raw=dg_raw,
                    df_filt=df, dg_filt=dg, v0=v0, s0=s0, g0=g0, lam=lam,
                    change=change, feasible=feasible)
                break

            np.savez(
                f"{self.history_dir}/gen_{gen:03d}.npz",
                free_names=np.array(self.free_names),
                x=x, x_phys=x_phys, x_new=x_new,
                df_raw=df_raw, dg_raw=dg_raw,
                df_filt=df, dg_filt=dg,
                v0=v0, s0=s0, g0=g0, lam=lam, change=change,
                feasible=feasible,
            )

            # Spool this design for asynchronous ccx-adjoint labelling.
            # TOPOPT_LABEL_SPOOL enables it; training/trajectory_labeller.py
            # picks the files up out of band. Only a ~10KB write happens here --
            # a label costs ~46s of CPU against a ~4s generation, so it is only
            # free in wall-clock if the optimiser never waits on it.
            #
            # These designs cannot be produced by any i.i.d. sampler: measured
            # 2026-09-10, ten random designs matched to a trajectory design's
            # mean/std/total-variation scored cosine +0.94..+0.996 against the
            # ccx adjoint, while the trajectory design itself scored +0.09. The
            # failing designs are selected by the surrogate's OWN errors, so a
            # live trajectory is the only place to observe them.
            _spool = os.environ.get("TOPOPT_LABEL_SPOOL")
            # Front-loaded cadence: the surrogate gradient decorrelates from the
            # ccx adjoint by gen ~20-30 (cosine +0.94 -> ~0), so labels are worth
            # most EARLY. Dense until TOPOPT_LABEL_DENSE_UNTIL, sparse after.
            _dense = int(os.environ.get("TOPOPT_LABEL_DENSE_UNTIL", "0"))
            _every = 1 if gen <= _dense else int(os.environ.get("TOPOPT_LABEL_EVERY", "5"))
            if _spool and gen % _every == 0:
                try:
                    os.makedirs(_spool, exist_ok=True)
                    # NOTE: np.savez APPENDS .npz unless the name already ends
                    # in it, so a "foo.npz.writing" temp silently becomes
                    # "foo.npz.writing.npz" and the rename below fails. Use a
                    # dot-prefixed .npz instead; the labeller skips dotfiles.
                    _tmp = os.path.join(_spool, f".gen_{gen:06d}.npz")
                    # The labeller solves and differentiates the PHYSICAL design;
                    # dg_raw here is d/dx_phys (pre chain rule) so the two compare.
                    np.savez(_tmp, free_names=np.array(self.free_names), x=x_phys,
                             dg_raw=dg_phys, v0=v0, s0=s0, gen=gen)
                    # publish atomically -- the labeller must never claim a
                    # half-written file
                    os.replace(_tmp, os.path.join(_spool, f"gen_{gen:06d}.npz"))
                    # A REJECTED generation `continue`s above before reaching this
                    # block, so it never spools -- and the lagged wait below used to
                    # block on its (never-coming) label for the full 1800s timeout.
                    # Seen in every lagged run (move08 gen 9, three more earlier).
                    if not hasattr(self, "_spooled_gens"):
                        self._spooled_gens = set()
                    self._spooled_gens.add(gen)

                    # TOPOPT_LABEL_SAMPLES>1 also spools perturbations AROUND the
                    # current design, giving local coverage of the region the
                    # optimiser is in rather than a single point on its path.
                    # Deliberately uneconomic -- 10 labels/gen at ~45s each is far
                    # slower than just running ccx -- so this is a ceiling-finding
                    # diagnostic, not a production setting. dg_raw is left empty on
                    # perturbations: the surrogate gradient there is unmeasured, so
                    # the labeller must not report a bogus agreement number.
                    _ns = int(os.environ.get("TOPOPT_LABEL_SAMPLES", "1"))
                    if _ns > 1:
                        # Perturbation scales. A single 5% jitter makes all four
                        # siblings near-duplicates of the iterate, so a labelled
                        # bundle covers a POINT and teaches the surrogate almost
                        # nothing about its neighbourhood. TOPOPT_LABEL_JITTER
                        # accepts a comma-separated list (e.g. "0.02,0.05,0.10,0.20")
                        # cycled over the siblings, so the same 5 ccx solves cover a
                        # range of distances; TOPOPT_LABEL_JITTER_DIR=1 sends one
                        # sibling along the OC step direction (where the design is
                        # actually going) instead of an isotropic draw.
                        _sds = [float(v) for v in os.environ.get("TOPOPT_LABEL_JITTER", "0.05").split(",") if v]
                        _rng = np.random.default_rng(gen)
                        _dir = os.environ.get("TOPOPT_LABEL_JITTER_DIR", "0") == "1"
                        _step = np.asarray(x_new, float) - np.asarray(x, float)
                        for _k in range(_ns - 1):
                            _sd = _sds[_k % len(_sds)]
                            if _dir and _k == 0 and np.any(_step):
                                # step direction, scaled so |dx| ~ _sd * |x|
                                _u = _step / (np.linalg.norm(_step) + 1e-12)
                                _xp = np.clip(np.asarray(x_phys, float)
                                              + _u * _sd * float(np.linalg.norm(x_phys)),
                                              self.lo, self.hi)
                            else:
                                _xp = np.clip(np.asarray(x_phys, float) *
                                              (1.0 + _rng.normal(0.0, _sd, size=np.size(x_phys))),
                                              self.lo, self.hi)
                            _t2 = os.path.join(_spool, f".gen_{gen:06d}_p{_k:02d}.npz")
                            np.savez(_t2, free_names=np.array(self.free_names), x=_xp,
                                     dg_raw=np.zeros(0), v0=v0, s0=s0, gen=gen)
                            os.replace(_t2, os.path.join(_spool, f"gen_{gen:06d}_p{_k:02d}.npz"))

                    # Deterministic lagged pipeline: TOPOPT_LABEL_LAG=L makes gen i
                    # block until every sample spooled at gen i-L has a label AND
                    # the trainer has published weights newer than those labels.
                    # So each generation takes exactly one update, from exactly L
                    # generations back -- a reproducible test of lagged learning,
                    # at the cost of running at label speed rather than gen speed.
                    _lag = int(os.environ.get("TOPOPT_LABEL_LAG", "0"))
                    _ldir = os.environ.get("TOPOPT_LABEL_DIR")
                    _live = os.environ.get("GNN_LIVE_CKPT")
                    _tg = gen - _lag
                    if _lag and _ldir and gen >= _lag and _tg not in getattr(self, "_spooled_gens", set()):
                        print(f"[TOPOPT] gen {gen}: gen {_tg} was never spooled (rejected step) -- no wait", flush=True)
                    elif _lag and _ldir and gen >= _lag:
                        import time as _time
                        _want = [f"gen_{_tg:06d}.npz"] + [f"gen_{_tg:06d}_p{_k:02d}.npz" for _k in range(_ns - 1)]
                        _t0 = _time.time()
                        while True:
                            _have = [os.path.exists(os.path.join(_ldir, w)) for w in _want]
                            if all(_have):
                                # Wait for the trainer to have CONSUMED gen i-L, not merely
                                # to have published since its labels landed -- with a
                                # backlog those are very different (see online_finetune).
                                _mk = os.path.join(_ldir, '.consumed_upto')
                                try:
                                    _upto = int(open(_mk).read().strip()) if os.path.exists(_mk) else -1
                                except ValueError:
                                    _upto = -1
                                if _upto >= _tg:
                                    break
                            if _time.time() - _t0 > 1800:
                                print(f"[TOPOPT] lag wait timeout at gen {gen} (have {sum(_have)}/{len(_want)})", flush=True)
                                break
                            _time.sleep(1.0)
                        print(f"[TOPOPT] gen {gen}: trained on gen {_tg} ({_time.time()-_t0:.0f}s wait)", flush=True)

                    # Backpressure: without it the optimiser outruns the labellers and
                    # every label describes a design it has already left.
                    _maxq = int(os.environ.get("TOPOPT_LABEL_MAXQ", "0"))
                    if _maxq:
                        import time as _time
                        _t0 = _time.time()
                        while len([f for f in os.listdir(_spool)
                                   if f.startswith("gen_") and f.endswith(".npz")]) > _maxq:
                            if _time.time() - _t0 > 600:
                                print("[TOPOPT] backpressure timeout, continuing", flush=True)
                                break
                            _time.sleep(2.0)
                except Exception as e:
                    print(f"[TOPOPT] spool write failed: {e}", flush=True)

            t_screenshot = 0.0
            if gen % self.screenshot_every == 0 or getattr(self, "_screenshot_owed", False):
                self._screenshot_owed = False
                banner = f"[TOPOPT gen {gen:03d}] vol: {v0:.4e}, stress: {s0:.4e}"
                save_path = f"{self.screenshot_dir}/topopt_gen_{gen:03d}.png"
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
                        gen, dg_phys, dg, x_phys,
                        f"{self.screenshot_dir}/topopt_gen_{gen:03d}_grad.png",
                        dims=dims_dicts[0], s_sur=s0)
                except Exception as e:
                    print(f"[TOPOPT] gradient heatmap failed: {e}", flush=True)
                t_screenshot = perf_counter() - t

            t = perf_counter()
            with open(self._checkpoint_path(), "wb") as f:
                pickle.dump(self._checkpoint_state(x_new, best_dims, best_vol, gen), f)
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
            return self._vec_to_dims(self._phys(x, start_gen + self.generations)[0])
        return best_dims
