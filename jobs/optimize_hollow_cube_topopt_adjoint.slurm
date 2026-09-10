#!/bin/bash
#SBATCH --job-name=optimize_hc_topopt_adjoint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --partition=public
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# SIXTH RUN. Exactly ONE thing differs from job 54664: the trust region no
# longer counts the deliberate constraint back-off as model error. Plus a
# stagnation stop, which cannot change the answer -- only when we stop paying
# for it.
#
# WHAT 54664 SETTLED. The C3D8I gradient is right: calib read 1.0000 at all
# ~120 checkpoints across 600 generations, agreeing with ccx to ~0.002 MPa on
# designs from vol 705 down to 247. It produced the best design of any run --
# vol 250.02, genuinely feasible -- against 51500's 269.1 (constraint ~10% too
# generous) and 53474's 342.1 (2.2% over yield on real ccx).
#
# WHAT 54664 EXPOSED. It found that design at generation 145 and then spent
# 455 generations, about 13 hours, getting WORSE: vol mean 251 -> 258 -> 267
# -> 274 -> 282 while stress sat pinned at 100.2-100.4 MPa. Not a plateau, a
# ratchet. move_scale was frozen at ~0.12 for the last 61 generations and
# could move in neither direction.
#
# THE MECHANISM, measured on 54664's own history: the OC bisection drives the
# predicted post-step constraint to exactly -g_backoff every generation
# (dg.dx came out as -(g0 + 0.02) to five decimals, every time), while the
# design actually lands at g ~ 0.000. So `over` = g0 - pred_g was ~= g_backoff
# STRUCTURALLY -- not model error, just the margin being counted twice. The
# dead band opens whenever 0.25*tr_tol < g_backoff < tr_tol, and the defaults
# sat squarely inside it: 0.0125 < 0.02 < 0.05. Nothing about the gradient
# could have fixed this, which is why 54664 froze despite calib = 1.0000.
#
# THE FIX. over = g0 - (pred_g + g_backoff), i.e. how far outside the
# constraint the step actually landed. Replayed over 54664's 575 recorded
# generations: frozen 519 -> 72, growth events 56 -> 503. Collapse protection
# is unchanged -- jobs 49780/50796 reached g=+0.56 and +11.7, an order of
# magnitude past tr_tol.
#
# --stagnation_gens 120 stops the run once best_feas_vol has not improved for
# 120 generations. On 54664 that would have stopped at ~gen 265 and forfeited
# nothing at all.
#
# ---------------------------------------------------------------------------
# FIFTH RUN (54664). Exactly ONE thing differed from job 53474, whose args this file
# still carries verbatim: ShapeAdjoint now solves on ccx's C3D8I instead of
# plain C3D8. Everything else -- move limits, trust region, back-off,
# calibration, KS rho, 600 generations, uniform t=0.30 start -- is unchanged
# on purpose, so the two trajectories are directly comparable.
#
# WHY. 53474's adjoint agreed with ccx to +0.3% at the uniform start and read
# 32% LOW once walls thinned (76 vs 100.4 MPa). A constant offset calibrates
# out; a drifting one moves the constraint under the optimizer. It did: the
# run fell 705.6 -> 342.1 in 29 generations while calib was still ~1.03, then
# calib climbed 1.11 -> 1.21 -> 1.38 and the design walked BACK to vol ~400,
# where it sat for 176 generations. Archived in artifacts_53474_deadband_*.
#
# THE FIX (core/fem_adjoint.py) needed two corrections, and the first alone
# was worse than the original:
#   1. _bubble_B carried a detJ0/detJ scaling. ccx maps the Wilson modes with
#      the element-CENTRE Jacobian and NO such scaling.
#   2. _nodal_stress_terms computed sigma = D B ue, dropping the incompatible
#      modes' own strain Ba ae (ae = -Kaa^-1 Kua^T ue, condensed out of K and
#      so needing explicit recovery). With only (1), stiffness was C3D8I but
#      stress recovery was C3D8-shaped -- neither element, ratios 0.89/0.82/
#      1.15/1.20 across designs, not even sign-stable.
#   Displacements could not catch (2): they are set by the condensed
#   stiffness and were already right. Only a stress-level check found it.
#
# VALIDATED: element stiffness vs ccx C3D8I <=1.7e-5% over 9 distortions;
# nodal von Mises <=0.0012% over 6; KS on four of 53474's own designs ratio
# 1.0000 (76.741/76.739, 108.617/108.614, 101.588/101.586, 100.349/100.347);
# gradient sum rule 0.9997 / 0.9987 / 1.0010 at thick, thin and thinned
# designs (the ccx-axis route scores -2.790 on this same grid). Cost +10.4%.
#
# WHAT TO WATCH. calib is now a MONITOR, not a crutch: it should sit at ~1.00
# all run. If it drifts like 53474's did, element agreement has broken again.
# The trust-region DEAD BAND is deliberately NOT fixed here: 53474 froze with
# move_scale pinned at 0.3393 for 42+ generations because over = g0 - pred_g
# sat permanently in [0.25*tr_tol, tr_tol] = [0.0125, 0.05], too small to
# shrink and too large to grow. An accurate gradient should shrink that error
# below 0.0125 and let move_scale recover on its own. If this run stalls the
# same way, THAT is the next thing to fix -- and changing both at once would
# have made it impossible to tell which one mattered.

# Fourth topopt run on the exact gradient (core/fem_adjoint.py). The trust
# region added for job 51500 WORKED and is unchanged here: 705.6 -> ~270
# volume by generation 55, a 62% reduction, with g bounded in +0.008..+0.054
# across 140 generations and 7 rejections, against g=+11.7 (job 50796) and
# +0.56 (49780) by generation 25. That collapse is fixed. 51500 then converged
# -- oscillating in vol 269-282 from gen ~55 to 141 -- and was cancelled as
# finished, not broken. Its history is in optimization/artifacts_51500_*; its
# best design is gen 63, vol 269.1.
#
# TWO THINGS MADE 51500's OUTPUT UNUSABLE, and this run fixes both.
#
# A. IT COULD NOT RECORD WHAT IT FOUND (--g_backoff 0.02). The bisection aims
#    at g_lin = 0 exactly, but stress is convex in thinning so the linear
#    model under-predicts the rise and the iterate lands just OUTSIDE the
#    boundary nearly every generation. 51500 never once tripped `feasible`
#    after gen 3: best_feas_vol stayed frozen at 623.9 and run() returns
#    best_dims, so the whole 2-hour run would have handed back the
#    generation-3 design and discarded the vol-270 one it actually found.
#    Aiming at g_lin = -g_backoff is the standard constraint margin. Verified
#    on 51500's own gen-140 state: g_pred goes 0.0000 -> -0.0100 -> -0.0200
#    for backoff 0/0.01/0.02 and the estimated volume change turns positive
#    (+1.5 -> +4.8 -> +9.8), i.e. it adds material to get inside instead of
#    hovering outside. 0.04 saturates at -0.0236 (move limit), so 0.02 is
#    comfortably reachable.
#
# B. THE CONSTRAINT WAS OPTIMISTIC (--calib_every 5). ShapeAdjoint runs plain
#    C3D8; ccx runs C3D8I. Cross-checked on 51500's own designs:
#        design                        adjoint      ccx     diff
#        uniform t=0.30                 76.958   76.739   +0.29%
#        thinned gen  8 (t .105-.367)  101.100  109.648   -7.80%
#        thinned gen 16 (t .016-.434)  103.276  108.677   -4.97%
#        thinned gen 24 (t .010-.470)  103.352  117.055  -11.71%
#    The adjoint reads 5-12% LOW on real designs -- the unsafe direction, and
#    roughly the margin the optimizer is trying to hold. At gen 24 it believed
#    103.4 MPa (on the boundary) while ccx said 117.1 (17% over yield), so
#    51500 converged to a design that is genuinely over yield. Cause is almost
#    certainly C3D8 shear locking: too stiff in bending, worst in slender
#    members, and tiles become slender as they thin. They agree to 0.29% at
#    the uniform t=0.30 start, which is the ONLY design the adjoint's forward
#    solve had ever been validated at -- hence 24 hours of work on top of it
#    without noticing. Turning on the existing incompatible=True is NOT the
#    fix: our Wilson variant is ~8% off ccx the other way (101.22 vs 94.30).
#    So: one real ccx solve every 5 generations, smoothed into a ratio applied
#    to BOTH s0 and dg_raw (scaling KS by c scales dKS/dt by c, so constraint
#    and gradient stay one functional). This TRACKS the error rather than
#    fixing it; implementing ccx's incompatible-mode formulation with static
#    condensation in ShapeAdjoint is the real fix and is a separate job.
#    Cost is one extra solve per 5 generations, ~7% wall time.
#
# EXPECT A HIGHER FINAL VOLUME THAN 51500's 269. That run's number was bought
# with a constraint ~10% too generous; this one holds the real one.
#
# THE BUG (found in job 50796, present in 49780 and in every ccx-gradient run
# before them). _oc_update's lambda bisection SOLVES g0 + dg.(x_new - x) = 0,
# so it always reports a step landing exactly on the constraint boundary, and
# nothing ever checked whether that prediction came true. Job 50796, per
# generation, predicted vs measured:
#     gen    v0        g0     g_pred   g_actual   pred_err
#       0   705.6   -0.2304   -0.1729   -0.1588    +0.0141
#       1   666.8   -0.1588   -0.0806   -0.0585    +0.0221
#       2   623.9   -0.0585   +0.0000   +0.0579    +0.0579
#       5   486.0   +0.0294   +0.0000   +0.1109    +0.1109
#       6   433.6   +0.1109   +0.0000   +0.4976    +0.4976
#       7   368.3   +0.4976   +0.0000   +1.0916    +1.0916
#      14   126.0   +4.8220   +0.0000  +11.7205   +11.7205
# From gen 2 on, g_pred is 0.0000 every single generation and the prediction
# error IS the entire constraint value. Because dg is huge on the few tiles at
# the stress peak, the bisection can always claim SOME step wipes out the
# violation, so it picks the smallest lambda that does -- i.e. the one that
# removes the most material. Volume falls monotonically and stress rises
# monotonically, forever: 705.6 -> 42.5 over 75 generations in job 49780,
# 705.6 -> 35 over 25 in job 50796, both then limit-cycling for hundreds of
# generations with ~800/840 tiles pinned at the 0.01 floor.
#
# The linearization is not wrong, it is being used outside its radius. Gens
# 0-1 predicted to within 0.014 and 0.022 at the SAME 0.0368 absolute step.
# Stress goes like 1/t, so curvature explodes as tiles thin and a step that is
# mild at t~0.3 is wild at t~0.05 -- no fixed absolute move limit is right at
# both ends. Measured directly, replaying job 50796's gen-6 design through the
# real _oc_update at four move scales and evaluating each candidate:
#     mscale   max|dx|    g_pred  g_actual       err      vol
#      1.000 3.680e-02   +0.0000   +0.4976   +0.4976    368.3  <- what it did
#      0.500 1.840e-02   +0.0000   +0.1282   +0.1282    405.7
#      0.250 9.200e-03   +0.0285   +0.0272   -0.0013    449.0
#      0.125 4.600e-03   +0.0697   +0.0651   -0.0046    441.4
# The error falls quadratically in step size, as second-order truncation
# predicts, and at mscale 0.25 the model is good to ~0.001. Note that row's
# g_pred is +0.0285, NOT 0.0000: with a small enough move limit the bisection
# can no longer reach the boundary and stops claiming it can, which is lambda
# becoming meaningful again. Its design is also the better one -- g=+0.027,
# essentially on the constraint, against the full step's g=+0.498, which is
# violently infeasible and therefore useless whatever its volume.
#
# THE FIX (--tr_tol 0.05, --tr_min_scale 0.02). Each generation compares the
# measured g against the previous step's prediction. Over-promise by more than
# tr_tol -> halve the move limit; if the step ALSO made an already-violated
# constraint worse, reject it outright -- revert to the previous design and
# retry from there, reusing that design's cached gradient, which is still
# exact, so a rejection costs no solve. tr_min_scale floors the multiplier so
# rejections terminate. The controller walks 1.0 -> 0.5 -> 0.25 on the gen-6
# data above and settles. Watch the new mscale field in the log line.
#
# THE THREE FIXES FROM JOB 50796 ARE KEPT. All were real bugs and all are
# still needed -- 50796 improved best_feas_vol 705.6 -> 666.8 -> 623.9 in its
# first three generations, which 49780 never managed once in 503 -- they were
# simply not sufficient on their own. The gradient itself was never implicated
# in any of this: it scores 824/840 sign and +0.9997 against the 1680-solve
# remesh FD map and is unchanged.
#
# 1. THE CLAMP TRAPDOOR (data/hollow_cube/CAD_model/dims.json, max 0.49 ->
#    0.47). model_lin.c:30 defines HOLLOW_CUBE_MAX_THICKNESS 0.48, so a tile
#    driven past 0.48 stops moving its nodes entirely: measured, 48 moving
#    nodes at t=0.479 and 0 at t=0.480, with volume 0.48 -> 0.49 changing by
#    exactly 0.000000. Its sensitivity is then structurally zero, and OC
#    reads dG=0 as B=0 -> x*0^eta = 0 -> the 0.01 floor on the next
#    generation. It fires first on the tiles with the STRONGEST negative
#    gradient, because those are the ones pushed hardest into the clamp --
#    i.e. exactly the support corners, which is how this was found. 0.47
#    leaves room for the move limit to probe above it without reaching 0.48,
#    and costs 0.2% of achievable volume (836.98 vs 838.66).
#    ShapeAdjoint.tile_gradient now raises on a zero-moving-node tile rather
#    than returning 0.0, so this can never again be silent.
#
# 2. THE TRUST THRESHOLD (--trust_rel_tol 0.5 -> 0.0). Measured on the
#    initial design at rho=0.15: 793 of 840 tiles have dg < 0, a coherent
#    stress-reducing direction. trust_rel_tol trusts only dg < -0.5*std(dg),
#    and |dg| spans 795x (p50 1.49e-04, max 1.19e-01), so std is set by the
#    few hot tiles at the load and the cutoff lands 24x above the median:
#    trusted 45/840, i.e. straight onto the min_trust_frac floor. The other
#    795 tiles -- with a correct gradient -- get the 10% leash, and each one
#    whose B < 1 is walked down to the floor, which is absorbing (a floored
#    tile is out of the load path, so its dg is ~0 and it is never trusted
#    again). That is the 705 -> 42.5 drain. The whole mechanism was built
#    against FD/ccx-axis noise where, in its own words, "the sign is close to
#    a coin flip"; the adjoint is exact and that premise no longer holds.
#    min_trust_frac and untrusted_move_frac are left in place as a safety
#    net -- at tol=0 they simply stop binding.
#
# 3. NO FEASIBLE HEADROOM (--stress_ks_rho_mpa 0.15 -> 0.25). At rho=0.15 the
#    KS aggregate overestimates the true peak nodal von Mises by 41.8%
#    (94.302 vs 66.501 MPa), so the run began at g0 = -0.057 and any material
#    removal at all made it infeasible. Sweep on the initial design:
#      rho    KS     over    g0       neg%   n@90% of |dg| mass   corr vs 0.15
#      0.15  94.30  +41.8%  -0.0570   94.4%        195                --
#      0.25  76.96  +15.7%  -0.2304   68.8%         79             +0.9771
#      0.30  74.09  +11.4%  -0.2591   58.3%         70             +0.9641
#      0.50  69.71   +4.8%  -0.3029   53.8%         50             +0.9181
#    Sharpening trades constraint accuracy for gradient support: the sign
#    flips are all on near-zero tiles (correlation stays >= +0.92), but the
#    informative set shrinks fast. 0.25 is the knee -- it recovers 26 of the
#    41.8 points of conservatism while keeping 68.8% of tiles carrying
#    signal. The remaining +15.7% is kept deliberately: KS is meant to be a
#    conservative bound on the peak, and the two stress hot spots are both
#    modelling artifacts (the rigid-tie load patch and the support-corner
#    singularity), so a real margin there is wanted, not tuned away.
#    core/fem_adjoint.py's tile_gradient took rho as an argument for this
#    run: it had hardcoded 0.15 while the evaluator passed
#    --stress_ks_rho_mpa to ks() alone, which was inert at the default but
#    would have optimized one functional's value against another's gradient
#    the moment rho moved.
#
# DAMPING (--move_limit 0.20 -> 0.08, --oc_eta 0.7 -> 0.5). Consequences of
# fix 2, not independent guesses: with 578 tiles now moving on a real step
# instead of 45, the old limit is far too hot. At 0.20 a tile crosses the
# whole 0.01-0.47 span in 5 generations, which is faster than the design can
# be corrected; 0.08 makes it 13. oc_eta returns to the classic SIMP 0.5.
# 600 generations at ~50 s leaves plenty of room for the slower step.
#
# WHAT TO WATCH. `calib` in the log line should settle near 1.05-1.12 and
# drift up as the design thins; if it swings wildly the smoothing is too weak
# or a ccx solve is faulting. `feasible=True` must now actually appear and
# best_feas_vol must improve past 623.9 -- if it still never trips, g_backoff
# is too small for the curvature. mscale should drop early (the design starts at t=0.30, where
# the linearization is still good, and stiffens as tiles thin) and then hold
# somewhere around 0.1-0.5 rather than riding the tr_min_scale floor. Volume
# should track the constraint boundary -- g hovering near 0 with `feasible`
# flipping either side of it -- instead of falling monotonically while stress
# climbs. best_feas_vol should keep improving; both previous runs froze it
# (705.6 for all 503 generations of 49780, 623.9 from gen 3 of 50796).
# Persistent REJECTED lines with mscale stuck at 0.02 would mean the step is
# not the problem and something else is nonlinear. If the run throws the
# AssertionError from tile_gradient, a tile reached the CAD clamp anyway and
# the cap needs to come down further.
#
# FRESH START. optimization/artifacts is scoped by neither geometry nor arch,
# so 51500's checkpoint was moved to optimization/artifacts_51500_converged_*
# (the two collapsed runs are in artifacts_backup_49780_* and _50796_*) --
# otherwise run() would resume from the previous design instead of starting at
# generation 0.

cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$PWD/irit/bin:$PATH"
export IRIT_PATH="$PWD/irit/bin/"

# The adjoint's cost is torch's batched element passes plus one sparse
# factorization, so unlike the ccx path this one WANTS threads in-process.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

mkdir -p optimization/artifacts optimization/screenshots

python -u -m optimization.optimize_dims \
    --geometry hollow_cube \
    --arch ccx \
    --optimizer topopt \
    --ccx_gradient adjoint \
    --generations 600 \
    --processes 8 \
    --mesh_threads 4 \
    --nproc 1 \
    --filter_radius 2 \
    --move_limit 0.08 \
    --oc_eta 0.5 \
    --fd_step 0.04 \
    --trust_rel_tol 0.0 \
    --min_trust_frac 0.05 \
    --tr_tol 0.05 \
    --tr_min_scale 0.02 \
    --g_backoff 0.02 \
    --calib_every 5 \
    --calib_smooth 0.5 \
    --stagnation_gens 120 \
    --untrusted_move_frac 0.1 \
    --screenshot_every 5 \
    --stress_ks_rho_mpa 0.25

rm -r data/hollow_cube/Instance_* 2>/dev/null
rm -r data/hollow_cube/ansys_* 2>/dev/null
echo TOPOPTDONE
true
