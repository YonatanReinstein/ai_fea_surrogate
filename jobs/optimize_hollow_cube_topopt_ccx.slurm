#!/bin/bash
#SBATCH --job-name=optimize_hc_topopt_ccx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=13:00:00
#SBATCH --partition=public
# Pinned to the node every earlier run of this script used. NOT because of
# the ccx library situation: ldd on galileo3/protagoras/socrates/newton9
# resolves identical lapack/blas/atlas from ~/opt/calculix and the same
# glibc 2.35, so that hypothesis is dead.
#
# UNEXPLAINED, WATCH FOR IT: job 48454 (galileo3) reported 758.57 MPa at
# gen 1 for the uniform t=0.3 design, g0=+6.59, against the 93.9316 MPa
# that design measures. The ccx solve did not fail -- the gen-0 screenshot
# renders a perfectly plausible field, because a heat map autoscales to
# its own min/max and so hides a uniformly wrong magnitude. Four attempts
# to reproduce it all returned 93.9316 exactly: static solve alone and
# static-after-parallel-axis-gradient, on galileo3 AND protagoras. So it
# is not the node and not the concurrent axis solves. One-off bad solve of
# unknown cause. The tell is the reported stress jumping ~8x in a single
# generation; that is worth checking in the log before trusting a result,
# since OC answers a g0 of +6.6 by pinning lambda at its 1e9 cap.
#SBATCH --nodelist=protagoras
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$PWD/irit/bin:$PATH"
export IRIT_PATH="$PWD/irit/bin/"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p optimization/artifacts optimization/screenshots

# FRESH START (gen 0). The checkpoint that was sitting in
# optimization/artifacts/ came from a *surrogate* topopt run, not a ccx
# one -- that directory is not scoped by --arch or by --geometry, so any
# run overwrites it. It was moved to optimization/artifacts_backup_ccx_*/
# before this job so run() starts clean instead of resuming a design the
# surrogate produced (and that real ccx scored 139.7 MPa against the
# surrogate's 100.9 -- see project_surrogate_ood_collapse_topopt).
# History of this run's trust-mechanism bugs (all on the same 840-tile
# hollow_cube design, see project_topopt_trust_deadlock.md memory):
#   job 45778: trusted set collapsed to 0 -> permanent deadlock. Fixed with
#     --min_trust_frac (floors the trusted count).
#   job 47985: min_trust_frac fixed the deadlock but the untrusted majority
#     (now ~95% of tiles) still snapped to a full-move prune on every
#     noise-level feasible generation -> limit cycle, best_feas_vol stuck at
#     394.4 for 55 generations. Fixed with --prune_margin (gated the snap on
#     a feasibility margin instead of g0<=0).
#   job 47989: prune_margin stopped the snap, but with the trust threshold
#     routinely excluding ~95% of tiles, those tiles just sat frozen instead
#     -- 800/840 tiles stopped moving for 90+ generations, design drifted to
#     a worse local point (394.4 -> 401).
#   job 47992: untrusted_move_frac=0.1, resumed from job 47989's checkpoint
#     -- fixed all three of the above, best_feas_vol broke through 394.4
#     down to 377.3 by gen 249, clean-looking truss bands in screenshots.
#   job 47999: fresh restart (gen 0) with untrusted_move_frac raised to 0.3
#     for speed -- the offline adjacency-disagreement metric said this
#     wouldn't add checkerboarding risk (step_cap only scales movement
#     magnitude, not direction), but the real run visibly disagreed: gen 99
#     screenshot shows scattered speckled tiles through the empty region,
#     and it converged *worse* (best_feas_vol=400.4 after 140 gens) than
#     47992's 0.1 run.
#   job 48002: fresh restart (gen 0) with untrusted_move_frac back to 0.1 --
#     STILL showed checkerboarding/speckle and converged worse (425.1 after
#     140 gens) than 47992. This rules out untrusted_move_frac as the cause:
#     both 0.1 and 0.3 fresh starts checkerboard. 47992's clean truss bands
#     only appeared after ~250+ cumulative generations resumed across
#     several checkpoints -- 48002 only got 140 gens from scratch. The
#     design was still actively moving generation-to-generation when it hit
#     the --generations cap (max|dx| in the 1e-3..2e-2 range on the last 10
#     gens, nowhere near the change<1e-6 early-stop), so more generations,
#     not a different frac, is the lever to try next.
# GRADIENT SOURCE: --ccx_gradient axis -- the patched solver's global-axis
# nodal gradient dKS/dX contracted with the geometric Jacobian dX/dt. Four
# solves per generation (3 axis + 1 static) instead of the FD sweep's 841,
# for any tile count.
#
# WHY THE ALLOCATION SHRANK 190 -> 48 CPUS. The FD sweep's 841 independent
# solves were what --processes was sized for; it scaled flat across cores.
# The axis path solves ONE mesh, so the only parallelism left is the three
# axis solves, now run concurrently (Mesh.solve_ccx_axis_gradient), each on
# --gradient_nproc threads. 3 x 16 = 48 is the whole budget; asking for 190
# would idle 140 cores for 13 hours and, under this account's
# QOSMaxCpuPerUserLimit (~192), block everything else.
#
# ACCURACY at this geometry's native mesh_resolution (3,3,3). Each tile is a
# hollow tube whose wall is subdivided (U-1)x(V-1)x(W-1), so (3,3,3) puts two
# elements through the wall -- below the >=5 the method was characterised on.
# Measured on the sibling 8x6x1 grid (hollow_cube_mid), all 48 tiles against
# central-difference remesh FD:
#     3,3,3  sign 48/48  spearman 0.9922  ratio median 0.586
#     5,5,5  sign 45/48  spearman 0.9920  ratio median 0.999
# Two elements per wall costs ~1.7x in MAGNITUDE and nothing in RANK, and OC
# consumes rank (it rescales by lambda every generation), so 3,3,3 stands.
# Refining is not an option here anyway: 5,5,5 on 840 tiles is 215k elements
# against 27k, and *SENSITIVITY over ~48k design nodes already costs ~14x a
# plain solve.
#
# BEWARE the obvious sanity check on this grid. Sampling 15 tiles spread over
# the 40x21 grid and comparing to remesh FD gives sign 4/15 and Spearman
# -0.02, which looks like a broken gradient. It is not: |remesh FD| across
# those tiles spans 1.8 to 5.4e6, i.e. 12 of the 15 sit at solver noise
# because one tile in 840 barely moves a global KS max. Restricted to the
# tiles carrying signal (|FD| >= 10% of max) the sign agreement is 3/3. Judge
# this gradient only on tiles with real sensitivity.
# MESH 5,3,3 -- ANISOTROPIC. mesh_resolution is per tile and each tile is a
# hollow tube whose wall is subdivided (U-1)x(V-1)x(W-1); only U runs THROUGH
# the wall. Measured on the meshes themselves: refining U halves the smallest
# element edge (0.5000 -> 0.2500) while V and W only shorten the long edges.
# So U is the one that matters for the bending the wall carries, and refining
# it alone buys 5,5,5's wall resolution at a quarter of its size:
#
#   mesh     el/wall   nodes    elements   3 axis solves   spearman vs FD
#   3,3,3       2      48249      26880        ~500 s          +0.288
#   4,4,4       3     138496      90720        2611 s          +0.161
#   5,3,3       4      80415      53760        1540 s          +0.700   <-
#   5,5,5       4     299425     215040      >7200 s (cap)     n/a
#   7,3,3       6     112581      80640      >8900 s           pending
#
# 5,3,3 has the best per-tile ranking by a wide margin and the most negative
# dG of any variant tried (11/48 against remesh FD's 37/48; 3,3,3 gave 7/48
# and 4,4,4 only 3/48). 7,3,3 reaches 6 elements per wall but its gradient
# alone runs past 2.5 h, which prices it out whatever its accuracy.
#
# --init_thickness 0.42, not dims.json's 0.30: reported KS climbs steeply with
# elements-through-wall (94.3 MPa at 3,3,3 -> 132.4 at 5,3,3 for the SAME
# t=0.3 design) because the thin-wall stress concentration is resolved for the
# first time. At 0.30 this mesh starts infeasible against the 100 MPa yield and
# OC has to claw material back before it can optimise anything. Note the tube
# volume law 4v(1-v) peaks at v=0.5, so 0.30 -> 0.42 is only +16% material and
# 0.49 (the cap) only +19% -- thickening cannot buy much, and if 132 MPa does
# not come under 100 the constraint itself is calibrated to the 3,3,3 mesh.
#
# Big steps kept: --move_limit 0.20 and --oc_eta 0.7.
#
# COST, measured on this geometry at (3,3,3), 26880 elements (protagoras):
#   geometric Jacobian (2 CAD builds + mesh)      6.9 s   negligible
#   value solve (plain C3D8I, for reported stress) 10 s   negligible
#   ONE axis sensitivity solve                    165 s   <- the whole cost
# ccx does not scale past ~8 threads on this problem (179.6 s at 8, 164.2 s
# at 16, 167.1 s at 32), and concurrent solves contend hard: three at once
# take 455-506 s whatever the thread split, barely better than 495 s serial.
# So raising --gradient_nproc past ~16 or --cpus-per-task past ~48 buys
# nothing; the floor is set by *SENSITIVITY over the design set, not by cores.
#
# A geometry-specific shortcut was measured and REJECTED, on purpose. The
# contraction is sum_n sum_j g[n][j] * dX_j/dt[n], so skipping axes and nodes
# whose dX/dt is zero is exact. This grid is 40x21x1 -- one tile through the
# x-thickness -- so dx/dt is identically 0 on all 48249 nodes and axis 0 is a
# 165 s solve multiplied by zero; per surviving axis only 30240 of 48249 nodes
# move. Doing both took 482 s -> 291 s per generation and agreed with the full
# gradient to 3.3e-05 relative over all 840 tiles. It is still not adopted:
# which axes and nodes are live is a property of one outline and tile design,
# and this path must behave identically for every geometry it is handed.
python -u -m optimization.optimize_dims \
    --geometry hollow_cube \
    --arch ccx \
    --optimizer topopt \
    --ccx_gradient axis \
    --generations 300 \
    --processes "$SLURM_CPUS_PER_TASK" \
    --gradient_nproc 16 \
    --mesh_threads 1 \
    --nproc 1 \
    --filter_radius 2 \
    --move_limit 0.20 \
    --oc_eta 0.7 \
    --mesh_res 5,3,3 \
    --init_thickness 0.42 \
    --fd_step 0.04 \
    --min_trust_frac 0.05 \
    --untrusted_move_frac 0.1 \
    --stress_ks_rho_mpa 0.15

rm -r data/hollow_cube/Instance_* 2>/dev/null
rm -r data/hollow_cube/ansys_* 2>/dev/null

true
