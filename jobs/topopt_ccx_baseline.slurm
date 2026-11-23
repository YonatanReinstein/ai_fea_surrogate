#!/bin/bash
#SBATCH --job-name=topopt_ccx_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=36:00:00
#SBATCH --partition=public
#SBATCH --nodelist=protagoras
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#
# GROUND-TRUTH BASELINE for the surrogate-driven topopt runs.
#
# This is run T's optimizer, parameter for parameter, with the ONE substitution
# that matters: the GNN surrogate is replaced by ccx itself
# (--arch ccx --ccx_gradient adjoint). Everything downstream of the gradient --
# OC, the trust region, the design filter, the skeleton sharpening, the move
# limit, the calibration monitor -- is byte-identical to
# optimization/artifacts_rho025_T_sched800_ep80/topopt_params.jsonl. Whatever
# this run reaches is therefore what the surrogate runs would reach if the
# surrogate were perfect, and the gap between them is attributable to the
# surrogate alone rather than to a schedule difference.
#
# GRADIENT SOURCE: adjoint, not axis. core/fem_adjoint.py's exact shape adjoint
# -- one factorization of our own C3D8I assembly plus two back-substitutions
# gives the reported stress AND all 840 tile sensitivities. Scored against the
# full 1680-solve remesh FD map on this grid: sign 824/840, spearman +0.9997,
# ratio median 0.9985, sum rule x0.9953, 46 s. The 'axis' path scores
# 56/840 / -0.4716 on the same grid and must not be used here (see
# project_axis_gradient_fails_on_840_tiles).
#
# KS rho = 0.25, NOT the 0.15 most surrogate runs used. rho 0.15 plateaus near
# 410 even when ccx drives it; the 250/279 reference used 0.25 and the smallest
# rho that still reaches 279 is 0.20 (project_online_topopt_plateau).
#
# CORES. This does not scale the way the old FD sweep did. The adjoint is ONE
# scipy splu factorization (serial), the meshing is threaded, and ccx itself
# stops scaling past ~8 threads on this problem (179.6 s at 8, 164.2 s at 16,
# 167.1 s at 32 -- measured on this geometry). 64 cpus covers the mesh build,
# the screenshot solve and the calibration solve with headroom; asking for 180
# would idle most of them for a day and a half. Spare capacity is better spent
# on a CONCURRENT schedule variant -- set RUN_NAME and the schedule vars below
# and sbatch this file again; artifact dirs are per-run so the two cannot
# collide (project_newton2_nfs_rename_race).
#
# COST, measured on protagoras at 32 cpus, mesh (3,3,3), rho sweep job 101609:
#   adjoint gradient + stress      52 s     the floor
#   screenshot (real ccx solve)    20 s     every gen here (--screenshot_every 1)
#   calibration (real ccx solve)   24 s     every 5 gens
#   => ~76 s/gen  ->  800 gens ~= 17 h
# The screenshot is not waste: it re-solves the design with real ccx C3D8I
# independently of the adjoint, so the per-gen PNG doubles as a running check
# on the adjoint's reported stress.
#
# CALIBRATION OUTLIER GUARD (--calib_max_dev 2.0). The reference solve is a
# real ccx run and it occasionally returns a wildly wrong stress for a
# perfectly ordinary design. Job 48454 read 758.57 MPa for the uniform t=0.30
# design that measures 93.93; the first attempt at THIS run (job 183931) read
# 5186.26 for the one that measures 76.74, a ratio of 67.6, while every
# calibration after it sat at ~1.04. That single sample entered the EMA, and
# since calib multiplies BOTH s0 and dg_raw it pinned lambda at its 1e9 cap and
# thickened the whole design for the ~30 generations the EMA needed to decay
# back. The adjoint tracks ccx's C3D8I to 1.7e-5% and the bias this correction
# exists for is a few per cent, so any ratio outside [0.5, 2.0] is a bad solve,
# not a bias: it is now dropped with a REJECTED line in the log and the
# previous calib is kept.
#
# TOPOPT_CCX_REF is deliberately NOT set. It exists to overlay a ccx reference
# gradient on the surrogate's; here the gradient already IS ccx, so setting it
# would buy a second identical panel for 46 s a generation.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/"
export IRIT_TMP=/tmp/irit_$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- run identity
GEOM=${GEOM:-hollow_cube}                # geometry dir under data/
RUN_NAME=${RUN_NAME:-ccx_baseline_A}
ART=optimization/artifacts_${RUN_NAME}
mkdir -p "$ART/screenshots"
export TOPOPT_ARTIFACT_DIR="$ART" TOPOPT_SCREENSHOT_DIR="$ART/screenshots"

# ---------------------------------------------------------------- parameters
GENS=${GENS:-800}                       # total generations
RHO=${RHO:-0.25}                        # KS sharpness, 1/MPa
MOVE=${MOVE:-0.08}                      # OC move limit, fraction of span (0.46) -> 0.0368
ETA=${ETA:-0.5}                         # OC exponent
FILT_R=${FILT_R:-2.0}                   # design-filter radius, phase I
FILT_R_END=${FILT_R_END:-1.2}           # design-filter radius, phase IV
FILT_START=${FILT_START:-100}           # gen the radius decay begins
FILT_GENS=${FILT_GENS:-300}             # gens the decay spans (100 -> 400)
FILTER_TARGET=${FILTER_TARGET:-design}  # design | gradient | density | none
SHARPEN=${SHARPEN:-skeleton}            # grey-level skeleton flow
SH_START=${SH_START:-200}               # gen sharpening begins
SH_END=${SH_END:-500}                   # gen sharpening reaches full strength
SH_W=${SH_W:-0.05}                      # sharpening rate ceiling (compounds per gen)
TAU=${TAU:-0.03}                        # donor floor at SH_START
TAU_END=${TAU_END:-0.015}               # donor floor at SH_END
SHOT_EVERY=${SHOT_EVERY:-1}             # screenshot + gradient grid cadence
CALIB=${CALIB:-5}                       # ccx cross-check cadence
CALIB_DEV=${CALIB_DEV:-2.0}             # reject a calibration ratio outside [1/x, x] as a bad solve

export STRESS_KS_RHO_MPA=$RHO

# The radius-decay and sharpening flags only mean anything for
# filter_target=design or density (density: x = H y, sharpening acts on y). With "none" (no smoothing at all, the control that shows
# what the filter is actually buying) or "gradient" (the classic Sigmund
# sensitivity filter the 250/279 reference used) they are omitted entirely, so
# params.env and topopt_params.jsonl record "off" rather than values that were
# parsed and ignored.
if [ "$FILTER_TARGET" = "design" ] || [ "$FILTER_TARGET" = "density" ]; then
    SCHED_FLAGS="--filter_radius_end $FILT_R_END --filter_decay_start $FILT_START --filter_decay_gens $FILT_GENS"
    SCHED_FLAGS="$SCHED_FLAGS --sharpen_mode $SHARPEN --proj_start_gen $SH_START --proj_end_gen $SH_END"
    SCHED_FLAGS="$SCHED_FLAGS --proj_w_max $SH_W --skel_tau $TAU --skel_tau_end $TAU_END"
    SCHED_DESC="$FILTER_TARGET filter r $FILT_R -> $FILT_R_END over gens $FILT_START-$((FILT_START+FILT_GENS)); sharpening $SHARPEN w 0 -> $SH_W, tau $TAU -> $TAU_END over gens $SH_START-$SH_END"
else
    SCHED_FLAGS=""
    SCHED_DESC="filter_target=$FILTER_TARGET -- no design smoothing schedule, no sharpening"
fi

# The exact values this run used, next to its results. The optimizer also
# appends its own resolved view (including the per-tile move vector) to
# $ART/topopt_params.jsonl on every start and resume.
cat > "$ART/params.env" <<PARAMS
# $RUN_NAME -- ccx ground-truth topopt baseline
# job $SLURM_JOB_ID on $(hostname), $(date -Is)
RUN_NAME=$RUN_NAME
GEOM=$GEOM
ARCH=ccx
CCX_GRADIENT=adjoint
GENS=$GENS
RHO=$RHO
MOVE=$MOVE
ETA=$ETA
FILTER_TARGET=$FILTER_TARGET
FILT_R=$FILT_R
FILT_R_END=${SCHED_FLAGS:+$FILT_R_END}${SCHED_FLAGS:-off}
FILT_START=${SCHED_FLAGS:+$FILT_START}${SCHED_FLAGS:-off}
FILT_GENS=${SCHED_FLAGS:+$FILT_GENS}${SCHED_FLAGS:-off}
SHARPEN=${SCHED_FLAGS:+$SHARPEN}${SCHED_FLAGS:-off}
SH_START=${SCHED_FLAGS:+$SH_START}${SCHED_FLAGS:-off}
SH_END=${SCHED_FLAGS:+$SH_END}${SCHED_FLAGS:-off}
SH_W=${SCHED_FLAGS:+$SH_W}${SCHED_FLAGS:-off}
TAU=${SCHED_FLAGS:+$TAU}${SCHED_FLAGS:-off}
TAU_END=${SCHED_FLAGS:+$TAU_END}${SCHED_FLAGS:-off}
TRUST_REL_TOL=0.5
MIN_TRUST_FRAC=0.05
UNTRUSTED_MOVE_FRAC=0.1
TR_TOL=0.05
TR_MIN_SCALE=0.02
G_BACKOFF=${G_BACKOFF:-0.02}
CALIB_EVERY=$CALIB
CALIB_SMOOTH=0.5
CALIB_MAX_DEV=$CALIB_DEV
STAGNATION_GENS=0
SHOT_EVERY=$SHOT_EVERY
MESH_RES=default_3_3_3
INIT_THICKNESS=${INIT_T:-default_0.30}
PARAMS

echo "[job] $RUN_NAME geom=$GEOM art=$ART node=$(hostname) cpus=$SLURM_CPUS_PER_TASK gens=$GENS rho=$RHO move=$MOVE"
echo "[job] $SCHED_DESC"
echo "[job] screenshot + gradient grid every $SHOT_EVERY gen(s) -> $ART/screenshots"

# OPTIMIZE_MODULE=optimization.optimize_dims_legacy runs the pre-simplification
# optimizer (commit 7bf07d9) for comparison.
python -u -m ${OPTIMIZE_MODULE:-optimization.optimize_dims} \
    --geometry "$GEOM" --arch ccx --optimizer topopt --ccx_gradient adjoint \
    --generations "$GENS" --stress_ks_rho_mpa "$RHO" \
    --processes 2 --mesh_threads 16 --nproc 8 \
    --filter_target "$FILTER_TARGET" --filter_radius "$FILT_R" \
    $SCHED_FLAGS \
    --move_limit "$MOVE" --oc_eta "$ETA" --fd_step 0.04 \
    --trust_rel_tol 0.5 --min_trust_frac 0.05 --untrusted_move_frac 0.1 \
    --tr_tol 0.05 --tr_min_scale 0.02 --g_backoff "${G_BACKOFF:-0.02}" \
    --calib_every "$CALIB" --calib_smooth 0.5 --calib_max_dev "$CALIB_DEV" --stagnation_gens 0 \
    --screenshot_every "$SHOT_EVERY" ${INIT_T:+--init_thickness "$INIT_T"}

echo "TOPOPT_DONE rc=$?"
rm -rf "$IRIT_TMP"
rm -rf data/hollow_cube/Instance_* data/hollow_cube/ansys_* 2>/dev/null
true
