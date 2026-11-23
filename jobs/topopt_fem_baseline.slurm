#!/bin/bash
#SBATCH --job-name=topopt_fem_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --partition=public
#SBATCH --nodelist=protagoras
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#
# EXACT-SOLVER REFERENCE for the cross4 surrogate topopt runs: the same
# optimizer and physics as jobs/topopt_gnn_dsig_ft.slurm, with the surrogate
# replaced by core/fem_adjoint.py's exact shape adjoint (--arch fem). Whatever
# this reaches is what the FT runs would reach with a perfect surrogate.
#
# Defaults = the cross4_freeload_row{0,5,10}_tmin003_F5e7_m442_mma reference
# recipe: pyramid tile_nodes load 5e7 N on FEA_LOAD_ROW, no load pad, anchored
# pad 0.1, mesh 4,4,2, avg legs, KS rho 0.25 over non-pad nodes, MMA, move
# 0.08, t0 0.2, tile floor 0.03, no design filter, 200 gens.
#   RUN_NAME=... FEA_LOAD_ROW=10 sbatch jobs/topopt_fem_baseline.slurm
#
# Other geometries / the old hollow_cube schedule: override GEOM and the knobs
# below (FILTER_TARGET=design enables the radius-decay + skeleton-sharpening
# schedule flags). The cross4 physics env vars are only exported for cross4.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$PWD/irit/bin:$PATH"
export IRIT_PATH="$PWD/irit/bin/" PYTHONUNBUFFERED=1
export IRIT_TMP=/tmp/irit_$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

# ---------------------------------------------------------------- run identity
GEOM=${GEOM:-cross4_11x11}               # geometry dir under data/
RUN_NAME=${RUN_NAME:-fem_${GEOM}_$SLURM_JOB_ID}
ART=optimization/artifacts_${RUN_NAME}
mkdir -p "$ART/screenshots"
export TOPOPT_ARTIFACT_DIR="$ART" TOPOPT_SCREENSHOT_DIR="$ART/screenshots"

# ---------------------------------------------------------------- physics (cross4)
if [[ "$GEOM" == cross4_* ]]; then
    export CROSS4_LEG_RULE=avg CROSS4_FIXED_END_RADIUS=0 FEA_LOAD_MODE=tile_nodes FEA_LOAD_WEIGHT=pyramid
    export CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0 FEA_MESH_RES=${FEA_MESH_RES:-4,4,2}
    export FEA_LOAD_N=${FEA_LOAD_N:-5e7} FEA_LOAD_ROW=${FEA_LOAD_ROW:-5} TOPOPT_T_MIN=${T_MIN:-0.03}
    export TOPOPT_KS_EXCLUDE=pad TOPOPT_UPDATE=${UPDATE:-mma} TOPOPT_MMA_ASY_INIT=0.1
    [ -n "$T_MAX" ] && export TOPOPT_T_MAX=$T_MAX
fi

# ---------------------------------------------------------------- optimizer
GENS=${GENS:-200}                       # generations (additional, on resume)
RHO=${RHO:-0.25}                        # KS sharpness, 1/MPa
MOVE=${MOVE:-0.08}                      # move limit, fraction of each tile's span
ETA=${ETA:-0.5}                         # OC exponent (unused by MMA)
G_BACKOFF=${G_BACKOFF:-0.005}           # constraint back-off
INIT_T=${INIT_T:-0.2}                   # uniform starting thickness
SHOT_EVERY=${SHOT_EVERY:-10}            # screenshot + gradient grid cadence
FILTER_TARGET=${FILTER_TARGET:-none}    # none | gradient | design | density
FILT_R=${FILT_R:-1.5}                   # filter radius (tile units)
export STRESS_KS_RHO_MPA=$RHO

# radius-decay + skeleton sharpening schedule: only meaningful for design/density
SCHED_FLAGS=""
if [ "$FILTER_TARGET" = "design" ] || [ "$FILTER_TARGET" = "density" ]; then
    SCHED_FLAGS="--filter_radius_end ${FILT_R_END:-1.2} --filter_decay_start ${FILT_START:-100} --filter_decay_gens ${FILT_GENS:-300}"
    SCHED_FLAGS="$SCHED_FLAGS --sharpen_mode ${SHARPEN:-skeleton} --proj_start_gen ${SH_START:-200} --proj_end_gen ${SH_END:-500}"
    SCHED_FLAGS="$SCHED_FLAGS --proj_w_max ${SH_W:-0.05} --skel_tau ${TAU:-0.03} --skel_tau_end ${TAU_END:-0.015}"
fi

# the exact values this run used, next to its results
{ env | grep -E "^(CROSS4|FEA_|TOPOPT_|STRESS_)" | sort
  echo "GEOM=$GEOM ARCH=fem GENS=$GENS RHO=$RHO MOVE=$MOVE ETA=$ETA G_BACKOFF=$G_BACKOFF INIT_T=$INIT_T FILTER_TARGET=$FILTER_TARGET FILT_R=$FILT_R SCHED_FLAGS='$SCHED_FLAGS' job=$SLURM_JOB_ID host=$(hostname)"
} > "$ART/params.env"
cat "$ART/params.env"

python -u -m optimization.optimize_dims \
    --geometry "$GEOM" --arch fem \
    --generations "$GENS" --stress_ks_rho_mpa "$RHO" \
    --processes 2 --mesh_threads 8 --nproc 8 \
    --filter_target "$FILTER_TARGET" --filter_radius "$FILT_R" $SCHED_FLAGS \
    --move_limit "$MOVE" --oc_eta "$ETA" --g_backoff "$G_BACKOFF" \
    --stagnation_gens 0 --screenshot_every "$SHOT_EVERY" --init_thickness "$INIT_T"

echo "TOPOPT_DONE rc=$?"
rm -rf "$IRIT_TMP"
true
