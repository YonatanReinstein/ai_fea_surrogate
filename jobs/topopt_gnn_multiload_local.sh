#!/bin/bash
# Surrogate topopt with the multi-load stress-tensor model
# (evaluators/gnn_multiload_evaluator.py), run directly on a GPU node:
#   CKPT=training/runs/multiload_v1_frozen/opt_ckpt.pt RUN_NAME=... bash jobs/topopt_gnn_multiload_local.sh
# Recipe = the last ccx cross4 runs (MMA, move 0.08, KS rho 0.25 smooth, t0 0.2,
# no filter), physics = the multi-load dataset's (load on all nodes of the
# middle free-end tile, 1e7 N, loaded tile excluded from the KS).
cd "$(dirname "$0")/.."
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CROSS4_LEG_RULE=avg FEA_LOAD_MODE=tile_nodes CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0
export FEA_MESH_RES=6,6,4 FEA_LOAD_N=${FEA_LOAD_N:-1e7} FEA_LOAD_ROW=${FEA_LOAD_ROW:-5} MULTILOAD_DIR=${MULTILOAD_DIR:-0}
export TOPOPT_KS_EXCLUDE=load_tile TOPOPT_UPDATE=mma TOPOPT_MMA_ASY_INIT=0.1 TOPOPT_TILE_JACOBIAN=colored
export MULTILOAD_CCX_CHECK_EVERY=${CCX_CHECK_EVERY:-5}
RUN_NAME=${RUN_NAME:-gnnml_row5_F1e7_t20_mma}
ART=optimization/artifacts_${RUN_NAME}
mkdir -p "$ART/screenshots"
export TOPOPT_ARTIFACT_DIR="$ART" TOPOPT_SCREENSHOT_DIR="$ART/screenshots"
env | grep -E "^(CROSS4|FEA_|TOPOPT_|STRESS_|MULTILOAD)" | sort > "$ART/params.env"
echo "CKPT=$CKPT GENS=${GENS:-200} MOVE=${MOVE:-0.08} INIT_T=${INIT_T:-0.2} RHO=${RHO:-0.25} G_BACKOFF=${G_BACKOFF:-0.005}" >> "$ART/params.env"
python -u -m optimization.optimize_dims \
    --geometry cross4_11x11 --arch gnn_multiload --gnn_ckpt_path "$CKPT" --optimizer topopt \
    --generations "${GENS:-200}" --stress_ks_rho_mpa "${RHO:-0.25}" --mesh_threads 8 \
    --filter_target none --move_limit "${MOVE:-0.08}" --oc_eta 0.5 --g_backoff "${G_BACKOFF:-0.005}" \
    --stagnation_gens 0 --screenshot_every "${SHOT_EVERY:-10}" --init_thickness "${INIT_T:-0.2}"
echo "TOPOPT_DONE rc=$?"
