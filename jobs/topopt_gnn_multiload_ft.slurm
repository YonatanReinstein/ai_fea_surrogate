#!/bin/bash
#SBATCH --job-name=topopt_gnnml_ft
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton2
#
# Surrogate topopt on cross4_11x11 with the multi-load stress-tensor model
# (evaluators/gnn_multiload_evaluator.py) and synchronous on-trajectory
# fine-tuning (training/multiload_online.py): every FT_EVERY gens the current
# design + (FT_SAMPLES-1) jittered copies are labelled exactly with the dataset
# builder's solver, the model takes FT_STEPS AdamW steps on them mixed with
# dataset replay, and a hold-out gate on the pretraining validation designs
# discards updates that hurt it. Same recipe/physics as
# jobs/topopt_gnn_multiload_local.sh (MMA, move 0.08, KS rho 0.25 smooth,
# t0 0.2, no filter; middle free-end tile, tile_nodes load, 1e7 N, loaded tile
# excluded from the KS).
#   CKPT=... RUN_NAME=... sbatch jobs/topopt_gnn_multiload_ft.slurm
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/" PYTHONUNBUFFERED=1
export IRIT_TMP=/tmp/irit_$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CROSS4_LEG_RULE=avg FEA_LOAD_MODE=tile_nodes CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0
export FEA_MESH_RES=6,6,4 FEA_LOAD_N=${FEA_LOAD_N:-1e7} FEA_LOAD_ROW=${FEA_LOAD_ROW:-5} MULTILOAD_DIR=${MULTILOAD_DIR:-0}
export TOPOPT_KS_EXCLUDE=load_tile TOPOPT_UPDATE=mma TOPOPT_MMA_ASY_INIT=0.1 TOPOPT_TILE_JACOBIAN=colored
export MULTILOAD_CCX_CHECK_EVERY=${CCX_CHECK_EVERY:-5}
export MULTILOAD_FT=1 MULTILOAD_FT_EVERY=${FT_EVERY:-1} MULTILOAD_FT_SAMPLES=${FT_SAMPLES:-4}
export MULTILOAD_FT_JITTER=${FT_JITTER:-0.05} MULTILOAD_FT_STEPS=${FT_STEPS:-20} MULTILOAD_FT_BATCH=${FT_BATCH:-4}
export MULTILOAD_FT_LR=${FT_LR:-1e-4} MULTILOAD_FT_REPLAY_FRAC=${FT_REPLAY_FRAC:-0.5} MULTILOAD_FT_BUFFER=${FT_BUFFER:-40}
export MULTILOAD_FT_GATE=${FT_GATE:-0.10} MULTILOAD_FT_LABEL_THREADS=${FT_LABEL_THREADS:-4}
export MULTILOAD_FT_REPLAY_DESIGNS=${FT_REPLAY_DESIGNS:-48} MULTILOAD_FT_HOLDOUT_DESIGNS=${FT_HOLDOUT_DESIGNS:-12}
CKPT=${CKPT:-training/runs/multiload_v1_ep7/opt_ckpt.pt}
RUN_NAME=${RUN_NAME:-gnnml_ft_$SLURM_JOB_ID}
ART=optimization/artifacts_${RUN_NAME}
mkdir -p "$ART/screenshots"
export TOPOPT_ARTIFACT_DIR="$ART" TOPOPT_SCREENSHOT_DIR="$ART/screenshots"
env | grep -E "^(CROSS4|FEA_|TOPOPT_|STRESS_|MULTILOAD)" | sort > "$ART/params.env"
echo "CKPT=$CKPT GENS=${GENS:-200} MOVE=${MOVE:-0.08} INIT_T=${INIT_T:-0.2} RHO=${RHO:-0.25} G_BACKOFF=${G_BACKOFF:-0.005} job=$SLURM_JOB_ID host=$(hostname)" >> "$ART/params.env"
cat "$ART/params.env"
python -u -m optimization.optimize_dims \
    --geometry cross4_11x11 --arch gnn_multiload --gnn_ckpt_path "$CKPT" --optimizer topopt \
    --generations "${GENS:-200}" --stress_ks_rho_mpa "${RHO:-0.25}" --mesh_threads 8 \
    --filter_target none --move_limit "${MOVE:-0.08}" --oc_eta 0.5 --g_backoff "${G_BACKOFF:-0.005}" \
    --stagnation_gens 0 --screenshot_every "${SHOT_EVERY:-10}" --init_thickness "${INIT_T:-0.2}"
echo "TOPOPT_DONE rc=$?"
