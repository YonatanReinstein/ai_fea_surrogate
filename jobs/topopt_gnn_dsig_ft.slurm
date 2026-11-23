#!/bin/bash
#SBATCH --job-name=topopt_dsig_ft
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton2
#
# Surrogate topopt on cross4_11x11 with the superposing stress model
# (evaluators/gnn_dsig_evaluator.py, training/train_dsig.py checkpoints) and
# synchronous on-trajectory fine-tuning (training/dsig_online.py): every FT_EVERY
# gens the current design + (FT_SAMPLES-1) jittered copies are labelled exactly
# (S and dS/dt), the model takes FT_STEPS AdamW steps on the pretraining loss
# mixed with dataset replay, and a hold-out gate discards updates that hurt it.
# Physics and optimizer recipe = the ccx reference runs
# cross4_freeload_row{0,5,10}_tmin003_F5e7_m442_mma (jobs/topopt_ccx_baseline.slurm):
# pyramid tile_nodes load 5e7 N on FEA_LOAD_ROW, no load pad, mesh 4,4,2, avg
# legs, KS rho 0.25 over non-pad nodes (raw KS <= yield), MMA move 0.08, t0 0.2,
# tile floor 0.03. DSIG_LOAD_WEIGHTS / DSIG_COMBINE pass through for combinations.
#   CKPT=... RUN_NAME=... sbatch jobs/topopt_gnn_dsig_ft.slurm
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/" PYTHONUNBUFFERED=1
export IRIT_TMP=/tmp/irit_$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CROSS4_LEG_RULE=avg CROSS4_FIXED_END_RADIUS=0 FEA_LOAD_MODE=tile_nodes FEA_LOAD_WEIGHT=pyramid
export CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0 FEA_MESH_RES=4,4,2
export FEA_LOAD_N=${FEA_LOAD_N:-5e7} FEA_LOAD_ROW=${FEA_LOAD_ROW:-5} TOPOPT_T_MIN=${T_MIN:-0.03}
export TOPOPT_KS_EXCLUDE=pad TOPOPT_UPDATE=mma TOPOPT_MMA_ASY_INIT=0.1 TOPOPT_TILE_JACOBIAN=colored
[ -n "$T_MAX" ] && export TOPOPT_T_MAX=$T_MAX
export STRESS_KS_RHO_MPA=${RHO:-0.25}
export DSIG_FT=${FT:-1} DSIG_FT_EVERY=${FT_EVERY:-1} DSIG_FT_SAMPLES=${FT_SAMPLES:-4} DSIG_FT_JITTER=${FT_JITTER:-0.03}
export DSIG_FT_STEPS=${FT_STEPS:-20} DSIG_FT_BATCH=${FT_BATCH:-4} DSIG_FT_LR=${FT_LR:-5e-5}
export DSIG_FT_REPLAY_FRAC=${FT_REPLAY_FRAC:-0.5} DSIG_FT_BUFFER=${FT_BUFFER:-40}
export DSIG_FT_GATE=${FT_GATE:-0.10} DSIG_FT_GATE_COS=${FT_GATE_COS:-0.05}
[ -n "$FT_GATE_KS_ABS" ] && export DSIG_FT_GATE_KS_ABS=$FT_GATE_KS_ABS
export DSIG_FT_LABEL_THREADS=${FT_LABEL_THREADS:-2} DSIG_FT_JAC_WORKERS=${FT_JAC_WORKERS:-6}
export DSIG_FT_REPLAY_DESIGNS=${FT_REPLAY_DESIGNS:-48} DSIG_FT_HOLDOUT_DESIGNS=${FT_HOLDOUT_DESIGNS:-12}
[ -n "$DSIG_LOAD_WEIGHTS" ] && export DSIG_LOAD_WEIGHTS
[ -n "$DSIG_COMBINE" ] && export DSIG_COMBINE
CKPT=${CKPT:-training/runs/cross4_dsig_sobolev_v1_ep9/opt_ckpt.pt}
RUN_NAME=${RUN_NAME:-dsig_ft_$SLURM_JOB_ID}
ART=optimization/artifacts_${RUN_NAME}
mkdir -p "$ART/screenshots"
export TOPOPT_ARTIFACT_DIR="$ART" TOPOPT_SCREENSHOT_DIR="$ART/screenshots"
env | grep -E "^(CROSS4|FEA_|TOPOPT_|STRESS_|DSIG)" | sort > "$ART/params.env"
echo "CKPT=$CKPT GENS=${GENS:-200} MOVE=${MOVE:-0.08} INIT_T=${INIT_T:-0.2} RHO=${RHO:-0.25} G_BACKOFF=${G_BACKOFF:-0.005} job=$SLURM_JOB_ID host=$(hostname)" >> "$ART/params.env"
cat "$ART/params.env"
python -u -m optimization.optimize_dims \
    --geometry cross4_11x11 --arch gnn_dsig --gnn_ckpt_path "$CKPT" --optimizer topopt \
    --generations "${GENS:-200}" --stress_ks_rho_mpa "${RHO:-0.25}" --mesh_threads 8 \
    --filter_target none --move_limit "${MOVE:-0.08}" --oc_eta 0.5 --g_backoff "${G_BACKOFF:-0.005}" \
    --stagnation_gens 0 --screenshot_every "${SHOT_EVERY:-10}" --init_thickness "${INIT_T:-0.2}"
echo "TOPOPT_DONE rc=$?"
