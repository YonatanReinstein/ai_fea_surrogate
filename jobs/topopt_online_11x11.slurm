#!/bin/bash
#SBATCH --job-name=topopt_online_11x11
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=40
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton2
#
# Surrogate-driven topopt with DENSE online fine-tuning: the configuration that
# held the surrogate gradient at ccx-adjoint quality on-trajectory.
#
# WHY DENSE. Every sparse configuration collapsed: no learning ~+0.10 cosine
# vs the ccx adjoint by gen 20-30; the best sparse schedule (1 label / 3 gens,
# 100 steps @ 2e-4) plateaued at +0.65 over 60 gens and +0.39 over 300. Ten
# labels per generation -- the current design plus nine perturbations at 5%
# jitter -- with an update on EVERY label held +0.98..+0.998 through gen 35
# (wrong-way mass 1-4%, 222 updates, 0 gate rollbacks) and produced the first
# FEASIBLE surrogate design: real ccx 97.77 MPa vs surrogate 98.58, a 0.8%
# error where every earlier run was 8-15% under-predicted. So +0.65 was a
# LABEL-BUDGET ceiling, not a learning ceiling.
#
# WHY THIS MANY CPUS. A ccx adjoint label costs ~45s and is CPU-bound; a
# generation is ~5s. On the 10-CPU session (3 labellers) 10 labels/gen forced
# the optimiser to wait ~150s/gen -- 3x SLOWER than plain ccx. With 12
# labellers here throughput is ~10 labels per ~37s, so a generation runs at
# roughly ccx parity while the surrogate keeps improving. Backpressure
# (TOPOPT_LABEL_MAXQ) keeps the optimiser from outrunning the labellers,
# which is what made labels stale in the sparse runs.
#
# NOT newton2: its X550T still links at 100baseT (measured 11 MB/s NFS today).
# This job is light on the dataset but a dozen ccx solvers writing .frd/.dat
# to the NFS home could saturate that link.
#
# DETERMINISTIC LAGGED PIPELINE (LABEL_LAG, LABEL_SAMPLES) (a test, not a production schedule). Gen i
# blocks until all 10 samples spooled at gen i-LAG are labelled and the trainer
# has published weights newer than those labels; the trainer bundles exactly
# those 10 into ONE update (--update_every 10). So every generation is trained
# on exactly one update from exactly LAG generations back -- reproducible,
# and it isolates 'does lagged learning hold the gradient' from throughput
# effects. LAG-8 / 10-SAMPLE RESULT (job 1388590): topology emerged for the
# first time from a surrogate run (13.8% voided, recognisable truss) but the
# optimiser stalled at vol 411 ~gen 180 during a gradient trough (cosine
# 0.14-0.64 at gens 159-219) and drifted to 467; ccx reaches 255. Rate was
# 37s/gen, trainer-bound (10 mesh rebuilds per update). Hence fewer samples
# per generation and a shorter lag.
#
# Earlier note on lag 8 vs 5: with ~38 labellers across two nodes labels
# arrive in BURSTS, and at lag 5 gen i-5 was occasionally not yet complete
# (waits of 54s / 142s). Three more generations of slack absorbs that.
# It runs at label speed (~10 labels / ~100s with 10 contended
# labellers), so ~100s/gen; that is the accepted cost of the test.
#
# FILTER_TARGET=density + PROJ_ETA: density filter with the chain rule and a
# gated threshold projection (see TopologyOptimizer.__init__). Run I:
# radius 2 constant to gen 100 then 2 -> 1.5 by gen 300 (never identity),
# beta 0 -> 32 over gens 100-300, eta 0.12. Replaces the radius-to-identity
# decay of runs G/H, which destabilised the constraint the moment the blur
# was gone and left a grey web either way.
#
# Screenshots every 10 gens with TOPOPT_CCX_REF=1, so each carries the
# per-tile gradient heatmap and the surrogate-vs-ccx-adjoint difference panel.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

GEOM=${GEOM:-hollow_cube}
# Fine-tuner losses match training/gnn_training.py (jobs/train_11x11_screened.slurm):
# 0.25 each of KS percent error, per-tile gradient percent error and sign loss, each
# divided by its know-nothing baseline. Replay/holdout default to the screened dataset
# (data/$GEOM/replay_screened = train shards 0-2, audit_screened = val shard 29900).
RUN=onpolicy/${RUN_NAME:-dense_$SLURM_JOB_ID}
SPOOL=$RUN/spool; LABELS=$RUN/labels; LIVE=$RUN/live
ART=optimization/artifacts_${RUN_NAME:-dense_$SLURM_JOB_ID}
mkdir -p "$SPOOL" "$LABELS" "$LIVE" "$ART" optimization/screenshots
# LAB_THREADS: torch threads per labeller for the adjoint backward (PARDISO
# stays sequential). Measured on hollow_cube, one design: 1 -> 21.3 s/label,
# 2 -> 16.7 s, 4 -> 14.9 s, 8 -> 13.2 s, results identical to 3e-11. torch
# reads MKL_NUM_THREADS as well as OMP_NUM_THREADS, so both are set.
LAB_THREADS=${LAB_THREADS:-2}
# UPDATE_EVERY: labels per trainer update (default LABEL_SAMPLES = one update per
# generation). 2*LABEL_SAMPLES trains once per two generations at the same
# --steps, i.e. half the training per generation.
NLAB=${NLAB:-10}   # labellers; each uses ~4 threads (2 workers x 2 mesh threads). Override cpus with `sbatch -c <4*NLAB+8>`.
BASE=${BASE_CKPT:-data/hollow_cube/checkpoints_sobolev_lowlr/epoch_008.pt}
# RESUME=1: continue a run in place. Everything a stop/continue needs is on disk:
#   $ART/topopt_checkpoint.pkl   design, best, V_ref, gen, trust-region state (move_scale,
#                                previous prediction, re-anchored under the resumed weights),
#                                calibrations, spooled-gen set, screenshot flag
#   $ART/surrogate_loaded.pt     the weights the optimiser was serving at that checkpoint
#   $ART/topopt_params.jsonl     schedule params of every segment (appended per start)
#   $LIVE/trainer_state.pt       trainer Adam moments, update count, EMA, ORIGINAL base gates
#   $LABELS/.consumed_upto       labels already trained on (trainer --skip_existing)
# Schedule params (filter decay, sharpening, gens) are CLI/env and gen-indexed, so a
# resume may change them; STOP_AT=<abs gen> ends the run at that generation regardless
# of GENS (which counts from the resumed gen). Spool, labels, screenshots accumulate.
TRAINER_EXTRA="--state_path $LIVE/trainer_state.pt"
# The hot-swapped weights live on node-local /tmp: the trainer writes and the
# optimiser reads them every generation, and on newton4's NFS that write alone
# was 2 s of a ~21 s update. The resume files (trainer_state.pt, the optimiser's
# surrogate_loaded.pt) stay on NFS; the final live weights are copied back to
# $LIVE on exit so the RESUME fallback below still finds them.
LIVE_CKPT=/tmp/topopt_live_$SLURM_JOB_ID/surrogate_live.pt
mkdir -p "$(dirname "$LIVE_CKPT")"
trap 'cp -f "$LIVE_CKPT" "$LIVE/surrogate_live.pt" 2>/dev/null; rm -rf "$(dirname "$LIVE_CKPT")"' EXIT
if [ "${RESUME:-0}" = "1" ]; then
  # Prefer the weights the optimiser was actually serving when it wrote its
  # last checkpoint (evaluators/gnn_evaluator.py snapshots them on every
  # reload); fall back to the trainer's live file for runs predating that.
  if [ -f "$ART/surrogate_loaded.pt" ]; then
    cp "$ART/surrogate_loaded.pt" "$LIVE/surrogate_resume_$SLURM_JOB_ID.pt" && BASE="$LIVE/surrogate_resume_$SLURM_JOB_ID.pt"
  elif [ -f "$LIVE/surrogate_live.pt" ]; then
    cp "$LIVE/surrogate_live.pt" "$LIVE/surrogate_resume_$SLURM_JOB_ID.pt" && BASE="$LIVE/surrogate_resume_$SLURM_JOB_ID.pt"
  fi
  TRAINER_EXTRA="$TRAINER_EXTRA --skip_existing"
  for f in "$SPOOL"/*.claimed.*npz; do [ -e "$f" ] && mv "$f" "${f%%.claimed.*}.npz"; done
  echo "[job] RESUME from $BASE, topopt checkpoint: $(ls -la "$ART/topopt_checkpoint.pkl" 2>/dev/null | awk '{print $6,$7,$8}')"
fi
export IRIT_TMP=/tmp/irit_$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p "$IRIT_TMP"
echo "[job] replay_frac=${REPLAY_FRAC:-0.8} ks_w=${KS_W:-0.25} base=$BASE lag=${LABEL_LAG:-8} samples=${LABEL_SAMPLES:-10} move_limit=${MOVE_LIMIT:-0.05} rho=${RHO:-0.15} stress_gate_abs=${STRESS_GATE_ABS:-0.03} dg_calib=${DG_CALIB:-0} filter_target=${FILTER_TARGET:-gradient} filter_end=${FILTER_END:-const} filter_decay=${FILTER_DECAY:-0} filter_decay_start=${FILTER_DECAY_START:-0} proj_eta=${PROJ_ETA:-off} proj_beta=${PROJ_BETA:-32} proj_gens=${PROJ_START:-100}-${PROJ_END:-300} proj_wmax=${PROJ_WMAX:-1.0} sharpen_mode=${SHARPEN_MODE:-gate} unsharp_r=${UNSHARP_R:-design} skel_tau=${SKEL_TAU:-0.03}->${SKEL_TAU_END:-const} steps=${TRAIN_STEPS:-100} batch=${TRAIN_BATCH:-4} traj_buf=${TRAJ_BUFFER:-0} val_every=${VAL_EVERY:-0} node_emb_lr=${NODE_EMB_LR_SCALE:-1} anchor=${ANCHOR_L2:-0} jitter=${LABEL_JITTER:-0.05}/dir${LABEL_JITTER_DIR:-0} nlab=${NLAB:-10} lab_threads=${LAB_THREADS:-2} train_gpus=${TRAIN_GPUS:-1} update_every=${UPDATE_EVERY:-${LABEL_SAMPLES:-10}} trainer_poll=${TRAINER_POLL:-1} shot_every=${SHOT_EVERY:-10}"

echo "[job] $NLAB labellers on $SLURM_CPUS_PER_TASK cpus ($(hostname)), spool=$SPOOL"
PIDS=()
for i in $(seq 1 $NLAB); do
  OMP_NUM_THREADS=$LAB_THREADS MKL_NUM_THREADS=$LAB_THREADS \
  python -u -m training.trajectory_labeller --geometry "$GEOM" --spool "$SPOOL" --labels "$LABELS" \
      --workers 2 --nproc 1 --mesh_threads 2 --claim_order fifo --max_lag 0 --idle_exit 1800 \
      --stress_ks_rho_mpa "${RHO:-0.15}" >> "$RUN/labeller_$i.log" 2>&1 &
  PIDS+=($!)
done
sleep 3
# TRAIN_GPUS=N trains data-parallel over N GPUs (submit with --gres=gpu:N): each
# update's batches are split across them, ~N x faster on the trainer, which is
# the pipeline's bottleneck once labels are fast. The optimiser stays on GPU 0.
# With MORE GPUs than TRAIN_GPUS (e.g. --gres=gpu:3, TRAIN_GPUS=2) the optimiser
# gets the first spare GPU to itself. Sharing GPU 0 with trainer rank 0 slowed
# its surrogate evaluation from ~3.6 s to 4.4-6 s per generation (job 1433768).
IFS=, read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
TRAINER_GPUS_ENV=""; OPT_GPUS_ENV=""
if [ "${#_GPUS[@]}" -gt "${TRAIN_GPUS:-1}" ]; then
  TRAINER_GPUS_ENV=$(IFS=,; echo "${_GPUS[*]:0:${TRAIN_GPUS:-1}}")
  OPT_GPUS_ENV=${_GPUS[${TRAIN_GPUS:-1}]}
  echo "[job] GPUs: trainer on $TRAINER_GPUS_ENV, optimiser on $OPT_GPUS_ENV"
fi
if [ "${TRAIN_GPUS:-1}" -gt 1 ]; then
  TRAINER_LAUNCH="torchrun --standalone --nproc_per_node ${TRAIN_GPUS} -m training.online_finetune"
else
  TRAINER_LAUNCH="python -u -m training.online_finetune"
fi
CUDA_VISIBLE_DEVICES=${TRAINER_GPUS_ENV:-$CUDA_VISIBLE_DEVICES} $TRAINER_LAUNCH --geometry "$GEOM" --replay_dataset data/$GEOM/${REPLAY:-replay_screened}/dataset.pt --holdout_dataset data/$GEOM/${HOLDOUT:-audit_screened}/dataset.pt --labels "$LABELS" --base_ckpt "$BASE" \
    --live_ckpt "$LIVE_CKPT" --lr 2e-4 --update_every "${UPDATE_EVERY:-${LABEL_SAMPLES:-10}}" --steps "${TRAIN_STEPS:-100}" --batch_size "${TRAIN_BATCH:-4}" ${FREEZE:+--freeze "$FREEZE"} ${TRAJ_BUFFER:+--traj_buffer $TRAJ_BUFFER} ${VAL_EVERY:+--val_every $VAL_EVERY} ${VAL_GATE:+--val_gate} ${NODE_EMB_LR_SCALE:+--node_emb_lr_scale $NODE_EMB_LR_SCALE} ${ANCHOR_L2:+--anchor_l2 $ANCHOR_L2} ${ANCHOR_ALL:+--anchor_all} \
    --replay_frac "${REPLAY_FRAC:-0.8}" --holdout_n 24 --max_stress_drift 0.01 \
    --graph_loss_weight "${KS_W:-0.25}" --grad_loss_weight 0.25 --grad_sign_loss_weight 0.25 \
    --grad_weight_power 0.15 --grad_rel_sat 2.0 --grad_sign_delta 0.05 --max_stress_abs "${STRESS_GATE_ABS:-0.03}" --idle_exit 3600 --poll "${TRAINER_POLL:-1}" $TRAINER_EXTRA >> "$RUN/trainer.log" 2>&1 &
PIDS+=($!)
sleep 5
echo "[job] trainer + labellers up, starting topopt"

export TOPOPT_ARTIFACT_DIR="$ART" TOPOPT_SCREENSHOT_DIR="$ART/screenshots"
export TOPOPT_LABEL_SPOOL="$SPOOL"
export TOPOPT_LABEL_EVERY=1
export TOPOPT_LABEL_SAMPLES="${LABEL_SAMPLES:-10}"
export TOPOPT_LABEL_JITTER="${LABEL_JITTER:-0.05}"
export TOPOPT_LABEL_JITTER_DIR="${LABEL_JITTER_DIR:-0}"
export TOPOPT_LABEL_MAXQ=0
export TOPOPT_LABEL_LAG="${LABEL_LAG:-8}"   # gen i blocks until gen i-LAG is labelled AND trained on
export TOPOPT_LABEL_DIR="$LABELS"
export GNN_LIVE_CKPT="$LIVE_CKPT"
export TOPOPT_CCX_REF=1
export TOPOPT_DG_CALIB="${DG_CALIB:-0}"   # 1 = scale dg_raw by smoothed |ccx|/|surrogate| from the reference solves
export STRESS_KS_RHO_MPA="${RHO:-0.15}"

CUDA_VISIBLE_DEVICES=${OPT_GPUS_ENV:-$CUDA_VISIBLE_DEVICES} python -u -m optimization.optimize_dims --geometry "$GEOM" --arch gnn --optimizer topopt \
    --gnn_ckpt_path "$BASE" \
    --generations "${GENS:-300}" ${STOP_AT:+--stop_at_gen $STOP_AT} --screenshot_every "${SHOT_EVERY:-10}" --processes 6 --batch_size 4 \
    --filter_radius 2 --filter_target "${FILTER_TARGET:-gradient}" ${FILTER_END:+--filter_radius_end $FILTER_END --filter_decay_gens ${FILTER_DECAY:-150}} ${FILTER_DECAY_START:+--filter_decay_start $FILTER_DECAY_START} \
    ${PROJ_ETA:+--proj_eta $PROJ_ETA --proj_beta_end ${PROJ_BETA:-32} --proj_start_gen ${PROJ_START:-100} --proj_end_gen ${PROJ_END:-300} --proj_w_max ${PROJ_WMAX:-1.0}} \
    ${SHARPEN_MODE:+--sharpen_mode $SHARPEN_MODE --proj_start_gen ${PROJ_START:-100} --proj_end_gen ${PROJ_END:-300} --proj_w_max ${PROJ_WMAX:-1.0}} ${UNSHARP_R:+--unsharp_radius $UNSHARP_R} ${SKEL_TAU:+--skel_tau $SKEL_TAU} ${SKEL_TAU_END:+--skel_tau_end $SKEL_TAU_END} \
    --move_limit "${MOVE_LIMIT:-0.05}" --fd_step 0.04 --min_trust_frac 0.05 \
    --untrusted_move_frac 0.1 --stress_ks_rho_mpa "${RHO:-0.15}"
RC=$?
echo "TOPOPT_DONE rc=$RC"
sleep 120                       # let the last labels and update land
kill "${PIDS[@]}" 2>/dev/null
echo "[job] final on-trajectory cosine (real designs):"
cat "$RUN"/labeller_*.log | grep -a "^\[labeller\] gen_[0-9]*:" \
  | sed -E 's/.*gen_0*([0-9]+): [0-9.]+s.*cosine ([+-][0-9.]+).*mass ([0-9.]+)%.*/\1 \2 \3/' | sort -n \
  | awk 'NR%25==0 {printf "  gen %3s cos %s wrong %s%%\n",$1,$2,$3}'
grep -a "published\|DISCARDED" "$RUN/trainer.log" | tail -1
python -u -m optimization.plot_trajectory_accuracy --labels "$LABELS" --out "$ART/trajectory_accuracy.png" \
    --title "${RUN_NAME:-dense_$SLURM_JOB_ID} (base $BASE, job $SLURM_JOB_ID)"
true
