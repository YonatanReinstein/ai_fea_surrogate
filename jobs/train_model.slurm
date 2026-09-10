#!/bin/bash
#SBATCH --job-name=train_model_ccx_gnn_probe_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=650G
#SBATCH --time=24:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A40:2
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
# Diagnostic: force synchronous CUDA execution so a crash traceback points at
# the actual faulting kernel launch instead of surfacing later at whatever
# unrelated .item()/sync call happens to run next. Slower, but needed to
# pin down the epoch-172-after-resume crash (reproduced across seed_offset
# 42/142, A6000/A40/H200, compile on/off -- data confirmed clean by a full
# 49900-sample scan). Remove once root-caused.
export CUDA_LAUNCH_BLOCKING=1

# Diagnostic: the epoch-2 hang seen on the A40:2 wp0.5_rho0.15_nw0.5_gw0.5_ns1000
# probe (rank0 asleep in futex_wait_queue_me, rank1 spinning at 100% CPU with
# wchan=0 -- the signature of two ranks stuck on mismatched NCCL collectives,
# not a slow epoch) hangs silently forever with no watchdog abort. These force
# a real Python exception naming the stuck collective instead. Remove once
# root-caused.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

ulimit -n 65536


# 3-GPU DDP run (torchrun). Config choices, all validated this session:
#  - hidden_dim=32 (reverted from 16 -- user wanted the previously-validated
#    width back), --use_checkpoint true --checkpoint_fine_only true: hidden_dim=32
#    needs checkpointing to fit at all; batch_size dropped to 28 to compensate
#    (batch=32 measured 43.81/47.54GB -- too tight for an unattended run,
#    batch=36 OOMs). Noticeably slower per-epoch than the hidden_dim=16 run
#    this superseded (bigger model + smaller batch + checkpoint recompute).
#  - --coarse_arch transformer + --use_compile false: torch.compile crashes
#    with a CUDA illegal memory access on this GPU/driver/torch stack when
#    compiling the HierarchicalGNN coarse self-attention stage (reproduced
#    independent of hidden_dim/head count, and independent of excluding just
#    the coarse block from the compiled graph via torch.compiler.disable in
#    gnn_surrogate.py).
#  - --coarse_arch gnn + --use_compile false: gnn coarse arch previously ran
#    fine compiled (job 1287234 to epoch 100, and further resumes to epoch
#    170), but 3 resumes in a row from 170_epochs.pt crashed deterministically
#    2 epochs after resume (epoch 172) with an out-of-bounds device-side
#    assert -- reproduced across seed_offset=42 vs 142 (rules out a specific
#    bad shuffled batch) AND across A6000 vs A40 GPUs (rules out a
#    hardware/driver flake), so it's suspected to be a torch.compile
#    guard/recompilation bug tied to the resume+warmup-then-next-epoch
#    transition, not the data itself. Disabled pending root-cause; revisit
#    once understood.
#  - DDP data loading is shard-partitioned per rank (see gnn_training.py), not
#    full-dataset-per-rank -- the naive approach measured ~575GB RSS per
#    process for the full 50k-sample set, which would OOM any node at 2+ ranks.
# Param-sweep probe (stress_weight_power / stress_ks_rho_mpa / node_loss_weight /
# graph_loss_weight) -- geometry is
# a decoy (hollow_cube_ddptest, symlinked boundary_conditions.py only) since
# gnn_training's best-checkpoint save path is data/{geometry}/gnn_surrogate.pt,
# fixed by --geometry alone regardless of --checkpoint_dir/--run_dir, and a
# scratch run against the real hollow_cube would silently overwrite the
# production surrogate the first time it beats the initial val loss (see
# [[feedback-decoy-geometry-for-benchmarks]]). --dataset still points at the
# real production dataset; only num_samples of it are actually used.
STRESS_WEIGHT_POWER="${STRESS_WEIGHT_POWER:-0.5}"
STRESS_KS_RHO_MPA="${STRESS_KS_RHO_MPA:-0.15}"
NODE_LOSS_WEIGHT="${NODE_LOSS_WEIGHT:-0.5}"
GRAPH_LOSS_WEIGHT="${GRAPH_LOSS_WEIGHT:-0.5}"
USE_COMPILE="${USE_COMPILE:-false}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-28}"
EPOCHS="${EPOCHS:-100}"
SEED_OFFSET="${SEED_OFFSET:-42}"

# Tag checkpoint_dir/run_dir by the swept values so back-to-back sweep
# submissions (e.g. STRESS_WEIGHT_POWER=1.0 then =0.1) land in separate
# directories instead of the second run resuming off the first's checkpoint
# (see [[feedback-checkpoint-resume-gotcha]]) -- that would either silently
# continue training under the wrong stress_weight_power, or no-op outright
# if the first run already reached $EPOCHS.
RUN_TAG="wp${STRESS_WEIGHT_POWER}_rho${STRESS_KS_RHO_MPA}_nw${NODE_LOSS_WEIGHT}_gw${GRAPH_LOSS_WEIGHT}_ns${NUM_SAMPLES}"
RUN_DIR="training/runs/train_ccx_gnn_probe_3gpu_${RUN_TAG}"
mkdir -p "${RUN_DIR}"

unset CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node=2 -m training.gnn_training \
    --geometry hollow_cube_ddptest \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --num_samples "${NUM_SAMPLES}" \
    --epochs "${EPOCHS}" \
    --lr 1e-3 \
    --batch_size "${BATCH_SIZE}" \
    --hidden_dim 64 \
    --conv_layers 6 \
    --node_loss_weight "${NODE_LOSS_WEIGHT}" \
    --graph_loss_weight "${GRAPH_LOSS_WEIGHT}" \
    --stress_weight_power "${STRESS_WEIGHT_POWER}" \
    --weight_decay 1e-3 \
    --use_checkpoint true \
    --checkpoint_fine_only true \
    --use_node_emb false \
    --coarse_arch gnn \
    --use_compile "${USE_COMPILE}" \
    --stress_ks_rho_mpa "${STRESS_KS_RHO_MPA}" \
    --seed_offset "${SEED_OFFSET}" \
    --checkpoint_dir "data/hollow_cube_ddptest/checkpoints_ccx_gnn_probe_3gpu_${RUN_TAG}" \
    --run_dir "${RUN_DIR}"
