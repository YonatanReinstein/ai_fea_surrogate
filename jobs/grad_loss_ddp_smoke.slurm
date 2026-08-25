#!/bin/bash
#SBATCH --job-name=grad_loss_ddp_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A40:2
#SBATCH --nodelist=newton3
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# 2-GPU DDP smoke test for the new gradient-matching loss (see
# project_gradient_matching_loss.md) -- the double-backward
# (torch.autograd.grad(create_graph=True)) + DDP combination has no
# precedent elsewhere in this repo, so this checks it doesn't hang or
# diverge before trusting a full-scale run. Reuses the tiny local smoke
# datasets already built under data/hollow_cube_ddptest/ (60-sample plain +
# 40-sample grad-labeled, M=4 tiles/sample) -- decoy geometry, so
# gnn_surrogate.pt here is disposable (see feedback_decoy_geometry_for_benchmarks).
# Batch sizes match the single-GPU smoke test that first hit CUDA OOM at
# batch_size=5/grad_batch_size=5 (peak 17.3GB at batch_size=2/grad_batch_size=1
# single-GPU) -- kept conservative here since the goal is confirming
# correctness/no-hang, not throughput.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536

echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
nvidia-smi -L

torchrun --standalone --nproc_per_node=2 -m training.gnn_training \
    --geometry hollow_cube_ddptest \
    --dataset data/hollow_cube_ddptest/dataset_smoke_main/dataset.pt \
    --grad_dataset_path data/hollow_cube_ddptest/dataset_smoke_grad/dataset.pt \
    --grad_loss_weight 0.1 \
    --epochs 3 \
    --batch_size 2 \
    --grad_batch_size 1 \
    --use_compile false \
    --coarse_arch cnn \
    --checkpoint_dir data/hollow_cube_ddptest/checkpoints_grad_ddp_smoke \
    --run_dir training/runs/train_grad_ddp_smoke
