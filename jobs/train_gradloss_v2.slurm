#!/bin/bash
#SBATCH --job-name=train_gradloss_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A40:1
#SBATCH --nodelist=newton3
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Real-scale validation run for the gradient-matching loss (see
# project_gradient_matching_loss.md): 4000 real hollow_cube samples +
# data/hollow_cube/dataset_grad (1500 samples, M=6 tiles/sample) at
# grad_loss_weight=0.1, production architecture (hidden_dim=32, coarse_arch=cnn).
# Moved from an inline session process to a tracked sbatch job (session process
# killed at epoch 12/60) -- resumes automatically from
# data/hollow_cube/checkpoints_gradloss_v2/10_epochs.pt via gnn_training.py's
# own latest-checkpoint auto-resume, losing only ~2 epochs of progress.
# torchrun --nproc_per_node=1 (not a plain python invocation) is deliberate:
# it takes the DDP-shard-based dataset loader path, which honors --num_samples
# by only reading the shard files it needs, instead of the single-process
# path's load_dataset() which loads the ENTIRE ~295GB dataset_ccx into memory
# before subsetting.
#
# --nodelist=newton3 pinned: this and train_gradloss_v2_baseline.slurm both
# run `torchrun --standalone` concurrently; when Slurm happened to co-locate
# both on the same node they collided on --standalone's rendezvous (both
# stuck ~50min, "failed to send a keep-alive heartbeat" in both logs, zero
# epochs logged). Pinning each to its own node (baseline -> newton2) avoids
# the collision instead of chasing a distinct-port fix.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536

torchrun --standalone --nproc_per_node=1 -m training.gnn_training \
    --geometry hollow_cube \
    --dataset data/hollow_cube/dataset_ccx/dataset.pt \
    --grad_dataset_path data/hollow_cube/dataset_grad/dataset.pt \
    --grad_loss_weight 0.1 \
    --num_samples 4000 \
    --epochs 60 --batch_size 8 --grad_batch_size 4 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube/checkpoints_gradloss_v2 \
    --run_dir training/runs/train_gradloss_v2
