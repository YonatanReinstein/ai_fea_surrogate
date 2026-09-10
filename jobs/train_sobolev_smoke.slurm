#!/bin/bash
#SBATCH --job-name=train_sobolev_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Smoke test for the Sobolev (derivative-supervision) path against the NEW
# adjoint gradient labels in data/hollow_cube/dataset_gradadj.
#
# Why this exists: gnn_training.py's --grad_dataset_path branch was written
# for the old FD grad dataset, whose comment (line ~500) states the grad set
# is "small by construction (each sample costs M+1 real FEA solves)" and is
# therefore "Loaded in full on every rank". The adjoint build broke that
# assumption -- every one of the 50,001 samples now carries grad labels at
# ~1.1 solves each -- so pointing --grad_dataset_path at the full set would
# load ~575GB per rank. Hence --grad_dataset_path points at a SUBSET dir of
# symlinked shards, drawn only from the train region (shards 0..400; val is
# 401..500) so the gradient supervision never sees a val sample.
#
# This run only answers: does the double-backward path execute against the
# real labels, and does the loss move? Weights are throwaway.
# gnn_surrogate.pt is restored from .bak_pre_sobolev afterwards.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
ulimit -n 65536

torchrun --standalone --nproc_per_node=1 -m training.gnn_training \
    --geometry hollow_cube \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --grad_dataset_path data/hollow_cube/gradsub_smoke/dataset.pt \
    --grad_loss_weight 0.1 \
    --num_samples 600 \
    --epochs 2 --batch_size 8 --grad_batch_size 4 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube/checkpoints_sobolev_smoke \
    --run_dir training/runs/train_sobolev_smoke
