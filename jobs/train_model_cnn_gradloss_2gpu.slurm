#!/bin/bash
#SBATCH --job-name=train_cnn_gradloss_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=650G
#SBATCH --time=40:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A40:2
#SBATCH --exclude=newton2
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd "$SLURM_SUBMIT_DIR"
mkdir -p training/runs/train_cnn_gradloss_2gpu
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536

# Full production-scale CNN retrain (jobs/train_model_cnn.slurm recipe: 50k
# samples, coarse_arch cnn, batch_size 44, hidden_dim 32) now with the
# gradient-matching loss added (--grad_dataset_path/--grad_loss_weight,
# see project_gradient_matching_loss.md), scaling up the single-GPU
# validation run (job 1327061, 4000 samples/60 epochs, use_compile false)
# to full 2-GPU DDP + torch.compile -- that DDP+compile interaction was not
# yet validated at small scale. gnn_training.py already conditions DDP's
# static_graph/broadcast_buffers on grad_dataset_path being set, and loads
# the (small, 1500-sample) grad dataset in full on every rank rather than
# sharding it, so no extra DDP wiring should be needed here.
#
# Time budget raised from 34h to 40h vs train_model_cnn.slurm: that job
# only reached epoch 260/300 in 34h (~470s/epoch), and the extra grad-loss
# forward/backward pass adds further per-epoch cost.
#
# --use_compile false (not true, unlike train_model_cnn.slurm): job 1337982
# crashed immediately with "torch.compile with aot_autograd does not
# currently support double backward" -- the grad-loss term's autograd.grad
# sensitivity computation, backpropped through again for the weight update,
# is exactly a double backward. The validated small-scale probe
# (train_gradloss_v2.slurm) already used --use_compile false for this
# reason; missed carrying that over here.
#
# Tried A6000:2 on newton2 (idle per Slurm) when neither A40 node had 2
# free GPUs -- newton2's CUDA driver turned out to be dead (both ranks:
# "ProcessGroupNCCL is only supported with GPUs, no GPUs found!", confirmed
# node-wide by a separate single-GPU job also showing cuda_available=False
# there). Reverted to A40:2 and excluded newton2; this will sit
# PENDING(Resources) until an A40 node frees up 2 GPUs rather than risk
# landing on a broken node again.
#
# New checkpoint_dir (not the plain run's checkpoints_ccx_cnn_2gpu) so this
# doesn't silently resume off a checkpoint trained without the grad-loss
# term (see [[feedback-checkpoint-resume-gotcha]]). Writes to the real
# hollow_cube geometry -- best-val-loss checkpoints will overwrite
# data/hollow_cube/gnn_surrogate.pt, the production surrogate; it was
# backed up first (gnn_surrogate.pt.bak_20260825_195327).
unset CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node=2 -m training.gnn_training \
    --geometry hollow_cube \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --grad_dataset_path data/hollow_cube/dataset_grad/dataset.pt \
    --grad_loss_weight 0.1 \
    --grad_batch_size 4 \
    --num_samples 50000 \
    --epochs 300 \
    --lr 1e-3 \
    --batch_size 44 \
    --hidden_dim 32 \
    --conv_layers 6 \
    --node_loss_weight 0.5 \
    --graph_loss_weight 0.5 \
    --stress_weight_power 0.1 \
    --weight_decay 1e-3 \
    --use_checkpoint true \
    --checkpoint_fine_only true \
    --use_node_emb false \
    --coarse_arch cnn \
    --use_compile false \
    --stress_ks_rho_mpa 0.15 \
    --checkpoint_dir data/hollow_cube/checkpoints_ccx_cnn_gradloss_2gpu \
    --run_dir training/runs/train_cnn_gradloss_2gpu
