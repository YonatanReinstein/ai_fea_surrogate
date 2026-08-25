#!/bin/bash
#SBATCH --job-name=train_model_cnn_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=650G
#SBATCH --time=34:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A40:2
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd "$SLURM_SUBMIT_DIR"
mkdir -p training/runs/train_ccx_cnn_2gpu
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536

# First full production run of the new --coarse_arch cnn (3D U-Net over the
# dense tile grid, utils/gnn_hierarchical_cnn.py), replacing the gnn
# message-passing coarse stage used by jobs/train_model_gnn.slurm. All other
# hyperparameters mirror that job's validated recipe (batch_size=44,
# hidden_dim=32, node/graph loss weights, stress_weight_power) so results are
# comparable apples-to-apples against the gnn-arch run; coarse_dim defaults
# to hidden_dim=32 and cnn_depth defaults to 3 (both unset here). The coarse
# U-Net operates on a much smaller footprint (~840 tile cells) than either
# the gnn message-passing or transformer self-attention coarse stages, so it
# should fit at this batch size with equal or more headroom -- not
# independently VRAM-probed before launch.
unset CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node=2 -m training.gnn_training \
    --geometry hollow_cube \
    --dataset data/hollow_cube/dataset_ccx/dataset.pt \
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
    --use_compile true \
    --stress_ks_rho_mpa 0.15 \
    --checkpoint_dir data/hollow_cube/checkpoints_ccx_cnn_2gpu \
    --run_dir training/runs/train_ccx_cnn_2gpu
