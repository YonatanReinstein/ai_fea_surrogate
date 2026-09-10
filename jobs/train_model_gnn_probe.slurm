#!/bin/bash
#SBATCH --job-name=train_model_gnn_probe_3gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=650G
#SBATCH --time=00:30:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A6000:3
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd "$SLURM_SUBMIT_DIR"
mkdir -p training/runs/train_ccx_gnn_probe_3gpu
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536

# VRAM probe for hidden_dim=64 (up from 32) at the batch_size=28 that's
# proven to converge well at hidden_dim=32/coarse_arch=gnn (see
# jobs/train_model.slurm) -- checking whether doubling hidden_dim still fits
# in 28's batch before committing to a full unattended 100-epoch run at that
# batch size. hidden_dim=32 measured 20.53/21.22GB peak/reserved at batch=28;
# hidden_dim=64 roughly doubles checkpointed fine-mesh activation memory, so
# this is close to the 47GB card ceiling once DDP/NCCL overhead is added.
# --coarse_arch transformer required --use_compile false (CUDA illegal memory
# access compiling the coarse self-attention block); gnn coarse arch is
# unaffected by that bug, so compile is enabled here.
# --geometry hollow_cube_ddptest (decoy): gnn_surrogate.pt save path is fixed
# per-geometry, not checkpoint_dir-scoped, so a real --geometry hollow_cube
# run here would clobber the deployed best transformer surrogate after just
# 3 epochs. --dataset/graph.pt still point at the real hollow_cube data.
unset CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node=3 -m training.gnn_training \
    --geometry hollow_cube_ddptest \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --num_samples 50000 \
    --epochs 3 \
    --lr 1e-3 \
    --batch_size 28 \
    --hidden_dim 64 \
    --conv_layers 6 \
    --node_loss_weight 1.0 \
    --graph_loss_weight 0.0 \
    --stress_weight_power 1.0 \
    --weight_decay 1e-3 \
    --use_checkpoint true \
    --checkpoint_fine_only true \
    --use_node_emb false \
    --coarse_arch gnn \
    --use_compile true \
    --stress_ks_rho_mpa 0.15 \
    --checkpoint_dir data/hollow_cube_ddptest/checkpoints_ccx_gnn_probe_3gpu \
    --run_dir training/runs/train_ccx_gnn_probe_3gpu
