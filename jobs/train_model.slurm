#!/bin/bash
#SBATCH --job-name=train_hier_gnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:A40:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log



cd "$SLURM_SUBMIT_DIR"
mkdir -p training/runs/train
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536


python -u -m training.gnn_training \
    --geometry hollow_cube \
    --dataset data/hollow_cube/dataset/dataset.pt \
    --num_samples 500 \
    --epochs 400 \
    --lr 1e-4 \
    --batch_size 4 \
    --hidden_dim 128 \
    --conv_layers 6 \
    --node_loss_weight 1.0 \
    --graph_loss_weight 0.0 \
    --weight_decay 2e-4 \
    --use_checkpoint false \
    --checkpoint_fine_only false \
    --use_node_emb false \