#!/bin/bash
#SBATCH --job-name=train_hier_gnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:A100:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1

python -u -m training.gnn_training \
    --geometry bistable \
    --dataset data/bistable/dataset/dataset_test.pt \
    --num_samples 500 \
    --epochs 100 \
    --lr 1e-3 \
    --batch_size 8 \
    --hidden_dim 128 \
    --conv_layers 6 \
    --node_loss_weight 1.0 \
    --graph_loss_weight 0.0 \
    --weight_decay 1e-3
