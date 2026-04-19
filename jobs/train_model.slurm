#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
    --geometry tile \
    --num_samples 5000 \
    --epochs 100 \
    --lr 4e-4\
    --batch_size  32 \
    --hidden_dim 256 \
    --conv_layers 10
