#!/bin/bash
#SBATCH --job-name=train_hier_gnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:A40:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yoyo.reinstein@gmail.com

cd "$SLURM_SUBMIT_DIR"
mkdir -p training/runs/train
source env/bin/activate

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

ulimit -n 65536

python -u -m training.gnn_training \
    --geometry bistable \
    --dataset data/bistable/dataset/dataset_6x6.pt \
    --num_samples 5000 \
    --epochs 202 \
    --lr 1e-3 \
    --batch_size 30 \
    --hidden_dim 128 \
    --conv_layers 6 \
    --node_loss_weight 1.0 \
    --graph_loss_weight 0.0 \
    --weight_decay 1e-3 \
    --use_checkpoint false \
    --checkpoint_fine_only false \