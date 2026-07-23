#!/bin/bash
#SBATCH --job-name=train_model_ccx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd "$SLURM_SUBMIT_DIR"
mkdir -p training/runs/train_ccx_ks
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536


python -u -m training.gnn_training \
    --geometry hollow_cube \
    --dataset data/hollow_cube/dataset_ccx/dataset.pt \
    --num_samples 5000 \
    --epochs 200 \
    --lr 1e-3 \
    --batch_size 12 \
    --hidden_dim 32 \
    --coarse_dim 128 \
    --conv_layers 6 \
    --node_loss_weight 1.0 \
    --graph_loss_weight 0.0 \
    --stress_weight_power 1.0 \
    --weight_decay 1e-3 \
    --use_checkpoint true \
    --checkpoint_fine_only true \
    --use_node_emb false \
    --coarse_arch gnn \
    --mp_dropout 0.1 \
    --stress_ks_rho_mpa 0.15 \
    --checkpoint_dir data/hollow_cube/checkpoints_ccx_ks \
    --run_dir training/runs/train_ccx_ks