#!/bin/bash
#SBATCH --job-name=optimize_tile_gnn
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

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

mkdir -p optimization/artifacts optimization/screenshots

python -u -m optimization.optimize_dims \
    --geometry bistable \
    --arch gnn \
    --pop_size 300 \
    --generations 30 \
    --processes 16 \
    --batch_size 28 \

