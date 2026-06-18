#!/bin/bash
#SBATCH --job-name=optimize_tile_gnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --partition=public
#SBATCH --gres=gpu:A40:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p optimization/artifacts optimization/screenshots

python -u -m optimization.optimize_dims \
    --geometry hollow_cube \
    --arch gnn \
    --optimizer cma \
    --generations 10000 \
    --search_space reflect \
    --constraint_mode quadratic \
    --processes 64 \
    --batch_size 30 \
    --reset_on_stagnation 

rm -r data/hollow_cube/Instance_*
rm -r data/hollow_cube/ansys_*

