#!/bin/bash
#SBATCH --job-name=optimize_tile_gnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --time=08:00:00
# CPU-bound job: bottleneck is serial IRIT mesh generation (irit2inp), so we
# scale width (one core per concurrent mesh worker), not GPU power. Target the
# L4 nodes (plotinus1/2): low-demand inference GPU + 128 fast Zen4 EPYC cores +
# 503GB RAM, usually idle => far shorter queue than the in-demand A40 nodes.
#SBATCH --partition=public
#SBATCH --gres=gpu:L4:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Reduce CUDA fragmentation on the 24GB L4 (GNN forward is chunked by --batch_size).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p optimization/artifacts optimization/screenshots


python -u -m optimization.optimize_dims \
    --geometry hollow_cube \
    --arch gnn \
    --optimizer cma \
    --generations 10000 \
    --pop_size 90 \
    --search_space reflect \
    --sigma0 0.5 \
    --constraint_mode quadratic \
    --processes 90 \
    --batch_size 4 \
    --penalty_weight 10.0 \
    --reset_on_stagnation \
    --max_unproductive_resets 5 \

rm -r data/hollow_cube/Instance_*
rm -r data/hollow_cube/ansys_*

