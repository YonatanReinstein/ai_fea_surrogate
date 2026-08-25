#!/bin/bash
#SBATCH --job-name=optimize_tile_ccx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --partition=public
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Direct-simulation counterpart to optimize_tile.slurm: evaluates each CMA
# candidate with a real CalculiX solve (evaluators/ccx_evaluator.py) instead
# of the GNN surrogate. No GPU needed (ccx is pure CPU), and CPU-only batch
# jobs are rejected on the Newton GPU cluster, so this runs here on Darwin
# (partition=public, no --gres) rather than needing the Newton submission
# jobs/optimize_tile.slurm and jobs/train_model.slurm require.
#
# generations/pop_size are much smaller than optimize_tile.slurm's
# 10000x90: each fitness eval is now a real ~1-3s FEA solve instead of a
# near-instant batched GPU forward pass, so this is scaled as a validation
# run (sanity-check the GNN-optimized result against real sims), not a
# from-scratch full search.
#
# NOTE: CMAOptimizer checkpoints to the fixed path
# optimization/artifacts/cma_checkpoint.pkl regardless of --arch, and
# auto-resumes from it if present. Don't interleave this with
# jobs/optimize_tile.slurm (GNN) without clearing that file in between, or
# one run will silently resume from the other's CMA state.
#
# Launch with:
#   sbatch jobs/optimize_tile_ccx.slurm

cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$PWD/irit/bin:$PATH"
export IRIT_PATH="$PWD/irit/bin/"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p optimization/artifacts optimization/screenshots


python -u -m optimization.optimize_dims \
    --geometry hollow_cube \
    --arch ccx \
    --optimizer cma \
    --generations 1000 \
    --pop_size 128 \
    --search_space reflect \
    --sigma0 0.5 \
    --constraint_mode quadratic \
    --processes 128 \
    --penalty_weight 10.0 \
    --reset_on_stagnation \
    --max_unproductive_resets 5

rm -r data/hollow_cube/Instance_* 2>/dev/null
rm -r data/hollow_cube/ansys_* 2>/dev/null

true
