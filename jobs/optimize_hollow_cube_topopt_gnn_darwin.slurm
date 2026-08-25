#!/bin/bash
#SBATCH --job-name=optimize_hc_topopt_gnn_darwin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=190
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --partition=public
#SBATCH --account=cs

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

# No --gres/GPU request: evaluators/gnn_evaluator.py falls back to CPU
# automatically (torch.device("cuda" if torch.cuda.is_available() else "cpu")).
# Meshing (CPU-bound, ~652s/gen with 16 processes on Newton) is the actual
# bottleneck, not GPU inference (~70s/gen) -- so trading the GPU for 200 CPUs
# here should net out ahead. batch_size raised back up since CPU RAM is not
# the ~47GB VRAM-constrained resource GPU inference was.
python -u -m optimization.optimize_dims \
    --geometry hollow_cube \
    --arch gnn \
    --optimizer topopt \
    --generations 300 \
    --processes "$SLURM_CPUS_PER_TASK" \
    --batch_size 32 \
    --filter_radius 1.5 \
    --move_limit 0.05 \
    --fd_step 0.02

true
