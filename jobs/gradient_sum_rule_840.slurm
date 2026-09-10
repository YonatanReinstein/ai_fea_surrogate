#!/bin/bash
#SBATCH --job-name=sumrule_840
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --partition=public
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Which mesh resolution gives a gradient whose TOTAL is right?
#
# The full per-tile FD map at (3,3,3) showed the axis contraction summing to
# -2.79x the directly measured d(KS)/d(uniform thickening), i.e. wrong in sign
# on the one aggregate that needs no per-tile attribution, while the per-tile
# remesh FD summed to 1.0026x. Over 32 sampled tiles at (5,3,3) the sign
# agreement was 32/32 against 13/32 at (3,3,3), so more elements through the
# wall clearly helps -- this asks whether it helps enough to make the total
# correct, at 5 solves per resolution instead of 1680.

cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$PWD/irit/bin:$PATH"
export IRIT_PATH="$PWD/irit/bin/"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python -u -m evaluators.gradient_sum_rule \
    --geometry hollow_cube \
    --mesh_res "3,3,3;5,3,3;7,3,3;9,3,3" \
    --element_type C3D8 \
    --nproc 16 --mesh_threads 4 \
    --out evaluators/reports/sum_rule_840_C3D8.json

rm -r data/hollow_cube/Instance_* 2>/dev/null
echo SUMRULEDONE
true
