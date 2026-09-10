#!/bin/bash
#SBATCH --job-name=tilegrad_840
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=4:00:00
#SBATCH --partition=public
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# The per-tile gradient over the WHOLE 840-tile grid, computed both ways at
# one design point (uniform t=0.30, C3D8, mesh 3,3,3), rendered as heat maps.
#
# Every number on this gradient so far has been a single summary statistic
# over either 48 tiles (hollow_cube_mid, where it scores +0.9922) or a
# 15-to-32-tile sample of the 840 (where it scores +0.288). A scalar cannot
# say WHERE the two disagree, and on this grid that is the whole question:
# |remesh FD| spans ~7 decades across the 840 tiles because one tile in 840
# barely moves a GLOBAL KS max, so an all-tiles correlation mostly ranks
# solver noise. The maps separate "wrong here" from "nothing to be right
# about here".
#
# 1680 remesh solves for the FD side against 3 for the axis side -- the exact
# asymmetry the axis route exists to avoid. Paid once, on purpose, as an audit.

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

python -u -m evaluators.tile_gradient_map \
    --geometry hollow_cube \
    --mesh_res 3,3,3 \
    --element_type C3D8 \
    --fd_frac 0.02 \
    --signal_frac 0.10 \
    --stress_ks_rho_mpa 0.15 \
    --workers 44 \
    --gradient_nproc 16 \
    --mesh_threads 4

rm -r data/hollow_cube/Instance_* 2>/dev/null
echo MAPDONE
true
