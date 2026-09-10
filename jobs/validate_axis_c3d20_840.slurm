#!/bin/bash
#SBATCH --job-name=validate_axis_c3d20_840
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --partition=public
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Does the C3D20 result carry to the REAL grid?
#
# Everything measured so far is on hollow_cube_mid (48 tiles), and for this
# gradient mid does not predict hollow_cube (840 tiles): C3D8 at (3,3,3)
# scores spearman +0.9922 on mid and +0.288 on the 840-tile grid, which is
# the whole reason the production run was pushed to (5,3,3). Mid can kill a
# candidate, it cannot bless one. Measured on mid, 48 tiles vs remesh FD:
#     C3D8            sign 48/48  spearman +0.9922  ratio median  0.586
#     C3D20 all nodes sign 13/48  spearman -0.2638  ratio median -0.264
#     C3D20R all      sign 28/48  spearman +0.3734  ratio median  0.040
#     C3D20 corners   sign 43/48  spearman +0.6055  ratio median  0.878
# (a CCX_DESI_STEP sweep at 1e-2/1e-3/1e-5 moved the all-nodes number not at
#  all: -0.2530 / -0.2701 / -0.2660, so the semi-analytical step is not it.)
#
# WHY CORNERS ONLY. ccx perturbs one design node at a time. Moving a single
# MIDSIDE node bends an element edge -- a deformation a CAD remesh never
# applies, since utils/hex20.py keeps every midside node at its edge midpoint,
# so its dX/dt is the mean of its corners'. Offering only corner nodes drops
# that mode. It also drops those nodes' own contribution, which is why this is
# a real approximation and not a free fix -- but on mid it took the per-NODE
# gradient from cosine +0.969 / ratio 0.918 to +0.9985 / 1.031, and it drops
# ndesi from 9973 back to 2823, i.e. to the linear mesh's own design-set size,
# which is what makes a 20-node *SENSITIVITY affordable at 840 tiles at all.
#
# THE NOISE TRAP on this grid, and how this job avoids it. |remesh FD| across
# the 840 tiles spans 1.8 to 5.4e6: one tile in 840 barely moves a GLOBAL KS
# max, so most tiles sit at solver noise and an all-tiles Spearman mostly
# ranks noise. So: arm A nets the 32 tiles its own contraction ranks largest
# and writes them to its JSON; arms B and C re-run the SAME 32 via
# --tiles_from, so all three are scored on one tile set; and each arm prints a
# second metrics block over just the tiles whose |FD| clears 10% of the
# largest, a filter applied on the FD (truth) side where it cannot flatter
# the method.
#
# ARMS. C is the configuration currently running as job 49264, re-measured
# with the same instrument so the comparison is like for like -- the +0.700
# on record for it was computed over all 840 tiles, noise included, and is
# not comparable to the filtered numbers here.
#   A  C3D8   @ 3,3,3            the cheap linear baseline
#   B  C3D20  @ 3,3,3, corners   two quadratic elements through the wall
#   C  C3D8   @ 5,3,3            four linear elements through the wall

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

mkdir -p evaluators/reports
A=evaluators/reports/axis_840_333_C3D8.json
B=evaluators/reports/axis_840_333_C3D20_corners.json
C=evaluators/reports/axis_840_533_C3D8.json

echo "################ ARM A: C3D8 @ 3,3,3 (selects the tile set)"
python -u -m evaluators.validate_axis_gradient \
    --geometry hollow_cube --mesh_res 3,3,3 --element_type C3D8 --interior \
    --tiles top:32 --max_nodes 8 --nproc 4 --workers 12 --mesh_threads 4 \
    --stress_ks_rho_mpa 0.15 --out "$A"

echo "################ ARM B: C3D20 @ 3,3,3, corner design nodes only"
python -u -m evaluators.validate_axis_gradient \
    --geometry hollow_cube --mesh_res 3,3,3 --element_type C3D20 --interior \
    --corners_only --tiles_from "$A" --max_nodes 8 --nproc 4 --workers 8 \
    --mesh_threads 4 --stress_ks_rho_mpa 0.15 --out "$B"

echo "################ ARM C: C3D8 @ 5,3,3 (what job 49264 is running)"
python -u -m evaluators.validate_axis_gradient \
    --geometry hollow_cube --mesh_res 5,3,3 --element_type C3D8 --interior \
    --tiles_from "$A" --max_nodes 8 --nproc 4 --workers 8 --mesh_threads 4 \
    --stress_ks_rho_mpa 0.15 --out "$C"

rm -r data/hollow_cube/Instance_* 2>/dev/null
echo ARMSDONE
true
