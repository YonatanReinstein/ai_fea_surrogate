#!/bin/bash
#SBATCH --job-name=validate_axis_c3d20_mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --partition=public
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Does a SECOND-ORDER element buy at mesh_resolution (3,3,3) what refining the
# linear mesh was being made to buy?
#
# The problem: hollow_cube's tile is a thin tube wall, the stress in it is
# bending stress, and a trilinear hex cannot represent bending -- it needs
# several elements through the wall before it stops locking. That is the only
# reason mesh_resolution has been climbing (3,3,3 -> 5,3,3 -> 7,3,3), at a cost
# that grows with the cube of the wall resolution and that priced 7,3,3 out
# entirely (>2.5 h for the gradient alone). A C3D20 spans a linear strain
# gradient exactly, so TWO of them through a wall -- which is what (3,3,3)
# gives -- is already bending-capable.
#
# utils/hex20.py builds the 20-node mesh by adding the 12 edge-midpoint nodes
# to irit2inp's linear output; the CAD side is untouched and the element
# geometry is identical (irit2inp's faces are straight-edged already). ccx
# supports it natively in the sensitivity path: getdesiinfo3d.f branches on
# nope=20 explicitly and accepts exactly 8 and 20 for hexes, and e_c3d_se.f has
# its own '20' branch. None of build_ccx_axis.sh's three patches look at the
# element type.
#
# Run on hollow_cube_mid: same physical problem as hollow_cube (same outline,
# same -5e7 N tip load, same 100 MPa yield) over 48 tiles instead of 840, so
# all 48 tiles can be checked against central-difference REMESH FD -- the same
# benchmark that gave the linear pipeline sign 48/48 / Spearman 0.9922 at
# (3,3,3) and that the 840-tile grid fails (see the mesh table in
# jobs/optimize_hollow_cube_topopt_ccx.slurm).
#
# Three arms, all at mesh_resolution (3,3,3), all --interior:
#   C3D8   the current gradient element -- the baseline to beat
#   C3D20  full integration, 27 Gauss points
#   C3D20R reduced integration, 8 Gauss points, ~3x cheaper
# What to read: sign agreement and Spearman are what OC consumes (it rescales
# by lambda every generation, so rank matters more than magnitude); the ratio
# median says whether the magnitude is right too. The per-arm wall times print
# alongside, which is the other half of the decision.

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

for ELT in C3D8 C3D20 C3D20R; do
    echo "################ ARM: ${ELT} @ 3,3,3"
    # --max_nodes caps the per-NODE brute-force FD at 6 solves per node; the
    # per-TILE contraction against remesh FD is the number that decides this,
    # the per-node cosine is the diagnostic for WHY when it disagrees.
    python -u -m evaluators.validate_axis_gradient \
        --geometry hollow_cube_mid \
        --mesh_res 3,3,3 \
        --element_type "${ELT}" \
        --interior \
        --tiles all \
        --max_nodes 24 \
        --nproc 4 \
        --workers 24 \
        --mesh_threads 4 \
        --stress_ks_rho_mpa 0.15 \
        --out "evaluators/reports/axis_mid_333_${ELT}.json"
done

rm -r data/hollow_cube_mid/Instance_* 2>/dev/null
echo ARMSDONE
true
