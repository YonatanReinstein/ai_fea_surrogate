#!/bin/bash
#SBATCH --job-name=validate_axis_c3d20_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --partition=public
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# WHY. jobs/validate_axis_c3d20_mid.slurm measured, on hollow_cube_mid at
# (3,3,3), all 48 tiles against central-difference remesh FD:
#     C3D8   sign 48/48  spearman +0.9922  ratio median  0.586
#     C3D20  sign 13/48  spearman -0.2638  ratio median -0.264
# but the per-NODE gradient stayed good at C3D20 (direction cosine +0.969
# median against brute-force node FD, ratio median 0.918). So ccx is
# differentiating the quadratic element sensibly and it is the CONTRACTION
# onto tile thickness that falls apart. Two candidate causes, one arm each:
#
#   step   ccx's semi-analytical sensitivity forward-differences the element
#          stiffness with step distmin = the smallest element edge length
#          (smalldist.f). Inserting midside nodes halves the node-to-node
#          spacing, so distmin halves, and a 20-node element's Jacobian
#          degrades much faster under node motion than an 8-node one's. The
#          step was verified converged for C3D8 (identical from 1e-2 to
#          1e-7); CCX_DESI_STEP re-tests that at C3D20.
#
#   corners ccx perturbs one design node at a time. Moving a single MIDSIDE
#          node bends an element edge -- a deformation the remesh FD never
#          applies, since a CAD remesh moves corners and midsides together
#          (utils/hex20.py puts each midside node at its edge's midpoint, so
#          its dX/dt is the mean of its corners'). Offering only corner nodes
#          as design variables drops that mode entirely. It also drops those
#          nodes' real contribution, so this arm is a DIAGNOSTIC, not a
#          candidate configuration -- read it as "is the midside perturbation
#          the thing poisoning the sum?", the same question CCX_DESI_INTERIOR
#          answered for interior nodes.

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

common="--geometry hollow_cube_mid --mesh_res 3,3,3 --element_type C3D20 \
        --interior --tiles all --max_nodes 8 --nproc 4 --workers 16 \
        --mesh_threads 4 --stress_ks_rho_mpa 0.15"

for STEP in 1e-2 1e-3 1e-5; do
    echo "################ ARM: C3D20 desi_step=${STEP}"
    python -u -m evaluators.validate_axis_gradient $common \
        --desi_step "${STEP}" \
        --out "evaluators/reports/axis_mid_333_C3D20_step${STEP}.json"
done

echo "################ ARM: C3D20 corner design nodes only"
python -u -m evaluators.validate_axis_gradient $common \
    --corners_only \
    --out "evaluators/reports/axis_mid_333_C3D20_corners.json"

rm -r data/hollow_cube_mid/Instance_* 2>/dev/null
echo ARMSDONE
true
