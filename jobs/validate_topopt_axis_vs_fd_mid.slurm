#!/bin/bash
#SBATCH --job-name=topopt_axis_vs_fd_mid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --partition=public
#SBATCH --nodelist=protagoras
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Does OC actually converge when it is driven by the patched solver's nodal
# gradient, rather than by the per-tile finite-difference sweep?
#
# This is the prerequisite question for training a surrogate to emit nodal
# dKS/dX: if OC does not converge on ccx's OWN gradients, a network trained to
# reproduce them cannot do better, and the dataset is not worth building.
#
# Run on hollow_cube_mid deliberately. It is the SAME physical problem as
# hollow_cube -- same outline.itd, same -5e7 N tip load on the middle tile,
# same 100 MPa yield -- over a coarser 8x6x1 tile grid, and it is the regime
# where the axis gradient is validated: against central-difference remesh FD
# over all 48 tiles at this mesh (3,3,3) it agrees 48/48 on sign with Spearman
# 0.9922. On the 840-tile grid it does not (56/840, dG negative on 13 tiles,
# and OC thinned the whole design 65% past yield -- job 48455). So a failure
# here is a failure of the METHOD, while a success here localises that problem
# to gradient quality at scale.
#
# Both arms are the same optimizer, geometry, filter and move limit; only
# --ccx_gradient differs. optimization/artifacts is NOT scoped by geometry or
# by arch, so each arm is archived and the directory cleared between them,
# otherwise arm 2 silently resumes arm 1's checkpoint.

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

STAMP=$(date +%Y%m%d_%H%M%S)

run_arm () {
    local grad="$1"
    local out="optimization/artifacts_mid_${grad}_${STAMP}"

    rm -rf optimization/artifacts/topopt_checkpoint.pkl optimization/artifacts/topopt_history
    mkdir -p optimization/artifacts optimization/screenshots
    rm -f optimization/screenshots/*.png

    echo "################ ARM: --ccx_gradient ${grad}"
    python -u -m optimization.optimize_dims \
        --geometry hollow_cube_mid \
        --arch ccx \
        --optimizer topopt \
        --ccx_gradient "${grad}" \
        --generations 300 \
        --processes "$SLURM_CPUS_PER_TASK" \
        --gradient_nproc 8 \
        --mesh_threads 1 \
        --nproc 1 \
        --filter_radius 1.5 \
        --move_limit 0.05 \
        --fd_step 0.04 \
        --min_trust_frac 0.05 \
        --untrusted_move_frac 0.1 \
        --screenshot_every 5 \
        --stress_ks_rho_mpa 0.15

    mkdir -p "${out}"
    mv optimization/artifacts/topopt_checkpoint.pkl "${out}/" 2>/dev/null
    mv optimization/artifacts/topopt_history "${out}/" 2>/dev/null
    mkdir -p "${out}/screenshots"
    mv optimization/screenshots/*.png "${out}/screenshots/" 2>/dev/null
    echo "################ ARM ${grad} archived to ${out}"
}

run_arm axis
run_arm fd

rm -r data/hollow_cube_mid/Instance_* 2>/dev/null
rm -r data/hollow_cube_mid/ansys_* 2>/dev/null

echo ARMSDONE
true
