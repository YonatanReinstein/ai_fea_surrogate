#!/bin/bash
#SBATCH --job-name=topopt_ccx_rho
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=public
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#
# KS-rho sweep of the ccx-adjoint-driven optimizer. Identical to job 54664 (the
# 250/279 reference: rho 0.25, move 0.08, filter 2, 8x4 solve threads) except
# --stress_ks_rho_mpa = $RHO. Smaller rho = smoother KS = a more learnable
# gradient for the surrogate, but a more conservative bound on the peak
# (initial design: rho 0.15 -> KS 94.3 vs true peak 66.5; rho 0.25 -> 77.0).
# Every surrogate run so far used 0.15 while the reference used 0.25.
# Question: what is the smallest rho that still converges near the reference?
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/"
export IRIT_TMP=/tmp/irit_$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONUNBUFFERED=1
RHO=${RHO:?set RHO}
ART=optimization/artifacts_ccx_rho${RHO}
mkdir -p "$ART"
export TOPOPT_ARTIFACT_DIR="$ART" TOPOPT_SCREENSHOT_DIR="$ART/screenshots" STRESS_KS_RHO_MPA=$RHO
echo "[job] rho=$RHO art=$ART cpus=$SLURM_CPUS_PER_TASK node=$(hostname)"
python -u -m optimization.optimize_dims --geometry hollow_cube --arch ccx --optimizer topopt --ccx_gradient adjoint \
    --generations ${GENS:-300} --processes 8 --mesh_threads 4 --nproc 1 \
    --filter_radius 2 --move_limit 0.08 --oc_eta 0.5 --fd_step 0.04 --trust_rel_tol 0.0 --min_trust_frac 0.05 \
    --tr_tol 0.05 --tr_min_scale 0.02 --g_backoff 0.02 --calib_every 5 --calib_smooth 0.5 --stagnation_gens 120 \
    --untrusted_move_frac 0.1 --screenshot_every 10 --stress_ks_rho_mpa $RHO
echo "TOPOPT_DONE rc=$?"; rm -rf "$IRIT_TMP"
