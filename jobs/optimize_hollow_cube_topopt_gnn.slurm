#!/bin/bash
#SBATCH --job-name=optimize_hc_topopt_gnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --partition=newton
#SBATCH --account=cslab

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

mkdir -p optimization/artifacts optimization/screenshots

# GNN-surrogate gut of jobs/optimize_hollow_cube_topopt_ccx.slurm: same
# --optimizer topopt trust-mechanism hyperparameters as that script's current
# (post job-47992/48002) tuning -- filter_radius/move_limit/fd_step/
# min_trust_frac/untrusted_move_frac/stress_ks_rho_mpa are copied verbatim,
# NOT from the older jobs/optimize_hollow_cube_topopt_gnn_darwin.slurm /
# original version of this file, which predate that tuning (see
# project_topopt_trust_deadlock.md memory). --arch gnn replaces --arch ccx,
# swapping the per-generation real CCX solve for GNN surrogate inference
# (data/hollow_cube_ddptest/gnn_surrogate.pt, via --gnn_ckpt_path -- lower
# val loss than the default data/hollow_cube/gnn_surrogate.pt checkpoint,
# 0.0442 vs 0.0510). --mesh_threads/--nproc (ccx-only) are dropped;
# --batch_size (gnn-only, controls inference batch size) is added.
#
# Fresh start: optimization/artifacts/topopt_checkpoint.pkl (+ history +
# screenshots) is NOT geometry/arch-scoped -- it's the same path the ccx
# job above resumes from, so it was moved aside before this job was
# submitted (see optimization/artifacts_ccx_backup_*) instead of deleted,
# to start this run at gen 0 without discarding the ccx run's progress.
python -u -m optimization.optimize_dims \
    --geometry hollow_cube \
    --arch gnn \
    --gnn_ckpt_path data/hollow_cube_ddptest/gnn_surrogate.pt \
    --optimizer topopt \
    --generations 200 \
    --processes "$((SLURM_CPUS_PER_TASK - 2))" \
    --batch_size 4 \
    --filter_radius 2 \
    --move_limit 0.05 \
    --fd_step 0.04 \
    --min_trust_frac 0.05 \
    --untrusted_move_frac 0.1 \
    --stress_ks_rho_mpa 0.15

true
