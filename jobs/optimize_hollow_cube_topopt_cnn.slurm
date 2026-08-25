#!/bin/bash
#SBATCH --job-name=optimize_hc_topopt_cnn
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

# Topopt against the CNN-coarse-arch surrogate that's actively being trained
# by job 1318940 (jobs/train_model_cnn.slurm, --coarse_arch cnn, epoch ~15/300
# as of launch) -- --gnn_ckpt_path omitted so it uses the default
# data/{geometry}/gnn_surrogate.pt, i.e. whatever that training job's best
# checkpoint currently is at the moment this job starts. Loaded once at
# startup, not re-read as training keeps improving it afterward. Same
# trust-mechanism hyperparameters as jobs/optimize_hollow_cube_topopt_gnn.slurm
# (post job-47992/48002 tuning, see project_topopt_trust_deadlock.md memory).
#
# Fresh start: optimization/artifacts/topopt_checkpoint.pkl (+ history +
# screenshots) was archived to optimization/artifacts_backup_gen68_* /
# screenshots_backup_* before this job was submitted (it held an unrelated
# gen-68 run's state, not scoped to geometry/arch -- see
# feedback_checkpoint_resume_gotcha.md).
python -u -m optimization.optimize_dims \
    --geometry hollow_cube \
    --arch gnn \
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
