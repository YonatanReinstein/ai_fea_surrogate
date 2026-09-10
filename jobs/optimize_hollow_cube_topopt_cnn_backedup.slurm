#!/bin/bash
#SBATCH --job-name=optimize_hc_topopt_cnn_bak
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton2
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

# Topopt pinned to a frozen snapshot of the CNN-coarse-arch surrogate
# (data/hollow_cube/gnn_surrogate.pt.bak_20260825_195327), taken right
# before job 1337973 (jobs/train_model_cnn_gradloss_2gpu.slurm) started
# overwriting data/hollow_cube/gnn_surrogate.pt with its own best
# checkpoints -- --gnn_ckpt_path points at the backup explicitly so this
# run isn't reading a surrogate that's changing under it mid-run (loaded
# once at startup either way, but the backup guarantees a known, stable
# checkpoint rather than whatever the training job has written by the time
# this job happens to start). Same trust-mechanism hyperparameters as
# jobs/optimize_hollow_cube_topopt_cnn.slurm (post job-47992/48002 tuning,
# see project_topopt_trust_deadlock.md memory).
#
# Fresh start: optimization/artifacts/topopt_checkpoint.pkl (+ history +
# screenshots) held an unrelated gen-81 run's state and was archived to
# optimization/artifacts_backup_gen81_*/screenshots_backup_gen81_* before
# this job was submitted (see feedback_checkpoint_resume_gotcha.md).
#
# --exclude=newton2: first submission (job 1337978) landed there and
# silently ran with cuda_available=False (newton2's CUDA driver is dead
# right now) instead of failing loudly -- no checkpoint/history was written
# before it was caught and cancelled, so this is a clean restart.
python -u -m optimization.optimize_dims \
    --geometry hollow_cube \
    --arch gnn \
    --optimizer topopt \
    --gnn_ckpt_path data/hollow_cube/gnn_surrogate.pt.bak_20260825_195327 \
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
