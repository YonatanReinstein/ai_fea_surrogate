#!/bin/bash
#SBATCH --job-name=grad_audit_holdout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=public
#SBATCH --nodelist=protagoras
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Held-out FD-sensitivity audit set for comparing the grad-loss treatment
# checkpoint (data/hollow_cube/gnn_surrogate.pt) against the no-grad-loss
# baseline (data/hollow_cube_ddptest/gnn_surrogate.pt) -- see
# project_gradient_matching_loss.md. Must be genuinely unseen by BOTH
# checkpoints: the treatment run trained on the FULL 1500-sample
# data/hollow_cube/dataset_grad with no held-out split, so reusing any of
# that here would be in-sample for treatment but out-of-sample for
# baseline, an unfair comparison. Fresh seed (12345), separate output dir.
# CPU-only (no --gres) -- dataset_builder's ccx/mesh pipeline needs no GPU.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$SLURM_SUBMIT_DIR/irit/bin:$PATH"
export IRIT_PATH="$SLURM_SUBMIT_DIR/irit/bin/"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ulimit -n 65536

python -u -m training.dataset_builder \
    --geometry hollow_cube \
    --engine ccx \
    --num_samples 150 \
    --pool_size 16 \
    --nproc 1 \
    --mesh_threads 1 \
    --seed 12345 \
    --output_dir data/hollow_cube/dataset_grad_audit \
    --stress_ks_rho_mpa 0.15 \
    --field_random_base true \
    --grad_mode true \
    --grad_tiles_per_sample 6 \
    --grad_fd_frac 0.02
