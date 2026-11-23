#!/bin/bash
#SBATCH --job-name=train_dsig
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#
# Sobolev training on the cross4 stress-sensitivity dataset
# (data/cross4_11x11/dataset_dsig_2500_tmin003): stress tensor per node and load
# row, plus the directional dS/dt loss; see training/train_dsig.py.
#   sbatch --export=ALL,RUN_DIR=training/runs/cross4_dsig_sobolev_v1 jobs/train_dsig.slurm
#   smoke: EXTRA="--max_steps 200 --val_max_designs 4"

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
ulimit -n 65536
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m training.train_dsig --run_dir "${RUN_DIR:-training/runs/cross4_dsig_sobolev_v1}" ${EXTRA:-}
