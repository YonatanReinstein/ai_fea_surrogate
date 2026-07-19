#!/bin/bash
#SBATCH --job-name=sweep_model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:L40:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log



cd "$SLURM_SUBMIT_DIR"
mkdir -p training/runs
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536


python -u -m training.sweep
