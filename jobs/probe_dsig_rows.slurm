#!/bin/bash
#SBATCH --job-name=probe_dsig_rows
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PYTHONUNBUFFERED=1
python -m analysis.probe_dsig_rows --ckpt "${CKPT:-training/runs/cross4_dsig_sobolev_v1/epoch_026.pt}"
