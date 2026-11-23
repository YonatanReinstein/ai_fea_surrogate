#!/bin/bash
#SBATCH --job-name=test_dsig_eval
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$PWD/irit/bin:$PATH" IRIT_PATH="$PWD/irit/bin/" IRIT_TMP=/tmp/irit_$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export FEA_LOAD_MODE=tile_nodes FEA_LOAD_WEIGHT=pyramid CROSS4_LOAD_PAD_THICKNESS=0 CROSS4_PAD_THICKNESS=0.1 FEA_LOAD_N=5e7
python tests/test_gnn_dsig_evaluator.py training/runs/cross4_dsig_sobolev_v1
