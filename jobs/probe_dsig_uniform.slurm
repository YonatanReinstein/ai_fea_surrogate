#!/bin/bash
#SBATCH --job-name=probe_dsig_unif
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
export CROSS4_LEG_RULE=avg CROSS4_FIXED_END_RADIUS=0 FEA_LOAD_MODE=tile_nodes FEA_LOAD_WEIGHT=pyramid
export CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0 FEA_MESH_RES=4,4,2 FEA_LOAD_N=5e7
python -m analysis.probe_dsig_uniform --ckpt "$CKPT" --t "${T:-0.2}"
