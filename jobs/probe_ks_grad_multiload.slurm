#!/bin/bash
#SBATCH --job-name=c4_ksgrad
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=public
#SBATCH --array=0-40
#SBATCH --output=logs_ksgrad/w%a_%A.log
# Solver side of evaluators/probe_ks_grad_multiload.py: 243 points (uniform
# design + central differences per tile), 6 per task. Resumable.
cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate
export PATH="$PWD/irit/bin:$PATH"
export IRIT_PATH="$PWD/irit/bin/" IRIT_TMP=/tmp/irit_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} TMPDIR=/tmp
mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CROSS4_LEG_RULE=avg FEA_LOAD_MODE=tile_nodes CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0
export FEA_MESH_RES=6,6,4 FEA_LOAD_N=1e7
unset FEA_LOAD_ROW TOPOPT_STIFF_PENALTY TOPOPT_STRESS_RELAX HOLLOW_CUBE_CORNER_FRAC
LO=$((SLURM_ARRAY_TASK_ID * 6)); HI=$((LO + 5)); [ $HI -gt 242 ] && HI=242
python -u -m evaluators.probe_ks_grad_multiload solve --out data/cross4_11x11/ksgrad_uniform030 --points $LO-$HI
rm -rf "$IRIT_TMP"
