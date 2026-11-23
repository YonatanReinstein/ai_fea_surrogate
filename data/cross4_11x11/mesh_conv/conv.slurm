#!/bin/bash
#SBATCH --job-name=c4_meshconv
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=public
#SBATCH --array=0-9
#SBATCH --output=data/cross4_11x11/mesh_conv/logs/%x_%A_%a.log
# Mesh convergence, cross4_11x11 multiload dataset physics. Tasks 0-6: one
# dataset design each, default rounding. Tasks 7-9: no rounding (sharp cross).
cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/" IRIT_TMP=/tmp/irit_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} TMPDIR=/tmp
mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONUNBUFFERED=1
export CROSS4_LEG_RULE=avg FEA_LOAD_MODE=tile_nodes CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0 FEA_LOAD_N=1e7
unset FEA_LOAD_ROW TOPOPT_STIFF_PENALTY TOPOPT_STRESS_RELAX HOLLOW_CUBE_CORNER_FRAC CROSS4_REFINE_ROW_K CROSS4_REFINE_COL_K
DESIGNS=(4225 2825 1650 1000 1975 2100 2700 1000 1975 2825)
D=${DESIGNS[$SLURM_ARRAY_TASK_ID]}
if [ "$SLURM_ARRAY_TASK_ID" -ge 7 ]; then
  export CROSS4_CNTR_SHAPE=0 CROSS4_BNDRY_SHAPE=0 CROSS4_FLARE_CURV=0
  OUT=data/cross4_11x11/mesh_conv/sharp
  RES="4,4,3 6,6,3 8,8,3 12,12,3 16,16,3 24,24,3"
else
  OUT=data/cross4_11x11/mesh_conv/round
  RES="4,4,3 6,6,4 6,6,3 6,6,2 8,8,2 8,8,3 8,8,4 8,8,6 5,5,3 10,10,3 12,12,2 12,12,3 12,12,5 4,8,3 8,4,3 16,16,3 20,20,3 24,24,3 32,32,3"
fi
python -u data/cross4_11x11/mesh_conv/conv_study.py --designs $D --res $RES --out $OUT
rm -rf "$IRIT_TMP"
