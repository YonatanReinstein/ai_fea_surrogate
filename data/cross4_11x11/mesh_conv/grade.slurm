#!/bin/bash
#SBATCH --job-name=c4_meshgrade
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=public
#SBATCH --array=0-6
#SBATCH --output=data/cross4_11x11/mesh_conv/logs/%x_%A_%a.log
# Junction grading (CROSS4_GRADE_LEVELS/FRAC, model_grade binary), same
# designs and physics as conv.slurm; W=2 (one element through the depth).
cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/" IRIT_TMP=/tmp/irit_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} TMPDIR=/tmp
mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONUNBUFFERED=1
export CROSS4_LEG_RULE=avg FEA_LOAD_MODE=tile_nodes CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0 FEA_LOAD_N=1e7
unset FEA_LOAD_ROW TOPOPT_STIFF_PENALTY TOPOPT_STRESS_RELAX HOLLOW_CUBE_CORNER_FRAC CROSS4_REFINE_ROW_K CROSS4_REFINE_COL_K
DESIGNS=(4225 2825 1650 1000 1975 2100 2700)
D=${DESIGNS[$SLURM_ARRAY_TASK_ID]}
M=data/cross4_11x11/mesh_conv/model_grade/model
for cfg in "1 0.3" "2 0.3" "3 0.3" "1 0.2" "2 0.2" "3 0.2"; do
  set -- $cfg
  CROSS4_GRADE_LEVELS=$1 CROSS4_GRADE_FRAC=$2 python -u data/cross4_11x11/mesh_conv/conv_study.py --model $M \
    --designs $D --res 6,6,2 8,8,2 10,10,2 --out data/cross4_11x11/mesh_conv/grade_L$1_F$2
done
CROSS4_GRADE_LEVELS=3 CROSS4_GRADE_FRAC=0.3 python -u data/cross4_11x11/mesh_conv/conv_study.py --model $M \
    --designs $D --res 16,16,2 --out data/cross4_11x11/mesh_conv/grade_L3_F0.3
rm -rf "$IRIT_TMP"
