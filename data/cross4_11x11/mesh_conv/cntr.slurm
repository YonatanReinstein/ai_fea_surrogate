#!/bin/bash
#SBATCH --job-name=c4_cntr
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=public
#SBATCH --array=0-41
#SBATCH --output=data/cross4_11x11/mesh_conv/logs/%x_%A_%a.log
# CntrShape x thickness-range sweep: fixed CntrShape on every leg (flare rule
# off) and the dataset designs mapped into a t range that keeps the centre
# below CROSS4_MAX_CENTRE (no capping). Plain meshes, W=2. Task = config*7 + design.
cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/" IRIT_TMP=/tmp/irit_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} TMPDIR=/tmp
mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 PYTHONUNBUFFERED=1
export CROSS4_LEG_RULE=avg FEA_LOAD_MODE=tile_nodes CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0 FEA_LOAD_N=1e7
unset FEA_LOAD_ROW TOPOPT_STIFF_PENALTY TOPOPT_STRESS_RELAX HOLLOW_CUBE_CORNER_FRAC CROSS4_REFINE_ROW_K CROSS4_REFINE_COL_K CROSS4_CNTR_SHAPE CROSS4_FLARE_CURV
DESIGNS=(4225 2825 1650 1000 1975 2100 2700)
#        name    cntr  tmin  tmax
CFGS=("ref     -     -     -"
      "c20     0.2   0.05  0.32"
      "c30     0.3   0.05  0.29"
      "c40     0.4   0.05  0.26"
      "c50     0.5   0.05  0.24"
      "c30t10  0.3   0.10  0.29"
      "c40t10  0.4   0.10  0.24"
      "c40t08  0.4   0.08  0.24")
i=$SLURM_ARRAY_TASK_ID
set -- ${CFGS[$((i / 7))]}
NAME=$1; D=${DESIGNS[$((i % 7))]}
EXTRA=""
if [ "$2" != "-" ]; then export CROSS4_CNTR_SHAPE=$2 CROSS4_FLARE_CURV=0; EXTRA="--tmap $3,$4"; fi
python -u data/cross4_11x11/mesh_conv/conv_study.py --designs $D --res 4,4,2 6,6,2 8,8,2 12,12,2 16,16,2 24,24,2 \
    $EXTRA --out data/cross4_11x11/mesh_conv/cntr_$NAME
rm -rf "$IRIT_TMP"
