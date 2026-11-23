#!/bin/bash
#SBATCH --job-name=c4_multiload
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=public
#SBATCH --array=0-46
#SBATCH --output=logs_multiload/w%a_%A.log
# Multi-load cross4_11x11 dataset (training/build_cross4_multiload_dataset.py):
# 47 workers sharing one queue (--claim); resumable -- resubmit to continue.
cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/" IRIT_TMP=/tmp/irit_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} TMPDIR=/tmp
mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CROSS4_LEG_RULE=avg FEA_LOAD_MODE=tile_nodes CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0
export FEA_MESH_RES=6,6,4 FEA_LOAD_N=1e7
unset FEA_LOAD_ROW TOPOPT_STIFF_PENALTY TOPOPT_STRESS_RELAX HOLLOW_CUBE_CORNER_FRAC
python -u -m training.build_cross4_multiload_dataset --output_dir data/cross4_11x11/dataset_multiload_5000 \
    --num_designs 5000 --worker ${SLURM_ARRAY_TASK_ID} --num_workers 47 --claim --max_gb 420 --min_free_gb 50
rm -rf "$IRIT_TMP"
