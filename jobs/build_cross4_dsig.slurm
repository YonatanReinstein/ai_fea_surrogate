#!/bin/bash
#SBATCH --job-name=cross4_dsig
#SBATCH --partition=public
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --output=logs_dsig/build_%A_%a.log
#
# cross4_11x11 stress-sensitivity dataset (training/build_cross4_dsig_dataset.py):
# 2500 designs (20% field / 40% high-contrast field / 40% two-level), 11 load
# cases each (-5e7 N pyramid nodal load on every free-end tile), S + dS/dt per
# node, KS + dKS/dt per case. Array tasks share one --claim queue, so any
# number of tasks on any nodes can run and be restarted; ~110 s/design/core.
# Geometry = the CAD defaults (mean leg rule, free load end, mesh 4,4,2).
# Tile floor T_MIN=0.03 (dims.json says 0.01): below ~0.03 |dS| overflows float16
# and the design is skipped -- half of the 0.01-floor build was.
#   sbatch --array=0-23 --nodelist=protagoras jobs/build_cross4_dsig.slurm
cd "$SLURM_SUBMIT_DIR"; source env/bin/activate
export PATH="$PWD/irit/bin:$PATH" IRIT_PATH="$PWD/irit/bin/"
export IRIT_TMP=/tmp/irit_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CROSS4_LOAD_PAD_THICKNESS=0 CROSS4_PAD_THICKNESS=0.1 FEA_LOAD_MODE=tile_nodes FEA_LOAD_WEIGHT=pyramid FEA_LOAD_N=5e7
unset FEA_MESH_RES CROSS4_LEG_RULE FEA_LOAD_ROW CROSS4_FIXED_END_RADIUS TOPOPT_KS_EXCLUDE
OUT=${OUT:-data/cross4_11x11/dataset_dsig_2500_tmin003}
python -u -m training.build_cross4_dsig_dataset --output_dir "$OUT" --num_designs 2500 --seed 42 --t_min "${T_MIN:-0.03}" \
    --claim --worker "${SLURM_ARRAY_TASK_ID:-0}" --max_gb 740 --min_free_gb 50 --mesh_workers 2
rm -rf "$IRIT_TMP"
