#!/bin/bash
#SBATCH --job-name=tileload_11x11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=66
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --partition=public
#SBATCH --nodelist=socrates
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#
# Tile-load dataset for hollow_cube_11x11 (training/build_tile_load_dataset.py):
# every sampled design solved under a -5e7 N (-y) load on EACH of the 11
# free-end-column tiles, from one factorization, with exact per-tile dKS/dt.
#
# SEED defaults to 43, not the pilot's 42: the scrambled Sobol sequence is the
# same prefix for the same seed, so seed 42 would repeat the 100 pilot designs
# and the pilot (dataset_tileload_pilot) could no longer serve as a held-out set.
#
#   NUM_GEOMETRIES=3000 sbatch --parsable jobs/build_tileload_11x11.slurm
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$SLURM_SUBMIT_DIR/irit/bin:$PATH"
export IRIT_PATH="$SLURM_SUBMIT_DIR/irit/bin/"
export PYTHONUNBUFFERED=1
# one process per core; keep every math library sequential inside it
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ulimit -n 65536

GEOMETRY="${GEOMETRY:-hollow_cube_11x11}"
NUM_GEOMETRIES="${NUM_GEOMETRIES:-3000}"
WORKERS="${WORKERS:-64}"
SEED="${SEED:-43}"
OUT="${OUT:-data/${GEOMETRY}/dataset_tileload_${NUM_GEOMETRIES}}"

echo "host $(hostname)  geometry ${GEOMETRY}  designs ${NUM_GEOMETRIES}  workers ${WORKERS}  seed ${SEED}  -> ${OUT}"
python -u -m training.build_tile_load_dataset \
    --geometry "${GEOMETRY}" \
    --num_geometries "${NUM_GEOMETRIES}" \
    --workers "${WORKERS}" \
    --seed "${SEED}" \
    --stress_ks_rho_mpa "${STRESS_KS_RHO_MPA:-0.25}" \
    --output_dir "${OUT}"
