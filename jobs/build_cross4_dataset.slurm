#!/bin/bash
#SBATCH --job-name=cross4_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --partition=public
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#
# cross4_11x11 dataset (datasets/build_cross4_dataset.py): every design solved
# with the load on each free-end tile row, on the cross4 topopt pipeline
# (pads, rule rounding, C3D8I adjoint, exact per-tile dKS/dt). One process per
# core, ~2 GB each. Resumable: resubmit with the same OUT and SEED and finished
# (design, row) tasks are skipped.
#
#   NUM_GEOMETRIES=30000 sbatch --parsable jobs/build_cross4_dataset.slurm
#
# Physics settings are the opts' (jobs/topopt_ccx_baseline.slurm); override
# with the same variable names.
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$SLURM_SUBMIT_DIR/irit/bin:$PATH"
export IRIT_PATH="$SLURM_SUBMIT_DIR/irit/bin/"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ulimit -n 65536
unset FEA_LOAD_ROW                 # set per task by the builder

export FEA_LOAD_N="${FEA_LOAD_N:-5e6}"
export FEA_MESH_RES="${FEA_MESH_RES:-6,6,4}"
export CROSS4_PAD_THICKNESS="${CROSS4_PAD_THICKNESS:-0.1}"
export CROSS4_LOAD_PAD_THICKNESS="${CROSS4_LOAD_PAD_THICKNESS:-0.1}"
export TOPOPT_KS_EXCLUDE="${TOPOPT_KS_EXCLUDE:-pad}"

NUM_GEOMETRIES="${NUM_GEOMETRIES:-30000}"
WORKERS="${WORKERS:-$SLURM_CPUS_PER_TASK}"
SEED="${SEED:-43}"
ROWS="${ROWS:-all}"
OUT="${OUT:-data/cross4_11x11/dataset_rows_${NUM_GEOMETRIES}}"

echo "host $(hostname)  designs ${NUM_GEOMETRIES}  rows ${ROWS}  workers ${WORKERS}  seed ${SEED}  -> ${OUT}"
echo "FEA_LOAD_N=$FEA_LOAD_N FEA_MESH_RES=$FEA_MESH_RES pads $CROSS4_PAD_THICKNESS/$CROSS4_LOAD_PAD_THICKNESS KS exclude $TOPOPT_KS_EXCLUDE"
# Pass 1: every design with the load on SCREEN_ROW (the middle end tile).
# Pass 2: the other rows only for designs whose pass-1 KS <= SCREEN_MAX_KS MPa
# (thin-tile weak-link designs reach 1e3-3e4 MPa and are useless for training).
# SCREEN_MAX_KS=none skips the screen and runs every row for every design.
SCREEN_ROW="${SCREEN_ROW:-5}"
SCREEN_MAX_KS="${SCREEN_MAX_KS:-300}"
build() {
    python -u -m datasets.build_cross4_dataset \
        --num_geometries "${NUM_GEOMETRIES}" \
        --workers "${WORKERS}" \
        --seed "${SEED}" \
        --stress_ks_rho_mpa "${STRESS_KS_RHO_MPA:-0.25}" \
        --output_dir "${OUT}" "$@"
}
if [ "$SCREEN_MAX_KS" = "none" ]; then
    build --rows "${ROWS}" || exit 1
else
    echo "== pass 1: screen on row ${SCREEN_ROW}"
    build --rows "${SCREEN_ROW}" || exit 1
    echo "== pass 2: rows ${ROWS} for designs with KS <= ${SCREEN_MAX_KS} MPa"
    build --rows "${ROWS}" --screen_row "${SCREEN_ROW}" --screen_max_ks "${SCREEN_MAX_KS}" || exit 1
fi
