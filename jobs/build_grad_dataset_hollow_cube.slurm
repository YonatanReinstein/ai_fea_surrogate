#!/bin/bash
#SBATCH --job-name=grad_dataset_ccx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=63
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --partition=public
#SBATCH --nodelist=protagoras
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Builds the ground-truth gradient-matching dataset (see
# training/dataset_builder.py's --grad_mode / _ccx_run_grad_sample): each
# sample is a base ccx run plus GRAD_TILES_PER_SAMPLE single-tile-perturbed
# reruns, finite-differencing the real solver's ks_stress. training/gnn_training.py's
# --grad_dataset_path consumes the merged output.
#
# Structurally mirrors jobs/run_dataset_ccx.slurm (same NUM_PARTS/CHUNK_SIZE/
# vm.max_map_count precautions -- see that file's comments for why), scaled
# down: each sample here costs (GRAD_TILES_PER_SAMPLE+1)x a plain dataset
# sample's mesh+solve time, so the target sample count is much smaller.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$SLURM_SUBMIT_DIR/irit/bin:$PATH"
export IRIT_PATH="$SLURM_SUBMIT_DIR/irit/bin/"

export PYTHONUNBUFFERED=1

# See jobs/run_dataset_ccx.slurm -- same reasoning (one worker process's own
# math-library thread pools fighting pool_size x other workers' for the same
# cores).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ulimit -n 65536

GEOMETRY="${GEOMETRY:-hollow_cube}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-1500}"
GRAD_TILES_PER_SAMPLE="${GRAD_TILES_PER_SAMPLE:-6}"
GRAD_FD_FRAC="${GRAD_FD_FRAC:-0.02}"

NUM_PARTS="${NUM_PARTS:-3}"
POOL_SIZE_PER_PART="${POOL_SIZE_PER_PART:-21}"
PER_PART_SAMPLES=$(( (NUM_SAMPLES + NUM_PARTS - 1) / NUM_PARTS ))
CHUNK_SIZE="${CHUNK_SIZE:-500}"

run_part() {
    local part_idx=$1 part_seed=$2 part_output_dir=$3 part_log=$4 final_target=$5
    local current_target=0
    while [ "${current_target}" -lt "${final_target}" ]; do
        current_target=$(( current_target + CHUNK_SIZE ))
        if [ "${current_target}" -gt "${final_target}" ]; then
            current_target=${final_target}
        fi
        echo "--- part ${part_idx}: chunk toward num_samples=${current_target}/${final_target} ---" >> "${part_log}"
        python -u -m training.dataset_builder \
            --geometry "${GEOMETRY}" \
            --engine ccx \
            --num_samples "${current_target}" \
            --pool_size "${POOL_SIZE_PER_PART}" \
            --nproc "${NPROC:-1}" \
            --mesh_threads "${MESH_THREADS:-1}" \
            --seed "${part_seed}" \
            --output_dir "${part_output_dir}" \
            --stress_ks_rho_mpa "${STRESS_KS_RHO_MPA:-0.15}" \
            --field_random_base "${FIELD_RANDOM_BASE:-true}" \
            --grad_mode true \
            --grad_tiles_per_sample "${GRAD_TILES_PER_SAMPLE}" \
            --grad_fd_frac "${GRAD_FD_FRAC}" \
            >> "${part_log}" 2>&1
    done
}

pids=()
for i in $(seq 0 $((NUM_PARTS - 1))); do
    PART_SEED=$((SEED + i))
    PART_OUTPUT_DIR="data/${GEOMETRY}/dataset_grad_part${i}"
    PART_LOG="grad_dataset_ccx_part${i}_${SLURM_JOB_ID}.log"
    run_part "${i}" "${PART_SEED}" "${PART_OUTPUT_DIR}" "${PART_LOG}" "${PER_PART_SAMPLES}" &
    pids+=($!)
    echo "Launched part ${i} (pid $!, seed ${PART_SEED}) -> ${PART_OUTPUT_DIR}, log: ${PART_LOG}, chunk_size=${CHUNK_SIZE}"
done

fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "Part ${i} (pid ${pids[$i]}) exited with a non-zero status -- see grad_dataset_ccx_part${i}_${SLURM_JOB_ID}.log"
        fail=1
    fi
done
if [ "${fail}" -ne 0 ]; then
    echo "One or more parts failed. Resubmit this same script (same GEOMETRY/SEED/NUM_SAMPLES/NUM_PARTS) to resume -- each part continues independently from its own existing shard count."
    exit "${fail}"
fi

echo "All parts done -- merging into data/${GEOMETRY}/dataset_grad ..."
python -u -m training.merge_ccx_parts \
    --geometry "${GEOMETRY}" \
    --num_parts "${NUM_PARTS}" \
    --parts_prefix "data/${GEOMETRY}/dataset_grad_part" \
    --output_dir "data/${GEOMETRY}/dataset_grad"
