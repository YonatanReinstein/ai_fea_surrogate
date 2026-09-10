#!/bin/bash
# SIZING IS MEMORY-BOUND, NOT CPU-BOUND. First attempt (job 65492) ran 159
# single-threaded workers and two of its three parts died before producing a
# single sample:
#
#   as_completed() saw 300s with zero new completions (0/2000 resolved)
#   -- pool's manager thread likely died silently.
#
# 159 workers x 2.89 GB = 460 GB against ~290 GB free. Under that pressure
# fork/exec is slow enough to overrun the torch_shm_manager handshake that
# dataset_builder.py's random.uniform(0, 4.0) stagger exists to spread (see
# the comment at its _worker_warmup, and [[project-dataset-builder-ccx-fd-leak]]);
# that window was tuned when a worker was a fraction of this size. The
# surviving part also ran ~16 min/sample against the ~90 s a 4-thread
# measurement predicted, i.e. it was thrashing rather than computing.
#
# So: 3 x 26 = 78 workers at 2 threads each (156 CPUs, 225 GB resident, well
# inside 300G), and part launches are staggered by PART_LAUNCH_STAGGER so the
# three fork bursts do not overlap. Fewer, fatter workers is the right trade
# here -- the adjoint's splu and einsum both thread, while an extra concurrent
# worker costs a further 2.89 GB of a budget that is already the binding
# constraint.
#
# THIS DATASET REPLACES data/hollow_cube/dataset_ccx, which was deleted to
# make room (295 G of a ~700 G quota). That is sound because an adjoint
# sample is a strict SUPERSET of a dataset_ccx sample: identical ccx stress
# labels, plus exact per-node and per-tile gradients. Sampler settings below
# are copied from jobs/run_dataset_ccx.slurm so the design distribution is
# unchanged and the value model is not perturbed for unrelated reasons.
#
# TWO DELIBERATE DIFFERENCES FROM run_dataset_ccx.slurm:
#
# 1. STRESS_KS_RHO_MPA is 0.25, not 0.15. The gradient dKS/dt depends on the
#    KS sharpness, and optimization runs at --stress_ks_rho_mpa 0.25. Labelling
#    at 0.15 would train the surrogate on one functional's gradient and then
#    consume it as another's -- the exact decoupling that was already found
#    and fixed once in ShapeAdjoint.tile_gradient, where rho was hardcoded to
#    0.15 while the evaluator passed 0.25 to ks() only. There is no
#    compatibility cost now that dataset_ccx is gone.
#
# 2. MEMORY. --mem is 600G, not 220G. A worker peaks at 2.89 GB RSS (measured:
#    0.65 GB after imports, 2.89 GB through a full adjoint sample) because the
#    adjoint holds an splu factorization of a ~145k-DOF system. 189 workers x
#    2.89 GB = 546 GB. The 220G that sufficed for ccx-only sampling would OOM
#    this immediately; so would this file's original 80G at 63 workers.
#    protagoras has 512 CPUs / 1 TB, so this fits with headroom. The CPU
#    count is 159 (3 x 53) rather than run_dataset_ccx.slurm's 189 because
#    the account's QOS caps total CPUs per user: 189 pended on
#    QOSMaxCpuPerUserLimit while a 16-CPU optimization job was running.
#
# Throughput: ~90 s/sample (ccx ~45 s + adjoint solve 27.6 + gradient 6.9 +
# Jacobian 6.9) over 159 workers = ~7.9 h for 50k. Resumable: each part
# continues from its own shard count, so resubmit this same script to extend
# or to recover from a timeout.
#
# EXACT gradient labels from the adjoint, replacing the finite-difference
# builder this file was forked from.
#
# WHAT CHANGED. core/fem_adjoint.py now reproduces ccx's C3D8I to 1e-5%
# (KS ratio 1.0000 on four of job 54664's own designs; gradient sum rule
# 0.9987-1.0010 at thick, thin and thinned designs). So one adjoint solve on
# the mesh ccx has already built labels EVERY node and EVERY tile exactly,
# where the FD builder spent 6 extra full solves (~166 s) to difference SIX
# tiles. Measured per sample on hollow_cube at 3,3,3:
#
#   adjoint solve (factorize + u)   27.6 s   unavoidable
#   gradient() -> dKS/dX             6.9 s   ~48k x 3 per-node labels
#   CAD Jacobian dX/dt               6.9 s   per-tile labels
#   chain to per-tile dKS/dt         0.2 s
#
# i.e. exact labels for every design variable cost ~40 s on top of a sample
# that already runs ccx, against ~166 s for six differenced tiles. Roughly a
# 20x improvement in gradient information per second, and exact rather than
# differenced.
#
# ccx still writes the stress labels, unchanged, so this dataset stays
# label-compatible with everything built before it. Each sample also carries
# ks_stress_adjoint: it should track ccx's KS to ~4 decimals, and a drift
# means the C3D8I agreement has broken. That guard exists because plain C3D8
# agreed at the uniform design and read 32% low on thinned ones, and nothing
# noticed for four optimization jobs.
#
# GRAD_STORE_NODES=false drops the per-node field (0.54 MB/sample) and keeps
# only the per-tile labels (~3 KB/sample) -- set it if disk, not solver time,
# is the binding constraint.
#
#SBATCH --job-name=grad_dataset_adjoint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=159
#SBATCH --mem=300G
#SBATCH --time=24:00:00
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
export OMP_NUM_THREADS=${FEA_THREADS:-2}
export MKL_NUM_THREADS=${FEA_THREADS:-2}
export OPENBLAS_NUM_THREADS=${FEA_THREADS:-2}
export NUMEXPR_NUM_THREADS=1

ulimit -n 65536

GEOMETRY="${GEOMETRY:-hollow_cube}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"

NUM_PARTS="${NUM_PARTS:-3}"
POOL_SIZE_PER_PART="${POOL_SIZE_PER_PART:-26}"
PER_PART_SAMPLES=$(( (NUM_SAMPLES + NUM_PARTS - 1) / NUM_PARTS ))
CHUNK_SIZE="${CHUNK_SIZE:-2000}"

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
            --mesh_threads "${MESH_THREADS:-2}" \
            --seed "${part_seed}" \
            --output_dir "${part_output_dir}" \
            --stress_ks_rho_mpa "${STRESS_KS_RHO_MPA:-0.25}" \
            --field_amplitude_frac_up "${FIELD_AMPLITUDE_FRAC_UP:-0.10}" \
            --field_amplitude_frac_down "${FIELD_AMPLITUDE_FRAC_DOWN:-0.10}" \
            --field_random_base "${FIELD_RANDOM_BASE:-true}" \
            --grad_mode adjoint \
            --grad_store_nodes "${GRAD_STORE_NODES:-true}" \
            >> "${part_log}" 2>&1
    done
}

pids=()
for i in $(seq 0 $((NUM_PARTS - 1))); do
    PART_SEED=$((SEED + i))
    PART_OUTPUT_DIR="data/${GEOMETRY}/dataset_gradadj_part${i}"
    PART_LOG="grad_dataset_ccx_part${i}_${SLURM_JOB_ID}.log"
    if [ "${i}" -gt 0 ]; then sleep "${PART_LAUNCH_STAGGER:-90}"; fi
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

echo "All parts done -- merging into data/${GEOMETRY}/dataset_gradadj ..."
python -u -m training.merge_ccx_parts \
    --geometry "${GEOMETRY}" \
    --num_parts "${NUM_PARTS}" \
    --parts_prefix "data/${GEOMETRY}/dataset_gradadj_part" \
    --output_dir "data/${GEOMETRY}/dataset_gradadj"
