#!/bin/bash
#SBATCH --job-name=dataset_builder_ccx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=189
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --partition=public
#SBATCH --nodelist=protagoras
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$SLURM_SUBMIT_DIR/irit/bin:$PATH"
export IRIT_PATH="$SLURM_SUBMIT_DIR/irit/bin/"

export PYTHONUNBUFFERED=1

# Each worker process forks its own copy of PyTorch's CPU intra-op thread
# pool the first time it touches a tensor (thread pools don't survive
# fork()), and that pool defaults to nproc() -- 190 here, since that's this
# job's full cpuset -- unless told otherwise. With pool_size workers each
# independently grabbing ~190 threads, the parallelism this job actually
# wants (one process per core) turns into pool_size x 190 threads fighting
# over the same 190 real cores. Pin every worker's math-library thread pools
# to 1: real parallelism already comes from running many separate worker
# processes, not from each process also being multithreaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Killed worker processes leave a zombie + a parent-side pipe/sentinel fd
# behind until reaped (see the p.join() calls added after every p.kill() in
# dataset_builder.py). The default soft fd limit (1024) gets exhausted after
# only a handful of retry passes at pool_size=180, after which
# ProcessPoolExecutor() itself can't even open its own internal pipe --
# every subsequent pass fails instantly with "Too many open files" without
# doing any real work (job 48064, 2026-08-17). Raise the soft limit to match
# more of the hard limit (131072) as a second line of defense.
ulimit -n 65536

GEOMETRY="${GEOMETRY:-hollow_cube}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"

# Split into NUM_PARTS independent dataset_builder.py processes instead of
# one big pool_size=130 ProcessPoolExecutor. A BrokenProcessPool crash (see
# project_dataset_pool_split) forces killing and re-forking every worker in
# whichever pool it happened in -- at pool_size=130 that's 130 workers to
# re-fork per retry, under whatever memory pressure caused the crash in the
# first place, which was observed spiraling through all 30 retry passes
# without ever succeeding (jobs 48186, 48187). Splitting the same total
# concurrency across NUM_PARTS independent pools bounds the blast radius: a
# crash in one part only tears down and re-forks that part's own
# POOL_SIZE_PER_PART workers, while the other parts keep completing samples
# uninterrupted, and each retry has a much smaller pool to successfully
# re-fork. NUM_PARTS x POOL_SIZE_PER_PART should match --cpus-per-task
# above (default 3 x 63 = 189, just under this account's 190-core cap) --
# if you override either at submit time, pass a matching --cpus-per-task to
# sbatch too, since the #SBATCH directive is fixed at submission and can't
# read these env vars.
NUM_PARTS="${NUM_PARTS:-3}"
POOL_SIZE_PER_PART="${POOL_SIZE_PER_PART:-63}"
PER_PART_SAMPLES=$(( (NUM_SAMPLES + NUM_PARTS - 1) / NUM_PARTS ))

# Each dataset_builder.py process's own memory-mapped-region count climbs
# monotonically over its lifetime -- one (or more) new mapping per shared
# tensor received back from a worker, apparently never unmapped again even
# once cleanup elsewhere is working correctly. Confirmed directly on job
# 48195 (2026-08-20): every part hit the kernel's per-process
# vm.max_map_count ceiling (65530) after ~4500-4600 samples, at which point
# *any* new mmap in that process -- a newly received tensor, or even
# ProcessPoolExecutor's own internal setup -- fails with "Cannot allocate
# memory", regardless of how much actual RAM or /dev/shm space is free (see
# project_dataset_pool_split.md; this is a separate bug from the
# torch_shm_manager /dev/shm leak fixed in _kill_and_reap_pool). Splitting
# into multiple pools doesn't help here at all -- the ceiling is per
# *process*, not shared across pools like /dev/shm was.
#
# Fix: never let one dataset_builder.py process live long enough to
# approach that ceiling. Run each part in CHUNK_SIZE-sample increments
# instead of one shot for its whole target -- each chunk's process exits
# (cleanly, thanks to the os._exit(0) added after build_dataset() returns)
# and a fresh one launches for the next chunk, so no single process's map
# count ever gets close to 65530. 2000 is a >2x safety margin under the
# ~4500 observed threshold.
CHUNK_SIZE="${CHUNK_SIZE:-2000}"

# field_random_base=true: each field-sampled example's pre-noise anchor is
# drawn fresh, uniformly across the tile's real [min, max] range, instead of
# every example being centered on one fixed default (0.3, dims.json's
# mid-range value). A fixed anchor only ever produced training examples
# clustered around one overall baseline volume, so the surrogate had ~zero
# coverage of the thin/low-volume designs a topopt trajectory actually walks
# toward -- confirmed directly: re-evaluating a CNN-driven topopt run's
# per-generation designs against real CCX showed the surrogate's stress
# error growing monotonically to -90% by generation 48 as designs got
# thinner (see project_gnn_anchor_sensitivity_bug.md). This dataset targets
# that gap. FIELD_BASE_THICKNESS is ignored whenever this is on.
EXTRA_ARGS=()
if [ -n "${FIELD_BASE_THICKNESS:-}" ]; then
    EXTRA_ARGS+=(--field_base_thickness "${FIELD_BASE_THICKNESS}")
fi

# Each part is a fully independent dataset_builder.py run -- its own seed
# (so parts don't sample identical designs) and its own --output_dir (so
# concurrent shard writers never collide). Each one already resumes on its
# own if killed mid-run: dataset_builder.py's start_idx is read fresh from
# however many samples are already checkpointed under that part's own
# output_dir, so resubmitting this exact script (same GEOMETRY/SEED/
# NUM_SAMPLES/NUM_PARTS) picks every part back up from wherever it left off
# -- no extra state tracking needed. Once all parts are done (or whenever a
# consolidated dataset is wanted), combine them with:
#   python -m training.merge_ccx_parts --geometry "${GEOMETRY}" --num_parts "${NUM_PARTS}"
#
# run_part loops dataset_builder.py itself in CHUNK_SIZE increments (see
# CHUNK_SIZE above) instead of invoking it once for the part's whole
# target, so no single process lives long enough to approach
# vm.max_map_count. --num_samples is always the CUMULATIVE target so far
# (dataset_builder.py resumes from its own on-disk shard count regardless
# of what target we ask for), so this is exactly the same resume mechanism
# as before, just invoked repeatedly instead of once.
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
            --field_amplitude_frac_up "${FIELD_AMPLITUDE_FRAC_UP:-0.10}" \
            --field_amplitude_frac_down "${FIELD_AMPLITUDE_FRAC_DOWN:-0.10}" \
            --field_random_base "${FIELD_RANDOM_BASE:-true}" \
            "${EXTRA_ARGS[@]}" \
            >> "${part_log}" 2>&1
    done
}

pids=()
for i in $(seq 0 $((NUM_PARTS - 1))); do
    PART_SEED=$((SEED + i))
    PART_OUTPUT_DIR="data/${GEOMETRY}/dataset_ccx_part${i}"
    PART_LOG="dataset_builder_ccx_part${i}_${SLURM_JOB_ID}.log"
    run_part "${i}" "${PART_SEED}" "${PART_OUTPUT_DIR}" "${PART_LOG}" "${PER_PART_SAMPLES}" &
    pids+=($!)
    echo "Launched part ${i} (pid $!, seed ${PART_SEED}) -> ${PART_OUTPUT_DIR}, log: ${PART_LOG}, chunk_size=${CHUNK_SIZE}"
done

fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "Part ${i} (pid ${pids[$i]}) exited with a non-zero status -- see dataset_builder_ccx_part${i}_${SLURM_JOB_ID}.log"
        fail=1
    fi
done
if [ "${fail}" -ne 0 ]; then
    echo "One or more parts failed. Resubmit this same script (same GEOMETRY/SEED/NUM_SAMPLES/NUM_PARTS) to resume -- each part continues independently from its own existing shard count."
fi
exit "${fail}"
