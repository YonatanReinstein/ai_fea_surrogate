#!/bin/bash
#SBATCH --job-name=labellers_n2
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=100
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=newton2
#
# Extra ccx-adjoint labellers joining the lagged-pipeline run RUN_NAME 1388342's spool.
# Labelling is the bottleneck there -- 93% of each ~82s generation is waiting
# for 10 labellers on 40 contended CPUs (84s/label). newton2 has ~108 idle
# CPUs. Its X550T links at 100baseT, but that does not matter for this work:
# ccx scratch already goes to tempfile.mkdtemp() = /tmp = local NVMe, the
# adjoint solve is in-memory, and only ~10KB spool/label files cross NFS.
# The one NFS writer was IRIT's relative "tmp" scratch -- IRIT_TMP now
# redirects it to local disk. The GPU is required by cluster policy for any
# newton job and sits idle here.
#
# No restart needed: the spool is a shared directory, so these labellers just
# start claiming from it alongside the ten on newton3.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/"
export IRIT_TMP=/tmp/irit_$USER_$SLURM_JOB_ID
export TMPDIR=/tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$IRIT_TMP"

RUN=onpolicy/${RUN_NAME:?set RUN_NAME}
NLAB=28
echo "[job] $NLAB labellers on newton2 joining $RUN/spool  (IRIT_TMP=$IRIT_TMP)"
for i in $(seq 1 $NLAB); do
  python -u -m training.trajectory_labeller --spool "$RUN/spool" --labels "$RUN/labels" \
      --workers 2 --nproc 1 --mesh_threads 2 --claim_order fifo --max_lag 0 --idle_exit 1800 \
      > "$RUN/labeller_n2_$i.log" 2>&1 &
done
wait
echo "[job] all labellers exited"
rm -rf "$IRIT_TMP"
