#!/bin/bash
#SBATCH --job-name=screened_hc30k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=128 --mem=200G --time=36:00:00
#SBATCH --partition=public --nodelist=socrates
#SBATCH --output=optimization/optseed_test/screened_hc30k_%j.log
#SBATCH --error=optimization/optseed_test/screened_hc30k_%j.log
# hollow_cube = 840 tiles / 48 249 nodes. A worker peaks at ~2.89 GB here
# (vs 0.98 GB on the 11x11 mesh) because the adjoint holds an splu of a
# ~145k-DOF system, so 48 workers x 2.89 = 139 GB of the 200 G requested.
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/"
export IRIT_TMP=/tmp/irit_hc$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p "$IRIT_TMP"
export PYTHONPATH=. PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=1
ulimit -n 65536
python -u optimization/optseed_test/build_screened.py \
    --geometry hollow_cube --target_n 30000 \
    --out data/hollow_cube/dataset_screened30k \
    --coef 3.1838,-0.8339,0.3602 \
    --pool_size 64 --mesh_threads 2 --nproc 1 \
    --rho 0.25 --T 100 --sigma_e 80 --alpha 1.0 --band 50,200
echo SCREENED_DONE rc=$?
rm -rf "$IRIT_TMP"
