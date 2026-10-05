#!/bin/bash
#SBATCH --job-name=screened20k
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=180 --mem=250G --time=10:00:00
#SBATCH --partition=public --nodelist=socrates
#SBATCH --output=optimization/optseed_test/screened20k_%j.log
#SBATCH --error=optimization/optseed_test/screened20k_%j.log
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/"
export IRIT_TMP=/tmp/irit_scr$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p "$IRIT_TMP"
export PYTHONPATH=. PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=1
ulimit -n 65536
python -u optimization/optseed_test/build_screened.py \
    --geometry hollow_cube_11x11 --target_n 20000 \
    --out data/hollow_cube_11x11/dataset_screened20k \
    --pool_size 90 --mesh_threads 2 --nproc 1 \
    --rho 0.25 --T 100 --sigma_e 80 --alpha 1.0 --band 50,200
echo SCREENED_DONE rc=$?
rm -rf "$IRIT_TMP"
