#!/bin/bash
#SBATCH --job-name=optseed12
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=96G --time=2:00:00
#SBATCH --partition=public --nodelist=socrates
#SBATCH --output=optimization/optseed_test/B_%j.log
#SBATCH --error=optimization/optseed_test/B_%j.log
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/"
export IRIT_TMP=/tmp/irit_B$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p $IRIT_TMP
export PYTHONPATH=. PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
rm -f optimization/optseed_test/B_[01][0-9].json
for i in $(seq 0 11); do python -u optimization/optseed_test/optseed1.py $i & done
wait
echo ALLDONE
rm -rf $IRIT_TMP
