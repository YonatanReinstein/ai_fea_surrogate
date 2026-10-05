#!/bin/bash
#SBATCH --job-name=cut_band
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=4:00:00
#SBATCH --partition=public --nodelist=socrates
#SBATCH --output=optimization/optseed_test/cut_band_%j.log
#SBATCH --error=optimization/optseed_test/cut_band_%j.log
cd "$SLURM_SUBMIT_DIR"; source env/bin/activate
export PYTHONPATH=. PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
python -u optimization/optseed_test/cut_band.py \
    --src data/hollow_cube_11x11/dataset_gradadj \
    --out data/hollow_cube_11x11/dataset_band40_220 \
    --lo 40 --hi 220
echo CUT_DONE rc=$?
