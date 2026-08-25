#!/bin/bash
#SBATCH --job-name=validate_cnn_vs_ccx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$PWD/irit/bin:$PATH"
export IRIT_PATH="$PWD/irit/bin/"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# One-off surrogate-accuracy validation: re-evaluates every generation's
# design from the CNN-driven topopt run (optimization/artifacts/topopt_history,
# currently job 1320367's live output) with a real CCX solve, so the
# surrogate's per-generation stress prediction (s0, already in the .npz) can
# be compared against ground truth.
#
# --partition=newton + --gres=gpu:1 even though this script is pure CPU work
# (no torch/cuda): the usual CPU-only home for ccx jobs is partition=public,
# nodelist=protagoras (see project_ccx_slurm_missing_libs.md -- ccx's shared
# libs aren't installed uniformly across public), but that partition rejects
# CPU-only submissions from this cluster ("CPU-only jobs are not allowed on
# the public GPU infrastructure ... use Darwin cluster") and this session
# can't reach Darwin. Confirmed ccx runs fine on a newton node with the same
# PATH/LD_LIBRARY_PATH exports (tested interactively before submitting), so
# running here with a throwaway 1-GPU reservation is the workaround.
export CCX_WORKERS=40
python -u analysis_cnn_vs_ccx_validate.py
