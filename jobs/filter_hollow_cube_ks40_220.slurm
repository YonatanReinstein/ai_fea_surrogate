#!/bin/bash
#SBATCH --job-name=filter_ks40_220
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --exclude=newton2
#SBATCH --gres=gpu:1
#
# Cut data/hollow_cube/dataset_gradadj (840 tiles, 50,001 samples, 322 GB) down to
# samples with KS stress in [40, 220] MPa -> data/hollow_cube/dataset_ks40_220.
# A 500-sample spot check (5 shards spread over the set) kept 60.6%, ~30k samples.
# Pure I/O, but the newton cluster rejects CPU-only jobs, hence the (idle) gpu:1.
# newton2 excluded: its NIC is ~100 Mb, and this job is pure NFS reading.
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u -m training.filter_dataset --src data/hollow_cube/dataset_gradadj \
    --dst data/hollow_cube/dataset_ks40_220 --ks_min 40 --ks_max 220 --workers 16
