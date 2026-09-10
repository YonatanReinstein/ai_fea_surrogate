#!/bin/bash
#SBATCH --job-name=perm_diag
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A6000:1
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PYTHONUNBUFFERED=1
echo "########## TRAIN-REGION (model was supervised on these) ##########"
python -m evaluators.probe_dg_permutation --ckpt data/hollow_cube/gnn_surrogate.pt \
    --dataset data/hollow_cube/grad_audit_train/dataset.pt --limit 60
echo
echo "########## HELD-OUT VAL ##########"
python -m evaluators.probe_dg_permutation --ckpt data/hollow_cube/gnn_surrogate.pt \
    --dataset data/hollow_cube/grad_audit/dataset.pt --limit 60
