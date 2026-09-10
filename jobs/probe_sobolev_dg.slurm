#!/bin/bash
#SBATCH --job-name=probe_sobolev_dg
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A6000:1
#
# The deciding metric for the Sobolev run (job 1368050): per-tile dg against
# the ccx adjoint, which is what optimization/topopt_optimizer.py's OC update
# actually consumes. Val stress loss cannot see this -- the original defect was
# a surrogate whose *values* were fine while its per-tile dg was 10-280x too
# large and sign-flipped on ~half of tiles.
#
# Audit set = the tail val shards of dataset_gradadj. Neither checkpoint
# trained on them, and the grad-loss subset (gradsub) drew only from shards
# 0..400, so they never fed the derivative supervision either.
#
# baseline = pre-Sobolev production model. SAME architecture (cnn / hidden 32 /
# coarse 32 / depth 3 / 6 conv layers), so the arch is not a confound -- but it
# ran 81 epochs on dataset_ccx vs the treatment's 30 on dataset_gradadj, so
# epochs and dataset ARE confounded. Read the dg numbers, not the val loss.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export PYTHONUNBUFFERED=1

python -m evaluators.probe_gradloss_sensitivity \
    --geometry hollow_cube \
    --treatment_ckpt data/hollow_cube/gnn_surrogate.pt \
    --baseline_ckpt data/hollow_cube/gnn_surrogate.pt.bak_pre_sobolev \
    --holdout_dataset data/hollow_cube/grad_audit/dataset.pt
