#!/bin/bash
#SBATCH --job-name=sobolev_cos_smoke
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A6000:1
#
# Smoke for the new --grad_cos_weight direction term before spending ~9h on it.
# Answers only: (a) does the cosine term execute inside the double-backward
# without crashing, (b) how large is 1-cos next to the graph/node/slog terms,
# so the real run's weight is chosen from a measurement instead of a guess.
#
# --geometry hollow_cube_ddptest is a DECOY: its boundary_conditions.py is
# byte-identical to hollow_cube's (same tile grid, same physics) but the
# per-geometry gnn_surrogate.pt save path is fixed and NOT checkpoint_dir
# scoped, so training under the real name would clobber the Sobolev model
# from job 1368050. The dataset still points at the real hollow_cube data.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
ulimit -n 65536

for W in 0.05 0.2; do
  echo "################ grad_cos_weight=$W ################"
  rm -rf data/hollow_cube_ddptest/checkpoints_cos_smoke_$W
  torchrun --standalone --nproc_per_node=1 -m training.gnn_training \
      --geometry hollow_cube_ddptest \
      --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
      --grad_dataset_path data/hollow_cube/gradsub_smoke/dataset.pt \
      --grad_loss_weight 0.5 \
      --grad_cos_weight $W \
      --num_samples 2000 \
      --epochs 3 --batch_size 8 --grad_batch_size 4 \
      --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
      --use_compile false \
      --checkpoint_dir data/hollow_cube_ddptest/checkpoints_cos_smoke_$W \
      --run_dir training/runs/cos_smoke_$W
done
