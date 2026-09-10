#!/bin/bash
#SBATCH --job-name=train_gradloss_v2_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodelist=newton2
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Controlled baseline for jobs/train_gradloss_v2.slurm (see
# project_gradient_matching_loss.md): IDENTICAL config (same 4000-sample
# subset of data/hollow_cube/dataset_ccx, same architecture, same epoch
# count, same fixed torch.manual_seed(42) init/shuffle) but with the
# gradient-matching loss OFF (no --grad_dataset_path). Without this, any
# sensitivity-accuracy difference found between the grad-loss run and the
# old 50k-sample/251-epoch production checkpoint would be confounded with
# the huge dataset-size/training-budget difference -- this run isolates the
# grad-loss term as the only variable.
#
# --geometry hollow_cube_ddptest (decoy): same underlying CAD_model/mesh as
# real hollow_cube (symlinked), but its own gnn_surrogate.pt save path, so
# this can run concurrently with train_gradloss_v2.slurm (--geometry
# hollow_cube) without both jobs racing to overwrite the same checkpoint
# file (see feedback_decoy_geometry_for_benchmarks.md).

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536

torchrun --standalone --nproc_per_node=1 -m training.gnn_training \
    --geometry hollow_cube_ddptest \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --num_samples 4000 \
    --epochs 60 --batch_size 8 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube_ddptest/checkpoints_gradloss_v2_baseline \
    --run_dir training/runs/train_gradloss_v2_baseline
