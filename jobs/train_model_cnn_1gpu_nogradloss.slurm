#!/bin/bash
#SBATCH --job-name=train_cnn_1gpu_nogradloss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=325G
#SBATCH --time=40:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton2
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd "$SLURM_SUBMIT_DIR"
mkdir -p training/runs/train_cnn_1gpu_nogradloss
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536

# Fast production CNN run: same recipe as jobs/train_model_cnn.slurm
# (50k samples, 300 epochs, coarse_arch=cnn, batch_size=44, hidden_dim=32)
# but 1 GPU instead of 2, --use_compile true, and NO --grad_dataset_path --
# job 1338703 (jobs/train_model_cnn_gradloss_2gpu.slurm) showed the
# grad-loss double-backward forces --use_compile false and costs ~3.9x
# per-epoch (1844s vs 470s), so this run trades the gradient-matching loss
# for compile's fusion speedup and no double-backward overhead, to get a
# checkpoint trained under the corrected tile-mapping (commit 907c0ef)
# faster, at the cost of not addressing the per-tile-sensitivity gap that
# job is aimed at.
#
# --geometry hollow_cube_ddptest (decoy, symlinked CAD_model/
# boundary_conditions.py to the real hollow_cube -- see
# feedback_decoy_geometry_for_benchmarks.md): gnn_training.py's best-
# checkpoint save path is data/{geometry}/gnn_surrogate.pt, fixed by
# --geometry alone. Job 1338703 is concurrently writing its own best
# checkpoints to the REAL data/hollow_cube/gnn_surrogate.pt; running this
# job under the same geometry would have both processes racing to
# overwrite that one file. --dataset still points at the real production
# dataset; only num_samples of it are actually used. Point --gnn_ckpt_path
# at data/hollow_cube_ddptest/gnn_surrogate.pt (this run's output) when
# using the result against the real hollow_cube geometry in topopt/eval --
# the checkpoint's weights don't reference the training geometry name.
#
# 1 GPU (not DDP-sharded): torchrun --nproc_per_node=1 is still used (not
# plain python) so the shard-based dataset loader path is taken, honoring
# --num_samples by only reading the shards it needs instead of loading the
# entire ~295GB dataset_ccx into memory first (see train_gradloss_v2.slurm).
#
# --exclude=newton2: confirmed still cuda_available=False as of this
# submission (test job 1339558) -- see feedback_dont_cancel_running_jobs.md
# sibling note in train_model_cnn_gradloss_2gpu.slurm's history for the
# original discovery.
unset CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node=1 -m training.gnn_training \
    --geometry hollow_cube_ddptest \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --num_samples 50000 \
    --epochs 300 \
    --lr 1e-3 \
    --batch_size 44 \
    --hidden_dim 32 \
    --conv_layers 6 \
    --node_loss_weight 0.5 \
    --graph_loss_weight 0.5 \
    --stress_weight_power 0.1 \
    --weight_decay 1e-3 \
    --use_checkpoint true \
    --checkpoint_fine_only true \
    --use_node_emb false \
    --coarse_arch cnn \
    --use_compile true \
    --stress_ks_rho_mpa 0.15 \
    --checkpoint_dir data/hollow_cube_ddptest/checkpoints_ccx_cnn_1gpu_nogradloss \
    --run_dir training/runs/train_cnn_1gpu_nogradloss
