#!/bin/bash
#SBATCH --job-name=train_model_gnn_3gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=650G
#SBATCH --time=34:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A6000:3
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd "$SLURM_SUBMIT_DIR"
mkdir -p training/runs/train_ccx_gnn_3gpu
source env/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

ulimit -n 65536

# Full 300-epoch gnn-coarse-arch run, mirrors jobs/train_model.slurm
# (transformer) except:
#  - --coarse_arch gnn + --use_compile true: gnn coarse-level message passing
#    is unaffected by the CUDA illegal-memory-access bug that forces
#    --use_compile false for the transformer coarse arch.
#  - --batch_size 44: single-GPU probes (200-sample decoy subset, same
#    hidden_dim=32/compile/checkpoint config) measured batch=36 at
#    26.4/32.8GB peak/reserved, batch=48 at 35.2/40.8GB, batch=64 OOM'd
#    (needed ~47.5GB, card has 47.33GB usable). gnn coarse level is much
#    cheaper than the transformer's attention block (which needed batch=28 to
#    stay under its 43.81/47.54GB batch=32 reading). 44 leaves headroom below
#    the 48 datapoint since the probe only covered the first 200/50k samples
#    and DDP adds some NCCL buffer overhead the single-GPU probe didn't capture.
unset CUDA_VISIBLE_DEVICES
torchrun --standalone --nproc_per_node=3 -m training.gnn_training \
    --geometry hollow_cube \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --num_samples 50000 \
    --epochs 300 \
    --lr 1e-3 \
    --resume_lr_scale 0.5 \
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
    --coarse_arch gnn \
    --use_compile true \
    --stress_ks_rho_mpa 0.15 \
    --checkpoint_dir data/hollow_cube/checkpoints_ccx_gnn_3gpu \
    --run_dir training/runs/train_ccx_gnn_3gpu
