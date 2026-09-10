#!/bin/bash
#SBATCH --job-name=train_sobolev_adj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=740G
#SBATCH --time=20:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:A6000:3
#SBATCH --nodelist=newton2
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# Sobolev (derivative-supervision) training on the 50,001-sample adjoint
# gradient dataset data/hollow_cube/dataset_gradadj.
#
# GOAL: fix the per-tile dg the OC topopt loop actually consumes. Base-stress
# error was already small; dg was 10-280x too large and sign-flipped on ~half
# of tested tiles. Plain MSE never looks at the slope, so the loss below adds
# torch.autograd.grad(..., create_graph=True) on the model's OWN derivative
# (gnn_training.py ~line 927) against the ccx adjoint labels, in signed-log
# space (~line 937) because |dKS/dX| spans four orders of magnitude.
#
# TWO THINGS THIS SCRIPT WORKS AROUND -- do not "simplify" them away:
#
# 1. --grad_dataset_path points at a SUBSET (data/hollow_cube/gradsub),
#    not the full dataset. gnn_training.py loads the grad set in full on
#    EVERY rank, which was fine when its comment (~line 500) was written --
#    the old FD builder cost M+1 real solves per sample, so that set was
#    small by construction. The adjoint builder broke that: all 50,001
#    samples now carry labels at ~1.1 solves each. gradsub is 41 symlinked
#    shards (~4100 samples) taken ONLY from the train region (shards
#    0..400); val is 401..500, so grad supervision never sees a val sample.
#
# 2. --grad_loss_weight 0.5, not the 0.1 default. With correct label units
#    the grad term sits near 0.20 while total loss is ~1.0, so at 0.1 it
#    contributes ~2% of the loss -- too weak to reshape a gradient field
#    that is currently 10-280x off. 0.5 puts it near 10%. This is the first
#    knob to revisit if graph/node stress accuracy regresses.
#
# 3. --num_samples 50001 is REQUIRED, not decorative. gnn_training.py's
#    --num_samples defaults to 3000 (not None/all), and the DDP shard path
#    trims the shard pool to ceil(num_samples/100) shards BEFORE the
#    train/val split. Omitting it silently trains on 3000 of the 50,001
#    samples -- job 1368049 did exactly that and was killed for it.
#
# Sizing: 3-way DDP shards the main set per rank (~13.4k train samples each,
# val on rank 0 only). newton2 was chosen because it was the only newton node
# with the free RAM -- newton4/5 had 20GB/47GB free at launch.
#
# use_compile stays false: torch.compile over the double-backward is the
# known-risky combination and was never validated here.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
ulimit -n 65536

torchrun --standalone --nproc_per_node=3 -m training.gnn_training \
    --geometry hollow_cube \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --grad_dataset_path data/hollow_cube/gradsub/dataset.pt \
    --grad_loss_weight 0.5 \
    --num_samples 50001 \
    --epochs 30 --batch_size 8 --grad_batch_size 4 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube/checkpoints_sobolev_adj \
    --run_dir training/runs/train_sobolev_adj
