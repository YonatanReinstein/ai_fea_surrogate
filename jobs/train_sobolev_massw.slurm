#!/bin/bash
#SBATCH --job-name=train_sobolev_massw
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=740G
#SBATCH --time=36:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton2
#
# MASS-WEIGHTED gradient loss, continuing from the end of the low-LR
# continuation (job 1391627, epochs 13-48).
#
# WHY: 31 extra epochs of the equal-weight slog MSE halved the train grad loss
# and changed NOTHING that OC uses. Per-|dg|-decile sign agreement, probed on
# a TRAINING shard and on held-out designs, epoch 8 vs 43:
#     near-zero deciles D1-D4:  51-57% -> 55-58%   (chance, both sets)
#     top decile (96% of mass): 77-80% train, 90-92% held-out, flat
#     median cosine:            +0.88 train / +0.90 held-out, flat
# No train/held-out gap -> not overfitting; the loss was spent on the ~60% of
# tiles that carry <0.01% of the gradient mass and are unlearnable from random
# designs. The needle OC follows got ~10% of the loss.
#
# CHANGE: --grad_mass_weight 1.0 (per-tile slog MSE weighted by the tile's share
# of its sample's |true dg| mass, mean weight 1) and --grad_cos_weight 0.2 -> 0.5.
# Both push the gradient objective onto the high-mass tiles. Everything else as
# before; LR continues from the restored optimizer state (~3.1e-4).
#
# RESUME: seeds checkpoints_sobolev_massw/ with the LATEST *_epochs.pt of the
# continuation dir at job start (48_epochs.pt once 1391627 finishes; submitted
# with --dependency=afterany:1391627), plus its losses.json. --epochs 64 -> 16
# more epochs (~11h). Probe every epoch_NNN.pt with probe_dg_alignment.py.

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
ulimit -n 65536

echo "[gpu] allocated CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
HEALTHY=$(python - <<'PY'
import subprocess, sys
ok = []
# Probe each ordinal in its own process: a wedged device can abort the caller.
for i in range(8):
    p = subprocess.run(
        [sys.executable, "-c",
         "import torch,sys; torch.zeros(8,device='cuda:0').sum().item()"],
        env={"CUDA_VISIBLE_DEVICES": str(i), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=120,
    )
    if p.returncode == 0:
        ok.append(str(i))
print(",".join(ok))
PY
)
echo "[gpu] cuda-healthy ordinals: '$HEALTHY'"
NGPU=$(awk -F, '{print NF}' <<< "$HEALTHY")
if [ "$HEALTHY" = "" ]; then
    echo "[gpu] FATAL: no cuda-healthy GPU in the allocation. Not starting."
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$HEALTHY

# CUDA-usable is NOT enough: NCCL additionally needs NVML, and it was NVML
# that broke in job 1368949 while the per-ordinal CUDA probe above passed.
# Run the real collective (both the all_reduce and the broadcast that
# actually died) before committing to a multi-hour run.
CHK="$SLURM_SUBMIT_DIR/.nccl_check_$SLURM_JOB_ID.py"
cat > "$CHK" <<'NCCLCHK'
import torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
t = torch.ones(4, device="cuda") * (r + 1); dist.all_reduce(t)
b = torch.zeros(1, dtype=torch.long, device="cuda") if r else torch.tensor([7], dtype=torch.long, device="cuda")
dist.broadcast(b, src=0)
exp = sum(range(1, dist.get_world_size() + 1))
assert int(t[0].item()) == exp and b.item() == 7
if r == 0: print("[gpu] NCCL probe OK")
dist.destroy_process_group()
NCCLCHK
if [ "$NGPU" -lt 2 ]; then
    echo "[gpu] single rank -- no collectives, skipping NCCL probe"
elif timeout 300 torchrun --standalone --nproc_per_node=$NGPU "$CHK" 2>&1 | grep -q "NCCL probe OK"; then
    echo "[gpu] NCCL verified across $NGPU ranks"
else
    echo "[gpu] NCCL FAILED on '$HEALTHY' -- falling back to single GPU (no collectives)."
    NGPU=1
    export CUDA_VISIBLE_DEVICES=${HEALTHY%%,*}
fi
rm -f "$CHK"
echo "[gpu] running $NGPU rank(s) on ordinals $CUDA_VISIBLE_DEVICES"

SRC=data/hollow_cube/checkpoints_sobolev_lowlr_cont
DST=data/hollow_cube/checkpoints_sobolev_massw
mkdir -p "$DST" training/runs/train_sobolev_massw
LATEST=$(ls "$SRC"/*_epochs.pt | sed -E 's/.*\/([0-9]+)_epochs\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
echo "[resume] seeding $DST from $LATEST"
cp "$LATEST" "$DST"/
cp training/runs/train_sobolev_lowlr_cont/losses.json training/runs/train_sobolev_massw/

torchrun --standalone --nproc_per_node=$NGPU -m training.gnn_training \
    --geometry hollow_cube_ddptest \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --grad_from_train true \
    --grad_loss_weight 0.5 \
    --grad_cos_weight 0.5 \
    --grad_mass_weight 1.0 \
    --num_samples 50001 \
    --lr 0.0005 \
    --epochs 64 --batch_size 8 --grad_batch_size 4 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube/checkpoints_sobolev_massw \
    --run_dir training/runs/train_sobolev_massw
