#!/bin/bash
#SBATCH --job-name=train_sobolev_rho025_fresh
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=740G
#SBATCH --time=48:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=newton2
#
# FRESH rho-0.25 run from epoch 0: the clean control for the warm-started
# jobs/train_sobolev_rho025.slurm. Same recipe as the low-LR Sobolev run
# (lr 5e-4, grad_from_train, grad_loss 0.5) plus the two later additions
# (--grad_mass_weight 1.0, --grad_cos_weight 0.5), with --stress_ks_rho_mpa 0.25
# so pooling matches the dataset's KS targets and dKS/dt labels from step one.
# The warm start may stay in the flattened-peak basin the 0.15 pooling put it
# in; this run says what the architecture does when its objective is
# consistent throughout. 40 epochs (~28h on 1 GPU + dataset load).
#
# Queued with --dependency=afterany on the warm-start job so the two do not
# share newton2's 100Mb NIC during the dataset load.

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

torchrun --standalone --nproc_per_node=$NGPU -m training.gnn_training \
    --geometry hollow_cube_ddptest \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --grad_from_train true \
    --grad_loss_weight 0.5 \
    --grad_cos_weight 0.5 \
    --grad_mass_weight 1.0 \
    --stress_ks_rho_mpa 0.25 \
    --num_samples 50001 \
    --lr 0.0005 \
    --epochs 40 --batch_size 8 --grad_batch_size 4 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube/checkpoints_sobolev_rho025_fresh \
    --run_dir training/runs/train_sobolev_rho025_fresh
