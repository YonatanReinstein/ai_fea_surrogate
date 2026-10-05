#!/bin/bash
#SBATCH --job-name=train_hc840_ks40_220
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=96:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton2
#
# hollow_cube (840 tiles, 29x29, 48,249-node mesh) on the KS-filtered 50k dataset:
# data/hollow_cube/dataset_ks40_220 = dataset_gradadj samples with KS in [40, 220] MPa
# (~30k of 50k, made by jobs/filter_hollow_cube_ks40_220.slurm). Its shards vary in
# size; gnn_training reads each shard's size from its name.
# Same loss as the 11x11 screened run (jobs/train_11x11_screened.slurm): ks, node,
# grad (percent error, ^0.15 weights, sat 2) and sign losses at 0.25 each, each
# divided by its know-nothing baseline. No cosine loss.
# ONE GPU. The 2-GPU attempt (job 1430529) died before epoch 1: rank 0 loads its
# train half PLUS all val shards (~60% of the data), and rank 1 timed out waiting
# for it at the normalization-stats broadcast after ~5 h of NFS reading. With a
# single process there is no cross-rank wait.
# Batch 12 / grad batch 6 (the 2-GPU smoke test peaked at 27.7 GB per GPU at 12/6).
# One process holds all ~30k samples (~200 GB), hence --mem 300G; no newton2 (100 Mb
# NIC). Expect several hours of loading, then roughly 40-60 min/epoch (the old 29x29
# 1-GPU run did ~42 min/epoch at batch 8), so 80 epochs is ~2.5-3.5 days; --time 96h.
# Checkpoints every 10 epochs, so a resubmit resumes.
# Decoy geometry hollow_cube_ks40_220, so data/hollow_cube/gnn_surrogate.pt is untouched.

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
    --geometry hollow_cube_ks40_220 \
    --dataset data/hollow_cube/dataset_ks40_220/dataset.pt \
    --grad_from_train true \
    --grad_err_mode rel --grad_weight_power 0.15 --grad_rel_sat 2.0 \
    --graph_loss_weight 0.25 --node_loss_weight 0.25 \
    --grad_loss_weight 0.25 \
    --grad_sign_loss_weight 0.25 --grad_sign_delta 0.05 \
    --stress_ks_rho_mpa 0.25 \
    --num_samples 60000 \
    --lr 0.0005 \
    --epochs 80 --batch_size 12 --grad_batch_size 6 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube/checkpoints_ks40_220 \
    --run_dir training/runs/train_hollow_cube_ks40_220
