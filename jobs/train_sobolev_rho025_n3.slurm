#!/bin/bash
#SBATCH --job-name=train_sobolev_rho025_n3
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=36:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=newton3   # fast NIC; takes the GPU the interactive session releases
#
# RHO 0.25: continue the mass-weighted run from epoch 64 with the KS pooling
# rho set to 0.25 -- the rho the dataset was actually built at.
#
# WHY: jobs/build_grad_dataset_adjoint_hollow_cube.slurm defaults to rho 0.25,
# and the stored ks_stress targets are exactly KS(node_stress) at 0.25 (sample
# 0: 77.388 stored, 77.388 recomputed; at 0.15 it would be 93.828). The
# adjoint gradient labels are dKS_0.25/dt. But gnn_training.py defaults
# --stress_ks_rho_mpa to 0.15 and no training job ever passed it, so every
# checkpoint pooled its node map at 0.15 against 0.25 targets. The network
# reconciled that by flattening its stress peak: pooled at 0.15 it predicts
# ~81 MPa on the uniform design (ccx: 93.9 at 0.15, 76.7 at 0.25), pooled at
# 0.25 it collapses to 57. Every optimizer/labeller run then used 0.15, i.e.
# solved the rho-0.15 problem (whose ccx-driven optimum is ~400, not 279).
#
# This run makes value target, gradient labels and pooling consistent for
# the first time. Expect the value loss to jump at epoch 65 (the peak has to
# be re-learned) and the Sobolev term to improve. A [rho-check] line at load
# now verifies dataset vs configured rho. Probe every epoch through the
# evaluator path (probe fixed 2026-09-15).
#
# newton3 variant: 12k SAMPLES (first 120 shards: 96 train / 24 val, ~77GB) --
# newton3 had ~111GB free at submit time. Separate checkpoint/run dirs from the
# newton2 20k job (1398349) so the two can run side by side.
# (original note:) 20k SAMPLES (first 200 shards: 160 train / 40 val, ~130GB) from 60_epochs.pt:
# every GPU on newton3-5 is allocated until ~03:26 next day, so newton2 now with
# a 3h load beats waiting. The 40 val shards were TRAIN shards in the 50k runs,
# so this job's val KS is not comparable to earlier logs -- judge by the
# held-out probe (grad_val_mid) only.
#
# --resume_lr_scale 8: the epoch-64 optimizer state carries lr 3.6e-5 after two
# plateau halvings -- too small to re-shape the peak. 8x restores ~2.9e-4, the
# lr the mass-weighted phase made its progress at; Adam moments are kept.

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

SRC=data/hollow_cube/checkpoints_sobolev_massw
DST=data/hollow_cube/checkpoints_sobolev_rho025_n3
mkdir -p "$DST" training/runs/train_sobolev_rho025_n3
LATEST="$SRC/60_epochs.pt"   # epoch 60 (lr 3.6e-5 x8 = 2.9e-4), per user
echo "[resume] seeding $DST from $LATEST"
cp "$LATEST" "$DST"/
cp training/runs/train_sobolev_massw/losses.json training/runs/train_sobolev_rho025_n3/

torchrun --standalone --nproc_per_node=$NGPU -m training.gnn_training \
    --geometry hollow_cube_ddptest \
    --dataset data/hollow_cube/dataset_gradadj/dataset.pt \
    --grad_from_train true \
    --grad_loss_weight 0.5 \
    --grad_cos_weight 0.5 \
    --grad_mass_weight 1.0 \
    --stress_ks_rho_mpa 0.25 \
    --resume_lr_scale 8 \
    --num_samples 12001 \
    --lr 0.0005 \
    --epochs 80 --batch_size 8 --grad_batch_size 4 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube/checkpoints_sobolev_rho025_n3 \
    --run_dir training/runs/train_sobolev_rho025_n3
