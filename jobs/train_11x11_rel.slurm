#!/bin/bash
#SBATCH --job-name=train_11x11_rel
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#
# hollow_cube_11x11 PERCENT-ERROR retrain. No log anywhere in the gradient loss:
#   per tile   r = |pred - true| / |true|,  e = r / (1 + r/2)   (smooth saturation at 200%)
#   weight     (|true| / sum|true|) ** 0.15, normalised to mean 1 per sample
#   cosine     on raw values (--grad_cos_space raw)
# Why: the slog scale (mean |dg| = 513) sat above 92% of tiles, so ~98% of the
# slog-MSE came from the top 10% of tiles even at mass weight 0.4
# (project_grad_slog_scale_dominance). Offline, on the gradbal model's trajectory,
# this loss splits bottom-50% / 50-75 / 75-90 / top-10% as 58 / 19 / 12 / 11%.
# Decoy geometry hollow_cube_11x11_rel.
#
# (previous run's notes follow)
# hollow_cube_11x11 GRADIENT-BALANCE retrain. Identical to jobs/train_11x11_30k.slurm
# (job 1428532) except for two changes:
#   --grad_mass_weight 1.0 -> 0.4   pure mass weighting gave the slog MSE ~7.5
#                                   effective tiles of 121
#   --grad_cos_space slog           the raw cosine gave ~2 effective tiles (the top
#                                   tile alone carried 73%)
# On the epoch-80 trajectory that model got the top decile's sign 98% right but
# squashed the dynamic range (top 0.58x, bottom quartile 2.8x), with near-chance
# sign on the bottom quartile. Decoy geometry hollow_cube_11x11_gradbal so the
# best-val gnn_surrogate.pt does not overwrite run 1428532's.
#
# (original notes for the base recipe follow)
# Same recipe as jobs/train_sobolev_rho025_fresh30k.slurm (the 29x29 run),
# re-sized for the smaller graph: 7,032 nodes/sample vs 48,249 (~6.9x).
# The 29x29 run peaked at 18.5 GB VRAM with batch 8 / grad batch 4; batch 96 /
# grad batch 48 is ~1.75x those node counts -> ~32 GB, fits a 48 GB A40/L40/A6000.
# Dataset is ~29 GB on disk (vs ~195 GB for 29x29), hence --mem 96G and no node pin.
# LR kept at 5e-4 despite the 12x batch: the Sobolev dg stability was an LR
# effect (see project_sobolev_lr_gradient_stability), so not scaled up blindly.

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
    --geometry hollow_cube_11x11_rel \
    --dataset data/hollow_cube_11x11/dataset_gradadj/dataset.pt \
    --grad_from_train true \
    --grad_loss_weight 0.5 \
    --grad_cos_weight 0.5 --grad_cos_space raw \
    --grad_err_mode rel --grad_weight_power 0.15 --grad_rel_sat 2.0 \
    --stress_ks_rho_mpa 0.25 \
    --num_samples 30000 \
    --lr 0.0005 \
    --epochs 80 --batch_size 96 --grad_batch_size 48 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube_11x11/checkpoints_rel_30k \
    --run_dir training/runs/train_11x11_rel
