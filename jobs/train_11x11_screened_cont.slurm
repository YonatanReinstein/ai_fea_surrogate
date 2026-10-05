#!/bin/bash
#SBATCH --job-name=train_11x11_screened_cont
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#
# CONTINUATION of job 1430006 (this same recipe) from its 80_epochs.pt to 160 epochs.
# Auto-resumes from checkpoint_dir (weights, AdamW state, LR) and from run_dir's loss
# history, which also restores the best-val record, so the decoy gnn_surrogate.pt is
# only replaced by a genuinely better epoch. At epoch 80: val ks 1.23%, node 2.03%,
# train sgnerr 4.2%, LR 2.24e-4.
#
# hollow_cube_11x11 on the SCREENED dataset (data/hollow_cube_11x11/dataset_screened30k,
# KS 66-199 MPa, adjoint dKS/dt labels, rho 0.25). Every loss term is now a
# readable fraction (no log, no normalized MSE), and there is no cosine loss:
#   ks     0.25 x mean |KS_pred - KS_true| / KS_true
#   node   0.25 x stress-weighted (alpha 1) mean |sigma_pred - sigma_true| / KS_true
#   grad   0.25 x saturated per-tile percent error r/(1+r/2), weights (|g|/sum|g|)^0.15
#   sign   0.25 x per-tile smooth wrong-sign loss 0.5*(1 - z/(0.05+|z|)), z = pred*sign(true)/|true|
# Each term is divided by its know-nothing baseline (median KS / zero nodes / zero
# gradient = 1/(1+1/sat) / coin-flip 0.5), computed from the train set at startup and
# printed as [loss-norm], so 0.25 each = equal importance relative to difficulty.
# Log prints ks/node/grad in raw %, plus cos (metric only) and sgnerr (exact wrong-sign rate).
# The samples lack edge_index/tile_idx; gnn_training re-attaches them from the graph.pt
# copied next to the shards. Decoy geometry hollow_cube_11x11_screened.

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
    --geometry hollow_cube_11x11_screened \
    --dataset data/hollow_cube_11x11/dataset_screened30k/dataset.pt \
    --grad_from_train true \
    --grad_err_mode rel --grad_weight_power 0.15 --grad_rel_sat 2.0 \
    --graph_loss_weight 0.25 --node_loss_weight 0.25 \
    --grad_loss_weight 0.25 \
    --grad_sign_loss_weight 0.25 --grad_sign_delta 0.05 \
    --stress_ks_rho_mpa 0.25 \
    --num_samples 30000 \
    --lr 0.0005 \
    --epochs 160 --batch_size 96 --grad_batch_size 48 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube_11x11/checkpoints_screened_30k \
    --run_dir training/runs/train_11x11_screened
