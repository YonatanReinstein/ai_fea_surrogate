#!/bin/bash
#SBATCH --job-name=train_nodegrad
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
# Sobolev run 2: adds the --grad_cos_weight direction term.
#
# WHY: run 1 (job 1368050, 30 epochs) drove the signed-log magnitude term down
# 7x (0.191 -> 0.028) but evaluators/probe_dg_alignment.py showed the thing OC
# actually consumes barely moved -- per-sample cosine(pred_dg, true_dg) median
# +0.019 (baseline) -> +0.064 (sobolev), i.e. still orthogonal to truth, with
# 21% of gradient MASS pointing the wrong way. The target carries ~6 decades of
# dynamic range with 97.6% of its mass in one decile of tiles, so signed-log is
# the only transform that makes it trainable AND the transform that throws away
# the structure OC follows. Cosine is scale-invariant and weights the high-mass
# tiles automatically.
#
# WEIGHT: 0.2, chosen from job 1368940's smoke, not guessed. Over 3 epochs,
#   w=0.05 -> cos +0.2537 -> +0.3862 -> +0.4617   (val g 0.0959)
#   w=0.20 -> cos +0.2925 -> +0.4901 -> +0.5334   (val g 0.0461)
# 0.2 wins on direction with no stress cost.
#
# EPOCHS 40 (not 30): run 1's best checkpoint was its FINAL epoch and the LR
# plateau trigger never fired -- it was still improving when the wall stopped
# it. 40 epochs is ~12.2h on 3 GPUs, ~17.8h on 2, both inside the 20h wall.
#
# GPU FILTER: newton2 has a wedged A6000 at PCI 0000:3E:00.0 ("Unable to
# determine the device handle ... Unknown Error"). Slurm still advertises 6 and,
# because a dead GPU is never allocated to anyone else, it sits permanently in
# the free pool and gets handed out FIRST -- it was in two consecutive fresh
# allocations. It is what hung job 1368050's teardown for 8.6h after training
# had actually finished.
#
# UPDATE 2026-09-06: newton2 was rebooted and 0000:3E:00.0 is healthy again --
# nvidia-smi enumerates every device and a 2-rank NCCL all_reduce+broadcast on
# 3D+3E passes (probe job 1371752). The filter below is kept anyway: the fault
# recurred across three consecutive allocations before the reboot, so it is
# worth 20s to confirm rather than assume.
#
# WHY THE OLD FILTER MISSED IT: run 2 (job 1368949) died 7 min in at the FIRST
# NCCL collective (gnn_training.py:361 dist.broadcast), 0 epochs done. The old
# probe only ran torch.zeros(...).cuda() per ordinal, which needs CUDA but NOT
# NVML -- and the actual breakage was NVML ("Can't initialize NVML" is right
# there in that log, ignored). NCCL needs NVML for topology/P2P, so it failed
# where plain CUDA had passed. The probe now runs the real 2-rank collective.
#
# GPUS 2, not 3: this session's own interactive allocation holds 1 of the
# account's 3-GPU QOSMaxGRESPerUser budget, so asking for 3 pends forever.
# 40 epochs is ~17.8h on 2 GPUs; wall is 36h (partition MaxTime=UNLIMITED) so a
# slower-than-expected epoch cannot silently truncate the run.
#
# OFF newton2 (2026-09-06 17:0x): after the 11:53 reboot newton2's Intel X550T
# 10G NIC links at 100baseT/Full on eno2 (eno1 NO-CARRIER, 7106 rx_dropped).
# Measured NFS ceiling there is 11.7 MB/s -- confirmed a hard wire limit, since
# adding 8 parallel readers left aggregate NIC rx unchanged and merely stole
# bandwidth from the job. Same 322GB dataset loaded in <7 min (~780 MB/s) on
# that node the day before, and dd gets 447 MB/s on newton4, so this is a
# physical-layer regression, not a code or filesystem problem. Loading 322GB
# there costs ~7.8h instead of ~10 min.
#
# Hence --exclude=newton2 and a GENERIC --gres=gpu:2: newton2 holds the only
# A6000s in the partition, so the old gpu:A6000:2 request silently pinned this
# job to the one broken node ("Requested node configuration is not available"
# for every alternative). A40/L40 are equally fine here -- Ampere/Ada, 48GB,
# bf16 autocast all supported. CPUs 24->16 so it fits alongside other work.

#
# RUN 3 (1 GPU): the fix is --grad_from_train, which draws the grad pool from
# the loaded train shards (~40k gradients) instead of data/hollow_cube/gradsub
# -- a 1-in-10 SYMLINK subsample (4,100) of those very same shards. Job
# 1373261 memorised that small pool: held-out dg peaked ~epoch 4 (median cosine
# +0.715 vs the +0.613 constant-direction control) then fell to +0.062 by
# epoch 15 while TRAIN cos rose to +0.9585.
#
# 1 GPU, not 2, purely on availability: sbatch --test-only puts a 2-GPU job at
# 2026-09-07T21:57 (~20h out) and a 1-GPU job at 01:26 (~6 min). Epochs run
# ~54min instead of ~27min, which still answers the ablation far sooner.
# torchrun --nproc_per_node=1 still sets LOCAL_RANK, so this takes the
# shard-based loader (all 401 train shards in one rank, last 100 stay val) --
# NOT the non-ddp load_dataset() path, so grad_audit stays held out.
#
# Model is snapshotted EVERY epoch (epoch_NNN.pt) so the best-dg epoch can be
# picked offline with evaluators/probe_dg_alignment.py -- best-val-KS selection
# is what hid run 2's collapse.

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
    --node_grad_weight 1.0 \
    --num_samples 50001 \
    --epochs 20 --batch_size 8 --grad_batch_size 4 \
    --hidden_dim 32 --coarse_arch cnn --cnn_depth 3 --conv_layers 6 \
    --use_compile false \
    --checkpoint_dir data/hollow_cube/checkpoints_nodegrad \
    --run_dir training/runs/train_nodegrad
