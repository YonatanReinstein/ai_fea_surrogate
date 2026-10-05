#!/bin/bash
#SBATCH --job-name=topopt_11x11
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=newton
#SBATCH --account=cslab
#SBATCH --gres=gpu:1
#
# Surrogate-driven topopt on hollow_cube_11x11 with a FROZEN surrogate: no
# on-policy labellers, no live fine-tuning. Schedule = run T's
# (artifacts_rho025_T_sched800_ep80/topopt_params.jsonl), KS rho 0.25 to match
# the dataset and the training run.
#
# CKPT is a COPY of data/hollow_cube_11x11/gnn_surrogate.pt taken at submit
# time: training job 1428532 keeps overwriting that path on every new best val.
#
# Ground truth alongside the surrogate: ccx calibration solve every CALIB gens,
# and TOPOPT_CCX_REF=1 overlays the exact adjoint gradient on every screenshot
# gen. best_feas_vol is judged by the surrogate -- quote the ccx stress next
# to it (project_topopt_best_feas_trusts_surrogate).

cd "$SLURM_SUBMIT_DIR"
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PWD/irit/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export IRIT_PATH="$PWD/irit/bin/"
export IRIT_TMP=/tmp/irit_$SLURM_JOB_ID TMPDIR=/tmp; mkdir -p "$IRIT_TMP"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

GEOM=${GEOM:-hollow_cube_11x11}
RUN_NAME=${RUN_NAME:-11x11_surrogate_A}
CKPT=${CKPT:?set CKPT to a frozen copy of the surrogate}
ART=optimization/artifacts_${RUN_NAME}
mkdir -p "$ART/screenshots"
export TOPOPT_ARTIFACT_DIR="$ART" TOPOPT_SCREENSHOT_DIR="$ART/screenshots"
export TOPOPT_CCX_REF=1
# Keep every reference solve (surrogate vs ccx stress + dG, labeller-label layout)
# for optimization/plot_trajectory_accuracy.py. One per screenshot gen.
export TOPOPT_CCX_REF_DIR="$ART/ccx_ref"

GENS=${GENS:-800}
RHO=${RHO:-0.25}
MOVE=${MOVE:-0.08}
SHOT_EVERY=${SHOT_EVERY:-10}
CALIB=${CALIB:-5}
# Schedule, defaults = run T (800 gens). Scale for shorter runs.
FILT_START=${FILT_START:-100}; FILT_GENS=${FILT_GENS:-300}
SH_START=${SH_START:-200}; SH_END=${SH_END:-500}; SH_W=${SH_W:-0.05}
export STRESS_KS_RHO_MPA=$RHO

cat > "$ART/params.env" <<PARAMS
# $RUN_NAME -- frozen-surrogate topopt, job $SLURM_JOB_ID on $(hostname), $(date -Is)
GEOM=$GEOM
ARCH=gnn
CKPT=$CKPT
GENS=$GENS RHO=$RHO MOVE=$MOVE SHOT_EVERY=$SHOT_EVERY CALIB=$CALIB
SCHEDULE: filter design r 2.0->1.2 from gen $FILT_START over $FILT_GENS gens; skeleton sharpen w 0 -> $SH_W gens $SH_START-$SH_END, tau 0.03->0.015
PARAMS
echo "[job] $RUN_NAME geom=$GEOM ckpt=$CKPT node=$(hostname) gens=$GENS rho=$RHO move=$MOVE"

python -u -m optimization.optimize_dims \
    --geometry "$GEOM" --arch gnn --optimizer topopt --gnn_ckpt_path "$CKPT" \
    --generations "$GENS" --stress_ks_rho_mpa "$RHO" \
    --processes 4 --batch_size 4 \
    --filter_target design --filter_radius 2.0 \
    --filter_radius_end 1.2 --filter_decay_start "$FILT_START" --filter_decay_gens "$FILT_GENS" \
    --sharpen_mode skeleton --proj_start_gen "$SH_START" --proj_end_gen "$SH_END" \
    --proj_w_max "$SH_W" --skel_tau 0.03 --skel_tau_end 0.015 \
    --move_limit "$MOVE" --oc_eta 0.5 --fd_step 0.04 \
    --trust_rel_tol 0.5 --min_trust_frac 0.05 --untrusted_move_frac 0.1 \
    --tr_tol 0.05 --tr_min_scale 0.02 --g_backoff 0.02 \
    --calib_every "$CALIB" --calib_smooth 0.5 --calib_max_dev 2.0 --stagnation_gens 0 \
    --screenshot_every "$SHOT_EVERY"

echo "TOPOPT_DONE rc=$?"
python -u -m optimization.plot_trajectory_accuracy --labels "$ART/ccx_ref" --out "$ART/trajectory_accuracy.png" \
    --title "$RUN_NAME -- frozen surrogate, no fine-tuning ($CKPT, job $SLURM_JOB_ID)"
rm -rf "$IRIT_TMP"
rm -rf data/$GEOM/Instance_* data/$GEOM/ansys_* 2>/dev/null
true
