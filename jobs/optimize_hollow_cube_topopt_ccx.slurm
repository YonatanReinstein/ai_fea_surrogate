#!/bin/bash
#SBATCH --job-name=optimize_hc_topopt_ccx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=190
#SBATCH --mem=192G
#SBATCH --time=13:00:00
#SBATCH --partition=public
#SBATCH --nodelist=protagoras
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

cd /home/ryonatan/ai_fea_surrogate
source env/bin/activate

export PATH="$HOME/opt/calculix/root/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/calculix/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="$PWD/irit/bin:$PATH"
export IRIT_PATH="$PWD/irit/bin/"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p optimization/artifacts optimization/screenshots

# Resuming from job 48002's checkpoint (gen 140, best_feas_vol=425.1) for
# 200 more generations -- NOT a fresh start, checkpoint/history/screenshots
# are left in place on purpose so run() picks up where 48002 left off.
# History of this run's trust-mechanism bugs (all on the same 840-tile
# hollow_cube design, see project_topopt_trust_deadlock.md memory):
#   job 45778: trusted set collapsed to 0 -> permanent deadlock. Fixed with
#     --min_trust_frac (floors the trusted count).
#   job 47985: min_trust_frac fixed the deadlock but the untrusted majority
#     (now ~95% of tiles) still snapped to a full-move prune on every
#     noise-level feasible generation -> limit cycle, best_feas_vol stuck at
#     394.4 for 55 generations. Fixed with --prune_margin (gated the snap on
#     a feasibility margin instead of g0<=0).
#   job 47989: prune_margin stopped the snap, but with the trust threshold
#     routinely excluding ~95% of tiles, those tiles just sat frozen instead
#     -- 800/840 tiles stopped moving for 90+ generations, design drifted to
#     a worse local point (394.4 -> 401).
#   job 47992: untrusted_move_frac=0.1, resumed from job 47989's checkpoint
#     -- fixed all three of the above, best_feas_vol broke through 394.4
#     down to 377.3 by gen 249, clean-looking truss bands in screenshots.
#   job 47999: fresh restart (gen 0) with untrusted_move_frac raised to 0.3
#     for speed -- the offline adjacency-disagreement metric said this
#     wouldn't add checkerboarding risk (step_cap only scales movement
#     magnitude, not direction), but the real run visibly disagreed: gen 99
#     screenshot shows scattered speckled tiles through the empty region,
#     and it converged *worse* (best_feas_vol=400.4 after 140 gens) than
#     47992's 0.1 run.
#   job 48002: fresh restart (gen 0) with untrusted_move_frac back to 0.1 --
#     STILL showed checkerboarding/speckle and converged worse (425.1 after
#     140 gens) than 47992. This rules out untrusted_move_frac as the cause:
#     both 0.1 and 0.3 fresh starts checkerboard. 47992's clean truss bands
#     only appeared after ~250+ cumulative generations resumed across
#     several checkpoints -- 48002 only got 140 gens from scratch. The
#     design was still actively moving generation-to-generation when it hit
#     the --generations cap (max|dx| in the 1e-3..2e-2 range on the last 10
#     gens, nowhere near the change<1e-6 early-stop), so more generations,
#     not a different frac, is the lever to try next.
python -u -m optimization.optimize_dims \
    --geometry hollow_cube \
    --arch ccx \
    --optimizer topopt \
    --generations 200 \
    --processes "$SLURM_CPUS_PER_TASK" \
    --mesh_threads 1 \
    --nproc 1 \
    --filter_radius 2 \
    --move_limit 0.05 \
    --fd_step 0.04 \
    --min_trust_frac 0.05 \
    --untrusted_move_frac 0.1 \
    --stress_ks_rho_mpa 0.15

rm -r data/hollow_cube/Instance_* 2>/dev/null
rm -r data/hollow_cube/ansys_* 2>/dev/null

true
