#!/bin/bash
# Reset all artifacts that become stale when the hollow_cube grid (fixed_dims())
# changes, then regenerate the grid-derived CAD + frozen-graph artifacts.
#
# Run from the project root AFTER editing data/hollow_cube/boundary_conditions.py:
#   bash jobs/reset_hollow_cube.sh
#
# Optional: PE_DIM env var overrides the Laplacian PE dimension (default 16).
#
# After this, build/train/optimize as usual:
#   qsub -l select=1:ncpus=26:mem=70gb -v GEOMETRY=hollow_cube,SEED=42,NUM_SAMPLES=1000,OUTPUT_DIR=data/hollow_cube/dataset,RUN_LOCATION=data/hollow_cube/instances jobs/run_dataset.pbs
#   qsub jobs/train_hollow_cube.pbs
#   cp data/hollow_cube/checkpoints/<best>_epochs.pt data/hollow_cube/gnn_surrogate.pt
#   qsub jobs/optimize_hollow_cube.pbs

set -euo pipefail

# Resolve project root from this script's location, so it works from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PE_DIM="${PE_DIM:-16}"

if [ ! -d env ]; then
    echo "ERROR: project venv 'env/' not found in $ROOT" >&2
    exit 1
fi
source env/bin/activate
export PATH="$ROOT/irit/bin:$PATH"
export IRIT_PATH="$ROOT/irit/bin/"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Cap BLAS/OpenMP thread fan-out. On a busy login node OpenBLAS otherwise tries to
# spawn one thread per core (~40), which trips the per-user RLIMIT_NPROC and segfaults
# numpy on import. The graph rebuild (mesh + scipy eigsh) does not need many threads.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

echo "==> Grid (fixed_dims) for this reset:"
#python -c "from data.hollow_cube.boundary_conditions import fixed_dims,; print('   fixed_dims =', fixed_dims()); print('   tile_grid  =', tile_grid())"

echo "==> [1/3] Regenerating CAD artifacts (outline.itd, dims.json)..."
python data/hollow_cube/CAD_model/gen_outline.py
python data/hollow_cube/CAD_model/gen_dims.py

echo "==> [2/3] Rebuilding frozen graph (dataset/graph.pt, pe_dim=$PE_DIM)..."
python -m core.fixed_graph --geometry hollow_cube --pe_dim "$PE_DIM"

echo "==> [3/3] Purging artifacts baked with the old topology / tile count..."
rm -fv data/hollow_cube/dataset/dataset.pt
rm -fv data/hollow_cube/checkpoints/*.pt data/hollow_cube/checkpoints/losses.json
rm -fv data/hollow_cube/gnn_surrogate.pt
rm -fv optimization/artifacts/cma_checkpoint.pkl

echo "==> Done. graph.pt:"
python -c "import torch; g=torch.load('data/hollow_cube/dataset/graph.pt', weights_only=False); print('   nodes=%d  edges=%d  pe=%s  tile_grid=%s' % (g['num_nodes'], g['edge_index'].shape[1], tuple(g['pe'].shape), g['tile_grid']))"
echo "==> Next: qsub the dataset build, then train, then promote a checkpoint + optimize (see header)."
