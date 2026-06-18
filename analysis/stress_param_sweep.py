"""
Parametric sweep of max stress vs frame thickness across mesh resolutions and
force magnitudes. Stress is normalized by total applied force so curves with
different forces are directly comparable.

Run from project root: python -m analysis.stress_param_sweep
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path
from data.bistable.boundary_conditions import fixed_dims as get_fixed_dims
from evaluators.mapdl_evaluator import MAPDLEvaluator

GEOMETRY  = "bistable"
N_POINTS  = 20
POOL_SIZE = 20  # 20 × nproc=4 = 80 CPUs; sized for the cluster (96 CPUs available)

# Mesh resolutions to try (U, V, W)
MESH_RESOLUTIONS = [(3, 3, 3), (5, 5, 5), (8, 8, 8)]

# Per-node force magnitudes to try (N)
FORCE_MAGNITUDES = [1e3, 1e4, 1e5]

OUT_DIR = Path("analysis/stress_param_sweep")
OUT_DIR.mkdir(exist_ok=True)

dims_template = json.loads(Path(f"data/{GEOMETRY}/CAD_model/dims.json").read_text())
d1_min = dims_template["d1"]["min"]
d1_max = dims_template["d1"]["max"]

thicknesses = np.linspace(d1_min, d1_max, N_POINTS)
dims_list = [{"d1": float(t)} for t in thicknesses]

_fd = get_fixed_dims()

def make_force_pattern(force_mag: float):
    z_top = _fd["d1"]
    def force_pattern(node, tol):
        if abs(node.coords[2] - z_top) < tol:
            return [0.0, -force_mag, 0.0]
        return [0.0, 0.0, 0.0]
    return force_pattern

evaluator = MAPDLEvaluator(geometry_name=GEOMETRY, pool_size=POOL_SIZE, nproc=4)

all_results = []

try:
    for mesh in MESH_RESOLUTIONS:
        evaluator.U, evaluator.V, evaluator.W = mesh
        mesh_label = "x".join(map(str, mesh))

        for force_mag in FORCE_MAGNITUDES:
            evaluator.force_pattern = make_force_pattern(force_mag)
            combo_label = f"mesh{mesh_label}_F{force_mag:.0e}"
            print(f"\n=== {combo_label} ===", flush=True)

            res = evaluator.evaluate(dims_list)

            rows = []
            n_force_nodes = None
            for t, s, v, d in zip(thicknesses, res["stress"], res["volume"], res["disp"]):
                rows.append({
                    "d1":    float(t),
                    "stress_MPa":  s,
                    "stress_per_N": (s / force_mag) if s is not None else None,
                    "volume": v,
                    "disp":  d,
                })

            combo = {
                "mesh":  list(mesh),
                "force_per_node": force_mag,
                "data":  rows,
            }
            all_results.append(combo)
            (OUT_DIR / f"{combo_label}.json").write_text(json.dumps(combo, indent=2))
            print(f"  saved {combo_label}.json")

finally:
    evaluator.close()

(OUT_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
print(f"\nSaved all_results.json")

# ── Plots ────────────────────────────────────────────────────────────────────
# One subplot per mesh resolution.
# Each subplot shows stress/force_per_node curves for each force magnitude.
# In a linear system all three curves must overlap — divergence = nonlinearity.

n_mesh = len(MESH_RESOLUTIONS)
fig, axes = plt.subplots(1, n_mesh, figsize=(5 * n_mesh, 4), sharey=True)
if n_mesh == 1:
    axes = [axes]

colors = [f"C{i}" for i in range(len(FORCE_MAGNITUDES))]

for ax, mesh in zip(axes, MESH_RESOLUTIONS):
    mesh_label = "x".join(map(str, mesh))
    combos = [c for c in all_results if c["mesh"] == list(mesh)]

    for combo, color in zip(combos, colors):
        rows  = combo["data"]
        t_arr = [r["d1"]            for r in rows if r["stress_per_N"] is not None]
        s_arr = [r["stress_per_N"]  for r in rows if r["stress_per_N"] is not None]
        n_ok  = len(t_arr)
        n_tot = len(rows)
        lbl   = f"F={combo['force_per_node']:.0e} ({n_ok}/{n_tot} ok)"
        ax.plot(t_arr, s_arr, marker="o", markersize=3, linewidth=1.5,
                color=color, label=lbl)

    ax.set_title(f"Mesh {mesh_label}")
    ax.set_xlabel("Frame thickness d1")
    ax.set_ylabel("Stress / force-per-node  (MPa / N)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle("Bistable tile: stress/force vs thickness  (curves should overlap if linear)")
plt.tight_layout()
fig.savefig(OUT_DIR / "by_mesh.png", dpi=150)
print("Saved by_mesh.png")

# Second plot: one subplot per force magnitude, curves = mesh resolutions.
# Shows how much the result changes with mesh refinement.
fig2, axes2 = plt.subplots(1, len(FORCE_MAGNITUDES),
                            figsize=(5 * len(FORCE_MAGNITUDES), 4), sharey=True)
mesh_colors = [f"C{i}" for i in range(n_mesh)]

for ax, force_mag in zip(axes2, FORCE_MAGNITUDES):
    combos = [c for c in all_results if c["force_per_node"] == force_mag]
    for combo, color in zip(combos, mesh_colors):
        mesh_label = "x".join(map(str, combo["mesh"]))
        rows  = combo["data"]
        t_arr = [r["d1"]            for r in rows if r["stress_per_N"] is not None]
        s_arr = [r["stress_per_N"]  for r in rows if r["stress_per_N"] is not None]
        ax.plot(t_arr, s_arr, marker="o", markersize=3, linewidth=1.5,
                color=color, label=f"mesh {mesh_label}")
    ax.set_title(f"F={force_mag:.0e} N/node")
    ax.set_xlabel("Frame thickness d1")
    ax.set_ylabel("Stress / force-per-node  (MPa / N)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig2.suptitle("Bistable tile: mesh convergence check  (curves should overlap if mesh-independent)")
plt.tight_layout()
fig2.savefig(OUT_DIR / "by_force.png", dpi=150)
print("Saved by_force.png")
