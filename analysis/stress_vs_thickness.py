"""
Plot max von Mises stress vs frame thickness for the bistable 1x1x1 tile.
Run from project root: python -m analysis.stress_vs_thickness
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import importlib
from evaluators.mapdl_evaluator import MAPDLEvaluator

GEOMETRY = "hollow_cube"  # "bistable" or "hollow_tile"
N_POINTS = 20
POOL_SIZE = 1

dims_template = json.loads(Path(f"data/{GEOMETRY}/CAD_model/dims.json").read_text())
d1_min = dims_template["d1"]["min"]
d1_max = dims_template["d1"]["max"]

bc = importlib.import_module(f"data.{GEOMETRY}.boundary_conditions")
nx, ny, nz = bc.tile_grid()
n_tiles = nx * ny * nz

thicknesses = np.linspace(d1_min, d1_max, N_POINTS)
dims_list = [{f"d{i+1}": float(t) for i in range(n_tiles)} for t in thicknesses]

evaluator = MAPDLEvaluator(geometry_name=GEOMETRY, pool_size=POOL_SIZE, nproc=4)
try:
    results = evaluator.evaluate(dims_list)
finally:
    evaluator.close()

stresses = results["stress"]
volumes  = results["volume"]
disps    = results["disp"]

json_out = Path("analysis/stress_vs_thickness.json")
json_out.parent.mkdir(exist_ok=True)
json_out.write_text(json.dumps(
    [
        {"d1": float(t), "stress": s, "volume": v, "disp": d}
        for t, s, v, d in zip(thicknesses, stresses, volumes, disps)
    ],
    indent=2,
))
print(f"Saved {json_out}")

valid = [(t, s) for t, s in zip(thicknesses, stresses) if s is not None]
if not valid:
    raise RuntimeError("All MAPDL evaluations failed — no data to plot.")

t_vals, s_vals = zip(*valid)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_vals, s_vals, marker="o", markersize=4, linewidth=1.5)
ax.set_xlabel("Frame thickness d1")
ax.set_ylabel("Max von Mises stress (MPa)")
ax.set_title("Bistable tile max stress vs frame thickness")
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = Path("analysis/stress_vs_thickness.png")
fig.savefig(out, dpi=150)
print(f"Saved {out}")
print(f"Failed points: {len(thicknesses) - len(valid)}/{N_POINTS}")
