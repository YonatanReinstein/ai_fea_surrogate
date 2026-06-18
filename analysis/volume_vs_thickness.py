"""
Plot volume vs frame thickness for the bistable 1x1x1 tile.
Run from project root: python -m analysis.volume_vs_thickness
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from core.IritModel import IritCModel
import importlib

GEOMETRY = "bistable"
N_POINTS = 40

module = importlib.import_module(f"data.{GEOMETRY}.boundary_conditions")
fixed_dims = module.fixed_dims()

dims_template = json.loads(Path(f"data/{GEOMETRY}/CAD_model/dims.json").read_text())
d1_min = dims_template["d1"]["min"]
d1_max = dims_template["d1"]["max"]

thicknesses = np.linspace(d1_min, d1_max, N_POINTS)
volumes = []

model_path = f"data/{GEOMETRY}/CAD_model/model"

for i, t in enumerate(thicknesses):
    print(f"[{i+1}/{N_POINTS}] d1={t:.4f}", flush=True)
    model = IritCModel(model_path, dims_dict={"d1": t}, fixed_dims=fixed_dims)
    volumes.append(model.get_volume())

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thicknesses, volumes, marker="o", markersize=3, linewidth=1.5)
ax.set_xlabel("Frame thickness d1")
ax.set_ylabel("Volume")
ax.set_title("Bistable tile volume vs frame thickness")
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = Path("analysis/volume_vs_thickness.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"Saved {out}")

json_out = Path("analysis/volume_vs_thickness.json")
json_out.write_text(json.dumps(
    [{"d1": float(t), "volume": float(v)} for t, v in zip(thicknesses, volumes)],
    indent=2,
))
print(f"Saved {json_out}")
