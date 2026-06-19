"""Print raw per-column node-feature ranges for one hollow_cube sample,
to locate which column holds the (large) applied force.
Run: python -m analysis.probe_features
"""
import importlib
import torch
from core.IritModel import IritCModel
from core.component import Component

geom = "hollow_cube"
bc = importlib.import_module(f"data.{geom}.boundary_conditions")
fixed = bc.fixed_dims()
grid = bc.tile_grid()
K = grid[0] * grid[1] * grid[2]
dims = {f"d{i+1}": 0.25 for i in range(K)}

cad = IritCModel("data/%s/CAD_model/model" % geom, dims_dict=dims, fixed_dims=fixed)
comp = Component(cad, 1.0, 0.3)
mr = bc.mesh_resolution()
comp.generate_mesh(U=mr[0], V=mr[1], W=mr[2])
comp.mesh.anchor_nodes_by_condition(bc.anchor_condition)
comp.mesh.apply_force_by_pattern(bc.force_pattern)
data = comp.to_graph_with_labels(with_labels=False, tile_grid=grid)
x = data.x
print("base x shape:", tuple(x.shape))
for c in range(x.shape[1]):
    col = x[:, c]
    print(f"col {c}: min={col.min():.4g} max={col.max():.4g} mean={col.mean():.4g} std={col.std():.4g}")
