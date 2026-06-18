"""
Verify that the hollow_cube mesh has FIXED topology across dim samples:
  - same node count
  - same element count + connectivity (edge set)
  - node index i -> same physical node every sample (ordering stable)
Only node COORDINATES should differ between samples.

Run from project root:  python analysis/verify_fixed_topology.py
"""
import json, random, importlib, os
from pathlib import Path

from core.IritModel import IritCModel

GEOM = "hollow_cube"
N_SAMPLES = 3
SEED = 42

base = Path(f"data/{GEOM}")
model_path = base / "CAD_model"
dims_json = model_path / "dims.json"

mod = importlib.import_module(f"data.{GEOM}.boundary_conditions")
U, V, W = mod.mesh_resolution()
fixed_dims = getattr(mod, "fixed_dims", lambda: {})()

with open(dims_json) as f:
    dims_template = json.load(f)

random.seed(SEED)
samples = [
    {k: random.uniform(v["min"], v["max"]) for k, v in dims_template.items()}
    for _ in range(N_SAMPLES)
]


def mesh_for(dims):
    cad = IritCModel(str(model_path / "model"), dims_dict=dims, fixed_dims=fixed_dims)
    nodes, elements, _ = cad.create_mesh(U=U, V=V, W=W)
    # canonical edge set (undirected) from hex element connectivity
    edges = set()
    for nlist in elements.values():
        ids = [n - 1 for n in nlist]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                edges.add(tuple(sorted((ids[a], ids[b]))))
    return nodes, elements, edges


print(f"Meshing {N_SAMPLES} samples at U,V,W=({U},{V},{W}), fixed_dims={fixed_dims}\n")
results = []
for i, dims in enumerate(samples):
    nodes, elements, edges = mesh_for(dims)
    results.append((nodes, elements, edges))
    print(f"sample {i}: nodes={len(nodes):6d}  elems={len(elements):6d}  edges={len(edges):7d}")

print("\n=== comparison vs sample 0 ===")
n0, e0, edg0 = results[0]
ok = True
for i in range(1, N_SAMPLES):
    ni, ei, edgi = results[i]
    same_n = len(ni) == len(n0)
    same_e = len(ei) == len(e0)
    same_edges = edgi == edg0
    same_keys = set(ni.keys()) == set(n0.keys())
    # how far does node index i move physically between samples?
    if same_keys:
        max_disp = max(
            sum((a - b) ** 2 for a, b in zip(ni[k], n0[k])) ** 0.5
            for k in n0
        )
    else:
        max_disp = float("nan")
    print(f"sample {i}: same_node_count={same_n} same_elem_count={same_e} "
          f"same_edge_set={same_edges} same_node_ids={same_keys} "
          f"max_coord_shift={max_disp:.4g}")
    ok = ok and same_n and same_e and same_edges and same_keys

print("\nRESULT:", "FIXED TOPOLOGY CONFIRMED ✅" if ok else "TOPOLOGY VARIES ❌")
