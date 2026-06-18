"""
Precompute the FROZEN graph for a fixed-topology geometry (e.g. hollow_cube).

For geometries where the tile grid and mesh resolution are fixed, the mesh
connectivity and node ordering are identical across every dim sample
(verified for hollow_cube: node index i is always the same physical node;
only coordinates change). That means edge_index, the per-node tile index, and
graph Laplacian positional encodings can be computed ONCE and shared by the
whole dataset, instead of being rebuilt per sample.

Artifacts saved to data/<geometry>/dataset/graph.pt:
    edge_index : LongTensor [2, E]   bidirectional, ids 0-based (n.id - 1)
    pe         : FloatTensor [N, k]  Laplacian eigenvector positional encoding
    tile_idx   : LongTensor [N]      0-based tile index per node
    num_nodes  : int
    tile_grid  : (NX, NY, NZ)

Run from project root (with irit/bin on PATH):
    python -m core.fixed_graph --geometry hollow_cube --pe_dim 16
"""
import json
import importlib
from pathlib import Path

import numpy as np
import torch

from core.IritModel import IritCModel


def build_edge_index(elements: dict) -> torch.Tensor:
    """Bidirectional edge_index from hex connectivity, matching
    Component.to_graph_with_labels (node id -> id - 1)."""
    edges = set()
    for nlist in elements.values():
        ids = [n - 1 for n in nlist]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edges.add((ids[i], ids[j]))
                edges.add((ids[j], ids[i]))
    return torch.tensor(list(zip(*edges)), dtype=torch.long)


def node_tile_index(elements: dict, elem_to_tile: dict, num_nodes: int) -> torch.Tensor:
    """Per-node 0-based tile index; first element wins on shared nodes
    (same rule as Component.to_graph_with_labels)."""
    node_to_tile = {}
    for eid, nlist in elements.items():
        t = elem_to_tile.get(eid, -1)
        if t < 0:
            continue
        for nid in nlist:
            if nid not in node_to_tile:
                node_to_tile[nid] = t
    # nodes are 1-based ids; map to 0-based position
    out = torch.zeros(num_nodes, dtype=torch.long)
    for nid, t in node_to_tile.items():
        out[nid - 1] = t
    return out


def laplacian_pe(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    """Top-k non-trivial eigenvectors of the symmetric normalized Laplacian.
    Computed once because the graph is frozen."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    row, col = edge_index.numpy()
    data = np.ones(row.shape[0], dtype=np.float64)
    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)  # symmetric, binary
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    dinv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    Dinv = sp.diags(dinv_sqrt)
    L = sp.eye(num_nodes) - Dinv @ A @ Dinv

    # smallest k+1 eigenvalues; drop the trivial ~0 mode
    vals, vecs = spla.eigsh(L, k=k + 1, which="SM", tol=1e-3)
    order = np.argsort(vals)
    vecs = vecs[:, order][:, 1:k + 1]
    return torch.tensor(vecs, dtype=torch.float)


def build_fixed_graph(geometry: str, pe_dim: int = 16, dims: dict = None):
    base = Path(f"data/{geometry}")
    model_path = base / "CAD_model"

    mod = importlib.import_module(f"data.{geometry}.boundary_conditions")
    U, V, W = mod.mesh_resolution()
    fixed_dims = getattr(mod, "fixed_dims", lambda: {})()
    grid = getattr(mod, "tile_grid", lambda d=None: None)(None)

    if dims is None:
        with open(model_path / "dims.json") as f:
            tmpl = json.load(f)
        dims = {k: v["default"] for k, v in tmpl.items()}

    exe = model_path / "model"
    cad = IritCModel(str(exe), dims_dict=dims, fixed_dims=fixed_dims)
    nodes, elements, elem_to_tile = cad.create_mesh(U=U, V=V, W=W)
    num_nodes = len(nodes)

    edge_index = build_edge_index(elements)
    tile_idx = node_tile_index(elements, elem_to_tile, num_nodes)
    pe = laplacian_pe(edge_index, num_nodes, pe_dim)

    artifact = {
        "edge_index": edge_index,
        "pe": pe,
        "tile_idx": tile_idx,
        "num_nodes": num_nodes,
        "tile_grid": grid,
    }
    out_dir = base / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "graph.pt"
    torch.save(artifact, out_path)
    print(f"nodes={num_nodes}  edges={edge_index.shape[1]}  pe={tuple(pe.shape)}  "
          f"tile_grid={grid}")
    print(f"saved -> {out_path}")
    return artifact


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", default="hollow_cube")
    p.add_argument("--pe_dim", type=int, default=16)
    args = p.parse_args()
    build_fixed_graph(args.geometry, pe_dim=args.pe_dim)
