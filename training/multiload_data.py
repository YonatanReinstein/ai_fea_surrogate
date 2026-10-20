"""Data access for the multi-load cross4 dataset
(training/build_cross4_multiload_dataset.py): one design file serves all its
load cases, and each case's per-node load is rebuilt from (row, direction).

A sample is one (design, load case). Inputs per node:
    coords (3), load (3) = this node's share of the total force times the unit
    direction (the FEA_LOAD_MODE=tile_nodes rule, Mesh._apply_node_load),
    anchored (1), thickness of the node's tile (1)               -> 8
per tile (tile_x): thickness, is-loaded-tile, direction y, direction z -> 4
per edge: dx, dy, dz, length                                    -> 4
Targets per node: displacement (3) and stress tensor (6, Pa, sxx syy szz sxy
sxz syz) -> 9.

Everything static (graph, node->tile, anchored, cases) comes from mesh.npz and
is computed once; a design's arrays are read once and shared by its cases.
"""
import json
import os

import numpy as np
import torch
from torch_geometric.data import Data

HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
DIR_VEC = {0: np.array([0.0, -1.0, 0.0]), 1: np.array([0.0, 0.0, -1.0])}


class MultiLoadStatic:
    """Design-independent arrays, built once from mesh.npz + config.json."""

    def __init__(self, root, NX=11, NY=11, length=11.0, height=11.0):
        m = np.load(os.path.join(root, "mesh.npz"))
        self.cfg = json.load(open(os.path.join(root, "config.json")))
        self.elements = m["elements"].astype(np.int64)               # (M,8)
        self.elem_tile = m["elem_tile"].astype(np.int64)             # (M,) IRIT index ix + NX*iy
        self.anchored = m["anchored"].astype(bool)                   # (N,)
        self.cases = [tuple(int(v) for v in c) for c in m["cases"]]  # (row, direction)
        self.N = len(m["node_ids"])
        self.NX, self.NY = NX, NY
        self.tz, self.ty = length / NX, height / NY
        self.length = length
        self.F = float(self.cfg["load_N"])
        # graph: hex edges, both directions, unique
        e = np.concatenate([self.elements[:, [a, b]] for a, b in HEX_EDGES], axis=0)
        e = np.unique(np.sort(e, axis=1), axis=0)
        self.edge_index = torch.as_tensor(np.concatenate([e, e[:, ::-1]], axis=0).T.copy())
        # node -> tile: the lowest tile index among the elements using it
        tile = np.full(self.N, 1 << 30, dtype=np.int64)
        np.minimum.at(tile, self.elements.reshape(-1), np.repeat(self.elem_tile, 8))
        self.node_tile = tile
        # per load row: the loaded tile's element set -> its nodes (fixed topology)
        self.load_nodes = {}
        for row in range(NY):
            t = (NX - 1) + NX * row
            self.load_nodes[row] = np.unique(self.elements[self.elem_tile == t].reshape(-1))

    def node_load(self, coords, row, direction):
        """(N,3) per-node force / F: weights 1 - r/r_max in the (y,z) plane from
        the loaded tile's centre, normalized to sum 1, times the unit direction
        -- exactly Mesh._apply_node_load for the builder's load_spec."""
        ids = self.load_nodes[row]
        c = np.array([0.5 * (2 * row + 1) * self.ty, self.length - 0.5 * self.tz])
        r = np.linalg.norm(coords[ids][:, 1:3] - c, axis=1)
        w = 1.0 - r / r.max()
        w = w / w.sum()
        out = np.zeros((self.N, 3))
        out[ids] = w[:, None] * DIR_VEC[direction][None, :]
        return out


def design_cases(static, z, keep):
    """Data objects for the kept cases of one design (z: loaded npz dict)."""
    coords = z["coords"].astype(np.float64)
    t = z["t"].astype(np.float32)                                    # (K,) d1..dK = ix*NY + iy
    NX, NY = static.NX, static.NY
    # tile thickness in IRIT tile order (ix + NX*iy) from dims order (ix*NY + iy)
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    t_irit = np.empty(NX * NY, np.float32)
    t_irit[(ix + NX * iy).reshape(-1)] = t[(ix * NY + iy).reshape(-1)]
    src, dst = static.edge_index
    d = torch.as_tensor(coords[dst] - coords[src], dtype=torch.float32)
    edge_attr = torch.cat([d, d.norm(dim=1, keepdim=True)], dim=1)
    pos = torch.as_tensor(coords, dtype=torch.float32)
    anch = torch.as_tensor(static.anchored, dtype=torch.float32)[:, None]
    tnode = torch.as_tensor(t_irit[static.node_tile])[:, None]
    tile_idx = torch.as_tensor(static.node_tile)
    out = []
    for k in keep:
        row, direction = static.cases[k]
        load = torch.as_tensor(static.node_load(coords, row, direction), dtype=torch.float32)
        x = torch.cat([pos, load, anch, tnode], dim=1)
        tile_x = np.zeros((NX * NY, 4), np.float32)
        tile_x[:, 0] = t_irit
        tile_x[(NX - 1) + NX * row, 1] = 1.0
        tile_x[:, 2:] = DIR_VEC[direction][1:] * tile_x[:, 1:2]
        y = torch.cat([torch.as_tensor(z["u"][k], dtype=torch.float32),
                       torch.as_tensor(z["S"][k], dtype=torch.float32)], dim=1)   # (N,9)
        out.append(Data(x=x, edge_index=static.edge_index, edge_attr=edge_attr, y=y,
                        tile_idx=tile_idx, tile_x=torch.as_tensor(tile_x),
                        node_id=torch.arange(static.N), case=torch.tensor([k]),
                        max_vm=torch.tensor([float(z["max_vm"][k])]),
                        max_vm_excl=torch.tensor([float(z["max_vm_excl"][k])])))
    return out


class DesignDataset(torch.utils.data.Dataset):
    """Item = one design: all of its cases whose peak (excluding the loaded
    tile) is <= peak_max_pa, as a list of Data. Reading the 60 MB file once
    and serving every kept case from it keeps NFS traffic per case low."""

    def __init__(self, root, design_ids, static, peak_max_pa=300e6):
        self.root, self.ids, self.static, self.peak_max = root, list(design_ids), static, peak_max_pa

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        path = os.path.join(self.root, "designs", f"{self.ids[i]:05d}.npz")
        with open(path, "rb") as f:                                   # one sequential NFS read
            z = dict(np.load(f))
        keep = [k for k in range(len(self.static.cases)) if z["max_vm_excl"][k] <= self.peak_max]
        return design_cases(self.static, z, keep)


def case_mask(root, design_ids, peak_max_pa):
    """(designs, cases) bool: which cases pass the peak filter (reads only the
    small per-case arrays)."""
    out = []
    for i in design_ids:
        z = np.load(os.path.join(root, "designs", f"{i:05d}.npz"))
        out.append(z["max_vm_excl"] <= peak_max_pa)
    return np.array(out)
