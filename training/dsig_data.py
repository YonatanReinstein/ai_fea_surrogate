"""Data access for the cross4 stress-sensitivity dataset
(datasets/build_cross4_dsig_dataset.py): per design, the nodal stress tensor S
of each of the 11 free-end tile loads and its exact derivative dS/dt with
respect to every tile thickness.

The model is a function of the tile thicknesses t only, so that autograd
d(model)/dt is the same total derivative the labels hold (node motion
included). Per-design node coordinates are therefore NOT an input -- they
depend on t and would hide a gradient path. Inputs per node:
    reference coords (3; the mean over designs, fixed), load weight (1; this
    node's share of the total force, from the design's own coordinates --
    the labels hold nodal forces fixed through the derivative, so it is a
    constant input), anchored (1), thickness of the node's tile (1)    -> 6
per tile (tile_x): thickness, is-loaded-tile                          -> 2
per edge: dx, dy, dz, length of the reference coords                  -> 4
Thickness features are rebuilt from a differentiable t inside the trainer
(thickness_features); the arrays here carry everything else.

t, dS's tile axis and dks are in dims-key order (d1..dK); tile_x, tile_idx
and elem_tile in IRIT order (core.tile_sensitivity.dims_key_order_permutation).
"""
import json
import os

import numpy as np
import torch

from core.tile_sensitivity import dims_key_order_permutation
from training.multiload_data import HEX_EDGES


class DsigStatic:
    """Design-independent arrays, built once from mesh.npz + config.json."""

    def __init__(self, root, NX=11, NY=11, length=11.0, height=11.0, ref_designs=None):
        m = np.load(os.path.join(root, "mesh.npz"))
        self.root = root
        self.cfg = json.load(open(os.path.join(root, "config.json")))
        self.elements = m["elements"].astype(np.int64)
        self.elem_tile = m["elem_tile"].astype(np.int64)             # IRIT index ix + NX*iy
        self.anchored = m["anchored"].astype(bool)
        self.pad = m["pad"].astype(bool)                             # left out of KS
        self.rows = [int(r) for r in m["load_rows"]]
        self.N = len(m["node_ids"])
        self.NX, self.NY, self.K = NX, NY, NX * NY
        self.tz, self.ty, self.length = length / NX, height / NY, length
        self.perm = np.array(dims_key_order_permutation(NX, NY, 1))  # t_irit = t[perm]
        e = np.concatenate([self.elements[:, [a, b]] for a, b in HEX_EDGES], axis=0)
        e = np.unique(np.sort(e, axis=1), axis=0)
        self.edge_index = torch.as_tensor(np.concatenate([e, e[:, ::-1]], axis=0).T.copy())
        tile = np.full(self.N, 1 << 30, dtype=np.int64)
        np.minimum.at(tile, self.elements.reshape(-1), np.repeat(self.elem_tile, 8))
        self.node_tile = tile
        # loaded tile of each row and its nodes. Mesh._apply_node_load takes
        # every node of the elements whose centroid is in the load box; the
        # box is exactly that tile, so with fixed topology the set is fixed.
        self.load_tile = {r: (NX - 1) + NX * r for r in self.rows}
        self.load_nodes = {r: np.unique(self.elements[self.elem_tile == t].reshape(-1))
                           for r, t in self.load_tile.items()}
        self.ref_coords = self._ref_coords(ref_designs)
        src, dst = self.edge_index
        d = torch.as_tensor(self.ref_coords[dst] - self.ref_coords[src], dtype=torch.float32)
        self.edge_attr = torch.cat([d, d.norm(dim=1, keepdim=True)], dim=1)

    def _ref_coords(self, ids):
        path = os.path.join(self.root, "ref_coords.npy")
        if os.path.exists(path):
            return np.load(path)
        ids = ids if ids is not None else available_designs(self.root)[:50]
        c = np.mean([np.load(design_path(self.root, i))["coords"].astype(np.float64) for i in ids], axis=0)
        np.save(path + ".tmp.npy", c)
        os.replace(path + ".tmp.npy", path)
        return c

    def load_weights(self, coords, row):
        """(N,) share of the total force at each node for this row's load --
        Mesh._apply_node_load's pyramid rule, on the design's coordinates."""
        ids = self.load_nodes[row]
        c = np.array([0.5 * (2 * row + 1) * self.ty, self.length - 0.5 * self.tz])
        half = np.array([0.5 * self.ty, 0.5 * self.tz])
        d = coords[ids][:, 1:3] - c
        w = np.clip(1.0 - np.max(np.abs(d) / half, axis=1), 0.0, None)
        out = np.zeros(self.N)
        out[ids] = w / w.sum()
        return out


def design_path(root, i):
    return os.path.join(root, "designs", f"{i:05d}.npz")


def available_designs(root):
    return sorted(int(f[:5]) for f in os.listdir(os.path.join(root, "designs"))
                  if f.endswith(".npz") and f[:5].isdigit())


def design_ks(root, ids):
    """(designs, rows) KS in MPa, reading only the small arrays."""
    return np.array([np.load(design_path(root, i))["ks"] for i in ids])


class DesignDataset(torch.utils.data.Dataset):
    """Item = one design, every load row (dict of tensors):
        t (K,) dims order; load_w (C,N); S (C,N,6) MPa; dS (C,K,N,6) float16;
        ks (C,), dks (C,K) dims order; id."""

    def __init__(self, static, design_ids, with_ds=True):
        self.static, self.ids, self.with_ds = static, list(design_ids), with_ds

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with open(design_path(self.static.root, self.ids[i]), "rb") as f:   # one sequential NFS read
            z = dict(np.load(f))
        coords = z["coords"].astype(np.float64)
        out = dict(id=self.ids[i], t=torch.as_tensor(z["t"], dtype=torch.float32),
                   load_w=torch.as_tensor(np.stack([self.static.load_weights(coords, r) for r in self.static.rows]),
                                          dtype=torch.float32),
                   S=torch.as_tensor(z["S"]), ks=torch.as_tensor(z["ks"]), dks=torch.as_tensor(z["dks"]))
        if self.with_ds:
            out["dS"] = torch.as_tensor(z["dS"])
        return out


# Voigt order (sxx, syy, szz, sxy, sxz, syz). Reflecting y flips the two shear
# components that carry one y index -- and it also turns the -y load into a +y
# load, while every dataset case is loaded in -y; by linearity that is one more
# overall sign. Net: normal and xz components flip, xy and yz keep their sign.
# (Checked on the uniform design: mirrored row r == row NY-1-r to round-off.)
MIRROR_SIGN = torch.tensor([-1.0, -1.0, -1.0, 1.0, -1.0, 1.0])


def mirror_maps(static):
    """The cross4 y-mirror (tile row iy -> NY-1-iy, load row r -> NY-1-r) as index
    maps: tmir (K,) in dims order, t'[k] = t[tmir[k]]; nmir (N,), node j of a design
    sits where node nmir[j] of its mirror image does. nmir comes from the mesh of
    the uniform design (exact_uniform_*.npz next to the dataset), which is exactly
    mirror-symmetric (match distance ~1e-15); the averaged reference mesh is not,
    and matching on it gets 32 nodes wrong."""
    import glob
    from scipy.spatial import cKDTree
    NX, NY = static.NX, static.NY
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    tmir = (ix * NY + (NY - 1 - iy)).reshape(-1)
    path = sorted(glob.glob(os.path.join(static.root, "exact_uniform_*.npz")))
    assert path, "mirror_maps needs a uniform design's mesh: run analysis/probe_dsig_uniform.py once"
    X = np.load(path[0])["coords"]
    d, nmir = cKDTree(X).query(X * np.array([1.0, -1.0, 1.0]) + np.array([0.0, NY * static.ty, 0.0]))
    assert d.max() < 1e-6 and len(np.unique(nmir)) == static.N, "uniform mesh is not mirror-symmetric"
    return torch.as_tensor(tmir), torch.as_tensor(nmir)


def mirror_case(t, load_w, S, dS, row, tmir, nmir, NY):
    """One training case (t (K,), load_w (N,), S (N,6), dS (K,N,6)) -> its mirror
    image: the exact solution of the mirrored design under the mirrored load."""
    S2, dS2, lw2 = torch.empty_like(S), torch.empty_like(dS), torch.empty_like(load_w)
    S2[nmir] = S * MIRROR_SIGN.to(S.dtype)
    lw2[nmir] = load_w
    dS2[:, nmir] = dS[tmir] * MIRROR_SIGN.to(dS.dtype)       # tile k of the mirror = tile tmir[k] here
    return t[tmir], lw2, S2, dS2, NY - 1 - row
