"""Geometric Jacobian of mesh node coordinates w.r.t. per-tile thickness.

For hollow_cube's microstructure (model_lin.c), each tile's geometry is
built from its own scalar ShellThickness alone -- BuildHollowCubeTile only
reads LclData->ShellThicknesses[idx] for its own tile index, and within a
tile only the *inner* hole-wall corners depend on thickness; the *outer*
corners (which get welded to neighboring tiles by irit2inp's point-merge)
are constant. So tiles never share a thickness-sensitive node, and bumping
every tile's thickness by the same eps and remeshing once recovers every
tile's dNode/dThickness in a single extra mesh generation, instead of one
remesh per tile.
"""
from typing import Callable, Dict, Optional, Tuple

from .IritModel import IritCModel, IritModelBase


def _resolve_dims(dims_dict: dict) -> Dict[str, float]:
    return {
        k: (v["default"] if isinstance(v, dict) else v)
        for k, v in dims_dict.items()
    }


def compute_thickness_jacobian(
    irt_script_path: str,
    dims_dict: dict,
    U: int, V: int, W: int,
    fixed_dims: Optional[dict] = None,
    mesh_threads: int = 1,
    eps: float = 1e-4,
    model_cls: Callable[..., IritModelBase] = IritCModel,
) -> Tuple[Dict[int, Tuple[float, float, float]], dict, dict, dict]:
    """Global simultaneous finite-difference geometric Jacobian.

    Builds the nominal mesh and one perturbed mesh (every tile's thickness
    bumped by +eps at once), then forward-differences node coordinates.

    Returns:
        dX_dt:       node_id -> (dx/dt, dy/dt, dz/dt), per unit thickness.
                     Zero (to FD precision) for nodes on the outer,
                     tile-shared faces; nonzero only for a tile's own
                     inner hole-wall nodes.
        nodes0:      nominal node_id -> (x, y, z), for reuse.
        elements0:   nominal elem_id -> [node_id, ...], for reuse.
        elem_to_tile: nominal elem_id -> 0-based tile index, for reuse.
    """
    resolved = _resolve_dims(dims_dict)

    nominal = model_cls(irt_script_path, dims_dict=resolved,
                         fixed_dims=fixed_dims, mesh_threads=mesh_threads)
    nodes0, elements0, elem_to_tile0 = nominal.create_mesh(U=U, V=V, W=W)

    perturbed_dims = {k: v + eps for k, v in resolved.items()}
    perturbed = model_cls(irt_script_path, dims_dict=perturbed_dims,
                           fixed_dims=fixed_dims, mesh_threads=mesh_threads)
    nodes1, elements1, elem_to_tile1 = perturbed.create_mesh(U=U, V=V, W=W)

    if nodes0.keys() != nodes1.keys():
        raise RuntimeError(
            "Node set changed between nominal and perturbed meshes "
            f"({len(nodes0)} vs {len(nodes1)} nodes) -- the global "
            "simultaneous-FD assumption (identical topology/merge outcome "
            "across the perturbation) does not hold here."
        )
    if elements0.keys() != elements1.keys() or elem_to_tile0 != elem_to_tile1:
        raise RuntimeError(
            "Element connectivity or elem->tile mapping changed between "
            "nominal and perturbed meshes -- topology is not stable across "
            "the perturbation."
        )

    dX_dt = {
        nid: tuple((nodes1[nid][k] - nodes0[nid][k]) / eps for k in range(3))
        for nid in nodes0
    }
    return dX_dt, nodes0, elements0, elem_to_tile0
