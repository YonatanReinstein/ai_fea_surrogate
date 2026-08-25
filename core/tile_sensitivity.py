"""Chain rule: CCX's shape sensitivity (d(ks_stress)/d(normal displacement)
per free-surface node, from Mesh.solve_ccx_sensitivity) composed with the
geometric Jacobian (d(node coords)/d(tile thickness), from
core.geometric_jacobian.compute_thickness_jacobian) to get
d(ks_stress)/d(thickness) per tile -- without the O(#tiles) per-tile finite
differences the rest of this codebase otherwise relies on.

    d(ks_stress)/d(thickness_tile) = sum_n  dfdn[n] * dot(normal[n], dX_dt[n])

summed over each tile's own nodes n (a node moves under at most one tile's
thickness -- see compute_thickness_jacobian's docstring -- so this sum only
ever has one tile's nodes contributing per term). Nodes CCX excluded from
the design set (dfdn/normal absent -- see Mesh.solve_ccx_sensitivity's
docstring) simply drop out of the sum.
"""
from typing import Dict, Tuple


def node_to_tile_map(elements: dict, elem_to_tile: dict) -> Dict[int, int]:
    """node_id -> IRIT-native tile index (first element wins on shared
    nodes), matching core.component.py's to_graph_with_labels convention."""
    node_to_tile = {}
    for eid, nlist in elements.items():
        t = elem_to_tile.get(eid, -1)
        if t < 0:
            continue
        for nid in nlist:
            if nid not in node_to_tile:
                node_to_tile[nid] = t
    return node_to_tile


def tile_idx_to_dims_idx(tile_idx: int, NX: int, NY: int, NZ: int) -> int:
    """Map elem_to_tile's IRIT-native tile numbering (X fastest, Z slowest --
    see UserMicroEvalTilingIndex in irit/user_lib/micro_strct.c) to the
    dims.json ShellThicknesses array position for the same physical tile
    (X-outermost, per model_lin.c's read_shell_thicknesses comment) --
    these are different orderings of the same 840 tiles, not the same
    index. Also needed the other way whenever a per-tile value already in
    dims-key order (e.g. a raw dims dict/vector) has to be re-laid-out to
    tile_idx-native order to line up with tile_idx-indexed per-node
    features or the coarse-grid position/adjacency tensors, which are all
    natively tile_idx-ordered (see core/geometric_jacobian.py,
    utils/gnn_surrogate.py's _build_tile_positions) -- a dims-key-ordered
    array indexed directly by tile_idx silently feeds most nodes another
    tile's value whenever NX != NY (confirmed: this was happening in every
    strut/thickness feature injection site -- gnn_evaluator.py,
    gnn_training.py's _preprocess, probe_gradloss_sensitivity.py -- for
    hollow_cube's real 40x21x1 grid).
    """
    ix = tile_idx % NX
    iy = (tile_idx // NX) % NY
    iz = tile_idx // (NX * NY)
    return ix * NY * NZ + iy * NZ + iz


def tile_index_to_dims_key(tile_idx: int, NX: int, NY: int, NZ: int, dims_keys) -> str:
    """Map elem_to_tile's IRIT-native tile index to the dims.json
    ShellThicknesses key for the same physical tile -- see
    tile_idx_to_dims_idx's docstring for the two conventions."""
    dims_idx = tile_idx_to_dims_idx(tile_idx, NX, NY, NZ)
    return dims_keys[dims_idx]


def dims_key_order_permutation(NX: int, NY: int, NZ: int):
    """Index permutation (plain list, length NX*NY*NZ) such that, for any
    sequence `v` laid out in dims-key order (v[i] = dims key "d{i+1}"'s
    value), `[v[i] for i in dims_key_order_permutation(NX,NY,NZ)]` (or the
    tensor equivalent `v[perm]`) reorders it to tile_idx-native order:
    position t holds the value for the physical tile elem_to_tile calls t.
    Callers building a per-tile torch tensor should wrap this in
    torch.tensor(..., dtype=torch.long) once (NX/NY/NZ are fixed for a
    given geometry) and reuse it, rather than recomputing per sample.
    """
    K = NX * NY * NZ
    return [tile_idx_to_dims_idx(t, NX, NY, NZ) for t in range(K)]


def assemble_tile_thickness_sensitivity(
    dX_dt: Dict[int, Tuple[float, float, float]],
    dfdn: Dict[int, float],
    normal: Dict[int, Tuple[float, float, float]],
    elements: dict,
    elem_to_tile: dict,
    NX: int, NY: int, NZ: int,
    dims_keys,
) -> Dict[str, float]:
    """Returns dims_key -> d(ks_value)/d(thickness) for that tile.

    A tile absent from the result had every one of its design nodes
    excluded by CCX (see Mesh.solve_ccx_sensitivity's docstring) -- its
    sensitivity is unknown, not zero; callers should not conflate the two.
    """
    node_to_tile = node_to_tile_map(elements, elem_to_tile)

    per_tile: Dict[str, float] = {}
    for nid, dxdt in dX_dt.items():
        n = normal.get(nid)
        d = dfdn.get(nid)
        if n is None or d is None:
            continue  # CCX excluded this node from the design set
        tile_idx = node_to_tile.get(nid)
        if tile_idx is None:
            continue
        proj = dxdt[0] * n[0] + dxdt[1] * n[1] + dxdt[2] * n[2]
        key = tile_index_to_dims_key(tile_idx, NX, NY, NZ, dims_keys)
        # Empirically determined (cross-checked against a direct
        # finite-difference of ks_stress w.r.t. tile thickness on
        # hollow_cube): CCX's DFDN is the derivative w.r.t. displacement
        # along the OPPOSITE of the reported NORM direction, not along it
        # -- so the chain-rule term needs a sign flip here. Without it, the
        # assembled sensitivity matches the true FD in magnitude (up to the
        # expected partial-coverage undercount from excluded design nodes)
        # but with the wrong sign on every tile.
        per_tile[key] = per_tile.get(key, 0.0) - d * proj

    return per_tile
