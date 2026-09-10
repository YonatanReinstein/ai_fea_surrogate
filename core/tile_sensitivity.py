"""Chain rule: CalculiX's discrete stress gradient w.r.t. node coordinates
composed with the geometric Jacobian d(node coords)/d(tile thickness) (from
core.geometric_jacobian.compute_thickness_jacobian, rebuilt at the current
design point every time it is asked for) to get d(ks_stress)/d(thickness)
per tile -- at a solver cost that does not grow with the tile count, instead
of the O(#tiles) remesh-and-resolve finite differences the rest of this
codebase otherwise relies on.

assemble_tile_thickness_sensitivity_axis, fed by Mesh.solve_ccx_axis_gradient
(which needs the patched binary from tools/build_ccx_axis.sh), computes

    d(ks_stress)/d(t_tile) = sum_n sum_j  grad[n][j] * dX_dt[n][j]

summed over each tile's own nodes n and over the three global axes j. A node
moves under at most one tile's thickness -- see compute_thickness_jacobian's
docstring -- so each term belongs to exactly one tile.

grad[n] is d(ks_stress)/d(x_n): a derivative w.r.t. the node's cartesian
POSITION, not a shape-gradient density over a design surface. So the
contraction is the plain chain rule -- there is no surface normal to project
onto, no boundary integral to discretize, and nothing to area-weight.
Interior nodes carry real terms here and are kept (CCX_DESI_INTERIOR=1):
IRIT's parametric tile mesher moves most of a tile's nodes when its shell
thickness changes, interior ones included, and dropping them is the single
largest error term in the contraction.

Per design point that is one extra CAD remesh (which recovers dX_dt for
every tile at once -- tiles share no thickness-sensitive node) plus three ccx
solves, one per axis, whatever the tile count. Nodes ccx rejects from the
design set come back as [0, 0, 0] and drop out of the sum on their own.

Accuracy is governed by elements-per-wall, not by tile count: give the
geometry a mesh_resolution that puts at least ~5 elements through its
thinnest wall or the per-tile numbers are wrong by several times, silently.
Measured per tile against remesh FD on hollow_cube_mid (48 tiles, mesh
5,5,5, C3D8), ratio adjoint/FD over the 26 tiles above the FD noise floor:
median 1.000, range 0.981..1.094. The gradient must be taken on C3D8, not
C3D8I -- ccx differentiates the statically condensed incompatible modes
badly (see ccx_evaluator.compute_topopt_gradient, which deliberately takes
the gradient on C3D8 and the reported stress on C3D8I).
"""
from typing import Dict, Set, Tuple


# C3D8 (Abaqus/CalculiX hex) face connectivity, 0-based into an element's own
# node list -- the *ELEMENT TYPE=C3D8 face numbering S1..S6.
_HEX_FACES = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
              (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))


# The midside nodes of those same six faces on a C3D20, again 0-based into
# the element's node list -- CalculiX's ifaceq rows 9..16 for each face. A
# 20-node mesh is a hex mesh whose faces carry four extra nodes; matching a
# face by its four CORNERS alone is still correct (two elements share a face
# iff they share its corners), so only the reported node ids change.
_HEX20_FACE_MIDS = ((8, 9, 10, 11), (15, 14, 13, 12), (16, 12, 17, 8),
                    (17, 13, 18, 9), (18, 14, 19, 10), (19, 15, 16, 11))


def _boundary_faces(elements: dict):
    """Free-surface quads of a hex mesh: faces belonging to exactly one
    element. Yields the face's node ids in their own element-local order."""
    count, ids_by_key = {}, {}
    for nlist in elements.values():
        quadratic = len(nlist) == 20
        for fi, f in enumerate(_HEX_FACES):
            ids = tuple(nlist[i] for i in f)
            if quadratic:
                ids = ids + tuple(nlist[i] for i in _HEX20_FACE_MIDS[fi])
            key = tuple(sorted(ids[:4]))
            count[key] = count.get(key, 0) + 1
            ids_by_key[key] = ids
    return [ids_by_key[k] for k, c in count.items() if c == 1]


def free_surface_nodes(elements: dict) -> Set[int]:
    """Node ids on the mesh's free surface.

    Not the design-node set: the axis route deliberately declares interior
    nodes too (CCX_DESI_INTERIOR=1), because IRIT's parametric tile mesher
    moves most of a tile's nodes when its shell thickness changes, interior
    ones included, and those nodes carry a real part of the derivative.
    This is here for callers that need the surface subset specifically --
    evaluators/validate_axis_gradient.py isolates the free-surface-only case
    in its A/B against the full-gradient contraction.
    """
    out: Set[int] = set()
    for ids in _boundary_faces(elements):
        out.update(ids)
    return out


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


def assemble_tile_thickness_sensitivity_axis(
    dX_dt: Dict[int, Tuple[float, float, float]],
    grad: Dict[int, Tuple[float, float, float]],
    elements: dict,
    elem_to_tile: dict,
    NX: int, NY: int, NZ: int,
    dims_keys,
) -> Dict[str, float]:
    """dims_key -> d(ks_value)/d(thickness), contracting the full
    global-axis nodal gradient from Mesh.solve_ccx_axis_gradient with the
    geometric Jacobian.

        d(ks)/d(t_tile) = sum_n sum_j  grad[n][j] * dX_dt[n][j]

    over each tile's own nodes n and the three global axes j -- the plain
    chain rule; see the module docstring for why no normal and no nodal area
    appear in it, and why interior nodes have to be in `grad`.

    Both inputs must come from the SAME nominal mesh: dX_dt, elements and
    elem_to_tile from compute_thickness_jacobian, `grad` from a
    solve_ccx_axis_gradient on the mesh it built. dX_dt is keyed by every
    node, so nodes ccx rejected from the design set (which come back as
    [0, 0, 0]) drop out of the sum on their own rather than being skipped.

    Every dims_key is always present in the result, zero-initialized: a tile
    with no contributing node genuinely gets 0.0 here.
    """
    node_to_tile = node_to_tile_map(elements, elem_to_tile)

    per_tile: Dict[str, float] = {k: 0.0 for k in dims_keys}
    for nid, dxdt in dX_dt.items():
        g = grad.get(nid)
        if g is None:
            continue
        tile_idx = node_to_tile.get(nid)
        if tile_idx is None:
            continue
        key = tile_index_to_dims_key(tile_idx, NX, NY, NZ, dims_keys)
        per_tile[key] += g[0] * dxdt[0] + g[1] * dxdt[1] + g[2] * dxdt[2]

    return per_tile


def hollow_tube_volume_sensitivity(thicknesses, total_volume: float):
    """d(total volume)/d(tile thickness) for every tile, analytically -- no
    solver, no FD sweep, not even an extra CAD call.

    hollow_cube's tile is a hollow square tube (model_lin.c's
    BuildHollowCubeTile), so in the tile's own local unit cube the material
    area is outer(1x1) - inner((1-2t)^2) = 4t(1-t), and every tile shares the
    same local->physical scale factor (a uniform tiling of one outline). The
    scale is calibrated from the ALREADY-COMPUTED total_volume rather than a
    hardcoded constant, so this is not tied to a particular outline size --
    but it IS tied to that tile's cross-section formula. A geometry with a
    different tile construction needs its own expression here.

    thicknesses: sequence of per-tile thickness values.
    Returns a list of dV/dt, one per entry, in the same order.
    """
    t = [float(v) for v in thicknesses]
    local = sum(4.0 * v * (1.0 - v) for v in t)
    scale = total_volume / local if local else 0.0
    return [scale * 4.0 * (1.0 - 2.0 * v) for v in t]
