"""IRIT tile index <-> dims.json key conversions shared by the topopt
gradient paths, the stiffness/stress-weight helpers in core.mesh and the dsig
dataset. (The older contraction and surface helpers that lived here are in
git fed57d9.)
"""


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
    tile's value whenever NX != NY (this bit the old hollow_cube GNN
    pipeline on its 40x21x1 grid).
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


