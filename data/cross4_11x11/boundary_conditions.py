import os
from typing import List
from core.node import Node

# A physically SQUARE 11 x 11 x 1 part on an 11x11x1 tile grid, for fast
# iteration.
#
# WHY THIS EXISTS. hollow_cube is a 40 x 21 x 1 part on a 40x21x1 grid = 840
# tiles, which at mesh_resolution (3,3,3) is 48 249 nodes / 26 880 elements and
# costs ~35-46 s per fem adjoint. This part is 121 tiles of the same 1x1x1 size,
# so the mesh is ~1/7 as large and the adjoint -- one scipy splu factorization,
# which dominates -- drops to a few seconds.
#
# THIS IS ITS OWN PART, NOT A COARSER VIEW OF hollow_cube. CAD_model/outline.itd
# here is a REAL FILE (X 0-1, Y 0-11, Z 0-11), deliberately not the symlink to
# hollow_cube's outline (X 0-1, Y 0-21, Z 0-40) that model/ still is. Two
# consequences:
#   * Never "edit" a symlinked outline in place -- writing through it would
#     rewrite hollow_cube's own geometry and silently invalidate the 840-tile
#     runs and the training set built on them.
#   * Results do NOT transfer by rescaling. The bounding box is 1x11x11 = 121
#     against hollow_cube's 840, so V_ref differs (at uniform t=0.3: 101.64 here
#     vs 705.60 there, both = box x 4t(1-t)), and the same -5e7 load on a
#     shorter, narrower cantilever is a different stress problem, so
#     material_properties.json's yield_strength has to be recalibrated for the
#     constraint to mean what it means on hollow_cube.
#
# Tiles are 1 x 1 x 1 here (11 units over 11 tiles), the same size as
# hollow_cube's, so tile coordinates and physical coordinates coincide -- but
# force_pattern still derives the span from HEIGHT / NY rather than assuming it,
# so changing the grid or the outline does not silently move the load off the
# part. That assumption is exactly what breaks when only these constants are
# edited: the load goes looking for a face that is not there.
LENGTH = 11.0   # Z axis
HEIGHT = 11.0   # Y axis
WIDTH = 1.0     # X axis


def anchor_condition(node: Node, tol: float = 1e-3) -> bool:
    if abs(node.coords[2]) < tol:
        return True
    return False


def load_rows() -> list[int]:
    """Every loaded free-end tile row. FEA_LOAD_ROW may be a comma list
    ("4,5,6": three stacked tiles whose load pads meet in one plate --
    model_lin.c reads the same list); the default is the middle row. The
    loaded faces are tied into one rigid body carrying force_pattern.resultant,
    so the TOTAL load is FEA_LOAD_N however many rows share it."""
    NX, NY, NZ = tile_grid()
    v = os.environ.get("FEA_LOAD_ROW")
    rows = sorted(int(t) for t in v.split(",")) if v not in (None, "") else [NY // 2]
    assert all(0 <= r < NY for r in rows), f"FEA_LOAD_ROW={v} outside 0..{NY - 1}"
    assert rows == list(range(rows[0], rows[-1] + 1)), f"FEA_LOAD_ROW={v}: rows must be adjacent"
    return rows


def load_row() -> int:
    """Tile row (0 = bottom .. NY-1 = top) of the free-end face that carries
    the load. Default the middle row, the production load. FEA_LOAD_ROW
    overrides it -- an env var, like FEA_MESH_RES, because force_pattern runs
    inside worker processes. GNNDsigEvaluator reads the same variable for
    the row it predicts, so the surrogate and the fem reference /
    screenshots load the same tile."""
    rows = load_rows()
    # callers of load_row() assume ONE loaded tile (load_tile/load_nbhd KS
    # masks, the fixed load block, the surrogate's load feature)
    assert len(rows) == 1, f"FEA_LOAD_ROW={rows}: several load rows, use load_rows()"
    return rows[0]


def force_pattern(node: Node, tol: float) -> List[float]:
    NX, NY, NZ = tile_grid()
    on_force_face = abs(node.coords[2] - LENGTH) < tol
    tile_height = HEIGHT / NY
    rows = load_rows()
    in_middle_tile = (rows[0] * tile_height - tol) <= node.coords[1] <= ((rows[-1] + 1) * tile_height + tol)
    if on_force_face and in_middle_tile:
        mr = mesh_resolution()
        # Total load stays -5e7 whatever the grid, so stress levels stay
        # comparable to hollow_cube's and the 100 MPa yield means the same thing.
        n_nodes_on_tile = mr[0] * mr[1] * NZ
        force_per_node = -5e7 / n_nodes_on_tile
        return [0.0, force_per_node, 0.0]
    return [0.0, 0.0, 0.0]


# cross4 (own copy of hollow_cube_11x11's file, which it used to symlink): the
# loaded face may be the load pad, three boxes rather than one n x n patch, so
# Mesh.apply_force_by_pattern rescales the face's forces to exactly this total.
force_pattern.resultant = (0.0, -float(os.environ.get("FEA_LOAD_N", "5e7")), 0.0)
# FEA_LOAD_N: load magnitude in N (default 5e7, the hollow_cube load). An env
# var, like FEA_MESH_RES, so evaluator worker processes inherit it.


def mesh_resolution() -> tuple[int]:
    # FEA_MESH_RES overrides, e.g. FEA_MESH_RES=5,5,5. Read from the
    # environment rather than passed in because this is called inside
    # ProcessPoolExecutor workers (force_pattern normalises the load by it),
    # where a parent monkeypatch would not be visible but an inherited env var
    # is. Default (4,4,2) since 2026-10-14 (was (3,3,3), hollow_cube's
    # default and what the fem adjoint was validated at); each tile wall gets
    # (U-1)x(V-1)x(W-1) elements, and depth resolution barely matters here
    # (the stress is in-plane).
    v = os.environ.get("FEA_MESH_RES")
    if v:
        u, vv, w = (int(t) for t in v.split(","))
        return (u, vv, w)
    return (4, 4, 2)


def fixed_dims() -> dict:
    # Grid topology, NOT lengths -- d1=NX (along Z/length), d2=NY (along
    # Y/height), d3=NZ (along X/width). Fixed, not optimization variables; the
    # CAD model (model_lin.c) and the per-tile thickness params in dims.json
    # depend on these. Note this d1/d2/d3 namespace is separate from dims.json's
    # d1..d121, which are the tile thicknesses.
    return {"d1": 11.0, "d2": 11.0, "d3": 1.0}


def tile_grid(dims: dict = None) -> tuple[int, int, int]:
    # d1=NX, d2=NY, d3=NZ match the ordering in model_lin.c
    fd = fixed_dims()
    return (int(fd["d1"]), int(fd["d2"]), int(fd["d3"]))


# FEA_LOAD_MODE=tile_nodes: instead of loading the free-end face (through the
# load pad and a rigid tie), spread FEA_LOAD_N over EVERY node of the loaded
# tile(s), weighted 1 - r / r_max by in-plane distance r from the loaded
# region's centre: largest at the centre junction, zero at the farthest node.
# Per node, no rigid tie (Mesh._apply_node_load). Use without a load pad
# (CROSS4_LOAD_PAD_THICKNESS=0), and freeze the loaded tile and its face
# neighbours in an optimization (TOPOPT_FIX_LOAD_NBHD) so the loaded nodes
# never move -- the adjoint holds the nodal forces fixed.
if os.environ.get("FEA_LOAD_MODE") == "tile_nodes":
    _NX, _NY, _ = tile_grid()
    _tz, _ty, _rows = LENGTH / _NX, HEIGHT / _NY, load_rows()
    _y0, _y1 = _rows[0] * _ty, (_rows[-1] + 1) * _ty
    force_pattern.node_load = {
        "box": ((-1e9, 1e9), (_y0, _y1), (LENGTH - _tz, LENGTH)),
        "center": (0.0, 0.5 * (_y0 + _y1), LENGTH - 0.5 * _tz),
        "total": force_pattern.resultant,
    }
    # FEA_LOAD_WEIGHT=pyramid: w = 1 - max(|dy|/(ty/2), |dz|/(tz/2)), zero on
    # the loaded region's whole edge instead of only at its corners
    if os.environ.get("FEA_LOAD_WEIGHT") == "pyramid":
        force_pattern.node_load.update(weight="pyramid", half=(0.5 * (_y1 - _y0), 0.5 * _tz))
    elif os.environ.get("FEA_LOAD_WEIGHT"):
        raise ValueError(f"FEA_LOAD_WEIGHT={os.environ['FEA_LOAD_WEIGHT']!r}: use pyramid or unset")
elif os.environ.get("FEA_LOAD_MODE"):
    raise ValueError(f"FEA_LOAD_MODE={os.environ['FEA_LOAD_MODE']!r}: use tile_nodes or unset")
