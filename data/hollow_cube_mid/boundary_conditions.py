from typing import List
from core.node import Node

# outline.irt's GBOX dimensions are hardcoded there (shared with hollow_cube's
# CAD_model) and don't scale with tile count, unlike hollow_cube's own
# boundary_conditions.py where NX*1unit/NY*1unit happen to equal length/height
# for the real 40x21x1 grid. This geometry uses a coarser 8x6x1 tile grid over
# the SAME physical outline, so tile spans (LENGTH/NX, HEIGHT/NY) must be
# computed explicitly rather than assumed to be 1 unit.
LENGTH = 40.0   # Z axis
HEIGHT = 21.0   # Y axis
WIDTH = 1.0     # X axis


def anchor_condition(node: Node, tol: float = 1e-3) -> bool:
    if abs(node.coords[2]) < tol:
        return True
    return False


def force_pattern(node: Node, tol: float) -> List[float]:
    NX, NY, NZ = tile_grid()
    on_force_face = abs(node.coords[2] - LENGTH) < tol
    tile_height = HEIGHT / NY
    mid_tile = NY // 2
    in_middle_tile = (mid_tile * tile_height - tol) <= node.coords[1] <= ((mid_tile + 1) * tile_height + tol)
    if on_force_face and in_middle_tile:
        mr = mesh_resolution()
        n_nodes_on_tile = mr[0] * mr[1] * NZ
        force_per_node = -5e7 / n_nodes_on_tile
        return [0.0, force_per_node, 0.0]
    return [0.0, 0.0, 0.0]


def mesh_resolution() -> tuple[int]:
    return (3, 3, 3)


def fixed_dims() -> dict:
    # Grid topology -- d1=NX, d2=NY, d3=NZ tile counts (small test grid; not
    # a physical length, unlike the coincidental 40/21 in hollow_cube's own
    # fixed_dims). The CAD model (model_lin.c) and the d4.. strut params
    # depend on these.
    return {"d1": 8.0, "d2": 6.0, "d3": 1.0}


def tile_grid(dims: dict = None) -> tuple[int, int, int]:
    fd = fixed_dims()
    return (int(fd["d1"]), int(fd["d2"]), int(fd["d3"]))
