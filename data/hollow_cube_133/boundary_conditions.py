from typing import List
from core.node import Node

# 1 x 3 x 3 tile grid (x, y, z) over the SAME shared outline.itd box:
# one tile through the 1-unit wall thickness (X), 3 up the height (Y),
# 3 along the length (Z) -> 9 tiles, d1..d9.
#
# NOTE the CAD model's own tile_grid() tuple is ordered (along Z, along Y,
# along X) -- see hollow_cube's real (40, 21, 1) against LENGTH/HEIGHT/WIDTH
# below -- so a 1x3x3 grid in (x, y, z) is fixed_dims d1=3, d2=3, d3=1.
LENGTH = 40.0   # Z axis
HEIGHT = 21.0   # Y axis
WIDTH = 1.0     # X axis


def anchor_condition(node: Node, tol: float = 1e-3) -> bool:
    return abs(node.coords[2]) < tol


def force_pattern(node: Node, tol: float) -> List[float]:
    NX, NY, NZ = tile_grid()
    on_force_face = abs(node.coords[2] - LENGTH) < tol
    tile_height = HEIGHT / NY
    mid_tile = NY // 2
    in_middle_tile = (mid_tile * tile_height - tol) <= node.coords[1] <= \
        ((mid_tile + 1) * tile_height + tol)
    if on_force_face and in_middle_tile:
        mr = mesh_resolution()
        n_nodes_on_tile = mr[0] * mr[1] * NZ
        return [0.0, -5e7 / n_nodes_on_tile, 0.0]
    return [0.0, 0.0, 0.0]


def mesh_resolution() -> tuple[int]:
    return (7, 7, 7)


def fixed_dims() -> dict:
    # grid topology, NOT lengths: (along Z, along Y, along X) = (3, 3, 1)
    return {"d1": 3.0, "d2": 3.0, "d3": 1.0}


def tile_grid(dims: dict = None) -> tuple[int, int, int]:
    fd = fixed_dims()
    return (int(fd["d1"]), int(fd["d2"]), int(fd["d3"]))
