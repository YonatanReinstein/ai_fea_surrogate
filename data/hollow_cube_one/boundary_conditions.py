from typing import List
from core.node import Node

# Single-tile grid (1x1x1) over the SAME shared outline.itd box, so the one
# tile spans the whole 40 x 21 x 1 domain. Used to inspect per-node sensitivity
# vectors without any tile-to-tile interaction.
LENGTH = 40.0   # Z axis
HEIGHT = 21.0   # Y axis
WIDTH = 1.0     # X axis


def anchor_condition(node: Node, tol: float = 1e-3) -> bool:
    return abs(node.coords[2]) < tol


def force_pattern(node: Node, tol: float) -> List[float]:
    # whole far face is loaded: with one tile there is no "middle tile" to pick
    if abs(node.coords[2] - LENGTH) < tol:
        mr = mesh_resolution()
        return [0.0, -5e7 / (mr[0] * mr[1]), 0.0]
    return [0.0, 0.0, 0.0]


def mesh_resolution() -> tuple[int]:
    return (3, 3, 3)


def fixed_dims() -> dict:
    return {"d1": 1.0, "d2": 1.0, "d3": 1.0}


def tile_grid(dims: dict = None) -> tuple[int, int, int]:
    return (1, 1, 1)
