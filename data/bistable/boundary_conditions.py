from typing import List
from core.node import Node

def anchor_condition(node: Node, tol: float = 1e-3) -> bool:
    if abs(node.coords[2]) < tol:
        return True
    return False

def force_pattern(node: Node, tol: float) -> List[float]:
    if abs(node.coords[2] - 1.0) < tol:
        return [0.0,-1e6, 0.0]
    return [0.0, 0.0, 0.0]

def mesh_resolution() -> tuple[int]:
    return (3, 3, 3)

def tile_grid(dims: dict) -> tuple[int, int, int]:
    # d1=NX, d2=NY, d3=NZ match the ordering in model_lin.c
    return (int(dims["d1"]), int(dims["d2"]), int(dims["d3"]))
