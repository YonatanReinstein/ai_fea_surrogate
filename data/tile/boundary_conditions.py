from typing import List
from core.node import Node

def anchor_condition(node: Node, tol: float = 1e-3) -> bool:
    if abs(node.coords[0]) < tol:
        return True
    return False

def force_pattern(node: Node, tol: float) -> List[float]:
    if abs(node.coords[0] - 1.0) < tol:
        return [0.0, 0.0, -1e7]
    return [0.0, 0.0, 0.0]

def mesh_resolution() -> tuple[int]:
    return (3, 3, 3)
           