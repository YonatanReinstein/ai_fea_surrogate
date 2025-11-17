from typing import List
from .node import Node

def anchor_condition(node: Node, tol: float = 1e-3) -> bool:
    if abs(node.coords[2]) < tol:
        return True
    return False


def force_pattern(node: Node, tol: float) -> List[float]:
    if abs(node.coords[2] - 9) < tol:
        return [1e6, 0.0, 0.0]
    return [0.0, 0.0, 0.0]
            


