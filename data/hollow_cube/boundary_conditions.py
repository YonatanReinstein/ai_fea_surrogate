from typing import List
from core.node import Node

def anchor_condition(node: Node, tol: float = 1e-3) -> bool:
    if abs(node.coords[2]) < tol:
        return True
    return False

def force_pattern(node: Node, tol: float) -> List[float]:
    fd = fixed_dims()
    if abs(node.coords[2] - fd["d1"]) < tol:
        mr = mesh_resolution()
        n_nodes_on_face =  mr[0]  * mr[1] * fd["d2"] * fd["d3"]
        #force_per_node = -1.5e5 * fd["d2"] * fd["d3"] / n_nodes_on_face
        force_per_node = -1e7 / n_nodes_on_face
        return [0.0, force_per_node, 0.0]
    return [0.0, 0.0, 0.0]

def mesh_resolution() -> tuple[int]:
    return (3, 3, 3)

def fixed_dims() -> dict:
    # Grid topology — d1=NX, d2=NY, d3=NZ. Fixed, not optimization variables.
    # The CAD model (model_lin.c) and the d4.. strut params depend on these.
    #d1 = length, d2 = height, d3 = width. all in tiles
    return {"d1": 10.0, "d2": 5.0, "d3": 1.0}

def tile_grid(dims: dict = None) -> tuple[int, int, int]:
    # d1=NX, d2=NY, d3=NZ match the ordering in model_lin.c
    fd = fixed_dims()
    return (int(fd["d1"]), int(fd["d2"]), int(fd["d3"]))
