from typing import List
from core.node import Node

def anchor_condition(node: Node, tol: float = 1e-3) -> bool:
    if abs(node.coords[2]) < tol:
        return True
    return False

def force_pattern(node: Node, tol: float) -> List[float]:
    fd = fixed_dims()
    on_force_face = abs(node.coords[2] - fd["d1"]) < tol
    # Concentrate the load on the middle tile (along the height/y direction)
    # of the d2 tiles on the force face. Tiles are indexed 0..d2-1 and span
    # one unit each in tile coords, so the middle tile covers [mid, mid+1].
    mid_tile = int(fd["d2"]) // 2  # 10 for d2 = 21
    in_middle_tile = (mid_tile - tol) <= node.coords[1] <= (mid_tile + 1.0 + tol)
    if on_force_face and in_middle_tile:
        mr = mesh_resolution()
        # Nodes on the force face that lie within the single middle tile.
        n_nodes_on_tile = mr[0] * mr[1] * fd["d3"]
        force_per_node = -5e7 / n_nodes_on_tile
        return [0.0, force_per_node, 0.0]
    return [0.0, 0.0, 0.0]

def mesh_resolution() -> tuple[int]:
    return (3, 3, 3)

def fixed_dims() -> dict:
    # Grid topology — d1=NX, d2=NY, d3=NZ. Fixed, not optimization variables.
    # The CAD model (model_lin.c) and the d4.. strut params depend on these.
    #d1 = length, d2 = height, d3 = width. all in tiles
    return {"d1": 40.0, "d2": 21.0, "d3": 1.0}
    
def tile_grid(dims: dict = None) -> tuple[int, int, int]:
    # d1=NX, d2=NY, d3=NZ match the ordering in model_lin.c
    fd = fixed_dims()
    return (int(fd["d1"]), int(fd["d2"]), int(fd["d3"]))
