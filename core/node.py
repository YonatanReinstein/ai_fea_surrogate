from typing import List
from .element import Element

class Node:
    def __init__(self, nid: int, coords: List[float]):
        self.id = nid
        self.coords = coords
        self.displacement = [0.0, 0.0, 0.0]
        self.stress = 0.0
        self.anchored = False
        self.forces = [0.0, 0.0, 0.0]   
        self.elements = []   # elements that reference this node
    
    def get_elements(self) -> List["Element"]:
        return self.elements
    
    def __repr__(self):
        return (
            f"Node(\n"
            f"  id={self.id},\n"
            f"  coords={self.coords},\n"
            f"  displacement={self.displacement},\n"
            f"  stress={self.stress},\n"
            f"  anchored={self.anchored},\n"
            f"  forces={self.forces}\n"
            f")"
        )
