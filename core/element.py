from typing import List
from .node import Node

class Element:
    def __init__(self, eid: int, nodes: List[Node]):
        self.id = eid
        self.nodes = nodes 
