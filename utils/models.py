from typing import List
from utils.arg_processing import face_nodes_by_axis, unit_vector
from ansys.mapdl.core import launch_mapdl
import torch
from torch_geometric.data import Data


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


class Element:
    def __init__(self, eid: int, nodes: List[Node]):
        self.id = eid
        self.nodes = nodes 
        for node in nodes:
            node.elements.append(self)  # back-reference


class Mesh:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements
        self.mapdl = None
        self.max_stress = None

    @classmethod
    def from_inp(cls, path: str) -> "Mesh":
        """Build a Mesh from an IRIT2INP-generated .inp file."""
        from utils.read_inp import build_mesh_from_inp  # lazy import (avoids circular)
        nodes, elements = build_mesh_from_inp(path)
        return cls(nodes, elements)


    def solve(self, young: float, poisson: float):
        # Launch MAPDL
        self.mapdl = launch_mapdl(mode="grpc", override=True, cleanup_on_exit=True)
        self.mapdl.clear()
        self.mapdl.prep7()
        self.mapdl.et(1, 185)                 # SOLID185
        self.mapdl.keyopt(1, 9, 0)            # (default integration)
        self.mapdl.mp("EX", 1, young)
        self.mapdl.mp("PRXY", 1, poisson)

        for node in self.all_nodes():
            self.mapdl.n(node.id, *node.coords)

        self.mapdl.type(1)
        self.mapdl.mat(1)

        for elem in self.all_elements():
            self.mapdl.en(elem.id, *[n.id for n in elem.nodes])
        
        # Apply anchors
        self.mapdl.allsel("ALL")
        self.mapdl.nsel("NONE")

        for node in self.all_nodes():
            if node.anchored:
                self.mapdl.nsel("A", "NODE", vmin=node.id, vmax=node.id)
            
        self.mapdl.d("ALL", "UX", 0)
        self.mapdl.d("ALL", "UY", 0)    
        self.mapdl.d("ALL", "UZ", 0)

        # Apply forces
        for node in self.all_nodes():
            self.mapdl.allsel("ALL")
            self.mapdl.nsel("NONE")
            fx, fy, fz = node.forces
            if any([fx, fy, fz]):
                self.mapdl.nsel("A", "NODE", vmin=node.id, vmax=node.id)
                if fx != 0.0:
                    self.mapdl.f("ALL", "FX", fx)
                if fy != 0.0:
                    self.mapdl.f("ALL", "FY", fy)
                if fz != 0.0:
                    self.mapdl.f("ALL", "FZ", fz)

        self.mapdl.allsel("ALL")     
        self.mapdl.run("/SOLU")
        self.mapdl.antype("STATIC")
        self.mapdl.outres("ALL","ALL")
        self.mapdl.solve()
        self.mapdl.post1()
        self.mapdl.set("last")

        # von Mises stress
        stress = self.mapdl.post_processing.nodal_eqv_stress()

        # Displacements
        ux = self.mapdl.post_processing.nodal_displacement("X")
        uy = self.mapdl.post_processing.nodal_displacement("Y")
        uz = self.mapdl.post_processing.nodal_displacement("Z")

        # Update nodes_xyz with displacement and stress
        for node_id, node in self.nodes.items():
            node.displacement = [ux[node_id-1], uy[node_id-1], uz[node_id-1]]
            node.stress = stress[node_id-1]

        self.max_stress = stress.max()
        #self.mapdl.post_processing.plot_nodal_eqv_stress()
        self.mapdl.exit()

    def add_anchor(self, element_id: int, face: str):
        element = self.get_element(element_id)
        axis, sign = face[-1], face[0]  # e.g. "+Z"
        face_n = face_nodes_by_axis(element.nodes, axis, sign)
        for node in face_n:
            node.anchored = True
    
    def add_force(self, element_id: int, face: str, value: float, dir_code: str):

        if element_id not in self.elements.keys():
            raise ValueError(f"Element {element_id} not found in mesh")

        axis, sign = face[-1], face[0]
        nlist = self.elements[element_id].nodes
        face_n = face_nodes_by_axis(nlist, axis, sign)
        if not face_n:
            raise ValueError(f"No face '{face}' found on element {element_id}")

        unit_vector = unit_vector(dir_code)
        for node in face_n: 
            node.forces += value * unit_vector

    def add_force(self, element_id: int, face: str, value: float, dir_code: str):
        element = self.get_element(element_id)
        axis, sign = face[-1], face[0]  # e.g. "+Z"
        face_n = face_nodes_by_axis(element.nodes, axis, sign)
        dir_vector = unit_vector(dir_code)
        for node in face_n:
            fx, fy, fz = node.forces
            node.forces = (fx + value * dir_vector[0],
                           fy + value * dir_vector[1],
                           fz + value * dir_vector[2])

    def get_element(self, eid: int) -> Element:
        return self.elements[eid]

    def get_node(self, nid: int) -> Node:
        return self.nodes[nid]

    def all_nodes(self) -> List[Node]:
        return list(self.nodes.values())

    def all_elements(self) -> List[Element]:
        return list(self.elements.values())
    
    def to_graph(self, include_features=True):
        # Convert the mesh into a PyTorch Geometric Data graph.
        node_features = []
        for node in self.nodes.values():
            feats = []
            if include_features:
                feats.extend(node.coords)                      
                feats.extend(node.displacement or [0, 0, 0])   
                feats.extend(node.stress or [0, 0, 0])   
                feats.extend(node.forces or [0, 0, 0])     
                feats.append(float(node.anchored))             
            node_features.append(feats)

        x = torch.tensor(node_features, dtype=torch.float)

        edges = set()
        for elem in self.elements.values():
            nids = [n.id - 1 for n in elem.nodes]  # zero-based indexing
            for i in range(len(nids)):
                for j in range(i + 1, len(nids)):
                    edges.add((nids[i], nids[j]))
                    edges.add((nids[j], nids[i]))  # make bidirectional

        if edges:
            edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        return data

    
