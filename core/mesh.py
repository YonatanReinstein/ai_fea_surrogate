from typing import List
from .node import Node
from .element import Element
from utils.data_processing import face_nodes_by_axis, unit_vector
from ansys.mapdl.core import launch_mapdl
import torch
from torch_geometric.data import Data
import subprocess
import os
import shutil



class Mesh:
    def __init__(self, nodes, elements, tolerance=1e-9):
        self.nodes = nodes
        self.elements = elements
        self.mapdl = None
        self.max_stress = None
        self.max_displacement = None
        self.tolerance = tolerance

    @classmethod
    def from_inp(cls, path: str) -> "Mesh":
        from utils.read_inp import build_mesh_from_inp  # lazy import (avoids circular)
        nodes, elements = build_mesh_from_inp(path)
        return cls(nodes, elements)


    def solve(self, young: float, poisson: float):

        self.mapdl = launch_mapdl(mode="grpc", override=True, cleanup_on_exit=True)
        #self.mapdl.print_commands = True

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
        displacements = [node.displacement for node in self.nodes.values()]
        displacements_magnitudes = [(disp[0]**2 + disp[1]**2 + disp[2]**2)**0.5 for disp in displacements]
        self.max_displacement = max(displacements_magnitudes) if displacements_magnitudes else 0.0
        self.mapdl.post_processing.plot_nodal_eqv_stress()
        self.mapdl.exit()

    def add_anchor(self, element_id: int, face: str):
        element = self.get_element(element_id)
        axis, sign = face[-1], face[0]  # e.g. "+Z"
        face_n = face_nodes_by_axis(element.nodes, axis, sign, tol=self.tolerance)
        for node in face_n:
            node.anchored = True

    def clear_anchors(self):    
        for node in self.all_nodes():
            node.anchored = False

    def add_force(self, element_id: int, face: str, value: float, dir_code: str):
        element = self.get_element(element_id)
        axis, sign = face[-1], face[0]  # e.g. "+Z"
        face_n = face_nodes_by_axis(element.nodes, axis, sign, tol=self.tolerance)
        dir_vector = unit_vector(dir_code)
        for node in face_n:
            fx, fy, fz = node.forces
            node.forces = (fx + value * dir_vector[0],
                           fy + value * dir_vector[1],
                           fz + value * dir_vector[2])
    
    def clear_forces(self):
        for node in self.all_nodes():
            node.forces = [0.0, 0.0, 0.0]

    def get_element(self, eid: int) -> Element:
        return self.elements[eid]

    def get_node(self, nid: int) -> Node:
        return self.nodes[nid]

    def all_nodes(self) -> List[Node]:
        return list(self.nodes.values())

    def all_elements(self) -> List[Element]:
        return list(self.elements.values())
    
    def to_graph_with_labels(self):
        node_feats = []
        y_node = []

        for node in self.nodes.values():
            # --- Inputs ---
            feats = []
            feats.extend(node.coords)           # (x,y,z)
            feats.extend(node.forces or [0,0,0])# (Fx,Fy,Fz)
            feats.append(float(node.anchored))  # 1
            node_feats.append(feats)

            # --- Targets ---
            ux, uy, uz = node.displacement
            sigma_vm = node.stress
            y_node.append([ux, uy, uz, sigma_vm])

        x = torch.tensor(node_feats, dtype=torch.float)
        y_node = torch.tensor(y_node, dtype=torch.float)
        
        # edges (bidirectional)
        edges = set()
        for elem in self.elements.values():
            ids0 = [n.id - 1 for n in elem.nodes]
            L = len(ids0)
            for i in range(L):
                for j in range(i+1, L):
                    edges.add((ids0[i], ids0[j]))
                    edges.add((ids0[j], ids0[i]))
        edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long) if edges else torch.empty((2,0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y_node=y_node)
        return data
    def get_max_stress(self) -> float:
        if self.max_stress is None:
            raise ValueError("Mesh has not been solved yet.")
        return self.max_stress

    def get_max_displacement(self) -> float:
        if self.max_displacement is None:
            raise ValueError("Mesh has not been solved yet.")
        return self.max_displacement

    


 

    