from typing import List, Callable
from .node import Node
from .element import Element
from ansys.mapdl.core import launch_mapdl

class Mesh:
    def __init__(self, nodes, elements, tolerance=1e-9):
        self.nodes = nodes
        self.elements = elements
        self.mapdl = None
        self.tolerance = tolerance
        self.solution_valid = False

    def solve(self, young: float, poisson: float):
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

        #self.mapdl.post_processing.plot_nodal_eqv_stress()
        self.mapdl.exit()
        self.solution_valid = True

    def anchor_node(self, node_id: int):
        node = self.get_node(node_id)
        node.anchored = True

    def anchor_nodes_by_condition(self, condition: Callable[[Node], bool]):
        for node in self.all_nodes():
            if condition(node, self.tolerance):
                node.anchored = True

    def clear_anchors(self):    
        for node in self.all_nodes():
            node.anchored = False

    def get_max_stress(self):
        if not self.solution_valid:
            raise ValueError("Solution is not valid. Please run the simulation first.")
        return max(node.stress for node in self.all_nodes() if node.stress is not None)
    
    def get_max_displacement(self):
        if not self.solution_valid:
            raise ValueError("Solution is not valid. Please run the simulation first.")
        return max(
            (node.displacement[0]**2 + node.displacement[1]**2 + node.displacement[2]**2)**0.5
            for node in self.all_nodes()
        )

    def apply_force_on_node(self, node_id: int, force: List[float]):
        node = self.get_node(node_id)
        node.forces = force

    def apply_force_by_pattern(self, force_pattern: Callable[[Node, float], List]):
        for node in self.all_nodes():
            force = force_pattern(node, self.tolerance)
            self.apply_force_on_node(node.id, force)

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
    
    import pyvista as pv
    import numpy as np

    def plot_mesh(nodes, elements):
        # Build array of points (N, 3)
        points = np.array([n.coords for n in nodes.values()], dtype=float)

        # Build cells
        # Flatten into: [num_points_in_elem, n1, n2, n3, n4, ...]
        cells = []
        cell_types = []

        for elem in elements.values():
            node_ids = [n.id - 1 for n in elem.nodes]  # zero-based indexing
            cells.append(len(node_ids))
            cells.extend(node_ids)
            cell_types.append(12)  # VTK_HEXAHEDRON = 12 (for solid185/c3d8 cubes)

        cells = np.array(cells)

        mesh = pv.UnstructuredGrid(cells, cell_types, points)

        mesh.plot(show_edges=True)



