from typing import List, Callable
from .node import Node
from .element import Element
from ansys.mapdl.core import launch_mapdl
from ansys.mapdl.core.errors import MapdlRuntimeError
import pyvista as pv
pv.OFF_SCREEN = True 

class Mesh:
    def __init__(self, nodes, elements, tolerance=1e-9):
        self.nodes = nodes
        self.elements = elements
        self.mapdl = None
        self.tolerance = tolerance
        self.solution_valid = False

    def solve(self, young: float, poisson: float, mapdl=None, screenshot_path: str = None):
        self.mapdl = mapdl
        created_mapdl = False
        try:
            if self.mapdl is None:
                created_mapdl = True
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
            # Create the plot but DO NOT display it
            plotter = self.mapdl.post_processing.plot_nodal_eqv_stress(
                show=False,
                return_plotter=True)
            #plotter.scene.camera.elevation = 270
            if screenshot_path is not None:
                plotter.scene.screenshot(screenshot_path)
            if created_mapdl:
                self.mapdl.exit()
            self.mapdl = None
            self.solution_valid = True

            #for element in self.elements.values():
            #    ancher_element = False
            #    for node in element.nodes:
            #        if node.anchored:
            #            ancher_element = True
            #            break
            #    if ancher_element:
            #        for node in element.nodes:
            #            node.stress = 0.0
                
        except MapdlRuntimeError as e:
            print(f"MapdlRuntimeError: {e}")
            if created_mapdl:
                self.mapdl.exit()
            raise e
            

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
        return max(node.stress for node in self.all_nodes() )    #if node.anchored is False
    
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
    
    def plot_mesh(self, save_path=None, resolution=(3840, 2160), aa_type="msaa", banner: str = None):
        import numpy as np
        import pyvista as pv
        from pyvista import CellType

        node_items = sorted(self.nodes.items())
        id_map = {nid: i for i, (nid, _) in enumerate(node_items)}
        points = np.array([node.coords for _, node in node_items], dtype=float)
        

        # Build unstructured grid
        cells = []
        cell_types = []
        for elem in self.elements.values():
            local_ids = [id_map[n.id] for n in elem.nodes]
            cells.append(len(local_ids))
            cells.extend(local_ids)
            cell_types.append(CellType.HEXAHEDRON)

        grid = pv.UnstructuredGrid(
            np.array(cells),
            np.array(cell_types),
            points
        )
        R = R = np.array([
                [ 0,  1, 0],
                [-1,  0, 0],
                [ 0,  0, 1]
            ])
        center = grid.center

        def apply_rot(p):
            return (R @ (p - center)) + center

        # ---------------------------------------------------------
        # ROTATE THE MESH (your existing line)
        # ---------------------------------------------------------
        grid.rotate_z(270, point=grid.center, inplace=True)



        # Rotated coordinates for force arrows & anchor points
        rotated_node_coords = {
            nid: apply_rot(np.array(node.coords, float))
            for nid, node in node_items
        }

        # ---------------------------------------------------------
        # PLOTTING
        # ---------------------------------------------------------
        off_screen = save_path is not None
        plotter = pv.Plotter(off_screen=off_screen, window_size=resolution)
        plotter.enable_anti_aliasing(aa_type)

        plotter.add_mesh(grid, show_edges=True, opacity=0.6, color="lightgray")

        # Optional banner text
        if banner is not None:
            plotter.add_text(banner, position="upper_left", font_size=14, color="black", shadow=True)

        # ---------------------------------------------------------
        # FORCES — now rotated
        # ---------------------------------------------------------
        for nid, node in node_items:
            force = np.array(node.forces, dtype=float)
            if np.linalg.norm(force) > 1e-9:
                start = rotated_node_coords[nid]
                arrow = pv.Arrow(start=start, direction=apply_rot(force), scale=0.1)
                plotter.add_mesh(arrow, color="red")

        # ---------------------------------------------------------
        # ANCHORS — now rotated
        # ---------------------------------------------------------
        anchored = [
            rotated_node_coords[nid]
            for nid, node in node_items
            if node.anchored is True
        ]
        if anchored:
            plotter.add_points(
                np.array(anchored),
                point_size=8,
                color="blue",
                render_points_as_spheres=True
            )
       #     plotter.add_points(np.array(anchored), point_size=18, color="blue")

        # ---------------------------------------------------------
        # CAMERA RESET
        # ---------------------------------------------------------
        plotter.camera.roll = 0
        plotter.camera.elevation = 0
        plotter.camera.azimuth = 0

        # ---------------------------------------------------------
        # SAVE OR SHOW
        # ---------------------------------------------------------
        if save_path is not None:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()



if __name__ == "__main__":
    # Example usage
    # Define some nodes and elements
    nodes = {
        1: Node(1, [0.0, 0.0, 0.0]),
        2: Node(2, [1.0, 0.0, 0.0]),
        3: Node(3, [1.0, 1.0, 0.0]),
        4: Node(4, [0.0, 1.0, 0.0]),
        5: Node(5, [0.0, 0.0, 1.0]),
        6: Node(6, [1.0, 0.0, 1.0]),
        7: Node(7, [1.0, 1.0, 1.0]),
        8: Node(8, [0.0, 1.0, 1.0]),
    }
    elements = {
        1: Element(1, [nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7], nodes[8]]),
    }
    mesh = Mesh(nodes, elements)
    mesh.plot_mesh()

    



