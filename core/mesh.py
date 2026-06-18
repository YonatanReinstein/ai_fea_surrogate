from typing import List, Callable
from .node import Node
from .element import Element
from ansys.mapdl.core import launch_mapdl
import time
from ansys.mapdl.core.errors import MapdlRuntimeError
import pyvista as pv
import warnings
warnings.filterwarnings("ignore")

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
                self.mapdl = launch_mapdl(mode="grpc", nproc=4, override=True, cleanup_on_exit=True)
            self.mapdl.clear()
            #self.mapdl.prep7()
            #self.mapdl.et(1, 185)                 # SOLID185
            #self.mapdl.keyopt(1, 9, 0)            # (default integration)
            #self.mapdl.mp("EX", 1, young)
            #self.mapdl.mp("PRXY", 1, poisson)
            #for node in self.all_nodes():
            #    self.mapdl.n(node.id, *node.coords)
            #self.mapdl.type(1)
            #self.mapdl.mat(1)
            #for elem in self.all_elements():
            #    self.mapdl.en(elem.id, *[n.id for n in elem.nodes])
      
            ## Apply anchors
            #self.mapdl.allsel("ALL")
            #self.mapdl.nsel("NONE")
#
            #for node in self.all_nodes():
            #    if node.anchored:
            #        self.mapdl.nsel("A", "NODE", vmin=node.id, vmax=node.id)
            #    
            #self.mapdl.d("ALL", "UX", 0)
            #self.mapdl.d("ALL", "UY", 0)    
            #self.mapdl.d("ALL", "UZ", 0)
#
            ## Apply forces
            #for node in self.all_nodes():
            #    self.mapdl.allsel("ALL")
            #    self.mapdl.nsel("NONE")
            #    fx, fy, fz = node.forces
            #    if any([fx, fy, fz]):
            #        self.mapdl.nsel("A", "NODE", vmin=node.id, vmax=node.id)
            #        if fx != 0.0:
            #            self.mapdl.f("ALL", "FX", fx)
            #        if fy != 0.0:
            #            self.mapdl.f("ALL", "FY", fy)
            #        if fz != 0.0:
            #            self.mapdl.f("ALL", "FZ", fz)
#
            #
            #self.mapdl.allsel("ALL")     
            #self.mapdl.run("/SOLU")
            #self.mapdl.antype("STATIC")
            #self.mapdl.outres("ALL","ALL")
#
            cmds = [
                "/PREP7",
                "ET,1,185",
                "KEYOPT,1,9,0",
                f"MP,EX,1,{young}",
                f"MP,PRXY,1,{poisson}",
            ]

            cmds += [f"N,{n.id},{n.coords[0]},{n.coords[1]},{n.coords[2]}" for n in self.all_nodes()]
            cmds += ["TYPE,1", "MAT,1"]
            cmds += [f"EN,{e.id}," + ",".join(str(n.id) for n in e.nodes) for e in self.all_elements()]

            for n in self.all_nodes():
                if n.anchored:
#                    cmds += [f"D,{n.id},UX,0", f"D,{n.id},UY,0", f"D,{n.id},UZ,0"]
                    cmds += [f"D,{n.id},UX,0", f"D,{n.id},UY,0", f"D,{n.id},UZ,0"]


            force_nodes = [n for n in self.all_nodes() if any(n.forces)]
            if force_nodes:
                total_fx = sum(n.forces[0] for n in force_nodes)
                total_fy = sum(n.forces[1] for n in force_nodes)
                total_fz = sum(n.forces[2] for n in force_nodes)
                cx = sum(n.coords[0] for n in force_nodes) / len(force_nodes)
                cy = sum(n.coords[1] for n in force_nodes) / len(force_nodes)
                cz = sum(n.coords[2] for n in force_nodes) / len(force_nodes)
                pilot_id = max(self.nodes.keys()) + 1
                cmds += [
                    f"N,{pilot_id},{cx},{cy},{cz}",
                    "ET,2,21",
                    "KEYOPT,2,3,0",
                    "R,2,1e-20,1e-20,1e-20,1e-20,1e-20,1e-20",
                    "TYPE,2", "REAL,2",
                    f"E,{pilot_id}",
                ]
                for n in force_nodes:
                    cmds.append(f"CERIG,{pilot_id},{n.id},UXYZ")
                if total_fx: cmds.append(f"F,{pilot_id},FX,{total_fx}")
                if total_fy: cmds.append(f"F,{pilot_id},FY,{total_fy}")
                if total_fz: cmds.append(f"F,{pilot_id},FZ,{total_fz}")

            cmds += ["/SOLU", "ANTYPE,STATIC", "OUTRES,ALL,ALL"]

            self.mapdl.input_strings("\n".join(cmds))
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
            if screenshot_path is not None:
                plotter = self.mapdl.post_processing.plot_nodal_eqv_stress(
                    return_plotter=True)
                plotter.screenshot(screenshot_path)
            if created_mapdl:
                self.mapdl.exit()
                self.mapdl = None
            self.solution_valid = True

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
    
    def plot_mesh(self, save_path=None, resolution=(3840, 2160), aa_type="msaa", banner: str = None, elev: float = 0, azim: float = 0):
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

        plotter.add_mesh(grid, show_edges=True, opacity=0.6, color="lightblue")

        # Optional banner text
        if banner is not None:
            plotter.add_text(banner, position="upper_left", font_size=20, color="black")

        # ---------------------------------------------------------
        # FORCES — now rotated
        # ---------------------------------------------------------
        for nid, node in node_items:
            if node.forces != [0.0, 0.0, 0.0]:
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
                point_size=25,
                color="blue",
                render_points_as_spheres=True
            )

        # ---------------------------------------------------------
        # CAMERA RESET
        # ---------------------------------------------------------
        plotter.view_xz()
        plotter.camera.up = (1.0, 0.0, 0.0)  # 90° CCW from standard XZ view
        plotter.enable_parallel_projection()

        # ---------------------------------------------------------
        # SAVE OR SHOW
        # ---------------------------------------------------------
        if save_path is not None:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()



    def plot_mesh_with_tiles(self, tile_centers, tile_edges=None, node_tile_map=None, save_path=None, resolution=(3840, 2160)):
        import numpy as np
        import pyvista as pv
        from pyvista import CellType

        node_items = sorted(self.nodes.items())
        id_map = {nid: i for i, (nid, _) in enumerate(node_items)}
        points = np.array([node.coords for _, node in node_items], dtype=float)

        cells = []
        cell_types = []
        for elem in self.elements.values():
            local_ids = [id_map[n.id] for n in elem.nodes]
            cells.append(len(local_ids))
            cells.extend(local_ids)
            cell_types.append(CellType.HEXAHEDRON)

        grid = pv.UnstructuredGrid(np.array(cells), np.array(cell_types), points)
        R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        center = grid.center

        def apply_rot(p):
            return (R @ (np.asarray(p, float) - center)) + center

        grid.rotate_z(270, point=grid.center, inplace=True)

        off_screen = save_path is not None
        plotter = pv.Plotter(off_screen=off_screen, window_size=resolution)
        plotter.enable_anti_aliasing("msaa")
        plotter.add_mesh(grid, show_edges=True, opacity=0.3, color="lightblue")

        rotated_tc = [apply_rot(c) for c in tile_centers]
        plotter.add_points(
            np.array(rotated_tc),
            point_size=30,
            color="orange",
            render_points_as_spheres=True,
        )

        if tile_edges is not None:
            for i, j in tile_edges:
                line = pv.Line(rotated_tc[i], rotated_tc[j])
                plotter.add_mesh(line, color="orange", line_width=4)

        rotated_node_coords = {
            nid: apply_rot(np.array(node.coords, float))
            for nid, node in node_items
        }

        if node_tile_map is not None:
            for nid, node in node_items:
                t = node_tile_map.get(nid)
                if t is not None:
                    p1 = rotated_node_coords[nid]
                    p2 = rotated_tc[t]
                    line = pv.Line(p1, p2)
                    plotter.add_mesh(line, color="green", line_width=1, opacity=0.2)

        # Forces
        for nid, node in node_items:
            if node.forces != [0.0, 0.0, 0.0]:
                force = np.array(node.forces, dtype=float)
                if np.linalg.norm(force) > 1e-9:
                    arrow = pv.Arrow(start=rotated_node_coords[nid], direction=apply_rot(force), scale=0.1)
                    plotter.add_mesh(arrow, color="red")

        # Anchors
        anchored = [rotated_node_coords[nid] for nid, node in node_items if node.anchored is True]
        if anchored:
            plotter.add_points(np.array(anchored), point_size=25, color="blue", render_points_as_spheres=True)

        plotter.view_xz()
        plotter.camera.up = (1.0, 0.0, 0.0)
        plotter.enable_parallel_projection()

        if save_path is not None:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()


if __name__ == "__main__":
    from core.IritModel import IritCModel
    model = IritCModel("data/bistable/CAD_model/model", "data/bistable/CAD_model/dims.json")
    model.__exec__script__()
    #volume = model.get_volume()
    #print(f"Volume: {volume}")
    nodes, elements = model.create_mesh(U=5, V=5, W=5)
    mesh = Mesh(nodes, elements)
    mesh.plot_mesh()

    



