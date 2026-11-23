from typing import List, Callable
from .node import Node
from .element import Element
import pyvista as pv
import warnings
warnings.filterwarnings("ignore")

pv.OFF_SCREEN = True

# ---------------------------------------------------------------------------
# Stiffness penalty for thin tiles (TOPOPT_STIFF_PENALTY = t_s, e.g. 0.03)
# ---------------------------------------------------------------------------
# A tile of thickness t gets Young's modulus E * s(t), s(t) = 1 - exp(-(t/t_s)^p) (p = TOPOPT_STIFF_P, default 3):
# 1 for normal tiles (s(2 t_s) = 0.9997), 0.036 at t = t_s / 3. A near-floor
# tile then carries almost no load -- the clean-up check on
# cross11_mean_tilenodes_F5e6_t20_mma_fixtile found its 52 floor tiles (t =
# 0.01, full stiffness) carrying real load: deleting them tripled the peak
# (100 -> 312 MPa). Stress is computed from each element's own softened
# material, sigma = s(t) E eps, so a near-empty tile's stress fades with it --
# the stress relaxation (exponent q = p) that lets such tiles leave the stress
# constraint. Smooth in t, so gradients and MMA keep working.
def stiffness_penalty_ts():
    import os
    v = os.environ.get("TOPOPT_STIFF_PENALTY")
    return float(v) if v not in (None, "") else None


def stiffness_exponent():
    """TOPOPT_STIFF_P, default 3. Use a steep exponent (e.g. 6) with
    TOPOPT_STRESS_RELAX so the stiffness drop and the stress relaxation switch
    on over the SAME narrow band above the floor: a tile that is soft but whose
    real-material stress counts in full overstates that stress by ~1/s (soft ->
    large strain); t_s 0.03, p 3 with relaxation read 436 MPa on a design whose
    honest full-stiffness peak was 144."""
    import os
    return float(os.environ.get("TOPOPT_STIFF_P") or 3.0)


def stiffness_scale(t, ts):
    import math
    p = stiffness_exponent()
    return 1.0 - math.exp(-(t / ts) ** p)


def stiffness_scale_dt(t, ts):
    import math
    p = stiffness_exponent()
    return p * t ** (p - 1) / ts ** p * math.exp(-(t / ts) ** p)


def element_stiffness_scales(dims, elem_to_tile, fixed_dims, dims_keys=None):
    """elem id -> s(t of its tile), or None when TOPOPT_STIFF_PENALTY is unset.
    dims: {key: t} (or {key: {"default": t}}); fixed_dims gives the tile grid
    (d1 = NX, d2 = NY, d3 = NZ); elem_to_tile maps to IRIT's 0-based tile index."""
    ts = stiffness_penalty_ts()
    if ts is None:
        return None
    from .tile_sensitivity import tile_index_to_dims_key
    keys = list(dims_keys) if dims_keys is not None else list(dims)
    val = {k: float(v["default"] if isinstance(v, dict) else v) for k, v in dims.items()}
    NX, NY, NZ = (int(fixed_dims[k]) for k in ("d1", "d2", "d3"))
    tile_s = {}
    out = {}
    for eid, ti in elem_to_tile.items():
        if ti not in tile_s:
            tile_s[ti] = stiffness_scale(val[tile_index_to_dims_key(ti, NX, NY, NZ, keys)], ts)
        out[eid] = tile_s[ti]
    return out


# Stress relaxation separate from the stiffness penalty (TOPOPT_STRESS_RELAX =
# t_w, e.g. 0.004; floor TOPOPT_RELAX_TMIN, default 0.01). The stress counted
# in the constraint becomes w(t) * (the stress the REAL material would carry at
# that deformation), w(t) = 1 - exp(-((t - tmin) / t_w)^2) for t > tmin, else 0:
# zero at the floor, 0.79 at tmin + t_w, ~1 by tmin + 2.5 t_w. Without it the
# stress is the penalized s(t) E eps, which also relaxes thin STRUCTURAL tiles:
# the cross run cross11_..._stiff03 read 100 MPa but 143.8 MPa at full
# stiffness, its peak in a 0.01-0.03 tile beside the load.
def stress_relax_params():
    import os
    v = os.environ.get("TOPOPT_STRESS_RELAX")
    if v in (None, ""):
        return None
    return float(v), float(os.environ.get("TOPOPT_RELAX_TMIN") or 0.01)


def stress_weight(t, tw, tmin):
    import math
    return 0.0 if t <= tmin else 1.0 - math.exp(-((t - tmin) / tw) ** 2)


def stress_weight_dt(t, tw, tmin):
    import math
    if t <= tmin:
        return 0.0
    a = (t - tmin) / tw
    return 2.0 * a / tw * math.exp(-a * a)


def element_stress_weights(dims, elem_to_tile, fixed_dims, dims_keys=None):
    """elem id -> w(t of its tile), or None when TOPOPT_STRESS_RELAX is unset."""
    prm = stress_relax_params()
    if prm is None:
        return None
    from .tile_sensitivity import tile_index_to_dims_key
    keys = list(dims_keys) if dims_keys is not None else list(dims)
    val = {k: float(v["default"] if isinstance(v, dict) else v) for k, v in dims.items()}
    NX, NY, NZ = (int(fixed_dims[k]) for k in ("d1", "d2", "d3"))
    cache, out = {}, {}
    for eid, ti in elem_to_tile.items():
        if ti not in cache:
            cache[ti] = stress_weight(val[tile_index_to_dims_key(ti, NX, NY, NZ, keys)], *prm)
        out[eid] = cache[ti]
    return out


class Mesh:
    def __init__(self, nodes, elements, tolerance=1e-9):
        self.nodes = nodes
        self.elements = elements
        self.tolerance = tolerance
        self.solution_valid = False

            

    def solve_fem(self, young: float, poisson: float):
        """Linear static solve with core.fem_adjoint's own C3D8I assembly (no
        CalculiX binary). Same boundary conditions as the topopt gradient path
        (FEMEvaluator.compute_topopt_gradient): anchored nodes fixed, loaded
        nodes in one rigid tie carrying the resultant, or each node's own
        force when rigid_load is False; elem_scale (TOPOPT_STIFF_PENALTY)
        scales each element's modulus. Fills node.displacement / node.stress
        (von Mises, Pa), as the deleted CalculiX solve did (git fed57d9)."""
        from core.fem_adjoint import ShapeAdjoint
        if self.hex_order() != 8:
            raise ValueError("solve_fem supports 8-node hexes only")
        nodes = self.all_nodes()
        coords = {n.id: list(n.coords) for n in nodes}
        conn = {e.id: [n.id for n in e.nodes] for e in self.all_elements()}
        anchored = [n.id for n in nodes if n.anchored]
        scales = getattr(self, "elem_scale", None)
        e_scale = [scales.get(e, 1.0) for e in sorted(conn)] if scales else None
        if getattr(self, "rigid_load", True):
            loaded = [n for n in nodes if any(n.forces)]
            sa = ShapeAdjoint(coords, conn, anchored, [n.id for n in loaded],
                              [sum(n.forces[j] for n in loaded) for j in range(3)],
                              young, poisson, e_scale=e_scale)
        else:
            sa = ShapeAdjoint(coords, conn, anchored, [], (0.0, 0.0, 0.0), young, poisson,
                              rigid_tie=False, e_scale=e_scale,
                              nodal_forces={n.id: list(n.forces) for n in nodes if any(n.forces)})
        u = sa.solve().detach().numpy().reshape(-1, 3)
        _, vm = sa.nodal_stress()
        vm = vm.numpy()
        for i, nid in enumerate(sa.ids):
            self.nodes[nid].displacement = [float(x) for x in u[i]]
            self.nodes[nid].stress = float(vm[i])
        self.solution_valid = True


    def anchor_nodes_by_condition(self, condition: Callable[[Node], bool]):
        for node in self.all_nodes():
            if condition(node, self.tolerance):
                node.anchored = True



    def get_aggregated_stress(self, rho: float = 0.1):
        """KS (Kreisselmeier-Steinhauser) upper bound on the nodal von Mises
        stress field, in the same units as node.stress (Pa).

        sigma_KS = sigma_max + (1/rho) * ln(sum_i exp(rho * (sigma_i - sigma_max)))

        Unlike get_max_stress() (a hard, non-smooth max over a single node),
        every node within ~1/rho of the peak contributes to the sum, so the
        aggregated value responds smoothly to material added near -- but not
        exactly at -- the current peak location. Always >= get_max_stress(),
        approaching it as rho -> inf. rho is in 1/Pa here; pass rho scaled
        for the stress units the caller works in (e.g. rho_MPa * 1e-6 if
        node.stress is in Pa and rho was tuned in 1/MPa).
        """
        if not self.solution_valid:
            raise ValueError("Solution is not valid. Please run the simulation first.")
        import numpy as np
        sigma = np.array([node.stress for node in self.all_nodes()], dtype=float)
        sigma_max = sigma.max()
        return float(sigma_max + np.log(np.sum(np.exp(rho * (sigma - sigma_max)))) / rho)

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

    def hex_order(self) -> int:
        """Nodes per element, read off the mesh itself (solve_fem needs 8)."""
        for e in self.elements.values():
            return len(e.nodes)
        return 8

    def corner_node_ids(self) -> set:
        """Node ids that are element CORNERS. On a linear mesh that is every
        node; on a 20-node mesh it is exactly the nodes of the linear mesh it
        was built from, the midside nodes being the rest."""
        return {n.id for e in self.elements.values() for n in e.nodes[:8]}

    # corner-node faces of an 8-node hex, in the element's node order
    _HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                  (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))

    def _apply_surface_traction(self, spec):
        """Uniform traction over the free INNER surface of a region.

        spec = {"box": ((x0, x1), (y0, y1), (z0, z1)), "total": (Fx, Fy, Fz)}.
        The region is every element whose centroid lies in the box (e.g. one
        tile); its loaded surface is each element face that is free (belongs
        to no other element) and does not lie on the part's bounding-box
        planes -- for a hollow-cube tile, the void's inner walls. The total
        is spread by area (force per unit area constant), each face passing
        a quarter of its share to each of its 4 corner nodes, which is the
        exact consistent load of a uniform traction on a bilinear face.
        The loaded nodes are NOT tied together (rigid_load = False)."""
        import numpy as np
        if self.hex_order() != 8:
            raise NotImplementedError("surface_load: 8-node hexes only")
        (x0, x1), (y0, y1), (z0, z1) = spec["box"]
        total = np.asarray(spec["total"], dtype=float)
        tol = 1e-6
        allx = np.array([n.coords for n in self.all_nodes()])
        lo, hi = allx.min(axis=0), allx.max(axis=0)
        counts, owner = {}, {}
        for e in self.all_elements():
            ids = [n.id for n in e.nodes][:8]
            c = np.mean([n.coords for n in e.nodes][:8], axis=0)
            inside = x0 <= c[0] <= x1 and y0 <= c[1] <= y1 and z0 <= c[2] <= z1
            for f in self._HEX_FACES:
                key = tuple(sorted(ids[i] for i in f))
                counts[key] = counts.get(key, 0) + 1
                if inside:
                    owner[key] = [ids[i] for i in f]
        faces, areas = [], []
        for key, ordered in owner.items():
            if counts[key] != 1:
                continue
            P = np.array([self.nodes[i].coords for i in ordered])
            on_box = any(np.all(np.abs(P[:, a] - lo[a]) < tol) or np.all(np.abs(P[:, a] - hi[a]) < tol)
                         for a in range(3))
            if on_box:
                continue
            area = 0.5 * (np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
                          + np.linalg.norm(np.cross(P[2] - P[0], P[3] - P[0])))
            faces.append(ordered)
            areas.append(area)
        if not faces:
            raise ValueError(f"surface_load: no free inner face in box {spec['box']}")
        A = float(np.sum(areas))
        for ordered, a in zip(faces, areas):
            share = total * (a / A) / 4.0
            for nid in ordered:
                f = self.nodes[nid].forces
                self.nodes[nid].forces = [f[j] + share[j] for j in range(3)]
        self.rigid_load = False
        self.surface_load_area = A

    def _apply_node_load(self, spec):
        """Distribute a total force over every node of a region, weighted by
        distance from a centre point.

        spec = {"box": ((x0, x1), (y0, y1), (z0, z1)), "total": (Fx, Fy, Fz),
                "center": (cx, cy, cz), "axes": (1, 2)}.
        The region's nodes are all nodes of the elements whose centroid lies
        in the box (interior and faces). Each node gets the weight
        w = 1 - r / r_max, r = its distance from `center` measured over
        `axes` only (default y and z, the part's plane), r_max = the
        farthest such node: largest at the centre, linearly down to zero at
        the farthest node. Per node, not per volume, so it follows the
        mesh's node density. No rigid tie (rigid_load = False)."""
        import numpy as np
        (x0, x1), (y0, y1), (z0, z1) = spec["box"]
        total = np.asarray(spec["total"], dtype=float)
        axes = list(spec.get("axes", (1, 2)))
        ids = set()
        for e in self.all_elements():
            c = np.mean([n.coords for n in e.nodes], axis=0)
            if x0 <= c[0] <= x1 and y0 <= c[1] <= y1 and z0 <= c[2] <= z1:
                ids.update(n.id for n in e.nodes)
        if not ids:
            raise ValueError(f"node_load: no element in box {spec['box']}")
        ids = sorted(ids)
        P = np.array([self.nodes[i].coords for i in ids])
        d = P[:, axes] - np.asarray(spec["center"], dtype=float)[axes]
        if spec.get("weight") == "pyramid":
            # 1 - max_i |d_i| / half_i: zero on the whole region boundary,
            # not just at its farthest node (FEA_LOAD_WEIGHT=pyramid)
            w = np.clip(1.0 - np.max(np.abs(d) / np.asarray(spec["half"], dtype=float), axis=1), 0.0, None)
        else:
            r = np.linalg.norm(d, axis=1)
            w = 1.0 - r / r.max()
        w /= w.sum()
        for nid, wi in zip(ids, w):
            f = self.nodes[nid].forces
            self.nodes[nid].forces = [f[j] + wi * total[j] for j in range(3)]
        self.rigid_load = False
        self.node_load_count = len(ids)

    def apply_force_by_pattern(self, force_pattern: Callable[[Node, float], List]):
        # A geometry's force_pattern may instead carry a distributed surface
        # load (FEA_LOAD_MODE=void_traction on hollow_cube_11x11): see
        # _apply_surface_traction. The resultant is then exact by construction.
        surface = getattr(force_pattern, "surface_load", None)
        if surface is not None:
            self.clear_forces()
            self._apply_surface_traction(surface)
            return
        # ...or a load spread over a region's nodes by distance from its
        # centre (FEA_LOAD_MODE=tile_nodes on cross4): see _apply_node_load.
        node_load = getattr(force_pattern, "node_load", None)
        if node_load is not None:
            self.clear_forces()
            self._apply_node_load(node_load)
            return
        for node in self.all_nodes():
            force = force_pattern(node, self.tolerance)
            self.apply_force_on_node(node.id, force)

        # The force_pattern of every geometry here spreads a FIXED resultant
        # over the loaded face by dividing it by a node count derived from
        # mesh_resolution() -- so adding midside nodes to that face would
        # silently multiply the total applied load by ~3x and compare a
        # different physical problem. Rescale back onto the resultant the
        # corner nodes (i.e. the linear mesh) carry. The forces themselves
        # are then only a distribution: solve_fem ties every loaded node
        # into one rigid body and applies the resultant at its pilot, so what matters is which nodes are loaded (all of the face's,
        # midside included, which is why the pattern is applied to them) and
        # what the total is. Same renormalisation fd_reference._build_mesh
        # does for a mesh-refinement study, for the same reason.
        if self.hex_order() == 20:
            corners = self.corner_node_ids()
            for j in range(3):
                have = sum(n.forces[j] for n in self.all_nodes())
                want = sum(n.forces[j] for n in self.all_nodes() if n.id in corners)
                if have and want:
                    k = want / have
                    for n in self.all_nodes():
                        n.forces[j] *= k

        # Opt-in exact resultant: a force_pattern carrying a `resultant`
        # attribute (Fx, Fy, Fz) is rescaled to it, so the total no longer
        # depends on the node count the pattern assumes for the loaded face
        # (cross4's padded load face is three boxes, not one n x n patch).
        target = getattr(force_pattern, "resultant", None)
        if target is not None:
            for j in range(3):
                have = sum(n.forces[j] for n in self.all_nodes())
                if target[j] and not have:
                    raise ValueError("force_pattern.resultant set but no node was loaded")
                if have:
                    k = target[j] / have
                    for n in self.all_nodes():
                        n.forces[j] *= k

    def clear_forces(self):
        for node in self.all_nodes():
            node.forces = [0.0, 0.0, 0.0]


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
            # nodes[:8] -- the corners. A 20-node element is drawn as the
            # straight-edged hexahedron it geometrically is; VTK's quadratic
            # cell would render identically and cost 2.5x the points.
            local_ids = [id_map[n.id] for n in elem.nodes[:8]]
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



    def plot_stress_heatmap(self, node_stress: dict, save_path=None, resolution=(3840, 2160),
                             aa_type="msaa", banner: str = None, cmap="jet",
                             clim=None, title="Von Mises Stress (MPa)", marks=None):
        """Render the mesh colored by per-node stress, matching Ansys's nodal
        equivalent-stress plot style (same camera as plot_mesh).

        node_stress: dict mapping mesh node id -> scalar stress value.
        Missing node ids default to 0.0.
        marks: optional list of (node_id, label, colour) drawn as labelled
        dots on top of the mesh (always visible), e.g. the max-stress node.
        """
        import numpy as np
        import pyvista as pv
        from pyvista import CellType

        node_items = sorted(self.nodes.items())
        id_map = {nid: i for i, (nid, _) in enumerate(node_items)}
        points = np.array([node.coords for _, node in node_items], dtype=float)
        scalars = np.array([node_stress.get(nid, 0.0) for nid, _ in node_items], dtype=float)

        cells = []
        cell_types = []
        for elem in self.elements.values():
            # nodes[:8] -- the corners. A 20-node element is drawn as the
            # straight-edged hexahedron it geometrically is; VTK's quadratic
            # cell would render identically and cost 2.5x the points.
            local_ids = [id_map[n.id] for n in elem.nodes[:8]]
            cells.append(len(local_ids))
            cells.extend(local_ids)
            cell_types.append(CellType.HEXAHEDRON)

        grid = pv.UnstructuredGrid(np.array(cells), np.array(cell_types), points)
        grid.point_data[title] = scalars

        centre = grid.center
        grid.rotate_z(270, point=centre, inplace=True)

        off_screen = save_path is not None
        plotter = pv.Plotter(off_screen=off_screen, window_size=resolution)
        plotter.enable_anti_aliasing(aa_type)

        plotter.add_mesh(
            grid,
            scalars=title,
            cmap=cmap,
            show_edges=True,
            clim=clim,
            scalar_bar_args={"title": title, "vertical": True},
        )

        # marks: a sphere on BOTH depth faces (the node may sit mid-depth,
        # hidden inside the material; one face is the one the camera sees)
        # and its label lifted a little so it does not cover the sphere
        x_lo, x_hi = float(points[:, 0].min()), float(points[:, 0].max())
        span = float(np.ptp(points[:, 1:], axis=0).max()) or 1.0
        for nid, label, colour in (marks or []):
            x, y, z = self.nodes[nid].coords
            for xf in (x_lo - 0.02 * span, x_hi + 0.02 * span):
                sph = pv.Sphere(radius=0.012 * span, center=(xf, y, z))
                sph.rotate_z(270, point=centre, inplace=True)
                plotter.add_mesh(sph, color=colour)
            lab = pv.PolyData(np.array([[x_hi + 0.02 * span, y + 0.04 * span, z]], dtype=float))
            lab.rotate_z(270, point=centre, inplace=True)
            plotter.add_point_labels(lab, [label], show_points=False, always_visible=True,
                                     font_size=40, text_color=colour, shape_color="white",
                                     shape_opacity=0.8, margin=6)

        if banner is not None:
            plotter.add_text(banner, position="upper_left", font_size=20, color="black")

        plotter.view_xz()
        plotter.camera.up = (1.0, 0.0, 0.0)
        plotter.enable_parallel_projection()

        if save_path is not None:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show()
