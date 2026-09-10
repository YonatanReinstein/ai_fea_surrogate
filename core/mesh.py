from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor
from .node import Node
from .element import Element
from utils.hex20 import is_quadratic, quadratic_element_type
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
            

    def _ccx_base_deck(self, young: float, poisson: float,
                        element_type: str = "C3D8I"):
        """Model-definition lines shared by solve_ccx() and
        solve_ccx_sensitivity(): nodes, elements, material, boundary
        conditions, and the rigid-body force coupling. Does not include any
        *STEP -- callers append their own step(s) after this (and, for
        solve_ccx_sensitivity, a *DESIGNVARIABLES card, which must precede
        the first *STEP).

        Returns (lines, pilot_id, total_force): pilot_id is the *RIGID BODY
        reference node holding the resultant force (None if no forces are
        applied), for use in a *CLOAD line in the caller's *STATIC step.

        element_type overrides the hex element (see the C3D8I note below).
        Only solve_ccx_axis_gradient uses anything but the default: ccx's
        semi-analytical shape sensitivity differentiates C3D8I's statically
        condensed incompatible modes badly, so that path asks for C3D8.
        """
        # ccx's card reader parses each comma-separated entry out of a
        # fixed 20-character field, so a coordinate must never be written
        # wider than that. repr() of a double runs to 17 significant digits
        # (e.g. -8.164970000000001e-05, 22 chars) and overflows it with a
        # bare "*ERROR reading *NODE" -- fine for a nominal mesh whose
        # coordinates happen to be short decimals, fatal for anything
        # perturbed (node-level shape FD, remeshed designs). %.13g keeps
        # ~1e-13 relative precision, far below any mesh-motion step used
        # here, while fitting in 19 characters worst case.
        lines = ["*NODE"]
        lines += [f"{n.id}, {n.coords[0]:.13g}, {n.coords[1]:.13g}, {n.coords[2]:.13g}"
                  for n in self.all_nodes()]

        # C3D8I (incompatible modes), not plain C3D8: measured against MAPDL's
        # SOLID185 (full integration + incompatible modes, ANSYS's default) on
        # hollow_cube, plain C3D8 under-predicted max stress by 6-9% and max
        # displacement by ~4% from shear locking. C3D8I closes stress to ~1-2%.
        # On a second-order mesh (utils/hex20.py) the C3D8/C3D8I choice the
        # caller made does not exist: there are no incompatible modes to add
        # for the value solve or to differentiate for the gradient, so both
        # callers land on the same element and the deck says so rather than
        # writing an 8-node type over 20-node connectivity.
        if self.hex_order() == 20 and not is_quadratic(element_type):
            element_type = quadratic_element_type()
        lines.append(f"*ELEMENT, TYPE={element_type}, ELSET=EALL")
        # ccx's fixed-field line parser rejects more than 16 entries per data
        # line (splitline: "there should not be more than 16 entries in a
        # line"), and a C3D20 needs 21 -- the element id plus 20 nodes. Abaqus
        # and CalculiX both continue such an element onto a second line after
        # the 15th node; an 8-node element is unaffected and stays on one.
        for e in self.all_elements():
            ids = [str(n.id) for n in e.nodes]
            lines.append(f"{e.id}, " + ", ".join(ids[:15]))
            if len(ids) > 15:
                lines.append(", ".join(ids[15:]))

        lines += [
            "*MATERIAL, NAME=MAT1",
            "*ELASTIC",
            f"{young!r}, {poisson!r}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1",
        ]

        anchored = [n for n in self.all_nodes() if n.anchored]
        if anchored:
            lines.append("*BOUNDARY")
            lines += [f"{n.id}, 1, 3" for n in anchored]

        force_nodes = [n for n in self.all_nodes() if any(n.forces)]
        pilot_id = None
        total_force = (0.0, 0.0, 0.0)
        if force_nodes:
            total_force = (
                sum(n.forces[0] for n in force_nodes),
                sum(n.forces[1] for n in force_nodes),
                sum(n.forces[2] for n in force_nodes),
            )
            cx = sum(n.coords[0] for n in force_nodes) / len(force_nodes)
            cy = sum(n.coords[1] for n in force_nodes) / len(force_nodes)
            cz = sum(n.coords[2] for n in force_nodes) / len(force_nodes)
            pilot_id = max(self.nodes.keys()) + 1
            rot_id = pilot_id + 1

            # ccx's fixed-field line parser rejects more than 16 entries per
            # data line, so a large force-node set must be wrapped.
            force_ids = [n.id for n in force_nodes]
            force_lines = [
                ", ".join(str(nid) for nid in force_ids[i:i + 16])
                for i in range(0, len(force_ids), 16)
            ]
            lines += [
                "*NODE",
                f"{pilot_id}, {cx:.13g}, {cy:.13g}, {cz:.13g}",
                f"{rot_id}, {cx:.13g}, {cy:.13g}, {cz:.13g}",
                "*NSET, NSET=FORCESET",
                *force_lines,
                f"*RIGID BODY, NSET=FORCESET, REF NODE={pilot_id}, ROT NODE={rot_id}",
            ]

        return lines, pilot_id, total_force

    def solve_ccx(self, young: float, poisson: float, ccx_path: str = "ccx",
                  nproc: int = 1, work_dir: str = None, keep_files: bool = False):
        """Linear static solve via CalculiX (ccx), as a free/open-source
        alternative to solve()'s MAPDL path. Writes an Abaqus-style .inp deck,
        runs ccx, and parses the .frd result file into the same node
        displacement/stress attributes solve() populates, so
        get_max_stress()/get_max_displacement() work unchanged.

        The rigid-region force distribution MAPDL does with a MASS21 pilot
        node + CERIG (solve()'s ET,2,21 / CERIG,...,UXYZ block) is
        reproduced here with CalculiX's *RIGID BODY: a reference node
        (translation) plus a rotational node give the force nodes a true
        6-DOF rigid coupling (translate + rotate together), with the
        resultant force applied at the reference node. A naive
        *EQUATION tie of translational DOFs only (no rotation) was tried
        first and measured ~4x too stiff against a closed-form cantilever
        check, since it prevents the attached face from rotating at all.
        """
        import os
        import shutil
        import subprocess
        import tempfile
        from utils.frd_parser import parse_frd

        cleanup = False
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="ccx_")
            cleanup = True
        else:
            os.makedirs(work_dir, exist_ok=True)

        job_name = "job"
        inp_path = os.path.join(work_dir, f"{job_name}.inp")

        lines, pilot_id, total_force = self._ccx_base_deck(young, poisson)

        lines += ["*STEP", "*STATIC"]
        if pilot_id is not None:
            lines.append("*CLOAD")
            for dof, val in zip((1, 2, 3), total_force):
                if val != 0.0:
                    lines.append(f"{pilot_id}, {dof}, {val!r}")
        lines += ["*NODE FILE", "U", "*EL FILE", "S", "*END STEP"]

        with open(inp_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(nproc)
        result = subprocess.run(
            [ccx_path, "-i", job_name],
            cwd=work_dir, env=env, capture_output=True, text=True, timeout=1200,
        )

        frd_path = os.path.join(work_dir, f"{job_name}.frd")
        if result.returncode != 0 or not os.path.isfile(frd_path):
            tail = (result.stdout or "")[-4000:] + "\n" + (result.stderr or "")[-2000:]
            if cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(f"CalculiX run failed (rc={result.returncode}):\n{tail}")

        displacement, stress = parse_frd(frd_path)

        for node_id, node in self.nodes.items():
            node.displacement = displacement.get(node_id, [0.0, 0.0, 0.0])
            node.stress = stress.get(node_id, 0.0)
        self.solution_valid = True

        if cleanup and not keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)

    def solve_ccx_sensitivity(self, young: float, poisson: float, design_node_ids,
                               stress_ks_rho_mpa: float = 0.15, ccx_path: str = "ccx",
                               nproc: int = 1, work_dir: str = None, keep_files: bool = False,
                               sensitivity_extra=None, element_type: str = "C3D8I",
                               env_extra: dict = None):
        """Static solve + one CalculiX *SENSITIVITY step for the KS-pooled
        von Mises stress w.r.t. the coordinates of design_node_ids.

        Low-level driver, not a gradient on its own: solve_ccx_axis_gradient
        calls it three times, once per global axis, and that is the method
        callers should be using. On its own (env_extra unset, i.e. stock ccx
        semantics) it returns the derivative along each node's surface NORMAL
        divided by the node's tributary area, which is a shape-gradient
        density over the design surface and cannot represent an interior
        node or a motion along a coordinate axis.

        design_node_ids must be free-surface nodes under those stock
        semantics -- ccx rejects the rest, a surface normal being undefined
        for an interior node. CCX_DESI_INTERIOR=1 (via env_extra) lifts that,
        which is why the axis route offers every node.

        The KS response CalculiX computes is
            sigma_ref * [ max(s/sigma_ref) + (1/rho) * ln(sum(exp(rho*(s/sigma_ref - max(s/sigma_ref))))) ]
        (s = nodal von Mises stress, rho, sigma_ref from *OBJECTIVE's STRESS
        line). Passing sigma_ref=1 here makes this numerically identical to
        Mesh.get_aggregated_stress(rho=stress_ks_rho_mpa * 1e-6) -- same
        metric, so the returned ks_value/sensitivities are directly
        comparable to that method's output and to the training/topopt KS
        target elsewhere in this codebase.

        Returns (ks_value, dfdn, normal): dfdn is node_id -> d(ks_value)/d
        (normal displacement); normal is node_id -> unit outward-normal
        (nx, ny, nz), CCX's own mean-of-adjacent-face-normals at that node.
        Both dicts are zero/absent for nodes CCX excluded from the design
        set (see getdesiinfo3d.f: SPC'd nodes, MPC'd nodes, and nodes that
        don't belong to any element face where design nodes are a majority
        of the face's corners -- confirmed by direct reproduction against
        ccx 2.17's own WarnNodeDesignReject.nam on hollow_cube).

        What dfdn MEANS depends on env_extra: with CCX_DESI_AXIS=j it is
        d(ks_value)/d(x_j) at that node and `normal` is irrelevant to it;
        with nothing set it is the stock per-area normal derivative
        described above.
        """
        import os
        import shutil
        import subprocess
        import tempfile
        from utils.frd_parser import parse_frd_sensitivity, parse_frd_normals

        cleanup = False
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="ccx_sens_")
            cleanup = True
        else:
            os.makedirs(work_dir, exist_ok=True)

        job_name = "job"
        inp_path = os.path.join(work_dir, f"{job_name}.inp")

        lines, pilot_id, total_force = self._ccx_base_deck(
            young, poisson, element_type=element_type)

        max_node_id = max(self.nodes.keys())
        # ccx's fixed-field line parser rejects more than 16 entries per
        # data line, so a large design-node set must be wrapped.
        design_ids = list(design_node_ids)
        design_lines = [
            ", ".join(str(nid) for nid in design_ids[i:i + 16])
            for i in range(0, len(design_ids), 16)
        ]
        lines += [
            "*NSET, NSET=NALL, GENERATE",
            f"1, {max_node_id}",
            "*NSET, NSET=DESIGNNODES",
            *design_lines,
            "*DESIGNVARIABLES, TYPE=COORDINATE",
            "DESIGNNODES",
        ]

        lines += ["*STEP", "*STATIC"]
        if pilot_id is not None:
            lines.append("*CLOAD")
            for dof, val in zip((1, 2, 3), total_force):
                if val != 0.0:
                    lines.append(f"{pilot_id}, {dof}, {val!r}")
        lines += ["*NODE FILE", "U", "*EL FILE", "S", "*END STEP"]

        # ccx's KS operates on stress normalized by sigma_ref, and rejects
        # rho < 1 -- so rho_pa (get_aggregated_stress's raw-Pa convention)
        # can't be passed directly. Pick the smallest allowed rho_ccx=1 and
        # solve sigma_ref = rho_ccx / rho_pa so that rho_ccx*(s/sigma_ref)
        # == rho_pa*s for every stress s: this makes ccx's normalized KS
        # equal (1/sigma_ref) * get_aggregated_stress(rho_pa)'s raw-Pa KS,
        # exactly and for any valid rho_ccx>=1. Rescale the outputs back by
        # sigma_ref below to undo the normalization.
        rho_pa = stress_ks_rho_mpa * 1e-6
        rho_ccx = 1.0
        sigma_ref = rho_ccx / rho_pa
        # sensitivity_extra: extra cards inserted right after *SENSITIVITY, e.g.
        # ["*FILTER, EDGEPRESERVATION=YES"]. Per ccx 2.17's normalsonsurface_se.f,
        # EDGEPRESERVATION switches the design-node normal from "mean of ALL
        # external faces at this node" to "mean of only the faces inside the
        # design set" -- which makes the design-node SET steer the reported NORM.
        lines += [
            "*STEP",
            "*SENSITIVITY",
            *(sensitivity_extra or []),
            "*OBJECTIVE",
            f"STRESS, NALL, {rho_ccx!r}, {sigma_ref!r}",
            "*NODE FILE",
            "SEN",
            "*END STEP",
        ]

        with open(inp_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(nproc)
        # env_extra reaches the ccx process only, never os.environ, so
        # concurrent solves in the same interpreter can't clobber each
        # other's CCX_DESI_* settings (solve_ccx_axis_gradient runs three).
        if env_extra:
            env.update({k: str(v) for k, v in env_extra.items()})
        # Sensitivity analysis runs several extra adjoint-style solves beyond
        # a plain static step (internal/external force sensitivity, then
        # stress w.r.t. coordinates and w.r.t. displacements) -- measured
        # to exceed solve_ccx's 1200s budget well before finishing on a
        # full hollow_cube mesh (386k DOF, 40320 design nodes).
        result = subprocess.run(
            [ccx_path, "-i", job_name],
            cwd=work_dir, env=env, capture_output=True, text=True, timeout=7200,
        )

        frd_path = os.path.join(work_dir, f"{job_name}.frd")
        dat_path = os.path.join(work_dir, f"{job_name}.dat")
        if result.returncode != 0 or not os.path.isfile(frd_path):
            tail = (result.stdout or "")[-4000:] + "\n" + (result.stderr or "")[-2000:]
            if cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(f"CalculiX run failed (rc={result.returncode}):\n{tail}")

        ks_value = None
        with open(dat_path, "r", errors="ignore") as f:
            dat_lines = f.readlines()
        for i, line in enumerate(dat_lines):
            if "OBJECTIVE" in line.upper() and "STRESS" in line.upper():
                for j in range(i + 1, min(i + 4, len(dat_lines))):
                    stripped = dat_lines[j].strip()
                    if stripped:
                        ks_value = float(stripped) * sigma_ref
                        break
                break

        dfdn = {nid: v * sigma_ref for nid, v in parse_frd_sensitivity(frd_path).items()}
        normal = parse_frd_normals(frd_path)

        if cleanup and not keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)

        return ks_value, dfdn, normal

    def solve_ccx_axis_gradient(self, young: float, poisson: float, design_node_ids,
                                 stress_ks_rho_mpa: float = 0.15,
                                 ccx_path: str = "ccx_axis", nproc: int = 1,
                                 interior: bool = True, element_type: str = "C3D8",
                                 desi_step: float = None, work_dir: str = None,
                                 parallel: bool = True):
        """Full discrete gradient d(ks_stress)/d(node coordinate), along the
        GLOBAL x/y/z axes, in three solves total regardless of tile count.

        Requires the patched CalculiX built by tools/build_ccx_axis.sh (see
        that script and evaluators/validate_axis_gradient.py). The patched
        binary is bit-identical to stock ccx with none of its environment
        variables set; this method sets three of them:

            CCX_DESI_AXIS=1|2|3  replace every design direction by a global
                                 axis instead of the node's surface normal
            CCX_NO_AREA_NORM=1   skip distributesens.f's division by the
                                 nodal tributary area, so the reported DFDN
                                 is d(objective)/d(nodal displacement) and
                                 not a per-unit-area density
            CCX_DESI_INTERIOR=1  keep interior nodes as design variables

        Why the patch is needed at all: stock ccx reports one scalar per
        design node, the derivative along that node's mean-of-adjacent-face-
        normals. At an edge or corner that direction has no relation to the
        tile parameter being differentiated, and no design-node set can
        steer it onto a coordinate axis. Measured per tile against remesh
        finite differences on hollow_cube_mid (48 tiles, mesh 5,5,5, C3D8),
        ratio to FD over the 26 tiles above the FD noise floor:

            stock, normal only      : median 1.007  0.084..2.645  spread 31.4x
            axis, free surface only : median 1.036  0.634..1.530  spread 2.41x
            axis + interior=True    : median 1.000  0.981..1.094  spread 1.12x

        interior=True is not optional in practice. Stock ccx only accepts
        free-surface nodes because a shape sensitivity is undefined without a
        normal; under CCX_DESI_AXIS there is no normal, so the restriction is
        lifted -- and it is the largest single error term. On hollow_cube_one
        at mesh 5,5,5 the interior nodes carry 16.6% of the surface gradient
        magnitude but contribute -9.0% of the tile derivative (opposite sign,
        cancelling part of the surface sum), moving the contraction from
        1.0755 to 0.9858 of the remesh FD. It also removes the pipeline's old
        mesh-refinement fragility: surface-only drifts 1.08 -> 1.55 from mesh
        5,5,5 to 13,13,13 while with interior nodes it holds at 0.995+-0.005.

        Accuracy is governed by elements-per-wall, NOT by tile count: at one
        element through a wall the gradient is wrong by ~5x. Give the geometry
        enough mesh_resolution that its thinnest feature spans >= 5 elements.

        Returns (ks_value, grad): grad is node_id -> [d/dx, d/dy, d/dz] in the
        same units as get_aggregated_stress (Pa per unit length). A node CCX
        rejected from the design set is exactly [0, 0, 0] -- callers that need
        to tell "rejected" from "genuinely zero" should compare against the
        design_node_ids they asked for.
        """
        import os

        def _one(j):
            env_extra = {"CCX_DESI_AXIS": j + 1, "CCX_NO_AREA_NORM": 1}
            if interior:
                env_extra["CCX_DESI_INTERIOR"] = 1
            if desi_step:
                env_extra["CCX_DESI_STEP"] = repr(desi_step)
            # Each axis gets its own scratch directory: with a caller-
            # supplied work_dir the three would otherwise write the same
            # .inp/.frd/.dat names. work_dir=None already mkdtemps per call.
            wd = os.path.join(work_dir, f"axis{j + 1}") if work_dir else None
            return self.solve_ccx_sensitivity(
                young, poisson, design_node_ids,
                stress_ks_rho_mpa=stress_ks_rho_mpa, ccx_path=ccx_path,
                nproc=nproc, work_dir=wd, element_type=element_type,
                env_extra=env_extra,
            )

        # The three axes are independent solves, so run them at once: this
        # is the whole cost of the gradient, and *SENSITIVITY over a large
        # design set runs ~14x a plain solve, so serialising them is what
        # made the route lose to the per-tile FD sweep on wall-clock even
        # while doing 200x fewer solves. Threads, not processes: each one
        # only waits on a ccx subprocess. env_extra is passed to that
        # subprocess's environment rather than set in os.environ precisely
        # so the three cannot clobber each other's CCX_DESI_AXIS.
        grad, ks_value = {}, None
        if parallel:
            with ThreadPoolExecutor(max_workers=3) as ex:
                results = list(ex.map(_one, range(3)))
        else:
            results = [_one(j) for j in range(3)]
        for j, (ks_value, dfdn, _) in enumerate(results):
            for nid, v in dfdn.items():
                grad.setdefault(nid, [0.0, 0.0, 0.0])[j] = v
        return ks_value, grad

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
        """Nodes per element -- 8 for a C3D8/C3D8I mesh, 20 for a C3D20[R]
        one (utils/hex20.py). Read off the mesh itself, so nothing downstream
        has to be told which kind it was handed."""
        for e in self.elements.values():
            return len(e.nodes)
        return 8

    def corner_node_ids(self) -> set:
        """Node ids that are element CORNERS. On a linear mesh that is every
        node; on a 20-node mesh it is exactly the nodes of the linear mesh it
        was built from, the midside nodes being the rest."""
        return {n.id for e in self.elements.values() for n in e.nodes[:8]}

    def apply_force_by_pattern(self, force_pattern: Callable[[Node, float], List]):
        for node in self.all_nodes():
            force = force_pattern(node, self.tolerance)
            self.apply_force_on_node(node.id, force)

        # The force_pattern of every geometry here spreads a FIXED resultant
        # over the loaded face by dividing it by a node count derived from
        # mesh_resolution() -- so adding midside nodes to that face would
        # silently multiply the total applied load by ~3x and compare a
        # different physical problem. Rescale back onto the resultant the
        # corner nodes (i.e. the linear mesh) carry. The forces themselves
        # are then only a distribution: _ccx_base_deck ties every loaded node
        # into one *RIGID BODY and applies the resultant at its reference
        # node, so what matters is which nodes are loaded (all of the face's,
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
                             clim=None, title="Von Mises Stress (MPa)"):
        """Render the mesh colored by per-node stress, matching Ansys's nodal
        equivalent-stress plot style (same camera as plot_mesh).

        node_stress: dict mapping mesh node id -> scalar stress value.
        Missing node ids default to 0.0.
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

        grid.rotate_z(270, point=grid.center, inplace=True)

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
            # nodes[:8] -- the corners. A 20-node element is drawn as the
            # straight-edged hexahedron it geometrically is; VTK's quadratic
            # cell would render identically and cost 2.5x the points.
            local_ids = [id_map[n.id] for n in elem.nodes[:8]]
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

    



