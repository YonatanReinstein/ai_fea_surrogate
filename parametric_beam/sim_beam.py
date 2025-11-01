import argparse
from ansys.mapdl.core import launch_mapdl

#import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
#
#from read_inp import read_inp

from utils.read_inp import read_inp


# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--inp", type=str,default="parametric_beam/beam.inp", help="Path to Abaqus .inp")
parser.add_argument("--young", type=float, default=2.1e11, help="Young's modulus [Pa]")
parser.add_argument("--poisson", type=float, default=0.3, help="Poisson ratio [-]")
parser.add_argument("--solve", action="store_true", help="Solve automatically (default: True)", default=True)

# Repeatable flags:
parser.add_argument("--anchor", action="append",
                    help="Anchor spec: cube=1,face=+Z  (faces: +X,-X,+Y,-Y,+Z,-Z)")
parser.add_argument("--force", action="append",
                    help="Force spec: cube=2,face=+Z,type=pressure,value=1e6 "
                         "or cube=2,face=+Z,type=nodal,value=1000 "
                         "(nodal=N force per node in +Z unless dir=+X/-X/+Y/-Y/+Z/-Z) "
                         "Optional dir=...; for pressure use Pa, distributes over face area.")

args = parser.parse_args()

# ---------- helpers ----------
def parse_kv_list(s):
    """Parse 'k=v, k=v' into dict with ints/floats when possible."""
    out = {}
    for part in s.split(","):
        k, v = [p.strip() for p in part.split("=", 1)]
        if v.lstrip("+-").replace(".","",1).isdigit():
            # int or float
            out[k] = float(v) if "." in v else int(v)
        else:
            out[k] = v
    return out


def face_nodes_by_axis(nodes_xyz, elem_node_ids, axis, sign, tol=1e-9):
    """Return node IDs on +/- extreme face along axis ('X','Y','Z') for one element."""
    idx = {"X":0, "Y":1, "Z":2}[axis]
    coords = [nodes_xyz[n][idx] for n in elem_node_ids]
    target = (max(coords) if sign == "+" else min(coords))
    face = [n for n in elem_node_ids if abs(nodes_xyz[n][idx] - target) <= tol]
    # Expect 4 nodes for a hexa face:
    if len(face) != 4:
        # Relax tolerance a bit if needed
        tol2 = max(1e-9, abs(target) * 1e-9)
        face = [n for n in elem_node_ids if abs(nodes_xyz[n][idx] - target) <= tol2]
    return face

def unit_vector(dir_code):
    return {
        "+X": (1,0,0), "-X": (-1,0,0),
        "+Y": (0,1,0), "-Y": (0,-1,0),
        "+Z": (0,0,1), "-Z": (0,0,-1),
    }[dir_code]

# ---------- main ----------
nodes_xyz, elems = read_inp(args.inp)

# Launch MAPDL
mapdl = launch_mapdl(mode="grpc", override=True, cleanup_on_exit=True)
mapdl.clear()
mapdl.prep7()
mapdl.et(1, 185)                 # SOLID185
mapdl.keyopt(1, 9, 0)            # (default integration)
mapdl.mp("EX", 1, args.young)
mapdl.mp("PRXY", 1, args.poisson)

# Create nodes
for nid, (x,y,z) in nodes_xyz.items():
    mapdl.n(nid, x, y, z)

# Create elements, set type+mat
mapdl.type(1)
mapdl.mat(1)
for eid, nn in elems.items():
    mapdl.en(eid, *nn)

# Apply anchors
if args.anchor:
    for a in args.anchor:
        spec = parse_kv_list(a)
        eid = int(spec["cube"])  # element ID provided by user
        face = spec["face"].upper()
        axis, sign = face[-1], face[0]  # e.g. "+Z"

        if eid not in elems:
            raise ValueError(f"Element {eid} not found in mesh")

        nlist = elems[eid]
        face_n = face_nodes_by_axis(nodes_xyz, nlist, axis, sign)

        print(f"Anchoring element {eid} on face {face} (nodes: {face_n})")

        mapdl.allsel("ALL")
        mapdl.nsel("NONE")
        for nid in face_n:
            mapdl.nsel("A", "NODE", vmin=nid, vmax=nid)
        mapdl.d("ALL", "UX", 0)
        mapdl.d("ALL", "UY", 0)
        mapdl.d("ALL", "UZ", 0)


# Apply forces
if args.force:
    for f in args.force:
        print(f"Applying force: {f}")
        spec = parse_kv_list(f)
        eid = int(spec["cube"])  # element ID to target
        face = spec["face"].upper()
        value = float(spec["value"])
        dir_code = str(spec.get("dir", face)).upper()
        axis, sign = face[-1], face[0]

        if eid not in elems:
            raise ValueError(f"Element {eid} not found in mesh")

        nlist = elems[eid]
        face_n = face_nodes_by_axis(nodes_xyz, nlist, axis, sign)
        if not face_n:
            raise ValueError(f"No face '{face}' found on element {eid}")

        print(f"Applying force on element {eid} face {face} (nodes: {face_n})")
        mapdl.allsel("ALL")
        mapdl.nsel("NONE")
        for nid in face_n:
            mapdl.nsel("A", "NODE", vmin=nid, vmax=nid)

        ux, uy, uz = unit_vector(dir_code)
        comp = {
            (1, 0, 0): "FX", (0, 1, 0): "FY", (0, 0, 1): "FZ",
            (-1, 0, 0): "FX", (0, -1, 0): "FY", (0, 0, -1): "FZ",
        }[(ux, uy, uz)]
        sgn = 1 if (ux + uy + uz) > 0 else -1
        mapdl.f("ALL", comp, sgn * value)


# Mesh is already the element set we created, so go solve
mapdl.allsel("ALL")
mapdl.finish()
mapdl.run("/SOLU")
mapdl.antype("STATIC")
mapdl.outres("ALL","ALL")
if args.solve:
    mapdl.solve()
    mapdl.finish()
    # --- Postprocessing: compute maximum von Mises stress ---
    mapdl.post1()
    mapdl.set("last")

    # Use high-level API to extract nodal von Mises stress
    stress = mapdl.post_processing.nodal_eqv_stress()

    vmax = stress.max()
    print(f"Maximum von Mises stress: {vmax:.3e} Pa")

    # Displacements
    ux = mapdl.post_processing.nodal_displacement("X")
    uy = mapdl.post_processing.nodal_displacement("Y")
    uz = mapdl.post_processing.nodal_displacement("Z")


    # Update nodes_xyz with displacement and stress
    for nid in nodes_xyz.keys():
        nodes_xyz[nid] = {
            "coords": nodes_xyz[nid],
            "ux": ux[nid-1],
            "uy": uy[nid-1],
            "uz": uz[nid-1],
            "stress": stress[nid-1]
        }
    print(nodes_xyz)
