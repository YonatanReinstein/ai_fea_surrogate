import argparse
from ansys.mapdl.core import launch_mapdl
from utils.read_inp import read_inp
from utils.models import Mesh
from utils.arg_processing import str_to_dict , face_nodes_by_axis, unit_vector


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
#nodes_xyz, elems = read_inp(args.inp)
#
#mesh = Mesh.from_inp(args.inp)
#
## Launch MAPDL
#mapdl = launch_mapdl(mode="grpc", override=True, cleanup_on_exit=True)
#mapdl.clear()
#mapdl.prep7()
#mapdl.et(1, 185)                 # SOLID185
#mapdl.keyopt(1, 9, 0)            # (default integration)
#mapdl.mp("EX", 1, args.young)
#mapdl.mp("PRXY", 1, args.poisson)
#
## Create nodes
#for nid, (x,y,z) in nodes_xyz.items():
#    mapdl.n(nid, x, y, z)
#
## Create elements, set type+mat
#mapdl.type(1)
#mapdl.mat(1)
#for eid, nn in elems.items():
#    mapdl.en(eid, *nn)
#
## Apply anchors
#if args.anchor:
#    for a in args.anchor:
#        spec = str_to_dict(a)
#        eid = int(spec["cube"])  # element ID provided by user
#        face = spec["face"].upper()
#
#        axis, sign = face[-1], face[0]  # e.g. "+Z"
#
#        if eid not in elems:
#            raise ValueError(f"Element {eid} not found in mesh")
#
#        nlist = elems[eid]
#        face_n = face_nodes_by_axis(nodes_xyz, nlist, axis, sign)
#
#        print(f"Anchoring element {eid} on face {face} (nodes: {face_n})")
#
#        mapdl.allsel("ALL")
#        mapdl.nsel("NONE")
#        for nid in face_n:
#            mapdl.nsel("A", "NODE", vmin=nid, vmax=nid)
#        mapdl.d("ALL", "UX", 0)
#        mapdl.d("ALL", "UY", 0)
#        mapdl.d("ALL", "UZ", 0)
#
#
## Apply forces
#if args.force:
#    for f in args.force:
#        print(f"Applying force: {f}")
#        spec = str_to_dict(f)
#        eid = int(spec["cube"])
#        face = spec["face"].upper()
#        value = float(spec["value"])
#        dir_code = str(spec.get("dir", face)).upper()
#
#        if eid not in elems:
#            raise ValueError(f"Element {eid} not found in mesh")
#
#        axis, sign = face[-1], face[0]
#        nlist = elems[eid]
#        face_n = face_nodes_by_axis(nodes_xyz, nlist, axis, sign)
#        if not face_n:
#            raise ValueError(f"No face '{face}' found on element {eid}")
#
#        print(f"Applying force on element {eid} face {face} (nodes: {face_n})")
#        mapdl.allsel("ALL")
#        mapdl.nsel("NONE")
#        for nid in face_n:
#            mapdl.nsel("A", "NODE", vmin=nid, vmax=nid)
#
#        axis = dir_code[-1]  # 'X', 'Y', or 'Z'
#        sign = dir_code[0]   # '+' or '-'
#        comp = f"F{axis}"
#        sgn = 1 if sign == "+" else -1
#        mapdl.f("ALL", comp, sgn * value)
#
#
## Mesh is already the element set we created, so go solve
#mapdl.allsel("ALL")
#mapdl.finish()
#mapdl.run("/SOLU")
#mapdl.antype("STATIC")
#mapdl.outres("ALL","ALL")
#if args.solve:
#    mapdl.solve()
#    mapdl.finish()
#    # --- Postprocessing: compute maximum von Mises stress ---
#    mapdl.post1()
#    mapdl.set("last")
#
#    # Use high-level API to extract nodal von Mises stress
#    stress = mapdl.post_processing.nodal_eqv_stress()
#
#    vmax = stress.max()
#    print(f"Maximum von Mises stress: {vmax:.3e} Pa")
#
#    # Displacements
#    ux = mapdl.post_processing.nodal_displacement("X")
#    uy = mapdl.post_processing.nodal_displacement("Y")
#    uz = mapdl.post_processing.nodal_displacement("Z")
#
#
#    # Update nodes_xyz with displacement and stress
#    for nid in nodes_xyz.keys():
#        nodes_xyz[nid] = {
#            "coords": nodes_xyz[nid],
#            "ux": ux[nid-1],
#            "uy": uy[nid-1],
#            "uz": uz[nid-1],
#            "stress": stress[nid-1]
#        }
#
#


mesh = Mesh.from_inp(args.inp)
for a in args.anchor:
    spec = str_to_dict(a)
    eid = int(spec["cube"])  # element ID provided by user
    face = spec["face"].upper()
    mesh.add_anchor(eid, face)

for f in args.force:
    print(f"Applying force: {f}")
    spec = str_to_dict(f)
    eid = int(spec["cube"])
    face = spec["face"].upper()
    value = float(spec["value"])
    dir_code = str(spec.get("dir", face)).upper()
    mesh.add_force(eid, face, value, dir_code)

if args.solve:
    mesh.solve(args.young, args.poisson)

