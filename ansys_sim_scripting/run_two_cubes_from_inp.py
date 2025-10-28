# run_two_cubes_from_inp.py
# Requires: pip install ansys-mapdl-core
# Launches MAPDL, parses an Abaqus .inp with *NODE/*ELEMENT,TYPE=C3D8,
# creates SOLID185 mesh, and applies anchors/forces per cube & face.

import argparse
from ansys.mapdl.core import launch_mapdl

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--inp", required=True, help="Path to Abaqus .inp (12 nodes, 2×C3D8)")
parser.add_argument("--young", type=float, default=2.1e11, help="Young's modulus [Pa]")
parser.add_argument("--poisson", type=float, default=0.3, help="Poisson ratio [-]")
parser.add_argument("--density", type=float, default=7800, help="Density [kg/m^3] (optional, only if you add inertia)")
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

def read_abaqus_inp(path):
    """Return nodes: {nid:(x,y,z)}, elems: {eid:[n1..n8]} for TYPE=C3D8."""
    nodes = {}
    elems = {}
    with open(path, "r") as f:
        lines = f.readlines()
    i = 0
    # Find *NODE
    while i < len(lines):
        if lines[i].strip().upper().startswith("*NODE"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("*"):
                line = lines[i].strip()
                if line:
                    nid, x, y, z = [p.strip() for p in line.split(",")]
                    nodes[int(nid)] = (float(x), float(y), float(z))
                i += 1
            continue
        if lines[i].strip().upper().startswith("*ELEMENT") and "C3D8" in lines[i].upper():
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("*"):
                line = lines[i].strip()
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    eid = int(parts[0])
                    nn = [int(p) for p in parts[1:]]
                    if len(nn) != 8:
                        raise ValueError(f"Element {eid} does not have 8 nodes.")
                    elems[eid] = nn
                i += 1
            continue
        i += 1
    if not nodes or not elems:
        raise ValueError("Did not find *NODE and/or *ELEMENT,TYPE=C3D8 blocks in the .inp")
    return nodes, elems

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
nodes_xyz, elems = read_abaqus_inp(args.inp)

#print("Nodes:")
#for nid in sorted(nodes_xyz.keys()):
#    print(f"  {nid}: {nodes_xyz[nid]}")
#
#print("Elements:")
#for eid in sorted(elems.keys()):
#    print(f"  {eid}: {elems[eid]}")


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



# Convenience “cube = element id”
cube_elems = {1: [1], 2: [2]}

# Build NSL for each cube (all nodes belonging to that element)
cube_nodes = {}
for cid, elist in cube_elems.items():
    mapdl.allsel("ALL")
    mapdl.esel("S", "ELEM", vmin=elist[0], vmax=elist[0])
    mapdl.nsle("S", 1)
    nodes = mapdl.mesh.nnum
    cube_nodes[cid] = set(nodes)

print(args.anchor)
# Apply anchors
if args.anchor:
    for a in args.anchor:
        #print(f"Applying anchor: {a}")
        spec = parse_kv_list(a)
        #print(spec)
        cube = int(spec["cube"])
        face = spec["face"].upper()
        axis, sign = face[-1], face[0]  # e.g., "+Z"
        # get the element id for this cube
        eid = cube_elems[cube][0]
        face_n = face_nodes_by_axis(nodes_xyz, elems[eid], axis, sign)
        # print(f"  Face nodes: {face_n}")
        # Select just those nodes and fix UX, UY, UZ
        mapdl.allsel("ALL")
        mapdl.nsel("NONE")
        mapdl.nsel("S", "NODE", vmin=face_n[0], vmax=face_n[0])
        for nid in face_n[1:]:
            mapdl.nsel("A", "NODE", vmin=nid, vmax=nid)
        mapdl.d("ALL", "UX", 0)
        mapdl.d("ALL", "UY", 0)
        mapdl.d("ALL", "UZ", 0)
        #print(mapdl.dlist())


# Apply forces
if args.force:
    for f in args.force:
        print(f"Applying force: {f}")
        spec = parse_kv_list(f)
        print(spec)
        cube = int(spec["cube"])
        face = spec["face"].upper()
        ftype = str(spec.get("type", "pressure")).lower()
        value = float(spec["value"])
        #print("hey")
        #print(f"  On cube {cube}, face {face}, type {ftype}, value {value}")
        dir_code = str(spec.get("dir", face)).upper()  # default same normal
        print("Direction code:", dir_code)
        print(dir_code)
        axis, sign = face[-1], face[0]
        eid = cube_elems[cube][0]
        #print(f"  Element ID: {eid}")
        face_n = face_nodes_by_axis(nodes_xyz, elems[eid], axis, sign)
        #print(f"  Face nodes: {face_n}")
        mapdl.allsel("ALL")
        mapdl.nsel("NONE")
        # select exactly the face nodes
        mapdl.nsel("S", "NODE", vmin=face_n[0], vmax=face_n[0])
        #for nid in face_n[1:]:
        #    #print(f"  Selecting node {nid}")
        #    mapdl.nsel("A", "NODE", vmin=nid, vmax=nid)

        if ftype == "nodal":
            ux, uy, uz = unit_vector(dir_code)
            print("Unit vector:", ux, uy, uz)   
            comp = { (1,0,0): "FX", (0,1,0): "FY", (0,0,1): "FZ",
                     (-1,0,0): "FX", (0,-1,0): "FY", (0,0,-1): "FZ"}[(ux,uy,uz)]
            
            print(f"  Applying nodal force component {comp}")
            sgn = 1 if (ux+uy+uz) > 0 else -1
            mapdl.f("ALL", comp, sgn * value) 


        #pressure does not work properly on student version
        elif ftype == "pressure":
            face_map = {("+X"): 3, ("-X"): 5, ("+Y"): 4, ("-Y"): 2, ("+Z"): 6, ("-Z"): 1}
            face_id = face_map[(face)]
            print(f"  Applying pressure on face {face_id}")
            mapdl.allsel("ALL")
            mapdl.esel("S", "ELEM", vmin=eid, vmax=eid)

            mapdl.sfe(eid, face_id, "PRES", float(value))  # pressure in Pa
            print("Applied pressure:", value)
            print(mapdl.shpp("SUMM"))
            print(mapdl.lswrite())
            print(mapdl.sfelist())

        else:
            raise ValueError("Unknown force type. Use 'nodal' or 'pressure'.")

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


    vmax = float(stress.max())
    print(f"Maximum von Mises stress: {vmax:.3e} Pa")

    # Optional plot
    mapdl.post_processing.plot_nodal_eqv_stress()



print("Done.")
