from utils.models import Node, Element

def detect_encoding(path):
    with open(path, "rb") as f:
        start = f.read(4)
        # --- Check for BOMs ---
        if start.startswith(b'\xff\xfe'):
            return "utf-16-le"
        elif start.startswith(b'\xfe\xff'):
            return "utf-16-be"
        elif start.startswith(b'\xef\xbb\xbf'):
            return "utf-8-sig"
        # --- No BOM: heuristic ---
        elif b'\x00' in start:  # null bytes → probably UTF-16 without BOM
            return "utf-16"
        else:
            return "utf-8"

def read_inp(path):
    """
    Parse an IRIT2INP-generated .inp file and return:
        nodes: {nid: (x, y, z)}
        elems: {eid: [n1..n8]}   for TYPE=C3D8 elements
    """
    nodes = {}
    elems = {}

    encoding = detect_encoding(path)
    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # --- NODE section ---
        if line.upper().startswith("*NODE"):
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("*"):  # End of NODE section
                    break
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    nid = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    nodes[nid] = (x, y, z)
                i += 1
            continue

        # --- ELEMENT section (C3D8 only) ---
        if line.upper().startswith("*ELEMENT") and "C3D8" in line.upper():
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("*"):
                    break
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 9:
                    eid = int(parts[0])
                    nn = [int(p) for p in parts[1:9]]
                    elems[eid] = nn
                i += 1
            continue

        i += 1

    if not nodes:
        raise ValueError("No *NODE section found in file.")
    if not elems:
        raise ValueError("No *ELEMENT,TYPE=C3D8 section found in file.")

    return nodes, elems


def build_mesh_from_inp(path):
    """Read .inp and return linked Node and Element objects."""
    nodes_xyz, elems_raw = read_inp(path)

    # Create Node objects
    nodes = {nid: Node(nid, xyz) for nid, xyz in nodes_xyz.items()}

    # Create Element objects with pointers to Node objects
    elements = {
        eid: Element(eid, [nodes[nid] for nid in nlist])
        for eid, nlist in elems_raw.items()
    }

    return nodes, elements


