def str_to_dict(s):
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

def face_nodes_by_axis(nodes, axis, sign, tol=1e-9):
    from utils.models import Node
    idx = {"X":0, "Y":1, "Z":2}[axis]
    coords = [node.coords[idx] for node in nodes]
    target = (max(coords) if sign == "+" else min(coords))
    face = [node for node in nodes if abs(node.coords[idx] - target) <= tol]
    return face

def unit_vector(dir_code):
    return {
        "+X": (1,0,0), "-X": (-1,0,0),
        "+Y": (0,1,0), "-Y": (0,-1,0),
        "+Z": (0,0,1), "-Z": (0,0,-1),
    }[dir_code]

