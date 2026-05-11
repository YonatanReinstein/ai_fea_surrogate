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
    nodes = {}
    elems = {}
    elem_to_tile = {}  # elem_id -> 0-based tile index

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
                if not line or line.startswith("*"):
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

        # --- ELSET sections (SET_k, GENERATE) — one per tile ---
        # Format: *ELSET, ELSET=SET_k, GENERATE
        #         start, end[, step]
        if line.upper().startswith("*ELSET") and "GENERATE" in line.upper():
            import re
            m = re.search(r'SET_(\d+)', line, re.IGNORECASE)
            if m:
                tile_idx = int(m.group(1)) - 1  # 0-based
                i += 1
                if i < len(lines):
                    parts = [p.strip() for p in lines[i].strip().split(",")]
                    start = int(parts[0])
                    end   = int(parts[1])
                    step  = int(parts[2]) if len(parts) > 2 else 1
                    for eid in range(start, end + 1, step):
                        elem_to_tile[eid] = tile_idx
            i += 1
            continue

        i += 1

    if not nodes:
        raise ValueError("No *NODE section found in file.")
    if not elems:
        raise ValueError("No *ELEMENT,TYPE=C3D8 section found in file.")
    return nodes, elems, elem_to_tile





