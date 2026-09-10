"""Minimal parser for CalculiX ASCII .frd result files.

Only extracts the two result blocks needed by the CalculiX evaluator:
nodal displacement (DISP) and nodal-averaged stress tensor (STRESS).
Values are extracted with a fixed-width float regex rather than
whitespace-splitting, since ccx's Fortran-style E12.5 fields can run
together with no separator when a value is negative.
"""
import re

_FLOAT_RE = re.compile(r'[+-]?\d\.\d+E[+-]\d+')
_DATA_LINE_RE = re.compile(r'^\s*-1\s*(\d+)(.*)$')


def _parse_block(lines, start_idx):
    """Parse a ' -4 ... -3' result block starting at the '-4' line.

    Returns (values_by_node_id, index_after_block).
    """
    values = {}
    i = start_idx + 1
    while i < len(lines) and lines[i].lstrip().startswith("-5"):
        i += 1
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("-3"):
            i += 1
            break
        m = _DATA_LINE_RE.match(line)
        if m:
            node_id = int(m.group(1))
            values[node_id] = [float(x) for x in _FLOAT_RE.findall(m.group(2))]
        i += 1
    return values, i


def _von_mises(sxx, syy, szz, sxy, syz, szx):
    return (0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
                   + 6 * (sxy ** 2 + syz ** 2 + szx ** 2))) ** 0.5


def parse_frd(path):
    """Return (displacement, stress): dicts keyed by node id.

    displacement[nid] = [ux, uy, uz]
    stress[nid] = von Mises equivalent stress (scalar)

    If a result type appears in multiple blocks (e.g. several increments),
    the last block encountered wins, matching "final converged state".
    """
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    disp_raw = {}
    stress_raw = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("-4") and "DISP" in stripped.upper():
            disp_raw, i = _parse_block(lines, i)
            continue
        if stripped.startswith("-4") and "STRESS" in stripped.upper():
            stress_raw, i = _parse_block(lines, i)
            continue
        i += 1

    displacement = {nid: vals[:3] for nid, vals in disp_raw.items() if len(vals) >= 3}
    stress = {
        nid: _von_mises(*vals[:6])
        for nid, vals in stress_raw.items() if len(vals) >= 6
    }
    return displacement, stress


def parse_frd_sensitivity(path, block: str = "SENSTRE"):
    """Return dfdn: dict mapping node id -> d(objective)/d(normal displacement),
    the raw (unfiltered) SENSTRE field written by a *SENSITIVITY step's
    `*NODE FILE / SEN` request. Zero (or absent) for nodes outside the
    *DESIGNVARIABLES,TYPE=COORDINATE set.

    SENSTRE has two sub-fields, DFDN (raw) and DFDNFIL (spatially filtered,
    normalized for direct use in an OC-style update) -- this returns DFDN,
    the physically-scaled derivative, for use as an exact chain-rule term.

    `block` selects which sensitivity field to read: CalculiX names it after
    the *OBJECTIVE, so a MASS objective writes SENMASS instead of SENSTRE
    (same DFDN/DFDNFIL sub-field layout). A MASS run is the calibration case
    -- with *DENSITY 1.0 the objective is exactly the volume, whose true
    derivative is known independently, which is how the DFDN convention was
    pinned down.
    """
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    sen_raw = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("-4") and block.upper() in stripped.upper():
            sen_raw, i = _parse_block(lines, i)
            continue
        i += 1

    return {nid: vals[0] for nid, vals in sen_raw.items() if len(vals) >= 1}


def parse_frd_normals(path):
    """Return normal: dict mapping node id -> unit outward-normal (nx, ny, nz),
    the NORM field CalculiX writes for every *DESIGNVARIABLES,TYPE=COORDINATE
    node during a *SENSITIVITY step (the mean of the normals on every
    external face the node belongs to). Needed to project a 3D geometric
    velocity (e.g. core.geometric_jacobian's dX_dt) onto the single
    normal-direction component that SENSTRE's DFDN is a derivative of.
    """
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    norm_raw = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("-4") and "NORM" in stripped.upper():
            norm_raw, i = _parse_block(lines, i)
            continue
        i += 1

    return {nid: tuple(vals[:3]) for nid, vals in norm_raw.items() if len(vals) >= 3}
