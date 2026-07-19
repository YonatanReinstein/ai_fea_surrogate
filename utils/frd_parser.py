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
