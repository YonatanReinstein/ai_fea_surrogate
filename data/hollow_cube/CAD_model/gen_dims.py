"""
Generate dims.json with one entry per tile in the NX*NY*NZ grid.
Run from the project root: python data/hollow_cube/CAD_model/gen_dims.py

Optional flags:
  --default FLOAT   per-tile FrameThickness default (default: 0.3)
  --min FLOAT       lower bound (default: 0.01)
  --max FLOAT       upper bound (default: 0.5)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))
from data.hollow_cube.boundary_conditions import tile_grid

CAD_DIR = Path(__file__).parent


def gen_dims(default: float = 0.3, lo: float = 0.01, hi: float = 0.49) -> dict:
    NX, NY, NZ = tile_grid()
    N = NX * NY * NZ
    return {f"d{i + 1}": {"default": default, "min": lo, "max": hi} for i in range(N)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--default", type=float, default=0.3)
    parser.add_argument("--min", type=float, default=0.01, dest="lo")
    parser.add_argument("--max", type=float, default=0.49, dest="hi")
    args = parser.parse_args()

    dims = gen_dims(default=args.default, lo=args.lo, hi=args.hi)
    out = CAD_DIR / "dims.json"
    out.write_text(json.dumps(dims, indent=4))
    NX, NY, NZ = tile_grid()
    print(f"Wrote {len(dims)} entries to {out}  (NX={NX} NY={NY} NZ={NZ})")
