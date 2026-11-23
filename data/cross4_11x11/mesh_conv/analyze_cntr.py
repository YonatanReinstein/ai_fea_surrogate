"""Summarize the CntrShape x t-range sweep (cntr.slurm): per config, error of
each plain mesh vs 24,24,2 on the peak (load tile excluded) and on per-tile
maxima (tiles above 30% of the case peak), the 16->24 change as a convergence
signal, inverted elements, and the design volume range (variety).
    python analyze_cntr.py [--ref 24,24,2]
"""
import argparse, glob, os
import numpy as np

B = "data/cross4_11x11/mesh_conv"
p = argparse.ArgumentParser()
p.add_argument("--ref", default="24_24_2")
a = p.parse_args()
RES = ["4_4_2", "6_6_2", "8_8_2", "12_12_2", "16_16_2"]
for cdir in ["cntr_ref", "cntr_c20", "cntr_c30", "cntr_c40", "cntr_c50", "cntr_c30t10", "cntr_c40t10", "cntr_c40t08"]:
    refs = sorted(glob.glob(f"{B}/{cdir}/d*_r{a.ref}.npz"))
    if not refs:
        print(cdir, "no reference yet"); continue
    pk = {r: [] for r in RES}; tl = {r: [] for r in RES}; vols, bad, per_design = [], 0, []
    for f in refs:
        R = np.load(f); vols.append(float(R["volume"])); bad += int(R["n_bad"])
        big = R["tile_max"] > 0.3 * R["max_vm_excl"][:, None]
        row = []
        for r in RES:
            g = f.replace(a.ref, r)
            if not os.path.exists(g):
                row.append(np.nan); continue
            A = np.load(g); bad += int(A["n_bad"])
            e = A["max_vm_excl"] / R["max_vm_excl"] - 1
            pk[r].append(e); tl[r].append(A["tile_max"][big] / R["tile_max"][big] - 1)
            row.append(np.median(e) * 100)
        per_design.append((os.path.basename(f)[1:6], R["max_vm_excl"].max() / 1e6, row))
    print(f"\n{cdir}: {len(refs)} designs, volume {min(vols):.1f}-{max(vols):.1f}, inverted elements {bad}")
    print(f"  {'mesh':>8} | peak err med  worst | tile err med   p5")
    for r in RES:
        if not pk[r]:
            continue
        e = np.concatenate(pk[r]); t = np.concatenate(tl[r])
        print(f"  {r.replace('_', ','):>8} | {np.median(e)*100:+8.1f}% {e.min()*100:+6.1f}% | {np.median(t)*100:+8.1f}% {np.percentile(t, 5)*100:+6.1f}%")
    print("  per design, peak@24 MPa and median peak err at", ",".join(r.split('_')[0] for r in RES))
    for d, p24, row in per_design:
        print(f"    {d} {p24:8.1f}  " + " ".join(f"{x:+6.1f}" for x in row))
