"""Summarize the cross4 mesh-convergence runs against a reference resolution.
    python analyze.py DIR [--ref 31,31,3]
Per resolution, over every design x 22 load cases:
  peak   : max von Mises excluding the loaded tile, rel. error vs ref
  ks     : KS (rho .25/MPa), rel. error
  tile   : per-tile max VM, rel. error on tiles above 30% of the case peak
  node   : nodal VM on x=0 face nodes shared with the ref mesh (nested res
           only), |err| / case peak, and rel. error on nodes above 50% of peak
"""
import argparse, glob, os, re
import numpy as np
from scipy.spatial import cKDTree


def vm(S):
    sxx, syy, szz, sxy, sxz, syz = np.moveaxis(S, -1, 0)
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--ref", default="31,31,3")
    a = p.parse_args()
    files = sorted(glob.glob(f"{a.dir}/d*_r*.npz"))
    runs = {}
    for f in files:
        d, r = re.match(r".*/d(\d+)_r([\d_]+)\.npz", f).groups()
        runs.setdefault(r.replace("_", ","), {})[int(d)] = f
    ref = runs.get(a.ref, {})
    print(f"reference {a.ref}: designs {sorted(ref)}")
    q = lambda x, s: np.percentile(np.abs(x), s) * 100
    print(f"{'res':>9} {'nd':>2} {'elem':>7} {'solve':>6} | {'peak med':>8} {'p95':>6} {'max':>6} | "
          f"{'ks med':>6} {'max':>6} | {'tile med':>8} {'p95':>6} | {'node/pk p95':>11} {'hi med':>6} {'hi p95':>6}")
    def key(r):
        u, v, w = map(int, r.split(","))
        return (u * v, w)
    for r in sorted(runs, key=key):
        pe, ke, te, ne, he, el, ts = [], [], [], [], [], [], []
        for d, f in runs[r].items():
            if d not in ref:
                continue
            A, R = np.load(f), np.load(ref[d])
            pe.append(A["max_vm_excl"] / R["max_vm_excl"] - 1)
            ke.append(A["ks"] / R["ks"] - 1)
            big = R["tile_max"] > 0.3 * R["max_vm_excl"][:, None]
            te.append((A["tile_max"][big] / R["tile_max"][big] - 1))
            el.append(int(A["n_elem"])); ts.append(float(A["t_solve"]))
            if r != a.ref:
                tree = cKDTree(R["face_xyz"])
                dist, j = tree.query(A["face_xyz"])
                ok = dist < 1e-5
                if ok.sum() > 0:
                    va, vr = vm(A["face_S"][:, ok]), vm(R["face_S"][:, j[ok]])
                    pk = R["max_vm_excl"][:, None]
                    ne.append(((va - vr) / pk).ravel())
                    hi = vr > 0.5 * pk
                    he.append((va[hi] / vr[hi] - 1))
        if not pe:
            continue
        pe, ke, te = map(np.concatenate, (pe, ke, te))
        s = (f"{r:>9} {len(el):>2} {int(np.mean(el)):>7} {np.mean(ts):>6.1f} | {np.median(pe)*100:>+8.2f} {q(pe,95):>6.2f} "
             f"{np.abs(pe).max()*100:>6.2f} | {np.median(ke)*100:>+6.2f} {np.abs(ke).max()*100:>6.2f} | "
             f"{np.median(te)*100:>+8.2f} {q(te,95):>6.2f} |")
        if ne:
            ne, he = np.concatenate(ne), np.concatenate(he)
            s += f" {q(ne,95):>11.2f} {np.median(he)*100:>+6.2f} {q(he,95):>6.2f}"
        print(s)


if __name__ == "__main__":
    main()
