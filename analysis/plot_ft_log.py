import re, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
log, ref, out = sys.argv[1:4]
model, job = (sys.argv[4:6] + ["v4 ep25", "1454287"][len(sys.argv[4:6]):])
num = r"([+-]?\d+\.?\d*)"
U = re.compile(rf"update (\d+) \(call \d+\): exact KS {num} MPa \(peak {num}\); before: KS {num} \({num}%\), dG cos {num}, wrong-way {num}%; after: KS {num} \({num}%\), cos {num};.*hold-out KS rel {num} cos {num}.*-> (\w+)")
rows = [m.groups() for m in map(U.search, open(log)) if m]
u = np.array([[float(x) for x in r[:-1]] for r in rows]); acc = np.array([r[-1] == "ACCEPTED" for r in rows])
n, ks_ex, pk_ex, ks_b, e_b, cos_b, ww, ks_a, e_a, cos_a, ho_ks, ho_cos = u.T
# |dG| surrogate / exact, logged by dsig_online.py since job 1456374 (absent before)
R = re.compile(rf"\|dG\| ratio before {num} after {num};.*hold-out .*\|dKS\| ratio {num}")
ratios = np.array([[float(x) for x in m.groups()] for m in (R.search(l) for l in open(log) if U.search(l)) if m])
gen = lambda f: [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in
                 (re.search(r"^\[TOPOPT\] Gen (\d+) \| vol=(\S+) stress=(\S+)", l) for l in open(f)) if m]
G, Rf = np.array(gen(log)), np.array(gen(ref))
# surrogate's reported constraint stress for the design labelled in update k (= gen k-1 design; gen 0 has none)
sur_peak = {int(g): s for g, _, s in G}
blue, orange, ink, ink2, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left", "lines.linewidth": 1.6})
fig, ax = plt.subplots(3, 2, figsize=(13, 11), facecolor="#fcfcfb")
for a in ax.flat: a.set_facecolor("#fcfcfb"); a.set_xlabel("generation of the labelled design")
x = n - 1
a = ax[0, 0]; a.plot(x, ks_ex, color=orange, label="exact (ccx-equivalent adjoint)"); a.plot(x, ks_b, color=blue, label="surrogate, before update")
a.set_title("KS stress on the trajectory (MPa)"); a.legend(frameon=False)
a = ax[0, 1]; a.axhspan(-2, 2, color=grid, zorder=0); a.axhline(0, color=ink2, lw=0.8)
a.plot(x, e_b, color=blue, label="before update"); a.plot(x, e_a, color=blue, alpha=0.4, label="after update")
a.set_title("Surrogate KS error vs exact (%)   (grey band: ±2%)"); a.legend(frameon=False)
a = ax[1, 0]; a.axhline(1, color=ink2, lw=0.8)
a.plot(x, cos_b, color=blue, label="trajectory design, before update"); a.plot(x, ho_cos, color=orange, label="36 hold-out designs (median)")
a.set_ylim(min(0, cos_b.min() - 0.05), 1.02); a.set_title("dKS/dt cosine vs exact"); a.legend(frameon=False, loc="lower right")
a = ax[1, 1]
if len(ratios) == len(x):
    a.axhline(1, color=ink2, lw=0.8)
    a.plot(x, ratios[:, 0], color=blue, label="trajectory design, before update")
    a.plot(x, ratios[:, 2], color=orange, label="36 hold-out designs (median)")
    a.set_title("dKS/dt size, surrogate / exact"); a.legend(frameon=False)
else:
    a.plot(x, ww, color=blue); a.set_title("Wrong-way gradient mass, before update (%)"); a.set_ylim(0, None)
a = ax[2, 0]; a.axhline(100, color=ink2, lw=0.8, ls="--"); a.text(x.max() * 0.45, 100.6, "limit 100 MPa", ha="center", color=ink2, fontsize=9)
xs = [k for k in x if int(k) in sur_peak]
# constraint = raw KS since the max normalization was removed (2026-10-22); the peak before
raw_ks = len(sys.argv) > 6 and sys.argv[6] == "rawks"
a.plot(x, ks_ex if raw_ks else pk_ex, color=orange, label="exact KS" if raw_ks else "exact nodal peak")
a.plot(xs, [sur_peak[int(k)] for k in xs], color=blue, label="surrogate constraint stress")
if raw_ks:
    a.plot(x, pk_ex, color=orange, alpha=0.4, ls=":", label="exact nodal peak")
a.set_ylim(80, max(120, 1)); a.set_title("Constraint stress: what the optimizer saw vs real (MPa)"); a.legend(frameon=False)
a = ax[2, 1]; a.plot(Rf[:, 0], Rf[:, 1], color=orange, label="ccx-driven reference"); a.plot(G[:, 0], G[:, 1], color=blue, label=f"surrogate + online FT ({model})")
a.set_xlabel("generation"); a.set_title("Volume"); a.legend(frameon=False)
a.set_ylim(55, 85)
fig.suptitle(f"Trajectory accuracy — cross4 row 5, surrogate FT job {job} ({acc.sum()}/{len(acc)} updates accepted)", x=0.01, ha="left", fontsize=13, fontweight="bold", color=ink)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
print(len(rows), "updates;", f"KS err before: median {np.median(e_b):+.1f}% (last 100 {np.median(e_b[-100:]):+.1f}%), cos median {np.median(cos_b):.3f} (last 100 {np.median(cos_b[-100:]):.3f}); "
      f"exact peak > 100 at {np.mean(pk_ex[-100:]>100):.0%} of last 100 labels"
      + (f"; |dG| ratio median {np.median(ratios[:, 0]):.3f} (last 100 {np.median(ratios[-100:, 0]):.3f})" if len(ratios) else ""))
