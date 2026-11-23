"""Merged trajectory accuracy report: a ccx-driven reference run and a
surrogate (dsig, online FT) run of the same problem, raw-KS constraint.

    python jobs/tests/plot_trajectory_report.py SUR_LOG SUR_ART CCX_LOG CCX_ART ON_CCX.npz OUT.png [title]

ON_CCX.npz comes from eval_surrogate_on_trajectory.py (surrogate checkpoints
scored on the ccx run's designs against its exact values).
"""
import glob, os, re, sys
sys.path.insert(0, os.getcwd())
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sur_log, sur_art, ccx_log, ccx_art, on_ccx, out = sys.argv[1:7]
title = sys.argv[7] if len(sys.argv) > 7 else "Trajectory accuracy -- ccx-driven vs surrogate-driven topopt"
num = r"([+-]?\d+\.?\d*)"
U = re.compile(rf"update (\d+) \(call \d+\): exact KS {num} MPa \(peak {num}\); before: KS {num} \({num}%\), dG cos {num}, "
               rf"wrong-way {num}%;.*\|dG\| ratio before {num} after {num};.*hold-out KS rel {num} cos {num} \|dKS\| ratio {num}"
               rf".*-> (\w+)")
rows = [m.groups() for m in map(U.search, open(sur_log)) if m]
u = np.array([[float(x) for x in r[:-1]] for r in rows])
kept = np.array([r[-1] == "ACCEPTED" for r in rows])
n, ks_ex, pk_ex, ks_b, e_b, cos_b, ww, size_b, size_a, ho_ks, ho_cos, ho_size = u.T
xs = n - 1                                                   # update k labels the gen k-1 design


def gens(f):
    m = (re.search(r"^\[TOPOPT\] Gen (\d+) \| vol=(\S+) stress=(\S+).*best_feas_vol=(\S+)", l) for l in open(f))
    return np.array([(int(a.group(1)), float(a.group(2)), float(a.group(3)),
                      np.nan if a.group(4) == "none" else float(a.group(4))) for a in m if a])


def design_asym(art):
    from core.tile_sensitivity import dims_key_order_permutation
    perm = np.asarray(dims_key_order_permutation(11, 11, 1))
    out = []
    for f in sorted(glob.glob(f"{art}/topopt_history/gen_*.npz")):
        z = np.load(f)
        full = np.zeros(121); full[[int(k[1:]) - 1 for k in z["free_names"]]] = z["x"]
        a = full[perm].reshape(11, 11)
        out.append((int(f[-7:-4]), np.linalg.norm(a - a[::-1]) / np.linalg.norm(a)))
    return np.array(out)


Gs, Gc = gens(sur_log), gens(ccx_log)
ccx_pk = np.array([float(p) for p in re.findall(r"\[ccx\] KS [\d.]+ MPa, peak ([\d.]+) MPa", open(ccx_log).read())])
ccx_ks = np.array([float(p) for p in re.findall(r"\[ccx\] KS ([\d.]+) MPa, peak", open(ccx_log).read())])
Z = np.load(on_ccx)
names = list(Z["names"])
As, Ac = design_asym(sur_art), design_asym(ccx_art)

blue, orange, aqua, ink, ink2, grid = "#2a78d6", "#eb6834", "#1baa9a", "#0b0b0b", "#52514e", "#e6e5e0"
model_col = {names[0]: aqua, names[-1]: "#7a4fd6"} if len(names) > 1 else {names[0]: aqua}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink2, "xtick.color": ink2,
                     "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": grid, "grid.linewidth": 0.8, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlelocation": "left", "lines.linewidth": 1.6})
fig, ax = plt.subplots(4, 2, figsize=(13, 15), facecolor="#fcfcfb")
for a in ax.flat:
    a.set_facecolor("#fcfcfb"); a.set_xlabel("generation")
lab_own = "surrogate on its own trajectory (before FT update)"
lab_m = {k: f"surrogate '{k}' on the ccx trajectory" for k in names}

a = ax[0, 0]
a.plot(Gc[:, 0], Gc[:, 3], color=orange, label=f"ccx-driven (best {np.nanmin(Gc[:, 3]):.2f})")
a.plot(Gs[:, 0], Gs[:, 3], color=blue, label=f"surrogate-driven (best {np.nanmin(Gs[:, 3]):.2f}, surrogate-judged)")
a.set_title("Best feasible volume"); a.legend(frameon=False)

a = ax[0, 1]; a.axhline(100, color=ink2, lw=0.8, ls="--")
a.plot(np.arange(len(ccx_ks)), ccx_ks, color=orange, label="ccx run: KS")
a.plot(np.arange(len(ccx_pk)), ccx_pk, color=orange, alpha=0.45, ls=":", label="ccx run: peak")
a.plot(xs, ks_ex, color=blue, label="surrogate run: exact KS")
a.plot(xs, pk_ex, color=blue, alpha=0.45, ls=":", label="surrogate run: exact peak")
a.set_xlabel("evaluation (ccx) / generation (surrogate)")
a.set_ylim(70, 120); a.set_title("Exact stress of each run's designs (MPa; limit KS 100)"); a.legend(frameon=False, fontsize=8)

a = ax[1, 0]; a.axhspan(-2, 2, color=grid, zorder=0); a.axhline(0, color=ink2, lw=0.8)
a.plot(xs, e_b, color=blue, label=lab_own)
for k in names:
    a.plot(Z["gen"], 100 * (Z[f"{k}/ks"] / Z["ks"] - 1), color=model_col[k], marker="o", ms=3, label=lab_m[k])
a.set_title("KS error vs exact (%)  (grey: ±2%)"); a.legend(frameon=False, fontsize=8)

a = ax[1, 1]; a.axhline(1, color=ink2, lw=0.8)
a.plot(xs, cos_b, color=blue, label=lab_own)
for k in names:
    a.plot(Z["gen"], Z[f"{k}/cos"], color=model_col[k], marker="o", ms=3, label=lab_m[k])
a.set_ylim(min(0.6, np.nanmin(cos_b) - 0.02), 1.01); a.set_title("dKS/dt cosine vs exact"); a.legend(frameon=False, fontsize=8)

a = ax[2, 0]; a.axhline(1, color=ink2, lw=0.8)
a.plot(xs, size_b, color=blue, label=lab_own)
for k in names:
    a.plot(Z["gen"], Z[f"{k}/size"], color=model_col[k], marker="o", ms=3, label=lab_m[k])
a.set_title("dKS/dt size, surrogate / exact"); a.legend(frameon=False, fontsize=8)

a = ax[2, 1]
for k in names:
    a.plot(Z["gen"], Z[f"{k}/asym"], color=model_col[k], marker="o", ms=3, label=lab_m[k])
a.set_ylim(0, None)
a.set_title("Surrogate dKS/dt mirror asymmetry on ccx's designs\n(exact = 0: all of it is error)")
a.legend(frameon=False, fontsize=8)

a = ax[3, 0]
a.plot(Ac[:, 0], Ac[:, 1], color=orange, label="ccx-driven")
a.plot(As[:, 0], As[:, 1], color=blue, label="surrogate-driven")
a.set_ylim(0, None); a.set_title("Design mirror asymmetry |t - mirror(t)| / |t|"); a.legend(frameon=False)

a = ax[3, 1]; a2 = a.twinx()
a.plot(xs, ho_cos, color=blue, label="hold-out dKS cos (median)")
a2.plot(xs, 100 * ho_ks, color=orange, label="hold-out KS error % (median)")
a2.spines["right"].set_visible(True); a2.grid(False)
a.set_title(f"Surrogate FT hold-out drift ({kept.sum()}/{len(kept)} updates kept)")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower left")
fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=ink)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(out, dpi=120, facecolor=fig.get_facecolor())
print(f"surrogate own trajectory: KS err median {np.median(e_b):+.2f}%, cos median {np.median(cos_b):.3f}, "
      f"size median {np.median(size_b):.3f}; exact peak max {pk_ex.max():.1f}, exact KS > 100 at "
      f"{np.mean(ks_ex[-100:] > 100):.0%} of last 100 labels")
for k in names:
    print(f"on ccx trajectory '{k}': KS err median {np.median(100 * (Z[f'{k}/ks'] / Z['ks'] - 1)):+.2f}%, cos median "
          f"{np.median(Z[f'{k}/cos']):.3f}, size median {np.median(Z[f'{k}/size']):.3f}, asym median {np.median(Z[f'{k}/asym']):.3f}")
print(f"best feasible vol: ccx {np.nanmin(Gc[:, 3]):.2f}, surrogate {np.nanmin(Gs[:, 3]):.2f}; "
      f"design asym at end: ccx {Ac[-1, 1]:.3f}, surrogate {As[-1, 1]:.3f}")
