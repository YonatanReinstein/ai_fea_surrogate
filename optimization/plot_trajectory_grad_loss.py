"""Per-generation gradient loss along topopt trajectories, for several runs.

At every generation the surrogate's dG/dx (as served) is scored against the ccx
adjoint with the SAME gradient terms training/gnn_training.py trains on:
  grad   saturated per-tile percent error r/(1+r/sat), r = |pred-true|/|true|,
         weighted by (|true|/sum|true|)^power (normalised to mean 1)
  sign   smooth per-tile wrong-sign loss 0.5*(1 - z/(delta+|z|)), z = pred*sign(true)/|true|
  sgnerr exact fraction of tiles with the wrong sign (every tile equal)
All three are relative, so the dG units (dKS/dx / yield) do not matter.
Reads the per-generation files written by the labeller (online runs) or by
TOPOPT_CCX_REF_DIR (frozen runs): gen_NNNNNN.npz with surrogate_dg / dg_ccx.

    python -m optimization.plot_trajectory_grad_loss \
        --run "frozen=optimization/artifacts_X/ccx_ref" --run "fine-tuned=onpolicy/Y/labels" \
        --out a.png --out b.png
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical order
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "gen_*.npz")):
        if not re.fullmatch(r"gen_\d+\.npz", os.path.basename(f)):
            continue                      # skip the labeller's _pNN jitter siblings
        z = np.load(f, allow_pickle=True)
        p, t = np.asarray(z["surrogate_dg"], float), np.asarray(z["dg_ccx"], float)
        if p.size and p.size == t.size:
            rows.append((int(z["gen"]), p, t, float(z["ccx_volume"])))
    rows.sort(key=lambda r: r[0])
    return rows


def terms(p, t, power, sat, delta):
    r = np.abs(p - t) / (np.abs(t) + 1e-12)
    w = (np.abs(t) / np.abs(t).sum()) ** power
    grad = float((w / w.mean() * (r / (1 + r / sat))).mean())
    z = p * np.sign(t) / (np.abs(t) + 1e-12)
    sign = float((0.5 * (1 - z / (delta + np.abs(z)))).mean())
    sgnerr = float((np.sign(p) != np.sign(t)).mean())
    return grad, sign, sgnerr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=dir (repeatable)")
    ap.add_argument("--out", action="append", required=True, help="output png (repeatable)")
    ap.add_argument("--power", type=float, default=0.15)
    ap.add_argument("--sat", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--title", default="Gradient loss along the optimization trajectory (surrogate vs ccx)")
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        label, d = spec.split("=", 1)
        rows = load(d)
        assert rows, f"no per-generation files with surrogate_dg under {d}"
        g = np.array([r[0] for r in rows])
        T = np.array([terms(r[1], r[2], a.power, a.sat, a.delta) for r in rows])
        v = np.array([r[3] for r in rows])
        runs.append((label, g, T, v))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True,
                           gridspec_kw={"height_ratios": [3, 3, 3, 2]})
    panels = [
        (0, 100, "gradient percent-error loss (%)", 100 / (1 + 1 / a.sat),
         f"know-nothing baseline {100 / (1 + 1 / a.sat):.1f}% (predict 0)"),
        (1, 1, "sign loss", 0.5, "know-nothing baseline 0.5 (coin flip)"),
        (2, 100, "tiles with the wrong sign (%)", 50, "coin flip 50%"),
    ]
    for k, scale, ylabel, base, base_lab in panels:
        axk = ax[k]
        axk.axhline(base, color=MUTED, ls="--", lw=1)
        axk.text(0.995, base, base_lab, transform=axk.get_yaxis_transform(), ha="right",
                 va="bottom", color=MUTED, fontsize=8.5)
        for i, (label, g, T, v) in enumerate(runs):
            y = scale * T[:, k]
            axk.plot(g, y, color=COLORS[i], lw=2, label=f"{label} (mean {np.mean(y):.3g})")
        axk.set_ylabel(ylabel)
        axk.set_ylim(bottom=0)
        axk.grid(axis="y", color=GRID, lw=0.8)
        axk.legend(loc="upper left", fontsize=9, frameon=False)
        for s in ("top", "right"):
            axk.spines[s].set_visible(False)
    for i, (label, g, T, v) in enumerate(runs):
        ax[3].plot(g, v, color=COLORS[i], lw=2, label=label)
    ax[3].set_ylabel("volume")
    ax[3].set_xlabel("generation")
    ax[3].grid(axis="y", color=GRID, lw=0.8)
    ax[3].legend(loc="upper right", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax[3].spines[s].set_visible(False)
    fig.suptitle(a.title, fontsize=12, color=INK)
    fig.tight_layout()
    for o in a.out:
        fig.savefig(o, dpi=120)
        print(f"[grad-loss-plot] -> {o}")
    for label, g, T, v in runs:
        early, late = g < 40, g >= 40
        print(f"[grad-loss-plot] {label}: grad {100*T[:,0].mean():.1f}% (gen<40 {100*T[early,0].mean():.1f}%, "
              f">=40 {100*T[late,0].mean():.1f}%), sign {T[:,1].mean():.3f}, wrong sign {100*T[:,2].mean():.1f}% "
              f"(gen<40 {100*T[early,2].mean():.1f}%, >=40 {100*T[late,2].mean():.1f}%)")


if __name__ == "__main__":
    main()
