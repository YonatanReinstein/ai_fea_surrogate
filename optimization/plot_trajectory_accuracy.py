"""Surrogate accuracy along an online topopt trajectory, scored against ccx.

Reads the labeller's unperturbed per-generation labels (<labels>/gen_NNNNNN.npz,
not the _pNN jitter siblings). Each holds the surrogate's stress and dG/dx as
served at that generation and the ccx adjoint's stress and dG/dx on the same
design.

Control for the gradient panel: the cosine of a CONSTANT predictor, gen 0's
ccx dG reused at every generation. The surrogate's cosine only means something
where it beats that line (see feedback_dg_metrics_need_a_control).

    python -m optimization.plot_trajectory_accuracy --labels onpolicy/RUN/labels \
        --out optimization/artifacts_RUN/trajectory_accuracy.png
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def wrong_mass(pred, true):
    ok = np.sign(pred) == np.sign(true)
    return float(np.abs(true)[~ok].sum() / (np.abs(true).sum() + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(a.labels, "gen_*.npz"))
             if re.fullmatch(r"gen_\d+\.npz", os.path.basename(f))]
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        sdg, cdg = np.asarray(d["surrogate_dg"], float), np.asarray(d["dg_ccx"], float)
        rows.append(dict(gen=int(d["gen"]), s_sur=float(d["surrogate_stress"]),
                         s_ccx=float(d["ccx_stress"]), vol=float(d["ccx_volume"]),
                         yld=float(d["yield_strength"]), sdg=sdg, cdg=cdg))
    assert rows, f"no unperturbed gen_*.npz labels under {a.labels}"
    rows.sort(key=lambda r: r["gen"])

    gen = np.array([r["gen"] for r in rows])
    s_sur = np.array([r["s_sur"] for r in rows])
    s_ccx = np.array([r["s_ccx"] for r in rows])
    vol = np.array([r["vol"] for r in rows])
    yld = rows[0]["yld"]
    err = 100.0 * (s_sur - s_ccx) / s_ccx
    has_dg = np.array([r["sdg"].size == r["cdg"].size and r["sdg"].size > 0 for r in rows])
    ref0 = rows[0]["cdg"]
    cos = np.array([cosine(r["sdg"], r["cdg"]) if h else np.nan for r, h in zip(rows, has_dg)])
    wm = np.array([100 * wrong_mass(r["sdg"], r["cdg"]) if h else np.nan for r, h in zip(rows, has_dg)])
    cos_ctl = np.array([cosine(ref0, r["cdg"]) for r in rows])
    wm_ctl = np.array([100 * wrong_mass(ref0, r["cdg"]) for r in rows])

    feas = s_ccx <= yld
    best = int(np.argmin(np.where(feas, vol, np.inf))) if feas.any() else None

    fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    ax[0].plot(gen, s_ccx, "k-", lw=1.8, label="ccx (truth)")
    ax[0].plot(gen, s_sur, "C3-", lw=1.4, label="surrogate (as served)")
    ax[0].axhline(yld, color="gray", ls="--", lw=1, label=f"yield {yld:.0f} MPa")
    if best is not None:
        ax[0].plot(gen[best], s_ccx[best], "g*", ms=14,
                   label=f"best ccx-feasible: gen {gen[best]}, vol {vol[best]:.1f}")
    ax[0].set_ylabel("KS stress (MPa)"); ax[0].legend(loc="best", fontsize=9)

    ax[1].plot(gen, err, "C3-", lw=1.4)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("stress error (%)\nsurrogate vs ccx")
    ax[1].set_title(f"mean {np.mean(err):+.1f}%   mean |err| {np.mean(np.abs(err)):.1f}%   "
                    f"max |err| {np.max(np.abs(err)):.1f}%", fontsize=10)

    ax[2].plot(gen, cos, "C0-", lw=1.4, label=f"surrogate dG (mean {np.nanmean(cos):+.3f})")
    ax[2].plot(gen, cos_ctl, "C7:", lw=1.4, label=f"control: gen-0 ccx dG held fixed (mean {np.mean(cos_ctl):+.3f})")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylim(-1.05, 1.05); ax[2].set_ylabel("dG/dx cosine vs ccx")
    ax[2].legend(loc="lower left", fontsize=9)
    ax2b = ax[2].twinx()
    ax2b.plot(gen, wm, "C1-", lw=1, alpha=0.7)
    ax2b.plot(gen, wm_ctl, "C1:", lw=1, alpha=0.7)
    ax2b.set_ylabel("wrong-way mass (%)", color="C1"); ax2b.set_ylim(0, 60)

    ax[3].plot(gen, vol, "k-", lw=1.4)
    ax[3].scatter(gen[~feas], vol[~feas], s=8, c="C3", label="ccx over yield", zorder=3)
    ax[3].set_ylabel("volume"); ax[3].set_xlabel("generation"); ax[3].legend(loc="best", fontsize=9)

    fig.suptitle(a.title or f"On-trajectory surrogate accuracy -- {a.labels}", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(f"[traj-acc] {len(rows)} gens -> {a.out}")
    print(f"[traj-acc] stress err mean {np.mean(err):+.1f}% |err| {np.mean(np.abs(err)):.1f}%; "
          f"dG cosine mean {np.nanmean(cos):+.3f} (control {np.mean(cos_ctl):+.3f}); "
          f"wrong-way mass {np.nanmean(wm):.1f}% (control {np.mean(wm_ctl):.1f}%)")
    if best is not None:
        print(f"[traj-acc] best ccx-feasible design: gen {gen[best]} vol {vol[best]:.2f} "
              f"ccx {s_ccx[best]:.2f} MPa (surrogate said {s_sur[best]:.2f})")


if __name__ == "__main__":
    main()
