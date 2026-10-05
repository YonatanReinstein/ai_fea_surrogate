"""MP4 of a topopt run: each generation's design screenshot next to running
traces of volume and of surrogate vs ccx KS stress, with a marker at the
current generation.

Inputs: the run's screenshots (topopt_gen_NNN.png) and its per-generation
files with surrogate_stress / ccx_stress / ccx_volume (labeller labels for an
online run, TOPOPT_CCX_REF_DIR for a frozen one).

    python -m optimization.make_trajectory_video \
        --screenshots optimization/artifacts_X/screenshots --labels onpolicy/X/labels \
        --out optimization/artifacts_X/topopt.mp4 --title "..."
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio

INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
C_CCX, C_SUR = "#1a1a19", "#eb6834"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshots", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--yield_mpa", type=float, default=100.0)
    # design region of the 5120x2880 screenshots (box: left, top, right, bottom)
    ap.add_argument("--crop", default="1480,360,3640,2520")
    # the screenshots' heatmap is a fixed 0-100 MPa jet scale (checked: identical
    # colorbar at gen 0 and gen 131); redrawn here so it is readable at video size
    ap.add_argument("--no_traces", action="store_true",
                    help="only the design (+ colorbar and the header line), no stress/volume graphs")
    ap.add_argument("--cmap_max", type=float, default=100.0)
    # topopt screenshots are colored by a REAL ccx solve of each design
    # (screenshot_heatmap defaults to True), not by the surrogate
    ap.add_argument("--cmap_label", default="von Mises stress, ccx (MPa)")
    a = ap.parse_args()

    shots = {}
    for f in glob.glob(os.path.join(a.screenshots, "topopt_gen_*.png")):
        m = re.fullmatch(r"topopt_gen_(\d+)\.png", os.path.basename(f))
        if m:
            shots[int(m.group(1))] = f
    lab = {}
    for f in glob.glob(os.path.join(a.labels, "gen_*.npz")):
        if re.fullmatch(r"gen_\d+\.npz", os.path.basename(f)):
            z = np.load(f, allow_pickle=True)
            lab[int(z["gen"])] = (float(z["surrogate_stress"]), float(z["ccx_stress"]), float(z["ccx_volume"]))
    gens = sorted(shots)
    assert gens, f"no topopt_gen_NNN.png under {a.screenshots}"
    lg = np.array(sorted(lab))
    s_sur = np.array([lab[g][0] for g in lg]); s_ccx = np.array([lab[g][1] for g in lg])
    vol = np.array([lab[g][2] for g in lg])
    crop = tuple(int(v) for v in a.crop.split(","))

    feas = s_ccx <= a.yield_mpa
    best = int(np.argmin(np.where(feas, vol, np.inf))) if feas.any() else None

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8,
                                macro_block_size=8, ffmpeg_params=[])
    for g in gens:
        img = Image.open(shots[g]).convert("RGB")
        design = img.crop(crop)
        fig = plt.figure(figsize=(16, 9), dpi=120)
        if a.no_traces:
            # design centred, colorbar to its right, no graphs
            axd = fig.add_axes([0.25, 0.03, 0.45, 0.83]); axd.imshow(design); axd.axis("off")
            axc = fig.add_axes([0.71, 0.20, 0.012, 0.50])
        else:
            # fixed columns: design | colorbar (+ its label) | the two traces
            axd = fig.add_axes([0.02, 0.05, 0.45, 0.80]); axd.imshow(design); axd.axis("off")
            axc = fig.add_axes([0.48, 0.20, 0.012, 0.50])
            axs = fig.add_axes([0.60, 0.52, 0.37, 0.33]); axv = fig.add_axes([0.60, 0.08, 0.37, 0.33], sharex=axs)
        cb = matplotlib.colorbar.ColorbarBase(axc, cmap=plt.get_cmap("jet"),
                                              norm=matplotlib.colors.Normalize(0, a.cmap_max))
        cb.set_label(a.cmap_label, color=INK); cb.outline.set_edgecolor(MUTED)

        if g in lab:
            su, cc, vv = lab[g]
            status = "feasible" if cc <= a.yield_mpa else f"{100*(cc/a.yield_mpa-1):.1f}% over yield"
            info = (f"gen {g:3d}    volume {vv:6.2f}    stress: surrogate {su:6.1f} MPa,  "
                    f"ccx {cc:6.1f} MPa  ({status})")
        else:
            info = f"gen {g:3d}    (no ccx label for this generation)"
        if not a.no_traces:
            axs.axhline(a.yield_mpa, color=MUTED, ls="--", lw=1)
            axs.text(lg.max(), a.yield_mpa, "yield ", ha="right", va="bottom", color=MUTED, fontsize=9)
            axs.plot(lg, s_ccx, color=GRID, lw=1.5); axs.plot(lg, s_sur, color=GRID, lw=1.5)
            k = lg <= g
            axs.plot(lg[k], s_ccx[k], color=C_CCX, lw=2, label="ccx (truth)")
            axs.plot(lg[k], s_sur[k], color=C_SUR, lw=2, label="surrogate")
            axs.set_ylabel("KS stress (MPa)"); axs.legend(loc="lower right", frameon=False, fontsize=10)
            axs.grid(axis="y", color=GRID, lw=0.8)
            axv.plot(lg, vol, color=GRID, lw=1.5)
            axv.plot(lg[k], vol[k], color=INK, lw=2)
            axv.set_ylabel("volume"); axv.set_xlabel("generation"); axv.grid(axis="y", color=GRID, lw=0.8)
            if g in lab:
                su, cc, vv = lab[g]
                axs.plot([g], [cc], "o", color=C_CCX, ms=8, mec="white", mew=2)
                axs.plot([g], [su], "o", color=C_SUR, ms=8, mec="white", mew=2)
                axv.plot([g], [vv], "o", color=INK, ms=8, mec="white", mew=2)
                status = "feasible" if cc <= a.yield_mpa else f"{100*(cc/a.yield_mpa-1):.1f}% over yield"
                info = (f"gen {g:3d}    volume {vv:6.2f}    stress: surrogate {su:6.1f} MPa,  "
                        f"ccx {cc:6.1f} MPa  ({status})")
            else:
                info = f"gen {g:3d}    (no ccx label for this generation)"
            if best is not None and g >= lg[best]:
                axv.plot([lg[best]], [vol[best]], "*", color="#1baf7a", ms=16, mec="white", mew=1.5)
                axv.text(lg[best], vol[best], f"  best ccx-feasible: gen {lg[best]}, vol {vol[best]:.1f}",
                         va="bottom", color=INK, fontsize=9)
            for ax_ in (axs, axv):
                for s in ("top", "right"):
                    ax_.spines[s].set_visible(False)
        fig.text(0.03, 0.95, a.title, fontsize=14, color=INK, weight="bold")
        fig.text(0.03, 0.915, info, fontsize=12, color=INK, family="monospace")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        writer.append_data(frame)
    # hold the last frame for 2 s
    for _ in range(int(2 * a.fps)):
        writer.append_data(frame)
    writer.close()
    print(f"[video] {len(gens)} generations, {a.fps:g} fps -> {a.out}")


if __name__ == "__main__":
    main()
