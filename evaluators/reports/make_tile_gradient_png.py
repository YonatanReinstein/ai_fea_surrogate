"""Render the 1x3x3 per-tile gradient heat map to PNG (slide-ready)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.cm import ScalarMappable

LO, HI = 5.0, 9.0                       # decades spanned by the colour scale
MID = "#eceff1"
NEG = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
POS = ["#fbd5d1", "#f3ada7", "#e06e68", "#ca4442", "#8c2828"]
INK, INK2, MUTED, RULE = "#14171a", "#4b5257", "#828a90", "#c9d1d6"

TILES = {  # id -> (y centre, z centre, col, row)   row 0 = top (y=17.5)
    "d1": (3.5, 6.67, 0, 2), "d2": (10.5, 6.67, 0, 1), "d3": (17.5, 6.67, 0, 0),
    "d4": (3.5, 20.0, 1, 2), "d5": (10.5, 20.0, 1, 1), "d6": (17.5, 20.0, 1, 0),
    "d7": (3.5, 33.33, 2, 2), "d8": (10.5, 33.33, 2, 1), "d9": (17.5, 33.33, 2, 0)}
RUNS = {
 "5,5,5": {"ks": 7.34519e7, "nodes": 3320, "v": {
   "d1": -3.0587e7, "d2": -1.0542e7, "d3": -3.0796e7, "d4": -2.4187e6, "d5": -1.0608e6,
   "d6": -2.4587e6, "d7": -2.9576e5, "d8": -4.3073e5, "d9": -2.9876e5}},
 "7,7,7": {"ks": 1.30641e8, "nodes": 10024, "v": {
   "d1": -1.4285e8, "d2": -1.0912e8, "d3": -1.4303e8, "d4": -3.3802e6, "d5": 7.2393e6,
   "d6": -3.3666e6, "d7": -3.2404e5, "d8": 4.3382e5, "d9": -3.3585e5}},
 "9,9,9": {"ks": 2.30954e8, "nodes": 22392, "v": {
   "d1": -3.0484e8, "d2": -2.7283e8, "d3": -3.0431e8, "d4": -4.9878e6, "d5": 1.7394e7,
   "d6": -5.0345e6, "d7": -4.5890e5, "d8": 9.5380e5, "d9": -4.2751e5}}}
RATIO = {"5,5,5": dict(zip(TILES, [0.8911, 0.9055, 0.8972, 0.6126, 0.5058, 0.6227, 0.3342, 0.3211, 0.3376])),
         "7,7,7": dict(zip(TILES, [0.9997, 0.9990, 1.0009, 0.9861, 1.0021, 0.9822, 0.9656, 1.0334, 1.0008])),
         "9,9,9": dict(zip(TILES, [1.0039, 1.0002, 1.0021, 0.9656, 0.9991, 0.9746, 1.1082, 0.9773, 1.0324]))}
ZE = [0.0, 40/3, 80/3, 40.0]            # tile edges along z (length)
YE = [0.0, 7.0, 14.0, 21.0]             # tile edges along y (height)

def build_cmap(n=1024):
    lut = np.zeros((n, 3))
    for i in range(n):
        p = i / (n - 1)
        t = abs(p - 0.5) * 2
        stops = [to_rgb(MID)] + [to_rgb(c) for c in (NEG if p < 0.5 else POS)]
        x = t * (len(stops) - 1)
        k = min(len(stops) - 2, int(x))
        a, b = np.array(stops[k]), np.array(stops[k + 1])
        lut[i] = a + (b - a) * (x - k)
    return LinearSegmentedColormap.from_list("dks", lut)

CMAP, NORM = build_cmap(), Normalize(0, 1)

def pos_of(v):
    """signed log magnitude -> [0,1] colour position (0.5 = zero)."""
    m = abs(v)
    t = 0.0 if m <= 0 else np.clip((np.log10(m) - LO) / (HI - LO), 0, 1)
    return 0.5 + 0.5 * np.sign(v) * t

def ink_on(rgb):
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * f(rgb[0]) + 0.7152 * f(rgb[1]) + 0.0722 * f(rgb[2])
    return INK if lum > 0.42 else "#ffffff"

def sci(v, d=2):
    s = f"{v:.{d}e}"
    m, e = s.split("e")
    return f"{m}e{'−' if e[0]=='-' else '+'}{abs(int(e)):02d}"

def draw_grid(ax, mesh, big=True):
    run = RUNS[mesh]["v"]
    for tid, (yc, zc, col, row) in TILES.items():
        v = run[tid]
        rgb = CMAP(NORM(pos_of(v)))[:3]
        z0, z1, y0, y1 = ZE[col], ZE[col + 1], YE[2 - row], YE[3 - row]
        ax.add_patch(Rectangle((z0, y0), z1 - z0, y1 - y0, facecolor=rgb,
                               edgecolor="white", linewidth=2.4, zorder=2))
        tc = ink_on(rgb)
        if big:
            ax.text(z0 + 1.0, y1 - 1.2, tid, color=tc, fontsize=11, weight="semibold",
                    alpha=.82, ha="left", va="top", zorder=3)
            ax.text((z0 + z1) / 2, (y0 + y1) / 2 - 0.4, sci(v), color=tc, fontsize=16,
                    ha="center", va="center", zorder=3)
        else:
            ax.text((z0 + z1) / 2, (y0 + y1) / 2, sci(v, 1), color=tc, fontsize=9.5,
                    ha="center", va="center", zorder=3)
    if big:
        ax.set_xlim(-7.5, 55.5)
        ax.set_ylim(-3.0, 24.5)
    else:
        ax.set_xlim(-0.6, 40.6)
        ax.set_ylim(-0.6, 21.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


def annotate(ax):
    # fixity hatch on the anchored root wall, z = 0
    ax.plot([0, 0], [0, 21], color=MUTED, lw=1.6, zorder=4)
    for yy in np.linspace(0, 20.0, 15):
        ax.plot([0, -1.5], [yy, yy + 1.3], color=MUTED, lw=1.0, zorder=4)
    ax.text(-6.4, 10.5, "R O O T   ·   A N C H O R E D", rotation=90,
            ha="center", va="center", fontsize=8, color=MUTED)
    # neutral axis, drawn through and labelled clear of the cells
    ax.plot([-1.8, 43.4], [10.5, 10.5], color=INK, lw=1.0, ls=(0, (1, 3.6)),
            alpha=.6, zorder=5)
    # tip load
    ax.annotate("", xy=(45.6, 6.6), xytext=(45.6, 14.4),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=INK, mutation_scale=14),
                zorder=4)
    ax.text(47.4, 14.0, "L O A D", fontsize=8, color=MUTED, va="center")
    ax.text(47.4, 12.2, "−5e7 N in −y", fontsize=8.6, color=INK2, va="center")
    ax.text(47.4, 9.6, "neutral axis", fontsize=8.6, color=MUTED, va="center")
    ax.text(47.4, 8.0, "y = 10.5", fontsize=8.6, color=MUTED, va="center")
    # axis ticks
    for zc, lab in ((6.67, "z=6.7"), (20.0, "z=20.0"), (33.33, "z=33.3")):
        ax.text(zc, -0.8, lab, fontsize=9, color=MUTED, ha="center", va="top")
    for yc in (17.5, 10.5, 3.5):
        ax.text(-2.4, yc, f"{yc}", fontsize=9, color=MUTED, ha="right", va="center")
    ax.text(20, 22.2, "z   →   L E N G T H   ( m m )", fontsize=8, color=MUTED,
            ha="center", va="bottom")
    ax.text(-2.4, 22.2, "y  (mm)", fontsize=8, color=MUTED, ha="right", va="bottom")


# ---------------------------------------------------------------- figure
plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white"})
fig = plt.figure(figsize=(12.4, 8.6), dpi=200)

fig.text(0.045, 0.972, "Per-tile stress gradient  ·  1×3×3 tile grid",
         fontsize=19.5, weight="semibold", color=INK, ha="left", va="top")
fig.text(0.955, 0.968, "hollow_cube_133", fontsize=10.5, color=MUTED,
         ha="right", va="top", family="DejaVu Sans Mono")
fig.text(0.045, 0.928,
         "d(KS stress)/d(tile thickness), from three patched-ccx solves.  "
         "Blue: thicker → lower stress.   Red: thicker → higher stress.",
         fontsize=11, color=INK2, ha="left", va="top")
fig.text(0.045, 0.898,
         "mesh 9,9,9 · C3D8 · interior nodes on · 22,392 nodes / 18,432 elements · "
         "KS = 2.310e+08 Pa",
         fontsize=9.5, color=MUTED, ha="left", va="top", family="DejaVu Sans Mono")

ax = fig.add_axes([0.045, 0.435, 0.91, 0.42])
draw_grid(ax, "9,9,9", big=True)
annotate(ax)

# ---- colour bar
cax = fig.add_axes([0.235, 0.392, 0.53, 0.020])
cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
ticks = [(0, "−1e9"), (.125, "−1e8"), (.25, "−1e7"), (.375, "−1e6"),
         (.5, "0"), (.625, "+1e6"), (.75, "+1e7"), (.875, "+1e8"), (1, "+1e9")]
cb.set_ticks([t for t, _ in ticks])
cb.ax.set_xticklabels([l for _, l in ticks], fontsize=8.4, color=MUTED)
cb.outline.set_edgecolor(RULE); cb.outline.set_linewidth(0.8)
cb.ax.tick_params(length=3, color=RULE, pad=2)
fig.text(0.5, 0.352, "dKS/dt, Pa per unit thickness  ·  log scale on each arm",
         fontsize=8.8, color=MUTED, ha="center", va="top")

# ---- mesh convergence strip
fig.text(0.045, 0.330, "M E S H   C O N V E R G E N C E", fontsize=8, color=MUTED,
         ha="left", va="top")
fig.text(0.955, 0.330,
         "ratio = adjoint contraction ÷ remesh finite difference",
         fontsize=8.8, color=MUTED, ha="right", va="top")
fig.add_artist(plt.Line2D([0.045, 0.955], [0.313, 0.313], color=RULE, lw=0.9))
for i, mesh in enumerate(("5,5,5", "7,7,7", "9,9,9")):
    x0 = 0.045 + i * 0.3125
    rs = sorted(RATIO[mesh].values())
    fig.text(x0, 0.295, f"mesh {mesh}", fontsize=10.5, weight="semibold", color=INK,
             ha="left", va="top")
    fig.text(x0, 0.270, f"median ratio {rs[4]:.4f}   ({rs[0]:.2f}–{rs[8]:.2f})",
             fontsize=8.8, color=MUTED, ha="left", va="top",
             family="DejaVu Sans Mono")
    a = fig.add_axes([x0, 0.105, 0.285, 0.150])
    draw_grid(a, mesh, big=False)
    a.text(0, -1.7, "root", fontsize=8, color=MUTED, ha="left", va="top", clip_on=False)
    a.text(40, -1.7, "tip", fontsize=8, color=MUTED, ha="right", va="top", clip_on=False)

fig.text(0.045, 0.030,
         "Validated tile by tile against central-difference remeshing (18 solves): sign "
         "agreement 9/9 at every resolution.\nAt 5,5,5 the 0.3-thick frame walls are "
         "under-resolved — the whole field is too weak and d5 / d8 point the wrong way.",
         fontsize=9, color=MUTED, ha="left", va="bottom", linespacing=1.5)

out = "/home/ryonatan/ai_fea_surrogate/evaluators/reports/tile_gradient_map_1x3x3.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
