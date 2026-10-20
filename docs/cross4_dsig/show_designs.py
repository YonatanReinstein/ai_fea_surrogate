import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, sys
sys.path.insert(0, ".")
from training.dsig_data import available_designs, design_path
root = "data/cross4_11x11/dataset_dsig_2500_tmin003"
kinds = json.load(open(f"{root}/designs.json"))["kinds"]
ids = available_designs(root)
ks = np.array([np.load(design_path(root, i))["ks"] for i in ids]); m = ks.max(1)
rng = np.random.default_rng(1)
kept = [ids[i] for i in np.where(m <= 800)[0]]; out = [ids[i] for i in np.where(m > 800)[0]]
def pick(pool, n):  # spread over the KS range of the pool
    pool = sorted(pool, key=lambda i: m[ids.index(i)]); return [pool[int(q)] for q in np.linspace(0, len(pool) - 1, n)]
sel = [("kept", i) for i in pick(kept, 10)] + [("filtered", i) for i in pick(out, 10)]
fig, axs = plt.subplots(4, 5, figsize=(17, 14.5))
for ax, (tag, i) in zip(axs.flat, sel):
    z = np.load(design_path(root, i)); t = z["t"].reshape(11, 11)      # [ix, iy]: ix = z (anchor -> load end), iy = y row
    im = ax.imshow(t.T, origin="lower", cmap="viridis", vmin=0.03, vmax=0.49)
    k = z["ks"]; r = int(k.argmax())
    ax.add_patch(plt.Rectangle((9.5, r - 0.5), 1, 1, fill=False, ec="red", lw=2))
    ax.set_title(f"{tag} #{i} ({kinds[i]})\nKS max {k.max():.0f} (row {r}), min {k.min():.0f} MPa\nt min {t.min():.3f}, mean {t.mean():.2f}",
                 fontsize=9, color="tab:green" if tag == "kept" else "tab:red")
    ax.set_xticks([0, 10]); ax.set_xticklabels(["anchor", "load end"], fontsize=8); ax.set_yticks([0, 10]); ax.set_yticklabels(["row 0", "row 10"], fontsize=8)
fig.colorbar(im, ax=axs, shrink=0.6, label="tile thickness t")
fig.suptitle("cross4 dsig designs: top two rows kept (KS <= 800 MPa on every load row), bottom two filtered out\n"
             "sorted by worst-row KS within each group; red box = load tile of the worst row", fontsize=12)
fig.savefig(sys.argv[1], dpi=90, bbox_inches="tight")
# kind breakdown
for kd in ("field", "field_hc", "two_level"):
    s = np.array([kinds[i] == kd for i in ids]); print(f"{kd:9s} kept {np.mean(m[s] <= 800):.1%} of {s.sum()}")
tmin = np.array([np.load(design_path(root, i))["t"].min() for i in ids])
for lo, hi in [(0.03, 0.035), (0.035, 0.05), (0.05, 0.1), (0.1, 1)]:
    s = (tmin >= lo) & (tmin < hi); print(f"thinnest tile {lo}-{hi}: kept {np.mean(m[s] <= 800):.1%} of {s.sum()}")
