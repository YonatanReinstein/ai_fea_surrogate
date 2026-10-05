"""Validation designs: the rendered geometry (as in the topopt screenshots),
the ccx adjoint gradient grid, and the surrogate's gradient grid side by side,
one row per design.

The geometry panel is optimization.screenshot.screenshot(heatmap=True), the
exact call the topopt screenshots use: the design's CAD mesh colored by a real
ccx solve's von Mises stress on a fixed 0..yield jet scale. Needs ccx and IRIT
on PATH. --render thickness shows the per-tile thickness grid instead.

The surrogate gradient comes from GNNEvaluator.compute_stress_and_gradient,
the same mesh + model path the topopt optimizer uses (not a hand-built probe).
Grids use the optimizer's tile layout (topopt_optimizer._save_tile_gradient_heatmap):
values permuted from dims-key order, reshaped (NZ, NY, NX), mid z-slice,
origin lower, so they match the optimization screenshots.

    python -m optimization.plot_val_gradient_grids --geometry hollow_cube_11x11 \
        --ckpt CKPT --shards SHARD [SHARD ...] --n 4 --out out.png
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED = "#1a1a19", "#6b6a63"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="hollow_cube_11x11")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--render", choices=["geometry", "thickness"], default="geometry")
    ap.add_argument("--crop", default="1480,360,3640,2520",
                    help="design region of the 5120x2880 screenshot (left, top, right, bottom)")
    a = ap.parse_args()

    from training.dataset_io import load_shard
    from evaluators.gnn_evaluator import GNNEvaluator
    from core.tile_sensitivity import dims_key_order_permutation

    names = list(json.load(open(f"data/{a.geometry}/CAD_model/dims.json")).keys())
    bc = importlib.import_module(f"data.{a.geometry}.boundary_conditions")
    NX, NY, NZ = bc.tile_grid()
    perm = np.asarray(dims_key_order_permutation(NX, NY, NZ))

    def grid(v):
        return np.asarray(v, float)[perm].reshape(NZ, NY, NX)[NZ // 2]

    pool = [s for p in a.shards for s in load_shard(p)]
    rng = np.random.default_rng(a.seed)
    picks = sorted(rng.choice(len(pool), size=a.n, replace=False))
    ev = GNNEvaluator(a.geometry, ckpt_path=a.ckpt, stress_ks_rho_mpa=a.rho)

    import tempfile
    from PIL import Image
    from optimization.screenshot import screenshot
    yld = float(json.load(open(f"data/{a.geometry}/CAD_model/material_properties.json"))["yield_strength"])
    crop = tuple(int(v) for v in a.crop.split(","))
    tmpdir = tempfile.mkdtemp()

    rows = []
    for i in picks:
        s = pool[i]
        x = s.dims.view(-1).numpy().astype(float)
        dims = {n: float(v) for n, v in zip(names, x)}
        true = np.zeros(len(names))
        true[s.grad_tile_idx.view(-1).numpy()] = s.grad_dks_dx.view(-1).numpy()   # dims-key order, MPa/unit
        st, dsdx, vol = ev.compute_stress_and_gradient(dims)
        pred = np.array([float(dsdx[n]) for n in names])
        ks_true = float(torch.as_tensor(s.ks_stress_adjoint).view(-1)[0])
        cos = float(pred @ true / (np.linalg.norm(pred) * np.linalg.norm(true) + 1e-30))
        wrong = float((np.sign(pred) != np.sign(true)).mean())
        r = np.abs(pred - true) / (np.abs(true) + 1e-12)
        w = (np.abs(true) / np.abs(true).sum()) ** 0.15
        gloss = float((w / w.mean() * (r / (1 + r / 2.0))).mean())
        img = None
        if a.render == "geometry":
            shot = os.path.join(tmpdir, f"val_{i}.png")
            screenshot(geometry=a.geometry, dims=dims, save_path=shot, heatmap=True)
            img = Image.open(shot).convert("RGB").crop(crop)
        rows.append((i, x, true, pred, ks_true, float(st), float(vol), cos, wrong, gloss, img))
        print(f"[val-grid] sample {i}: KS ccx {ks_true:.1f} surrogate {float(st):.1f} MPa, vol {float(vol):.1f}, "
              f"cos {cos:+.3f}, wrong sign {100*wrong:.1f}%, grad loss {100*gloss:.1f}%")

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for r_, (i, x, true, pred, kt, ks, vol, cos, wrong, gloss, img) in enumerate(rows):
        lim = float(np.percentile(np.abs(np.concatenate([true, pred])), 98)) or 1.0
        ax = axes[r_, 0]
        ttl = f"design (val sample {i})\nKS: ccx {kt:.1f} / surrogate {ks:.1f} MPa, volume {vol:.1f}"
        if img is not None:
            ax.imshow(img)
            sm = matplotlib.cm.ScalarMappable(cmap="jet", norm=matplotlib.colors.Normalize(0, yld))
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03, label="von Mises stress, ccx (MPa)")
        else:
            im = ax.imshow(grid(x), cmap="viridis", origin="lower", vmin=0.01, vmax=0.47)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="tile thickness")
        ax.set_title(ttl, fontsize=10, color=INK)
        for c, (v, lab) in enumerate([(true, "gradient, ccx adjoint (truth)"),
                                      (pred, f"gradient, surrogate\ncos {cos:+.3f}, wrong sign {100*wrong:.1f}%, "
                                             f"grad loss {100*gloss:.1f}%")], start=1):
            ax = axes[r_, c]
            im = ax.imshow(grid(v), cmap="seismic", origin="lower", vmin=-lim, vmax=lim)
            ax.set_title(lab, fontsize=10, color=INK)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="dKS/dt (MPa per unit)")
        if img is not None:
            axes[r_, 0].axis("off")
        for ax in (axes[r_] if img is None else axes[r_, 1:]):
            ax.set_xticks(range(0, NX, 2)); ax.set_yticks(range(0, NY, 2))
            # The rendered geometry (and every topopt screenshot) shows tile X
            # mirrored relative to this grid layout: dims key d1 sits at grid
            # (X=0, Y=0) but in the bottom-RIGHT corner of the render, next to
            # the anchored edge (checked 2026-10-01 with a single thick tile).
            # Reverse the X axis so the grids line up with the picture.
            if img is not None:
                ax.invert_xaxis()
            ax.set_xlabel("tile X", color=MUTED); ax.set_ylabel("tile Y", color=MUTED)
    fig.suptitle(a.title, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"[val-grid] -> {a.out}")


if __name__ == "__main__":
    main()
