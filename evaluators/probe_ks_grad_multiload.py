"""Per-tile dKS/dt of the multi-load surrogate vs the solver, by central
finite differences of the WHOLE pipeline on both sides (CAD mesh -> load ->
solve/predict -> KS), so neither side needs its own chain rule.

KS: rho 0.25/MPa over all nodes except the loaded tile's (the dataset's
max_vm_excl node set), per load case.

  solve  (cluster, CPU): point 0 = base design, 1+2i / 2+2i = tile i +h / -h.
         Saves coords and the solver's per-case KS for each point.
  plot   (GPU): runs the surrogate on the saved coords, finite-differences
         both, writes heat maps and per-case agreement.

    python -m evaluators.probe_ks_grad_multiload solve --out DIR --points 0-242
    python -m evaluators.probe_ks_grad_multiload plot --out DIR --ckpt training/runs/multiload_v1/best.pt
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch

GEOMETRY = "cross4_11x11"
NX = NY = 11
RHO_MPA = 0.25


def ks_excl(vm, excl_nodes, rho_mpa=RHO_MPA):
    v = np.delete(np.asarray(vm, np.float64), excl_nodes) / 1e6        # MPa
    m = v.max()
    return m + np.log(np.exp(rho_mpa * (v - m)).sum()) / rho_mpa, m


def vm_np(S):
    S = np.asarray(S, np.float64)
    a, b, c, d, e, f = (S[..., i] for i in range(6))
    return np.sqrt(0.5 * ((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2) + 3 * (d * d + e * e + f * f))


def point_dims(base, j, h):
    t = np.array(base, float)
    if j > 0:
        i = (j - 1) // 2
        t[i] += h if (j - 1) % 2 == 0 else -h
    return {f"d{k + 1}": float(t[k]) for k in range(len(t))}


def cmd_solve(a):
    from training.build_cross4_multiload_dataset import solve_design
    from training.multiload_data import MultiLoadStatic
    bc = importlib.import_module(f"data.{GEOMETRY}.boundary_conditions")
    static = MultiLoadStatic(a.data)
    cases = static.cases
    F = float(os.environ.get("FEA_LOAD_N") or 1e7)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS") or 4))
    lo, hi = (int(v) for v in a.points.split("-"))
    base = [a.t0] * (NX * NY)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"t0": a.t0, "h": a.h, "rho_mpa": RHO_MPA}, open(f"{a.out}/probe.json", "w"))
    for j in range(lo, hi + 1):
        path = f"{a.out}/p{j:03d}.npz"
        if os.path.exists(path):
            continue
        dims = point_dims(base, j, a.h)
        sample, _ = solve_design(dims, bc, cases, F, RHO_MPA)
        vm = vm_np(sample["S"])
        ks, pk = zip(*[ks_excl(vm[k], static.load_nodes[r]) for k, (r, _) in enumerate(cases)])
        extra = {"S": sample["S"]} if j == 0 else {}
        np.savez(path + ".tmp.npz", coords=sample["coords"], t=np.array(list(dims.values()), np.float32),
                 ks=np.array(ks), peak=np.array(pk), **extra)
        os.replace(path + ".tmp.npz", path)
        print(f"point {j}: ks {np.round(ks[:4], 2)} ...", flush=True)


@torch.no_grad()
def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch_geometric.data import Batch
    from training.multiload_data import MultiLoadStatic, design_cases
    from utils.gnn_hierarchical_cnn import HierarchicalGNNCNN

    meta = json.load(open(f"{a.out}/probe.json"))
    h = meta["h"]
    static = MultiLoadStatic(a.data)
    C = len(static.cases)
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = HierarchicalGNNCNN(**ck["model_kwargs"]).to(dev)
    model.set_tile_grid(NX, NY, 1)
    model.load_state_dict(ck["model"])
    model.eval()
    st = {k: v.to(dev) for k, v in ck["stats"].items()}
    stress_only = ck["args"].get("stress_only", 0)
    cols = slice(3, 9) if stress_only else slice(0, 9)
    ym, ys = st["y_mean"], st["y_std"]
    if ym.numel() == 9 and stress_only:
        ym, ys = ym[cols], ys[cols]

    def surrogate(z):
        out_ks, out_pk, out_vm = [], [], []
        data = design_cases(static, z, range(C))
        for j in range(0, C, 4):
            b = Batch.from_data_list(data[j:j + 4]).to(dev)
            x = (b.x - st["x_mean"]) / st["x_std"]
            ea = (b.edge_attr - st["edge_mean"]) / st["edge_std"]
            tx = (b.tile_x - st["tile_mean"]) / st["tile_std"]
            _, p = model(x, b.edge_index, ea, b.batch, tile_idx=b.tile_idx, tile_NX=NX, tile_NY=NY,
                         tile_NZ=1, tile_x=tx, num_graphs=b.num_graphs, node_id=b.node_id)
            S = (p.float() * ys + ym)[:, -6:].double().cpu().numpy()
            for g in range(b.num_graphs):
                k = j + g
                vm = vm_np(S[(b.batch == g).cpu().numpy()])
                ks, pk = ks_excl(vm, static.load_nodes[static.cases[k][0]])
                out_ks.append(ks); out_pk.append(pk); out_vm.append(vm)
        return np.array(out_ks), np.array(out_pk), out_vm

    def load(j):
        z = dict(np.load(f"{a.out}/p{j:03d}.npz"))
        z.update(u=np.zeros((C, static.N, 3), np.float32), S=np.zeros((C, static.N, 6), np.float32),
                 max_vm=np.zeros(C), max_vm_excl=np.zeros(C))
        return z

    K = NX * NY
    z0 = load(0)
    sks0, spk0, svm0 = surrogate(z0)
    fks0, fpk0 = z0["ks"], z0["peak"]
    gF, gS = np.zeros((C, K)), np.zeros((C, K))
    for i in range(K):
        zp, zm = load(1 + 2 * i), load(2 + 2 * i)
        gF[:, i] = (zp["ks"] - zm["ks"]) / (2 * h)
        gS[:, i] = (surrogate(zp)[0] - surrogate(zm)[0]) / (2 * h)
    np.savez(f"{a.out}/grads.npz", solver=gF, surrogate=gS, solver_ks=fks0, surrogate_ks=sks0,
             solver_peak=fpk0, surrogate_peak=spk0)

    # dims order d(ix*NY + iy): grid[iy, ix], ix = 0 anchored end -> NX-1 loaded end
    grid = lambda g: g.reshape(NX, NY).T
    rows = []
    for k, (r, d) in enumerate(static.cases):
        a_, b_ = gF[k], gS[k]
        cos = float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-30))
        sign = float(np.mean(np.sign(a_) == np.sign(b_)))
        # control: the case-independent mean solver gradient direction
        ctrl = gF.mean(0)
        cos_c = float(a_ @ ctrl / (np.linalg.norm(a_) * np.linalg.norm(ctrl) + 1e-30))
        rows.append(dict(case=k, row=r, dir=("down", "anchor")[d], ks_solver=fks0[k], ks_surr=sks0[k],
                         peak_solver=fpk0[k], peak_surr=spk0[k], cos=cos, sign=sign, cos_ctrl=cos_c,
                         norm_ratio=float(np.linalg.norm(b_) / np.linalg.norm(a_))))
    json.dump(rows, open(f"{a.out}/summary.json", "w"), indent=1)
    for r_ in rows:
        print(f"case {r_['case']:2d} row {r_['row']:2d} {r_['dir']:6s}  KS {r_['ks_solver']:6.1f}/{r_['ks_surr']:6.1f}  "
              f"peak {r_['peak_solver']:6.1f}/{r_['peak_surr']:6.1f} MPa  cos {r_['cos']:+.3f} "
              f"(control {r_['cos_ctrl']:+.3f})  sign {r_['sign']:.0%}  |surr|/|solver| {r_['norm_ratio']:.2f}")

    show = [int(v) for v in a.cases.split(",")]
    fig = plt.figure(figsize=(10, 4.1 * len(show)), layout="constrained")
    fig.suptitle(f"dKS/dt per tile, uniform design t = {meta['t0']}  (central differences, h = {h}; "
                 f"KS excludes the loaded tile, black box)", fontsize=11)
    for sf, k in zip(np.atleast_1d(fig.subfigures(len(show), 1)), show):
        r, d = static.cases[k]
        rr = rows[k]
        sf.suptitle(f"case {k}: load row {r}, {('down (−y)', 'toward anchor (−z)')[d]}  —  cosine {rr['cos']:+.2f} "
                    f"(control {rr['cos_ctrl']:+.2f}), sign agreement {rr['sign']:.0%}, "
                    f"|surrogate|/|solver| {rr['norm_ratio']:.2f}", fontsize=10, fontweight="bold")
        lim = max(np.abs(gF[k]).max(), np.abs(gS[k]).max())
        axs = sf.subplots(1, 2)
        for ax, g, name, ks in ((axs[0], gS[k], "surrogate", sks0[k]), (axs[1], gF[k], "solver", fks0[k])):
            im = ax.imshow(grid(g), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.add_patch(plt.Rectangle((NX - 1.5, r - 0.5), 1, 1, fill=False, ec="k", lw=2))
            ax.set_title(f"{name}  (KS {ks:.1f} MPa)", fontsize=10)
            ax.set_xlabel("tile ix   (anchored end → loaded end)")
            ax.set_ylabel("tile iy (row)")
        sf.colorbar(im, ax=axs, shrink=0.9, label="dKS/dt  (MPa per unit thickness)")
    fig.savefig(f"{a.out}/ks_grad_heatmaps.png", dpi=130)
    print("wrote", f"{a.out}/ks_grad_heatmaps.png")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=["solve", "plot"])
    p.add_argument("--data", default="data/cross4_11x11/dataset_multiload_5000")
    p.add_argument("--out", default="data/cross4_11x11/ksgrad_uniform030")
    p.add_argument("--points", default="0-242")
    p.add_argument("--t0", type=float, default=0.3)
    p.add_argument("--h", type=float, default=0.005)
    p.add_argument("--ckpt", default="training/runs/multiload_v1/best.pt")
    p.add_argument("--cases", default="10,11,0,20", help="cases to draw (index = 2*row + dir)")
    a = p.parse_args()
    (cmd_solve if a.cmd == "solve" else cmd_plot)(a)


if __name__ == "__main__":
    main()
