"""Score dsig surrogate checkpoints along another run's topopt trajectory (e.g.
the ccx-driven reference), against the exact values that run recorded: KS
(history s0), dKS/dt (history dg_raw x yield) and the nodal peak (the run's
"[ccx] KS .., peak .." log lines). Writes an npz for plot_trajectory_report.py.

    python jobs/tests/eval_surrogate_on_trajectory.py RUN_ART RUN_LOG OUT.npz EVERY name=ckpt [name=ckpt ...]
"""
import glob, os, re, sys
import numpy as np


def main():
    art, log, out, every = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    ckpts = dict(a.split("=", 1) for a in sys.argv[5:])
    from evaluators.gnn_dsig_evaluator import GNNDsigEvaluator
    # exact peak of each evaluated design, keyed by its KS (the history only stores s0)
    peaks = {round(float(k), 2): float(p) for k, p in
             re.findall(r"\[ccx\] KS ([\d.]+) MPa, peak ([\d.]+) MPa", open(log).read())}
    evs = {n: GNNDsigEvaluator("cross4_11x11", ckpt_path=p) for n, p in ckpts.items()}
    ev0 = next(iter(evs.values()))
    res = {"gen": [], "ks": [], "peak": []} | {f"{n}/{k}": [] for n in evs for k in ("ks", "peak", "cos", "size", "asym")}
    files = sorted(glob.glob(os.path.join(art, "topopt_history", "gen_*.npz")))
    for f in files:
        gen = int(f[-7:-4])
        if gen % every and f != files[-1]:
            continue
        z = np.load(f)
        names = list(z["free_names"])
        assert names == ev0.dims_keys[:len(names)] and len(names) == len(ev0.dims_keys), "free tiles != all dims"
        dims = dict(zip(names, map(float, z["x"])))
        g_ex = z["dg_raw"] * ev0.yld                                             # MPa per unit t
        X = ev0._mesh(dims)[0]
        res["gen"].append(gen); res["ks"].append(float(z["s0"]))
        res["peak"].append(peaks.get(round(float(z["s0"]), 2), np.nan))
        for n, ev in evs.items():
            G, peak, g = ev.predict(dims, X)
            res[f"{n}/ks"].append(G); res[f"{n}/peak"].append(peak)
            res[f"{n}/cos"].append(float(g @ g_ex / (np.linalg.norm(g) * np.linalg.norm(g_ex))))
            res[f"{n}/size"].append(float(np.linalg.norm(g) / np.linalg.norm(g_ex)))
            res[f"{n}/asym"].append(mirror_asym(g, ev0))
        print(f"gen {gen}: exact KS {res['ks'][-1]:.2f} peak {res['peak'][-1]:.2f} | " + " | ".join(
            f"{n} KS {res[f'{n}/ks'][-1]:.2f} cos {res[f'{n}/cos'][-1]:+.3f} size {res[f'{n}/size'][-1]:.3f} "
            f"asym {res[f'{n}/asym'][-1]:.3f}" for n in evs), flush=True)
    np.savez(out, names=list(evs), **{k: np.asarray(v) for k, v in res.items()})
    for ev in evs.values():
        ev.close()


def mirror_asym(v, ev):
    """|v - mirror(v)| / |v| over the tile rows (top/bottom), v in dims order."""
    from core.tile_sensitivity import dims_key_order_permutation
    perm = np.asarray(dims_key_order_permutation(ev.NX, ev.NY, ev.NZ))
    a = np.asarray(v)[perm].reshape(ev.NY, ev.NX)
    return float(np.linalg.norm(a - a[::-1]) / (np.linalg.norm(a) + 1e-30))


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
