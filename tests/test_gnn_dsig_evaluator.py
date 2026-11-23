"""Checks evaluators/gnn_dsig_evaluator.py on a held-out dataset design:
(1) the CAD mesh matches mesh.npz, (2) its exact path reproduces the stored
ks / dks for one-hot weights, (3) a superposed combination's exact path matches
the same combination of the stored S, dS, (4) prints surrogate vs exact."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, ".")
def main():
    root = "data/cross4_11x11/dataset_dsig_2500_tmin003"
    run = sys.argv[1]
    i = json.load(open(f"{run}/split.json"))["val"][0]
    dims = json.load(open(f"{root}/designs.json"))["designs"][i]
    z = np.load(f"{root}/designs/{i:05d}.npz")
    print("design", i, "stored ks rows 5:", z["ks"][5], flush=True)
    from evaluators.gnn_dsig_evaluator import GNNDsigEvaluator
    os.environ["FEA_LOAD_ROW"] = "5"
    ev = GNNDsigEvaluator("cross4_11x11", ckpt_path=f"{run}/epoch_000.pt")
    vol, val, yld, dV, dG = ev.compute_topopt_gradient(dims)
    G, peak, g = ev.exact(dims)
    print(f"one-hot row 5: exact KS {G:.4f} vs stored {z['ks'][5]:.4f}; exact dks rel diff "
          f"{np.linalg.norm(g - z['dks'][5]) / np.linalg.norm(z['dks'][5]):.2e}", flush=True)
    a = np.array([dG[k] for k in ev.dims_keys]) * yld
    print(f"surrogate KS {val:.2f}, cos vs exact {a @ g / np.linalg.norm(a) / np.linalg.norm(g):+.3f}; volume {vol:.3f}, "
          f"sum dV {sum(dV.values()):.3f}", flush=True)
    # superposed combination: exact path vs stored arrays
    w = np.random.default_rng(0).dirichlet(np.ones(11))
    os.environ["DSIG_LOAD_WEIGHTS"] = ",".join(f"{x:.6f}" for x in w)
    ev2 = GNNDsigEvaluator("cross4_11x11", ckpt_path=f"{run}/epoch_000.pt")
    G2, _, g2 = ev2.exact(dims)
    wt = torch.as_tensor([float(x) for x in os.environ["DSIG_LOAD_WEIGHTS"].split(",")], dtype=torch.float64)
    S = torch.einsum("c,cnk->nk", wt, torch.as_tensor(z["S"]).double()).requires_grad_(True)
    Gs = ev2.bt.ks(S.to(ev2.device), ev2.rho); gS, = torch.autograd.grad(Gs, S)
    gs = torch.einsum("c,nk,cjnk->j", wt, gS, torch.as_tensor(z["dS"]).double()).numpy()
    print(f"combo: exact-path KS {G2:.4f} vs stored-array KS {float(Gs):.4f}; dG rel diff "
          f"{np.linalg.norm(g2 - gs) / np.linalg.norm(gs):.2e}", flush=True)
    Gp, _, gp = ev2.predict(dims)
    print(f"combo surrogate KS {Gp:.2f} vs exact {G2:.2f}; cos {gp @ g2 / np.linalg.norm(gp) / np.linalg.norm(g2):+.3f}", flush=True)


if __name__ == "__main__":
    main()
