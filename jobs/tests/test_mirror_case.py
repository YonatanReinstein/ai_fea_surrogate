"""Physics check of training.dsig_data.mirror_case: on the uniform design (its
own mirror image) the mirror of the row-r case must equal the row-(NY-1-r) case
exactly, up to solver round-off -- S, dS, load weights."""
import importlib
import json

import numpy as np
import torch

from training.dsig_data import DsigStatic, mirror_case, mirror_maps


def main():
    root = "data/cross4_11x11/dataset_dsig_2500_tmin003"
    s = DsigStatic(root)
    tmir, nmir = mirror_maps(s)
    from training.build_cross4_dsig_dataset import solve_design
    bc = importlib.import_module("data.cross4_11x11.boundary_conditions")
    keys = list(json.load(open("data/cross4_11x11/CAD_model/dims.json")))
    rows = [1, 3, 7, 9]
    out, _ = solve_design({k: 0.2 for k in keys}, bc, rows, float(s.cfg["load_N"]), 0.25, workers=8)
    X = out["coords"]
    t = torch.full((s.K,), 0.2)
    rel = lambda a, b: float((a - b).norm() / b.norm())
    for a, b in ((0, 3), (1, 2)):
        ra, rb = rows[a], rows[b]
        t2, lw2, S2, dS2, r2 = mirror_case(t, torch.as_tensor(s.load_weights(X, ra)), torch.as_tensor(out["S"][a]),
                                           torch.as_tensor(out["dS"][a]), ra, tmir, nmir, s.NY)
        assert r2 == rb
        print(f"row {ra} mirrored vs row {rb}: S rel {rel(S2, torch.as_tensor(out['S'][b])):.2e}, "
              f"dS rel {rel(dS2, torch.as_tensor(out['dS'][b])):.2e}, "
              f"load_w rel {rel(lw2, torch.as_tensor(s.load_weights(X, rb))):.2e}; "
              f"dks: {rel(torch.as_tensor(out['dks'][a])[tmir], torch.as_tensor(out['dks'][b])):.2e}", flush=True)
    # worst nodes, to see whether any mapping is wrong
    S2 = mirror_case(t, torch.as_tensor(s.load_weights(X, 3)), torch.as_tensor(out["S"][1]),
                     torch.as_tensor(out["dS"][1]), 3, tmir, nmir, s.NY)[2]
    e = (S2 - torch.as_tensor(out["S"][2])).norm(dim=1) / torch.as_tensor(out["S"][2]).norm(dim=1).mean()
    print("per-node S error / mean |S|: max", float(e.max()), " count > 1e-3:", int((e > 1e-3).sum()), "of", s.N)


if __name__ == "__main__":
    main()
