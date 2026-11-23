"""Load-linear dsig model: the prediction is exactly linear in the load
(superposition and homogeneity, for every row and random loads) at random
weights with non-trivial geometry gates, and dS/dt flows through the geometry.

    python -m tests.test_load_linear
"""
import json
import os
import numpy as np
import torch

from training.dsig_data import DsigStatic
from training.train_dsig import Batcher, build_model


def main():
    torch.manual_seed(0)
    torch.backends.cudnn.allow_tf32 = False       # TF32 convs round at ~1e-4; linearity is exact up to fp32
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = DsigStatic("data/cross4_11x11/dataset_dsig_2500_tmin003")
    st = {"y_std": torch.rand(6) + 0.5, "y_mean": torch.zeros(6), "t_mean": torch.tensor(0.2),
          "t_std": torch.tensor(0.1), "lw_scale": torch.tensor(0.02),
          "ref_mean": torch.as_tensor(s.ref_coords).float().mean(0), "ref_std": torch.as_tensor(s.ref_coords).float().std(0),
          "edge_mean": s.edge_attr.mean(0), "edge_std": s.edge_attr.std(0).clamp_min(1e-12)}
    st = {k: v.to(dev) for k, v in st.items()}
    bt = Batcher(s, st, dev)
    model = build_model(dict(arch="load_linear", node_in_dim=5, edge_in_dim=4, hidden_dim=32, tile_in_dim=1,
                             num_pos_nodes=s.N, coarse_dim=32, cnn_depth=2, lin_dim=8, lin_layers=2, gate_bound=1.0, full_readout=True,
                             **json.loads(os.environ.get("LL_TEST_KW", "{}")))).to(dev)
    model.set_tile_grid(s.NX, s.NY, 1)
    model.float()
    with torch.no_grad():                       # zero-init gate heads would make the geometry dependence trivial
        for m in [model.node_gates.lins[-1], model.edge_gates.lins[-1], *model.tile_gates]:
            m.weight.normal_(0, 0.3); m.bias.normal_(0, 0.3)
    model.eval()
    B, C = 2, len(s.rows)
    t = (0.05 + 0.4 * torch.rand(B, s.K, device=dev)).requires_grad_(True)
    X = s.ref_coords
    lw = torch.as_tensor(np.array([[s.load_weights(X, r) for r in s.rows]] * B), dtype=torch.float32, device=dev)  # (B,C,N)
    S = bt.forward_rows(model, t, lw)                                                # (B,C,N,6)
    scale = S.abs().max()

    w = torch.randn(B, 4, C, device=dev)                                             # 4 random combos per design
    S_combo = bt.forward_rows(model, t, torch.einsum("bwc,bcn->bwn", w, lw))
    err = (S_combo - torch.einsum("bwc,bcnk->bwnk", w, S)).abs().max() / scale
    print(f"superposition of rows: max err {err:.2e} (rel to max |S|)")
    assert err < 1e-5

    f = torch.randn(B, 3, s.N, device=dev)                                          # arbitrary nodal loads
    S_f = bt.forward_rows(model, t, f)
    err2 = (bt.forward_rows(model, t, 2.5 * f[:, :1] - f[:, 1:2]) - (2.5 * S_f[:, :1] - S_f[:, 1:2])).abs().max() \
        / S_f.abs().max()
    print(f"arbitrary loads, a f1 + b f2: max err {err2:.2e}")
    assert err2 < 1e-5
    print(f"zero load -> max |S| {bt.forward_rows(model, t, torch.zeros_like(lw)).abs().max():.1e}")

    # per-case path (validate / online FT) == batched rows
    S_case = bt.forward(model, t[:1].expand(C, -1), lw[0], s.rows)
    err3 = (S_case - S[0]).abs().max() / scale
    print(f"per case at B={C} vs all rows of 2 designs: max err {err3:.2e} (designs must not see their batch)")
    assert err3 < 1e-5

    g, = torch.autograd.grad(bt.ks(S, 0.25).sum(), t)
    print(f"dKS/dt: norm {g.norm():.3e}, nonzero tiles {(g.abs() > 0).float().mean():.0%}")
    assert g.norm() > 0
    print("OK")


if __name__ == "__main__":
    main()
