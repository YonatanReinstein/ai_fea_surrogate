"""How usable is the surrogate's dg field for the OC update, really?

Per-tile sign agreement is the wrong summary: optimization/topopt_optimizer.py's
OC step moves along the dg VECTOR, so what matters is (a) how much of the
gradient magnitude points the wrong way and (b) whether sign errors sit on the
tiles that carry the field or on the near-zero tiles where the true gradient is
noise-scale anyway.

Reports, per checkpoint:
  cosine(pred, true) per sample -- the direction OC actually steps along
  sign agreement stratified by |true dg| decile
  fraction of total sum|true dg| carried by sign-wrong tiles  <- the verdict
"""
import argparse, importlib, torch
from torch_geometric.data import Batch
from training.dataset_io import load_dataset
from training.gnn_training import gnn_input_fn, LOG_STRESS_EPS
from core.tile_sensitivity import dims_key_order_permutation
from evaluators.probe_gradloss_sensitivity import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True, help="name=path pairs")
ap.add_argument("--dataset", required=True)
ap.add_argument("--geometry", default="hollow_cube")
ap.add_argument("--limit", type=int, default=120)
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mod = importlib.import_module(f"data.{a.geometry}.boundary_conditions")
soff = 0 if getattr(mod, "fixed_dims", lambda: {})() else 3
ds = load_dataset(a.dataset)
s0 = ds[0]
NX, NY, NZ = int(s0.tile_NX), int(s0.tile_NY), int(s0.tile_NZ)
K = NX * NY * NZ
perm = torch.tensor(dims_key_order_permutation(NX, NY, NZ), dtype=torch.long, device=dev)
perm_inv = torch.argsort(perm)


def field(model, stats):
    x_mean, x_std, e_mean, e_std, nt_mean, nt_std = stats
    P, T = [], []
    for i, s in enumerate(ds):
        if i >= a.limit:
            break
        b = Batch.from_data_list([s])
        x, ei, ea, bidx, tidx, tNX, tNY, tNZ, tx, nid = gnn_input_fn(b)
        t_leaf = s.dims[0, soff:soff + K].float().to(dev)[perm].view(-1, 1).clone().requires_grad_(True)
        bidx, tidx = bidx.to(dev), tidx.to(dev)
        ns = t_leaf.view(1, K)[bidx, tidx].unsqueeze(1)
        x = x.float().to(dev); x[:, 3:6] = x[:, 3:6] / 1e6
        xn = (torch.cat([x, ns], dim=1) - x_mean) / x_std
        gp, _ = model(xn, ei.to(dev), (ea.float().to(dev) - e_mean) / e_std, bidx,
                      tile_idx=tidx, tile_NX=tNX, tile_NY=tNY, tile_NZ=tNZ,
                      tile_x=t_leaf, num_graphs=1,
                      node_id=nid.to(dev) if nid is not None else None)
        gpm = (gp * nt_std + nt_mean).clamp_min(LOG_STRESS_EPS)
        d = torch.autograd.grad(gpm.sum(), t_leaf)[0].view(-1)
        t = s.grad_dks_dx.float().to(dev)
        if not hasattr(s, "ks_stress_adjoint"):
            t = t / 1e6                      # fd builder stores Pa/unit
        P.append(d[perm_inv[s.grad_tile_idx.to(dev)]].detach().cpu())
        T.append(t.cpu())
    return torch.stack(P), torch.stack(T)     # [S, K]


def report(name, P, T):
    cos = torch.nn.functional.cosine_similarity(P, T, dim=1)
    p_, t_ = P.reshape(-1), T.reshape(-1)
    ok = torch.sign(p_) == torch.sign(t_)
    mass_wrong = t_.abs()[~ok].sum() / t_.abs().sum()
    print(f"\n=== {name}  (S={P.shape[0]} samples x {K} tiles) ===")
    print(f"  cosine(pred,true) per sample: median {cos.median():+.3f}   "
          f"mean {cos.mean():+.3f}   p10 {cos.quantile(0.1):+.3f}   p90 {cos.quantile(0.9):+.3f}")
    print(f"  overall sign agreement:       {ok.float().mean()*100:5.1f}%")
    print(f"  >> fraction of total |dg| MASS pointing the wrong way: {mass_wrong*100:5.1f}% <<")
    q = torch.quantile(t_.abs(), torch.linspace(0, 1, 11))
    print("  sign agreement by |true dg| decile (low -> high):")
    for i in range(10):
        lo, hi = q[i], q[i + 1]
        m = (t_.abs() >= lo) & ((t_.abs() <= hi) if i == 9 else (t_.abs() < hi))
        if m.sum() == 0:
            continue
        print(f"    D{i+1:<2d} |dg| {lo:9.2f}-{hi:9.2f}   sign {ok[m].float().mean()*100:5.1f}%   "
              f"({t_.abs()[m].sum()/t_.abs().sum()*100:4.1f}% of mass)")


# ---------------------------------------------------------------- control
# A cosine near +0.6 on this data is NOT a good score -- it is the FLOOR.
# The true dg fields are strongly correlated across designs (every design
# wants to thin roughly the same low-stress regions), so a model that ignores
# its input entirely and always emits the mean direction already scores a
# median cosine of ~+0.61 on hollow_cube. Report it first, every time, so a
# checkpoint is judged by its MARGIN over trivial rather than against zero.
#
# This is not hypothetical: measured 2026-09-07, the pre-sobolev baseline
# (+0.022) and sobolev run 1 (+0.051) both scored BELOW this control on every
# metric, and job 1373261 fell below it too once it overfit its 4,100-sample
# grad pool (epoch 15: +0.062, 20.7% wrong-way mass vs the control's 12.4%).
_T_all = torch.stack([(s.grad_dks_dx.float() if hasattr(s, "ks_stress_adjoint")
                       else s.grad_dks_dx.float() / 1e6)
                      for s in ds[:a.limit]])
_mean_dir = torch.nn.functional.normalize(
    torch.nn.functional.normalize(_T_all, dim=1).mean(0, keepdim=True), dim=1)
report("CONSTANT-DIRECTION CONTROL (no input; beat this)",
       _mean_dir.expand_as(_T_all).contiguous(), _T_all)

for spec in a.ckpts:
    name, path = spec.split("=", 1)
    model, stats = load_model(path, a.geometry, dev)
    P, T = field(model, stats)
    report(name, P, T)
