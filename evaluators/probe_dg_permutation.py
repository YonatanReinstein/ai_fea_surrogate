"""Disambiguate whether the probe/training tile permutation is the right one.

If dg sign agreement is ~coin-flip under the mapping training uses but jumps
under a different pairing, the permutation direction is wrong (this repo has
had two prior tile-index transpose bugs). If ALL pairings are ~50%, the
permutation is fine and the model genuinely does not predict dg.
"""
import argparse, importlib, torch
from torch_geometric.data import Batch
from training.dataset_io import load_dataset
from training.gnn_training import gnn_input_fn, LOG_STRESS_EPS
from core.tile_sensitivity import dims_key_order_permutation
from evaluators.probe_gradloss_sensitivity import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--geometry", default="hollow_cube")
ap.add_argument("--limit", type=int, default=60)
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mod = importlib.import_module(f"data.{a.geometry}.boundary_conditions")
fixed = getattr(mod, "fixed_dims", lambda: {})()
soff = 0 if fixed else 3

ds = load_dataset(a.dataset)
model, stats = load_model(a.ckpt, a.geometry, dev)
x_mean, x_std, edge_mean, edge_std, nt_mean, nt_std = stats

s0 = ds[0]
NX, NY, NZ = int(s0.tile_NX), int(s0.tile_NY), int(s0.tile_NZ)
perm = torch.tensor(dims_key_order_permutation(NX, NY, NZ), dtype=torch.long, device=dev)
perm_inv = torch.argsort(perm)
gti = s0.grad_tile_idx
print(f"grad_tile_idx == arange: {bool((gti == torch.arange(gti.numel())).all())}  K={NX*NY*NZ}")

modes = {"training/probe (perm_inv)": perm_inv, "raw (no perm)": torch.arange(NX*NY*NZ, device=dev), "fwd (perm)": perm}
acc = {k: [] for k in modes}
tru = []
for i, sample in enumerate(ds):
    if i >= a.limit: break
    batch = Batch.from_data_list([sample])
    x, ei, ea, bidx, tidx, tNX, tNY, tNZ, tx, nid = gnn_input_fn(batch)
    K = NX*NY*NZ
    tsd = sample.dims[0, soff:soff+K].float().to(dev)
    t_leaf = tsd[perm].view(-1,1).clone().requires_grad_(True)
    bidx = bidx.to(dev); tidx = tidx.to(dev)
    ns = t_leaf.view(1,K)[bidx, tidx].unsqueeze(1)
    x = x.float().to(dev); x[:,3:6] = x[:,3:6]/1e6
    xn = (torch.cat([x, ns],dim=1) - x_mean)/x_std
    ean = (ea.float().to(dev) - edge_mean)/edge_std
    gp,_ = model(xn, ei.to(dev), ean, bidx, tile_idx=tidx, tile_NX=tNX, tile_NY=tNY,
                 tile_NZ=tNZ, tile_x=t_leaf, num_graphs=1,
                 node_id=nid.to(dev) if nid is not None else None)
    gpm = (gp*nt_std + nt_mean).clamp_min(LOG_STRESS_EPS)
    dpred = torch.autograd.grad(gpm.sum(), t_leaf)[0].view(-1)
    t = sample.grad_dks_dx.float().to(dev)
    if not hasattr(sample, "ks_stress_adjoint"):
        t = t/1e6
    tru.append(t.cpu())
    g = sample.grad_tile_idx.to(dev)
    for k, m in modes.items():
        acc[k].append(dpred[m[g]].detach().cpu())

tru = torch.cat(tru)
print(f"\nN={tru.numel()} pairs, mean|true|={tru.abs().mean():.4f} MPa/unit")
for k in modes:
    p = torch.cat(acc[k])
    sign = (torch.sign(p)==torch.sign(tru)).float().mean().item()*100
    med = ((p.abs()+1e-8)/(tru.abs()+1e-8)).log10().abs().median().item()
    corr = torch.corrcoef(torch.stack([p, tru]))[0,1].item()
    print(f"  {k:28s} sign={sign:5.1f}%  magerr=10^{med:.2f}  pearson={corr:+.3f}")
