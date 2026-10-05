"""Simulate two-criterion cut yields on hollow_cube's raw 50k. Read-only, no FEA."""
import torch, numpy as np, glob, sys
sys.path.insert(0,'.')
from training import dataset_io
fs = dataset_io.shard_paths('data/hollow_cube/dataset_gradadj')
N_TOT = dataset_io.num_existing_samples('data/hollow_cube/dataset_gradadj')
MB = 6.58
sel = fs[::50]
print(f"source: {len(fs)} shards / {N_TOT} samples; sampling {len(sel)} shards", flush=True)
X, KS = [], []
for p in sel:
    for s in dataset_io.load_shard(p):
        X.append(s.dims.numpy().ravel().astype(float))
        KS.append(float(np.asarray(s.ks_stress).ravel()[0])/1e6)
X = np.array(X); KS = np.array(KS)
C = np.array([d.max()/max(d.min(),1e-9) for d in X])
rng = np.random.default_rng(0)
print(f"sampled n={len(KS)}  KS median {np.median(KS):.0f}  contrast median {np.median(C):.1f}")
LO, HI, T = 40.0, 220.0, 100.0
band = (KS >= LO) & (KS <= HI)
print(f"hard band {LO:.0f}-{HI:.0f} MPa alone: {100*band.mean():.1f}%  -> ~{int(band.mean()*N_TOT)} samples, ~{band.mean()*N_TOT*MB/1024:.0f} GB\n")
def div(D, m=120, reps=12):
    if len(D) < m: return np.nan, np.nan
    pw, er = [], []
    for _ in range(reps):
        S = (D[rng.choice(len(D), m, replace=False)] - 0.01)/0.46
        i = rng.choice(m, (400,2)); i = i[i[:,0]!=i[:,1]]
        pw.append(np.abs(S[i[:,0]]-S[i[:,1]]).mean())
        Z = S - S.mean(0); ev = np.linalg.svd(Z, compute_uv=False)**2
        er.append((ev.sum()**2)/(ev**2).sum())
    return np.mean(pw), np.mean(er)
h = f"{'sigma_e':>8}{'alpha':>7}{'accept':>8}{'est n':>8}{'est GB':>8}{'KS med':>8}{'contrast':>10}{'pairwise':>10}{'effrank':>9}"
print(h); print("-"*len(h))
pw0, er0 = div(X)
print(f"{'raw (no cut)':>15}{100.0:>8.1f}{N_TOT:>8d}{N_TOT*MB/1024:>8.0f}{np.median(KS):>8.0f}{np.median(C):>10.1f}{pw0:>10.4f}{er0:>9.2f}")
for se in (40., 60., 80.):
    for al in (0.0, 0.5, 1.0, 1.5):
        W = np.exp(-0.5*((KS-T)/se)**2) * (C**al) * band     # band is a hard pre-filter
        if W.max() <= 0: continue
        m = rng.random(len(KS)) < W/W.max()
        if m.sum() < 130:
            print(f"{se:>8.0f}{al:>7.1f}{100*m.mean():>8.2f}{int(m.mean()*N_TOT):>8d}"
                  f"{m.mean()*N_TOT*MB/1024:>8.0f}{'':>8}{'  too few to score':>10}"); continue
        pw, er = div(X[m])
        print(f"{se:>8.0f}{al:>7.1f}{100*m.mean():>8.2f}{int(m.mean()*N_TOT):>8d}"
              f"{m.mean()*N_TOT*MB/1024:>8.0f}{np.median(KS[m]):>8.0f}{np.median(C[m]):>10.1f}{pw:>10.4f}{er:>9.2f}")
