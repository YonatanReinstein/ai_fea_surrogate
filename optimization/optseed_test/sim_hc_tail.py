"""Does hollow_cube's raw 50k contain ANY high-contrast designs near yield?"""
import numpy as np, sys
sys.path.insert(0,'.')
from training import dataset_io
fs = dataset_io.shard_paths('data/hollow_cube/dataset_gradadj')
X, KS = [], []
for p in fs[::17]:                                  # ~30 shards / ~3000 samples
    for s in dataset_io.load_shard(p):
        X.append(s.dims.numpy().ravel().astype(float))
        KS.append(float(np.asarray(s.ks_stress).ravel()[0])/1e6)
X = np.array(X); KS = np.array(KS)
C = np.array([d.max()/max(d.min(),1e-9) for d in X])
N_TOT = 50001; MB = 6.58
print(f"n={len(KS)} sampled from {N_TOT}\n")
print("contrast distribution of the SOURCE:")
for q in (50,75,90,95,99,100):
    print(f"   p{q:<4} {np.percentile(C,q):7.2f}")
print(f"\n{'contrast >=':>12}{'in 40-220':>11}{'count':>7}{'% of pool':>11}{'est n in 50k':>14}{'est GB':>8}")
print("-"*64)
band = (KS>=40)&(KS<=220)
for cm in (0,3,5,8,12,20,30):
    m = band & (C>=cm)
    print(f"{cm:>12}{'yes':>11}{m.sum():>7}{100*m.mean():>10.2f}%{int(m.mean()*N_TOT):>14}{m.mean()*N_TOT*MB/1024:>8.0f}")
rng=np.random.default_rng(0)
best = band & (C>=np.percentile(C[band],95))
if best.sum()>20:
    D=(X[best]-0.01)/0.46; k=len(D)
    i=rng.choice(k,(600,2)); i=i[i[:,0]!=i[:,1]]
    Z=D-D.mean(0); ev=np.linalg.svd(Z,compute_uv=False)**2
    print(f"\ntop-5% contrast inside the band: n={best.sum()}  contrast median {np.median(C[best]):.1f}  "
          f"KS median {np.median(KS[best]):.0f}  pairwise {np.abs(D[i[:,0]]-D[i[:,1]]).mean():.4f}  "
          f"effrank {(ev.sum()**2)/(ev**2).sum():.2f}")
