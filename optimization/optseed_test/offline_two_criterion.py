"""OFFLINE ONLY -- reads the built dataset, writes nothing into the pipeline.

Two-criterion screen: accept prop. to  W_stress(s) * contrast^alpha
  W_stress = exp(-0.5 ((s-T)/sigma_e)^2)        Picheny et al. Eqn 25, s_K -> 0
  contrast^alpha                                 pulls toward member/void structure
alpha = 0 reduces to the plain W-acceptance already tested.
"""
import torch, numpy as np, glob
T = 100.0
fs = sorted(glob.glob('data/hollow_cube_11x11/dataset_gradadj/shards/*.pt'))[::6]
X, KS = [], []
for f in fs:
    for s in torch.load(f, weights_only=False):
        X.append(s['dims'].numpy().ravel().astype(float)); KS.append(float(s['ks_stress'])/1e6)
X = np.array(X); KS = np.array(KS)
C = np.array([d.max()/max(d.min(),1e-9) for d in X])
n = len(X)
rng = np.random.default_rng(0)
print(f"pool: {n} designs (true ccx stress)\n")

def diversity(D, m=120, reps=25):
    if len(D) < m: return np.nan, np.nan
    pw, er = [], []
    for _ in range(reps):
        S = (D[rng.choice(len(D), m, replace=False)] - 0.01)/0.46
        i = rng.choice(m, (500,2)); i = i[i[:,0]!=i[:,1]]
        pw.append(np.abs(S[i[:,0]]-S[i[:,1]]).mean())
        Z = S - S.mean(0); ev = np.linalg.svd(Z, compute_uv=False)**2
        er.append((ev.sum()**2)/(ev**2).sum())
    return float(np.mean(pw)), float(np.mean(er))

def run(s_est, se, alpha, lab):
    W = np.exp(-0.5*((s_est-T)/se)**2) * (C**alpha)
    m = rng.random(n) < W/W.max()
    if m.sum() < 130:
        print(f"{lab:30s}{100*m.mean():7.2f}%  only {m.sum()} accepted -- too few to score"); return
    pw, er = diversity(X[m])
    print(f"{lab:30s}{100*m.mean():7.2f}%{np.median(KS[m]):9.0f}"
          f"{100*np.mean((KS[m]>50)&(KS[m]<200)):8.0f}%{np.median(C[m]):10.1f}"
          f"{pw:10.4f}{er:9.2f}{1/max(m.mean(),1e-9):11.0f}")

hdr = f"{'':30s}{'accept':>7}{'KS med':>9}{'in band':>9}{'contrast':>10}{'pairwise':>10}{'effrank':>9}{'cand/kept':>11}"
print(hdr); print("-"*len(hdr))
run(KS, 1e9, 0.0, "raw sampler (no screen)")
print()
for se in (40., 80.):
    for a in (0.0, 0.5, 1.0, 1.5):
        run(KS, se, a, f"true stress  se={se:.0f} alpha={a}")
    print()

# what a real pre-solve screen would have: predicted stress, EXACT contrast (free)
co = np.array([3.346, -0.819, 0.235])
pred = np.exp(co[0] + co[1]*np.log(np.median(X,1)) + co[2]*np.log(C))
print(f"predictor corr(log) = {np.corrcoef(np.log(pred), np.log(KS))[0,1]:.3f}   (contrast is exact, no model)")
print("-"*len(hdr))
for se in (40., 80.):
    for a in (1.0, 1.5):
        run(pred, se, a, f"PREDICTED   se={se:.0f} alpha={a}")
