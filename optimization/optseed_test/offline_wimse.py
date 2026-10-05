"""OFFLINE ONLY -- reads the built dataset, writes nothing into the pipeline.

Tests the Picheny et al. weighted-IMSE idea as a SELECTION rule rather than a
deformation: keep each design untouched, accept it with probability prop. to
    W = exp( -0.5 (s - T)^2 / (sigma_e^2 + s_err^2) )
and measure what that does to stress, contrast and between-design diversity.
"""
import torch, numpy as np, glob, json
T = 100.0
fs = sorted(glob.glob('data/hollow_cube_11x11/dataset_gradadj/shards/*.pt'))[::12]
X, KS = [], []
for f in fs:
    for s in torch.load(f, weights_only=False):
        X.append(s['dims'].numpy().ravel().astype(float)); KS.append(float(s['ks_stress'])/1e6)
X = np.array(X); KS = np.array(KS)
con = lambda D: np.array([d.max()/max(d.min(),1e-9) for d in D])
C = con(X)
print(f"candidate pool: {len(X)} designs from the built 30k (true ccx stress, no model error)\n")

rng = np.random.default_rng(0)
def diversity(D, n=250, reps=12):
    """pairwise distance + effective rank at FIXED n, so groups are comparable."""
    pw, er = [], []
    for _ in range(reps):
        if len(D) < n: return np.nan, np.nan
        S = (D[rng.choice(len(D), n, replace=False)] - 0.01)/0.46
        idx = rng.choice(n, (400,2))
        idx = idx[idx[:,0]!=idx[:,1]]
        pw.append(np.abs(S[idx[:,0]]-S[idx[:,1]]).mean())
        Z = S - S.mean(0); ev = np.linalg.svd(Z, compute_uv=False)**2
        er.append((ev.sum()**2)/(ev**2).sum())
    return float(np.mean(pw)), float(np.mean(er))

def report(mask, lab):
    D, s, c = X[mask], KS[mask], C[mask]
    if len(D) < 260: print(f"{lab:28s} only {len(D)} accepted -- too few"); return
    pw, er = diversity(D)
    print(f"{lab:28s}{100*mask.mean():6.1f}%{np.median(s):9.0f}"
          f"{100*np.mean((s>50)&(s<200)):9.0f}%{np.median(c):10.1f}{pw:11.4f}{er:10.2f}")

print(f"{'':28s}{'accept':>6}{'KS med':>9}{'in 50-200':>10}{'contrast':>10}{'pairwise':>11}{'eff.rank':>10}")
print("-"*84)
report(np.ones(len(X), bool), "raw sampler (all)")
for se in (10., 20., 40., 80.):
    W = np.exp(-0.5*((KS-T)/se)**2)
    report(rng.random(len(X)) < W/W.max(), f"W-accept, sigma_e={se:.0f} MPa")

# same rule driven by the CHEAP PREDICTOR instead of true stress -- what a real
# build would have to use, since it screens before paying for the solve
co = np.array([3.346, -0.819, 0.235])
pred = np.exp(co[0] + co[1]*np.log(np.median(X,1)) + co[2]*np.log(C))
print(f"\npredictor check: corr(log pred, log true) = {np.corrcoef(np.log(pred),np.log(KS))[0,1]:.3f}")
print("-"*84)
for se in (20., 40.):
    W = np.exp(-0.5*((pred-T)/se)**2)
    m = rng.random(len(X)) < W/W.max()
    report(m, f"predictor-screened, s_e={se:.0f}")
    print(f"{'':28s}  -> of those accepted, TRUE stress in 50-200: {100*np.mean((KS[m]>50)&(KS[m]<200)):.0f}%")
