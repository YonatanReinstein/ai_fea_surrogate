"""Audit the screened 20k by provenance segment. OFFLINE, read-only."""
import torch, numpy as np, glob, re, os
D='data/hollow_cube_11x11/dataset_screened30k/shards'
SEGS=[(0,9999,"A  0-9999      (job 195036)"),
      (10000,18699,"B  10000-18699  (UNKNOWN source)"),
      (18700,19999,"C  18700-19999  (job 195112)"),(20000,29999,"D  20000-29999  (job 198761)")]
files=sorted(glob.glob(D+'/*.pt'))
def seg_of(p):
    s,e=map(int,re.search(r'shard_(\d+)_(\d+)',p).groups())
    for lo,hi,lab in SEGS:
        if s>=lo and e<=hi: return lab
    return "SPANS A BOUNDARY"
buckets={lab:[] for _,_,lab in SEGS}; buckets["SPANS A BOUNDARY"]=[]
# subsample within each segment: every 3rd shard is plenty
by={}
for p in files: by.setdefault(seg_of(p),[]).append(p)
print(f"{'segment':<32}{'shards':>7}{'n':>7}{'KS med':>9}{'in band':>9}{'contrast':>10}{'pairwise':>10}{'effrank':>9}")
print("-"*93)
rng=np.random.default_rng(0)
for lab in [s[2] for s in SEGS]+["SPANS A BOUNDARY"]:
    ps=by.get(lab,[])
    if not ps: continue
    X,KS=[],[]
    for p in ps[::max(1,len(ps)//12)]:
        for s in torch.load(p, weights_only=False):
            X.append(s['dims'].numpy().ravel().astype(float)); KS.append(float(s['ks_stress'])/1e6)
    X=np.array(X); KS=np.array(KS)
    C=np.array([d.max()/max(d.min(),1e-9) for d in X])
    m=min(150,len(X)); pw=[];er=[]
    for _ in range(15):
        S=(X[rng.choice(len(X),m,replace=False)]-0.01)/0.46
        i=rng.choice(m,(500,2)); i=i[i[:,0]!=i[:,1]]
        pw.append(np.abs(S[i[:,0]]-S[i[:,1]]).mean())
        Z=S-S.mean(0); ev=np.linalg.svd(Z,compute_uv=False)**2
        er.append((ev.sum()**2)/(ev**2).sum())
    print(f"{lab:<32}{len(ps):>7}{len(X):>7}{np.median(KS):>9.0f}"
          f"{100*np.mean((KS>50)&(KS<200)):>8.0f}%{np.median(C):>10.1f}{np.mean(pw):>10.4f}{np.mean(er):>9.2f}")
    buckets[lab]=KS
print()
segs=[buckets[s[2]] for s in SEGS]
names=[s[2].split()[0] for s in SEGS]
from scipy.stats import ks_2samp
import itertools
pairs=[(segs[i],segs[j],f"{names[i]} vs {names[j]}") for i,j in itertools.combinations(range(len(segs)),2)]
for x,y,lab in pairs:
    st,p=ks_2samp(x,y)
    print(f"KS-test {lab}: statistic {st:.3f}  p={p:.3g}  -> {'SAME distribution' if p>0.01 else 'DIFFERENT'}")
