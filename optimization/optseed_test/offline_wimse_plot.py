"""OFFLINE ONLY -- renders what W-weighted selection actually keeps."""
import torch, numpy as np, glob, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core.tile_sensitivity import dims_key_order_permutation
SURF,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e1e0d9"
ORANGE,BLUE,AQUA,VIOLET="#eb6834","#2a78d6","#1baf7a","#4a3aa7"
T=100.0
fs=sorted(glob.glob('data/hollow_cube_11x11/dataset_gradadj/shards/*.pt'))[::12]
X,KS=[],[]
for f in fs:
    for s in torch.load(f,weights_only=False):
        X.append(s['dims'].numpy().ravel().astype(float)); KS.append(float(s['ks_stress'])/1e6)
X=np.array(X); KS=np.array(KS)
C=np.array([d.max()/max(d.min(),1e-9) for d in X])
NX=NY=11;NZ=1; perm=np.asarray(dims_key_order_permutation(NX,NY,NZ))
g=lambda v: np.asarray(v,float)[perm].reshape(NZ,NY,NX)[0]
rng=np.random.default_rng(3)
def pick(mask,n=12):
    idx=np.flatnonzero(mask); return rng.choice(idx,min(n,len(idx)),replace=False)
rows=[("raw sampler\n(all 30k)", pick(np.ones(len(X),bool)), ORANGE)]
for se,col in ((20.,BLUE),(80.,AQUA)):
    W=np.exp(-0.5*((KS-T)/se)**2)
    rows.append((f"W-accepted\nσε = {se:.0f} MPa", pick(rng.random(len(X))<W/W.max()), col))
# and the highest-contrast designs the pool contains near target, for contrast
near=(KS>70)&(KS<140)
top=np.flatnonzero(near)[np.argsort(C[near])[::-1][:12]]
W2=np.exp(-0.5*((KS-T)/40.)**2)*C
rows.append((f"two-criterion screen\n\u03c3\u03b5=40, \u03b1=1", pick(rng.random(len(X))<W2/W2.max()), VIOLET))
plt.rcParams.update({"font.family":"DejaVu Sans"})
n=12
fig,axes=plt.subplots(len(rows),n,figsize=(1.28*n+2.2,1.32*len(rows)+1.5),facecolor=SURF)
for ri,(lab,idx,col) in enumerate(rows):
    for j in range(n):
        a=axes[ri,j]
        if j<len(idx):
            k=idx[j]
            im=a.imshow(g(X[k]),origin="lower",cmap="Blues",vmin=0.01,vmax=0.47,interpolation="nearest")
            a.set_title(f"{KS[k]:.0f} MPa\nc {C[k]:.1f}",fontsize=7,color=INK2,pad=2.5)
        a.set_xticks([]);a.set_yticks([])
        for sp in a.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(1.2)
    axes[ri,0].set_ylabel(lab,fontsize=8.4,color=col,fontweight="bold",labelpad=8)
cb=fig.colorbar(im,ax=axes,fraction=0.011,pad=0.011)
cb.set_label("tile thickness",color=INK2,fontsize=8.5); cb.ax.tick_params(colors=INK2,length=0)
cb.outline.set_edgecolor(GRID)
fig.suptitle("Screening on stress ALONE keeps the bland designs; adding a contrast term recovers real structure",
             fontsize=12.5,fontweight="bold",color=INK,x=0.035,ha="left",y=1.015)
fig.text(0.035,0.975,"contrast median over the full accepted set: raw 8.6  ->  stress-only 1.9-2.3  ->  two-criterion (W x contrast) 21.6, with effective rank rising 2.3 -> 6.5. Twelve shown per row are a random draw, so single panels vary.",fontsize=8.8,color=INK2,ha="left")
fig.savefig("docs/runs/figs/t11_two_criterion_screen.png",dpi=150,facecolor=SURF,bbox_inches="tight")
print("saved docs/runs/figs/t11_two_criterion_screen.png")

