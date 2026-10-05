"""24 samples from the screened 10k, spanning its stress range. OFFLINE."""
import torch, numpy as np, glob, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core.tile_sensitivity import dims_key_order_permutation
SURF,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e1e0d9"
X,KS=[],[]
for f in sorted(glob.glob('data/hollow_cube_11x11/dataset_screened10k/shards/*.pt'))[::5]:
    for s in torch.load(f, weights_only=False):
        X.append(s['dims'].numpy().ravel().astype(float)); KS.append(float(s['ks_stress'])/1e6)
X=np.array(X); KS=np.array(KS)
C=np.array([d.max()/max(d.min(),1e-9) for d in X])
NX=NY=11;NZ=1; perm=np.asarray(dims_key_order_permutation(NX,NY,NZ))
g=lambda v: np.asarray(v,float)[perm].reshape(NZ,NY,NX)[0]
# order by stress and take 24 evenly spaced, so the grid spans the whole band
idx=np.argsort(KS)[np.linspace(0,len(KS)-1,24).astype(int)]
plt.rcParams.update({"font.family":"DejaVu Sans"})
fig,axes=plt.subplots(4,6,figsize=(13.2,9.4),facecolor=SURF)
for ax,k in zip(axes.ravel(), idx):
    im=ax.imshow(g(X[k]),origin="lower",cmap="Blues",vmin=0.01,vmax=0.47,interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.set_title(f"{KS[k]:.0f} MPa   contrast {C[k]:.1f}",fontsize=8.6,color=INK2,pad=4)
cb=fig.colorbar(im,ax=axes,fraction=0.016,pad=0.013)
cb.set_label("tile thickness",color=INK2,fontsize=9); cb.ax.tick_params(colors=INK2,length=0)
cb.outline.set_edgecolor(GRID)
fig.suptitle("24 samples from the screened 10k, ordered by stress across the 50–200 MPa band",
             fontsize=13,fontweight="bold",color=INK,x=0.05,ha="left",y=0.97)
fig.text(0.05,0.943,f"n={len(KS)} sampled from the set · stress median {np.median(KS):.0f} MPa, "
         f"contrast median {np.median(C):.1f} · light = thin (0.01), dark = thick (0.47)",
         fontsize=9.5,color=INK2,ha="left")
fig.savefig("docs/runs/figs/t11_screened10k_24samples.png",dpi=150,facecolor=SURF,bbox_inches="tight")
print(f"saved. pool n={len(KS)}; shown stress {KS[idx].min():.0f}-{KS[idx].max():.0f} MPa, "
      f"contrast {C[idx].min():.1f}-{C[idx].max():.1f}")
