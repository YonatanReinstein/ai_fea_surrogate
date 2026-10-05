"""The 30 screened hollow_cube designs (40x21 tiles). OFFLINE, read-only."""
import torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core.tile_sensitivity import dims_key_order_permutation
SURF,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e1e0d9"
d=torch.load('data/hollow_cube/dataset_screened30/shards/shard_000000_000029.pt', weights_only=False)
X=np.array([s['dims'].numpy().ravel().astype(float) for s in d])
KS=np.array([float(s['ks_stress'])/1e6 for s in d])
C=np.array([x.max()/max(x.min(),1e-9) for x in X])
NX,NY,NZ=int(d[0]['tile_NX']),int(d[0]['tile_NY']),int(d[0]['tile_NZ'])
perm=np.asarray(dims_key_order_permutation(NX,NY,NZ))
g=lambda v: np.asarray(v,float)[perm].reshape(NZ,NY,NX)[NZ//2]
order=np.argsort(KS)                     # ascending stress, so the band reads left->right
plt.rcParams.update({"font.family":"DejaVu Sans"})
fig,axes=plt.subplots(6,5,figsize=(15.0,9.6),facecolor=SURF)
for ax,k in zip(axes.ravel(), order):
    im=ax.imshow(g(X[k]),origin="lower",cmap="Blues",vmin=0.01,vmax=0.47,
                 interpolation="nearest",aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.set_title(f"{KS[k]:.0f} MPa   c {C[k]:.1f}",fontsize=8.4,color=INK2,pad=3)
cb=fig.colorbar(im,ax=axes,fraction=0.013,pad=0.012)
cb.set_label("tile thickness",color=INK2,fontsize=9); cb.ax.tick_params(colors=INK2,length=0)
cb.outline.set_edgecolor(GRID)
fig.suptitle("All 30 screened hollow_cube designs (40 × 21 = 840 tiles), ordered by stress",
             fontsize=13,fontweight="bold",color=INK,x=0.045,ha="left",y=0.975)
fig.text(0.045,0.948,f"KS {KS.min():.0f}–{KS.max():.0f} MPa, all inside the 50–200 band · contrast median {np.median(C):.1f} "
         f"(raw 50k: 1.9) · anchored edge left, load right · light = thin (0.01), dark = thick (0.47)",
         fontsize=9.5,color=INK2,ha="left")
fig.savefig("docs/runs/figs/hc_screened30_designs.png",dpi=150,facecolor=SURF,bbox_inches="tight")
print(f"saved docs/runs/figs/hc_screened30_designs.png  | grid {NX}x{NY}x{NZ}")
print(f"stress {KS.min():.1f}-{KS.max():.1f} MPa, contrast {C.min():.1f}-{C.max():.1f}")
