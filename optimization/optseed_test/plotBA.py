import numpy as np, json, glob, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core.tile_sensitivity import dims_key_order_permutation
W="optimization/optseed_test"
SURF,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e1e0d9"; ORANGE,BLUE,AQUA="#eb6834","#2a78d6","#1baf7a"
R=sorted([json.load(open(f)) for f in glob.glob(W+"/B_[01][0-9].json")], key=lambda r:r["i"])
idx=[r["i"] for r in R]
O=np.load(W+"/A_orig.npy")[:12][idx]
N=np.array([r["x"] for r in R])
A=np.load(W+"/A_rescaled_100.npy")[:12][idx]
NX=NY=11;NZ=1; perm=np.asarray(dims_key_order_permutation(NX,NY,NZ))
g=lambda v: np.asarray(v,float)[perm].reshape(NZ,NY,NX)[0]
con=lambda d: d.max()/max(d.min(),1e-9)
plt.rcParams.update({"font.family":"DejaVu Sans"})
n=len(R)
fig,axes=plt.subplots(3,n,figsize=(1.32*n+1.6,5.5),facecolor=SURF)
rows=[(O,"as sampled",ORANGE,[r["s0"] for r in R]),
      (A,"option A\n(uniform scale)",BLUE,[None]*n),
      (N,"your idea\n(optimise to 100)",AQUA,[r["s"] for r in R])]
for ri,(D,lab,col,ss) in enumerate(rows):
    for j in range(n):
        a=axes[ri,j]
        im=a.imshow(g(D[j]),origin="lower",cmap="Blues",vmin=0.01,vmax=0.47,interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(1.3)
        t=f"{ss[j]:.0f} MPa\n" if ss[j] is not None else "~100 MPa\n"
        a.set_title(t+f"c {con(D[j]):.1f}",fontsize=7.2,color=INK2,pad=3)
    axes[ri,0].set_ylabel(lab,fontsize=9,color=col,fontweight="bold",labelpad=8)
cb=fig.colorbar(im,ax=axes,fraction=0.012,pad=0.012)
cb.set_label("tile thickness",color=INK2,fontsize=8.5); cb.ax.tick_params(colors=INK2,length=0)
cb.outline.set_edgecolor(GRID)
fig.suptitle("Same 12 sampled designs, driven to ~100 MPa two ways",
             fontsize=13,fontweight="bold",color=INK,x=0.045,ha="left",y=1.02)
fig.text(0.045,0.965,"uniform scaling flattens structure into the 0.47 ceiling; adjoint-guided optimisation redistributes it  "
         "(contrast median 6.2 -> 2.6 vs 6.2 -> 9.2; diversity retained 72% vs 88%)",
         fontsize=9,color=INK2,ha="left")
fig.savefig(W+"/BA_designs.png",dpi=155,facecolor=SURF,bbox_inches="tight")
print("saved", W+"/BA_designs.png")
