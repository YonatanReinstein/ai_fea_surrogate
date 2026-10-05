"""Histogram of ks_stress in the screened 10k, against the raw 30k. OFFLINE."""
import torch, numpy as np, glob, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SURF,INK,INK2,GRID="#fcfcfb","#0b0b0b","#52514e","#e1e0d9"; BLUE,ORANGE="#2a78d6","#eb6834"
def load(pat, step):
    v=[]
    for f in sorted(glob.glob(pat))[::step]:
        for s in torch.load(f, weights_only=False):
            v.append(float(s['ks_stress'])/1e6)
    return np.array(v)
new = load('data/hollow_cube_11x11/dataset_screened10k/shards/*.pt', 1)   # all 100 shards
raw = load('data/hollow_cube_11x11/dataset_gradadj/shards/*.pt', 10)
plt.rcParams.update({"font.family":"DejaVu Sans"})
fig,(ax,ax2)=plt.subplots(1,2,figsize=(12.4,4.4),facecolor=SURF)
for a in (ax,ax2):
    a.set_facecolor(SURF)
    for s in ("top","right","left"): a.spines[s].set_visible(False)
    a.spines["bottom"].set_color(GRID); a.tick_params(length=0,colors=INK2,labelsize=9)
    a.grid(axis="y",color=GRID,lw=0.8); a.set_axisbelow(True)
# log axis: raw spans 60-3400 MPa
lb=np.logspace(np.log10(50),np.log10(4000),46)
ax.hist(raw,bins=lb,density=True,color=ORANGE,alpha=.72,label=f"raw 30k  (median {np.median(raw):.0f})")
ax.hist(new,bins=lb,density=True,color=BLUE,alpha=.78,label=f"screened 10k  (median {np.median(new):.0f})")
ax.axvline(100,color=INK,lw=1.5,ls="--"); ax.text(104,ax.get_ylim()[1]*.92,"100 MPa\ntarget",fontsize=8.5,color=INK)
ax.set_xscale("log"); ax.set_xlabel("KS stress (MPa), log scale",color=INK2,fontsize=9.5)
ax.set_ylabel("density",color=INK2,fontsize=9.5)
ax.legend(frameon=False,fontsize=8.8,labelcolor=INK2)
ax.set_title("Full range — the raw set's long tail is gone",fontsize=11,fontweight="bold",color=INK,loc="left")
# linear zoom on the band actually kept
ax2.hist(new,bins=np.linspace(50,200,46),color=BLUE,alpha=.85)
ax2.axvline(100,color=INK,lw=1.5,ls="--")
for q,lab in ((25,"p25"),(50,"median"),(75,"p75")):
    v=np.percentile(new,q); ax2.axvline(v,color=INK2,lw=1,ls=":")
    ax2.text(v,ax2.get_ylim()[1]*(0.97 if q==50 else 0.87),f" {lab} {v:.0f}",fontsize=8,color=INK2)
ax2.set_xlabel("KS stress (MPa)",color=INK2,fontsize=9.5); ax2.set_ylabel("designs",color=INK2,fontsize=9.5)
ax2.set_title("Screened 10k, inside the 50–200 MPa band",fontsize=11,fontweight="bold",color=INK,loc="left")
fig.suptitle("KS stress distribution: screened 10k vs the raw 30k it was screened from",
             fontsize=12.5,fontweight="bold",color=INK,x=0.045,ha="left",y=1.03)
fig.tight_layout(); fig.savefig("docs/runs/figs/t11_screened10k_ks_hist.png",dpi=160,facecolor=SURF,bbox_inches="tight")
print(f"screened 10k  n={len(new)}")
for q in (0,1,5,25,50,75,95,99,100):
    print(f"  p{q:<3d} {np.percentile(new,q):8.1f} MPa")
print(f"  in 50-200: {100*np.mean((new>50)&(new<200)):.1f}%   in 80-120: {100*np.mean((new>80)&(new<120)):.1f}%")
print(f"raw 30k median {np.median(raw):.0f}  p95 {np.percentile(raw,95):.0f}  max {raw.max():.0f}")
