"""One sampled design -> drive it to TARGET MPa with adjoint-guided steps.

Contrast with option A: that scaled every tile by the same factor, so it jammed
against the 0.47 ceiling and flattened max/min. Here the step is proportional to
dG, so material lands on the tiles that actually set the stress and stays off
the ones that do not -- redistribution rather than inflation.
"""
import numpy as np, json, sys, os
sys.path.insert(0, "/home/ryonatan/ai_fea_surrogate")
from evaluators.ccx_evaluator import CCXEvaluator
SC = os.path.dirname(os.path.abspath(__file__))
IDX    = int(sys.argv[1])
TARGET, TOL, MAXIT, MOVE = 100.0, 0.03, 30, 0.30
LO, HI = 0.01, 0.47
dj = json.load(open("data/hollow_cube_11x11/CAD_model/dims.json"))
keys = sorted([k for k in dj if k[0]=="d" and k[1:].isdigit()], key=lambda k: int(k[1:]))
span = HI - LO
ev = CCXEvaluator("hollow_cube_11x11", max_workers=1, mesh_threads=2, nproc=2,
                  stress_ks_rho_mpa=0.25, gradient_source="adjoint")
con = lambda d: float(d.max()/max(d.min(),1e-9))
x = np.load(SC+"/A_orig.npy")[IDX].astype(float).copy()
x0 = x.copy(); tr=[]
for it in range(MAXIT):
    v,s,yld,dV,dG = ev.compute_topopt_gradient({k: float(t) for k,t in zip(keys,x)})
    g = np.array([dG[k] for k in keys])*yld           # ds/dx in MPa
    tr.append([it, float(s), float(v), con(x)])
    if abs(s-TARGET)/TARGET < TOL: break
    n2 = float(g@g)
    if n2 <= 0: break
    dx = ((TARGET-s)/n2)*g
    cap = MOVE*span
    x = np.clip(x + np.clip(dx,-cap,cap), LO, HI)
v,s,yld,dV,dG = ev.compute_topopt_gradient({k: float(t) for k,t in zip(keys,x)})
tr.append([len(tr), float(s), float(v), con(x)])
json.dump({"i":IDX,"its":len(tr)-1,"s0":tr[0][1],"s":float(s),"v0":tr[0][2],"v":float(v),
           "c0":con(x0),"c":con(x),"trace":tr,"x":x.tolist()}, open(f"{SC}/B_{IDX:02d}.json","w"))
print(f"[{IDX:2d}] {tr[0][1]:7.1f} -> {s:6.1f} MPa in {len(tr)-1:2d} it | "
      f"contrast {con(x0):5.1f} -> {con(x):5.1f} | vol {tr[0][2]:6.1f} -> {v:6.1f}", flush=True)
