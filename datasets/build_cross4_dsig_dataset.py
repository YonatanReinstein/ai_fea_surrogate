"""Stress-sensitivity dataset for cross4_11x11: every design solved with each
free-end tile as the load tile, plus the exact derivative of every node's
stress tensor with respect to every tile's thickness, for every load case.

Load cases (per design): for each free-end tile row r = 0..NY-1, FEA_LOAD_N in
-y spread over every node of that tile with the pyramid weight
w = 1 - max(|dy|, |dz|) / (tile / 2) (zero on the tile's edge; Mesh._apply_node_load,
FEA_LOAD_WEIGHT=pyramid). No load pad and no fixed load end
(CROSS4_FIXED_END_RADIUS=0, the CAD default), so the geometry -- and the
stiffness matrix -- is the same for every case: one factorization per design.

Sensitivities (direct differentiation, nodal forces held fixed):
    du_j  = -K^-1 (dK/dX[V_j]) u          V_j = node motion of tile j (dX/dt_j)
    dS_j  = (dR/dX[V_j]) u + R du_j       R  = nodal stress recovery
V_j comes from core.geometric_jacobian.compute_tile_jacobian_colored (10 extra
meshes, exact per-tile motion including shared leg ends). The pyramid weights
of the loaded nodes are held fixed through the derivative (the loaded nodes
move with the load tile's thickness); see the validation in config.json.
Stress is linear in the load, so S and dS/dt of ANY combination of the cases
are the same combination of the stored ones -- exact superposition, and an
exact dKS/dt for any combination via KS's chain rule.

Designs: sample_mix -- 20% near-uniform correlated fields, 40% high-contrast
short-correlation fields, 40% two-level (thin web + thick members), shuffled
over the indices (--mix). No stress screening or targeting.

Layout of --output_dir:
    mesh.npz            stored ONCE (identical topology for every design):
        node_ids (N,), elements (M,8) int32 (indices into node_ids),
        elem_tile (M,) int32 (IRIT tile index ix + NX*iy), anchored (N,) bool,
        pad (N,) bool (anchored-pad nodes, z <= CROSS4_PAD_THICKNESS; left
        out of ks), load_rows (C,) int32
    config.json, designs.json
    designs/NNNNN.npz   one per design:
        t (K,) float32            tile thicknesses, dims keys d1..dK
        coords (N,3) float32
        S (C,N,6) float32         nodal stress tensor, MPa, Voigt order
                                  (sxx, syy, szz, sxy, sxz, syz)
        dS (C,K,N,6) float16      dS/dt_k, MPa per unit thickness
        ks (C,) float32           KS (rho MPa^-1) of von Mises over non-pad nodes, MPa
        dks (C,K) float32         dKS/dt_k, computed in float64 before storing
        max_vm (C,) float32       max von Mises over non-pad nodes, MPa
K follows dims.json key order (d1..dK); tile index of key d{k+1} is NOT k --
use core.tile_sensitivity.tile_index_to_dims_key.

    python -m datasets.build_cross4_dsig_dataset --output_dir DIR --num_designs 2500 --claim
"""
import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

GEOMETRY = "cross4_11x11"


def dir_size_gb(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1e9


def batched_nodal_stress(sa, U):
    """Nodal stress tensors (C,N,6) for C displacement fields U (C,N,3) at once
    -- ShapeAdjoint.nodal_stress batched over load cases: each element's B,
    detJ and incompatible-mode condensation (Kaa^-1 Kua^T) are built ONCE and
    applied to every case (stress is linear in u), instead of per case."""
    from core.fem_adjoint import _elastic, _geom, _bubble_B, _btdb, gauss_to_node
    C = U.shape[0]
    Ut = torch.as_tensor(np.ascontiguousarray(U.reshape(C, -1)))
    acc = torch.zeros(C, sa.N, 6)
    cnt = torch.zeros(sa.N)
    mult = sa.e_scale if sa.stress_scale is None else sa.stress_scale
    with torch.no_grad():
        for a in range(0, len(sa.conn), sa.chunk):
            blk = sa.conn[a:a + sa.chunk]
            coords = sa.X[blk]
            D = _elastic(sa.young, sa.poisson).to(coords)
            B, detJ = _geom(coords)
            ue = Ut[:, sa._dof(blk)]                                    # (C,m,24)
            eps = torch.einsum("mplj,cmj->cmpl", B, ue)
            if sa.incompatible:
                Ba = _bubble_B(coords)
                Kua = _btdb(B, D, Ba, detJ)
                Kaa = _btdb(Ba, D, Ba, detJ)
                M = torch.linalg.solve(Kaa, Kua.transpose(1, 2))       # Kaa^-1 Kua^T (m,9,24)
                ae = -torch.einsum("maj,cmj->cma", M, ue)
                eps = eps + torch.einsum("mpla,cma->cmpl", Ba, ae)
            sig = torch.einsum("kl,cmpl->cmpk", D, eps)
            sn = torch.einsum("np,cmpk->cmnk", gauss_to_node().to(coords), sig)
            sn = sn * torch.as_tensor(mult[a:a + sa.chunk])[None, :, None, None]
            flat = torch.as_tensor(blk.reshape(-1))
            acc.index_add_(1, flat, sn.reshape(C, -1, 6))
            cnt.index_add_(0, flat, torch.ones(len(flat)))
    return acc / cnt[None, :, None].clamp(min=1)


REQUIRED_ENV = {"FEA_LOAD_MODE": "tile_nodes", "FEA_LOAD_WEIGHT": "pyramid",
                "CROSS4_LOAD_PAD_THICKNESS": "0"}


def load_spec(bc, row, F):
    NX, NY, _ = bc.tile_grid()
    tz, ty = bc.LENGTH / NX, bc.HEIGHT / NY
    y0, y1 = row * ty, (row + 1) * ty
    return {"box": ((-1e9, 1e9), (y0, y1), (bc.LENGTH - tz, bc.LENGTH)),
            "center": (0.0, 0.5 * (y0 + y1), bc.LENGTH - 0.5 * tz),
            "total": (0.0, -F, 0.0), "weight": "pyramid", "half": (0.5 * ty, 0.5 * tz)}


def sample_mix(n, seed, frac=(0.2, 0.4, 0.4), amp=(0.05, 0.5), amp_hc=(0.5, 2.0), sigma_hc=(1.0, 4.0), t_min=None):
    """n designs from three samplers, kinds shuffled over the indices (so any
    prefix of the list has the same mix):
      field      datasets.design_sampling.sample_designs' correlated
                 field (sigma 1-12, amplitude amp): near-uniform designs
      field_hc   the same field with short correlation (sigma_hc) and large
                 amplitude (amp_hc): strong contrast over the full range
      two_level  sample_designs' members-on-a-thin-background designs
    Each family's anchor volume is drawn uniform in volume, as in sample_designs.
    t_min raises dims.json's thickness floor for all three."""
    from scipy.stats import qmc
    from datasets.design_sampling import (sample_designs, volume_at,
                                          _build_or_load_volume_inverse, _sample_correlated_field)
    bc = importlib.import_module(f"data.{GEOMETRY}.boundary_conditions")
    NX, NY, NZ = bc.tile_grid()
    keys = [f"d{i + 1}" for i in range(NX * NY * NZ)]
    tmpl = json.load(open(f"data/{GEOMETRY}/CAD_model/dims.json"))
    lo = np.array([tmpl[k]["min"] for k in keys])
    hi = np.array([tmpl[k]["max"] for k in keys])
    counts = [int(round(f * n)) for f in frac]
    counts[2] = n - counts[0] - counts[1]
    A = sample_designs(GEOMETRY, counts[0], seed, 0.0, amp, t_min)[0]
    C = sample_designs(GEOMETRY, counts[2], seed + 2, 1.0, amp, t_min)[0]
    v_lo, v_hi, v_inv = _build_or_load_volume_inverse(f"data/{GEOMETRY}", f"data/{GEOMETRY}/CAD_model",
                                                      bc.fixed_dims(), keys, lo[0], hi[0])
    if t_min is not None:
        lo, v_lo = np.maximum(lo, t_min), volume_at(f"data/{GEOMETRY}", t_min)
    m = int(np.ceil(np.log2(max(counts[1], 2))))
    sob = qmc.Sobol(d=4, scramble=True, seed=seed + 1).random_base2(m)[:counts[1]]
    rng = np.random.default_rng(seed + 1)
    lu = lambda u, a, b: float(np.exp(np.log(a) + u * (np.log(b) - np.log(a))))
    B = [{k: float(v) for k, v in _sample_correlated_field(
            keys, lo, hi, np.full(len(keys), float(v_inv(v_lo + u[0] * (v_hi - v_lo)))), NX, NY, NZ, rng,
            amplitude_frac_up=lu(u[2], *amp_hc), amplitude_frac_down=lu(u[3], *amp_hc),
            sigma=sigma_hc[0] + u[1] * (sigma_hc[1] - sigma_hc[0])).items()} for u in sob]
    pool = [(d, "field") for d in A] + [(d, "field_hc") for d in B] + [(d, "two_level") for d in C]
    order = np.random.default_rng(seed + 3).permutation(n)
    return [pool[i][0] for i in order], [pool[i][1] for i in order]


def solve_design(dims, bc, rows, F, rho_mpa, workers=10, need_ds=True):
    """Returns (sample, shared) as described in the module docstring, values
    in float64 (the caller casts)."""
    from core.IritModel import IritCModel
    from core.mesh import Mesh
    from core.node import Node
    from core.element import Element
    from core.fem_adjoint import ShapeAdjoint, element_stiffness, _vm
    from core.geometric_jacobian import compute_tile_jacobian_colored

    tm = {}
    _t = time.time()
    exe = f"data/{GEOMETRY}/CAD_model/model"
    U, V, W = bc.mesh_resolution()
    D_tile, nodes0, elements0, e2t, _ = compute_tile_jacobian_colored(
        exe, dims, U, V, W, bc.tile_grid(), fixed_dims=bc.fixed_dims(),
        mesh_threads=1, model_cls=IritCModel, workers=workers)
    tm["mesh+jacobian"] = time.time() - _t; _t = time.time()
    keys = list(dims)
    nodes = {nid: Node(nid, list(xyz)) for nid, xyz in nodes0.items()}
    mesh = Mesh(nodes, {eid: Element(eid, [nodes[n] for n in nl]) for eid, nl in elements0.items()})
    mesh.anchor_nodes_by_condition(bc.anchor_condition)
    anchored = [n.id for n in mesh.all_nodes() if n.anchored]
    mp = json.load(open(f"data/{GEOMETRY}/CAD_model/material_properties.json"))
    forces = []
    for r in rows:
        mesh.clear_forces()
        mesh._apply_node_load(load_spec(bc, r, F))
        f = {n.id: list(n.forces) for n in mesh.all_nodes() if any(n.forces)}
        assert np.allclose(np.sum(list(f.values()), axis=0), (0.0, -F, 0.0), rtol=1e-9, atol=1e-3)
        forces.append(f)
    elements = {eid: list(nl)[:8] for eid, nl in elements0.items()}
    sa = ShapeAdjoint(nodes0, elements, anchored, [], (0.0, 0.0, 0.0), mp["young_modulus"],
                      mp["poisson_ratio"], rigid_tie=False, nodal_forces=forces[0])
    assert sa.stress_scale is None and np.all(sa.e_scale == 1.0)
    sa.solve()
    C, N, idx = len(rows), sa.N, sa.index
    Ff = np.zeros((3 * N, C))
    for k in range(C):
        for nid, fv in forces[k].items():
            Ff[3 * idx[nid]:3 * idx[nid] + 3, k] = fv
    Uc = np.asarray(sa.T @ sa._lu.solve(sa.T.T @ Ff)).T.reshape(C, N, 3)
    S = batched_nodal_stress(sa, Uc).numpy() / 1e6                         # (C,N,6) MPa

    coords = sa.X.detach().numpy()
    pad_t = float(os.environ.get("CROSS4_PAD_THICKNESS") or 0.1)
    pad = coords[:, 2] <= pad_t + 1e-6 if pad_t > 0 else np.zeros(N, bool)
    keep = torch.as_tensor(~pad)
    rho = rho_mpa
    St = torch.as_tensor(S).requires_grad_(True)
    vm = _vm(St)                                                          # (C,N)
    vk = torch.where(keep[None], vm, torch.full_like(vm, -float("inf")))
    m0 = vk.max(1, keepdim=True).values
    ks = (m0[:, 0] + torch.logsumexp(rho * (vk - m0), 1) / rho)
    ks.sum().backward()
    gS = St.grad.numpy()                                                  # dKS_c/dS_c (C,N,6)
    out = dict(coords=coords, S=S, ks=ks.detach().numpy(),
               max_vm=m0[:, 0].detach().numpy())
    conn = sa.conn
    shared = dict(node_ids=np.array(sa.ids, np.int64), elements=conn.astype(np.int32),
                  elem_tile=np.array([e2t[e] for e in sa.eids], np.int32),
                  anchored=np.array([i in set(anchored) for i in sa.ids], bool), pad=pad,
                  load_rows=np.array(rows, np.int32))
    tm["solve+stress"] = time.time() - _t; _t = time.time()
    out["_timing"] = tm
    if not need_ds:
        return out, shared

    # ---- direct sensitivities, one tile at a time on its few elements ----
    K = len(keys)
    cnt = np.bincount(conn.reshape(-1), minlength=N).astype(np.float64)
    elems_of = [[] for _ in range(N)]
    for e, nl in enumerate(conn):
        for n in nl:
            elems_of[n].append(e)
    Xt = sa.X.detach()
    Ut = torch.as_tensor(Uc)                                              # (C,N,3)
    R = np.zeros((3 * N, C * K))                                          # -(dK u) per (case, tile)
    dS_part = np.zeros((C, K, N, 6))                                      # (dR/dX V) u, accumulated
    stress_terms = lambda X_, ue: sa._nodal_stress_terms(None, X_, ue)
    E, nu, inc = sa.young, sa.poisson, sa.incompatible
    for j, key in enumerate(keys):
        dX = D_tile[key]
        if not dX:
            continue
        Vn = np.zeros((N, 3))
        for nid, v in dX.items():
            Vn[idx[nid]] = v
        el = np.unique(np.concatenate([elems_of[idx[nid]] for nid in dX]))
        blk = conn[el]
        Xe, Ve = Xt[blk], torch.as_tensor(Vn[blk])
        _, dKe = torch.func.jvp(lambda X_: element_stiffness(X_, E, nu, inc), (Xe,), (Ve,))
        dof = (3 * blk[:, :, None] + np.arange(3)).reshape(len(blk), 24)
        ue = Ut[:, torch.as_tensor(blk)].reshape(C, len(blk), 24)        # (C,m,24)
        r = torch.einsum("mij,cmj->cmi", dKe, ue).numpy()                # (C,m,24)
        for c in range(C):
            np.add.at(R[:, c * K + j], dof.reshape(-1), -r[c].reshape(-1))
        Xr = Xe.repeat(C, 1, 1)
        Vr = Ve.repeat(C, 1, 1)
        uer = ue.reshape(C * len(blk), 24)
        _, dsn = torch.func.jvp(lambda X_: stress_terms(X_, uer), (Xr,), (Vr,))
        dsn = dsn.reshape(C, len(blk), 8, 6).numpy()
        for c in range(C):
            np.add.at(dS_part[c, j], blk.reshape(-1), dsn[c].reshape(-1, 6))
    tm["jvp loop"] = time.time() - _t; _t = time.time()
    dQ = sa._lu.solve(sa.T.T @ R)                                         # one multi-RHS solve
    dU = np.asarray(sa.T @ dQ)                                            # (3N, C*K)
    tm["multi-RHS solve"] = time.time() - _t; _t = time.time()
    dS = np.empty((C, K, N, 6))
    for c in range(C):
        dUc = dU[:, c * K:(c + 1) * K].T.reshape(K, N, 3)
        dS[c] = batched_nodal_stress(sa, dUc).numpy() + dS_part[c] / cnt[None, :, None]
    dS /= 1e6
    tm["dS recovery"] = time.time() - _t
    out["dS"] = dS
    out["dks"] = np.einsum("cnk,cjnk->cj", gS, dS)
    out["_D_tile"] = D_tile
    out["_sa"] = sa
    # displacement view of the same solve (not saved by main): u, du/dt and the
    # part of dS/dt from the mesh moving at fixed u -- dS = R du + _dS_geo
    out["_U"] = Uc
    out["_dU"] = dU.T.reshape(C, K, N, 3)
    out["_dS_geo"] = dS_part / cnt[None, None, :, None] / 1e6
    return out, shared


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_designs", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amplitude_range", default="0.05,0.5", help="field (A) and two-level amplitude")
    p.add_argument("--mix", default="0.2,0.4,0.4", help="shares of field / field_hc / two_level")
    p.add_argument("--t_min", type=float, default=None,
                   help="tile thickness floor (default dims.json's 0.01). Below ~0.03 |dS| can overflow float16 and the design is skipped")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--max_gb", type=float, default=740.0)
    p.add_argument("--min_free_gb", type=float, default=50.0)
    p.add_argument("--stress_ks_rho_mpa", type=float, default=0.25)
    p.add_argument("--mesh_workers", type=int, default=10)
    p.add_argument("--claim", action="store_true", help="shared work queue instead of a fixed split")
    p.add_argument("--stale_min", type=float, default=30.0)
    args = p.parse_args()
    for k, v in REQUIRED_ENV.items():
        if os.environ.get(k) != v:
            sys.exit(f"set {k}={v}")
    if os.environ.get("CROSS4_FIXED_END_RADIUS", "0") not in ("", "0"):
        sys.exit("CROSS4_FIXED_END_RADIUS must be unset/0: a fixed load end makes the geometry depend on the load row")
    out = args.output_dir
    os.makedirs(f"{out}/designs", exist_ok=True)
    bc = importlib.import_module(f"data.{GEOMETRY}.boundary_conditions")
    NX, NY, NZ = bc.tile_grid()
    rows = list(range(NY))
    F = float(os.environ.get("FEA_LOAD_N") or 5e7)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS") or 8))

    amp = tuple(float(v) for v in args.amplitude_range.split(","))
    frac = tuple(float(v) for v in args.mix.split(","))
    designs, kinds = sample_mix(args.num_designs, args.seed, frac, amp, t_min=args.t_min)
    keys = [f"d{i + 1}" for i in range(NX * NY * NZ)]
    if args.worker == 0:
        cfg = {"geometry": GEOMETRY, "num_designs": args.num_designs, "seed": args.seed,
               "amplitude_range": amp, "t_min": args.t_min, "mix": {"field": frac[0], "field_hc": frac[1], "two_level": frac[2]},
               "field_hc": {"amplitude_range": [0.5, 2.0], "sigma_range": [1.0, 4.0]}, "load_N": F, "load_dir": [0.0, -1.0, 0.0],
               "load_rows": rows, "mesh_resolution": list(bc.mesh_resolution()),
               "ks_rho_per_mpa": args.stress_ks_rho_mpa, "ks_nodes": "all but anchored pad",
               "stress_order": ["sxx", "syy", "szz", "sxy", "sxz", "syz"], "stress_unit": "MPa",
               "dS_unit": "MPa per unit tile thickness", "dS_dtype": "float16",
               "sensitivity": "direct differentiation, nodal forces held fixed",
               "env": {k: v for k, v in os.environ.items() if k.startswith(("FEA_", "CROSS4_"))}}
        for name, obj in (("config.json", cfg), ("designs.json", {"designs": designs, "kinds": kinds})):
            if not os.path.exists(f"{out}/{name}"):
                json.dump(obj, open(f"{out}/{name}.tmp{os.getpid()}", "w"), indent=1)
                os.replace(f"{out}/{name}.tmp{os.getpid()}", f"{out}/{name}")
    os.makedirs(f"{out}/claims", exist_ok=True)

    def claim(i):
        """True if this worker now owns design i (atomic O_EXCL create)."""
        cp = f"{out}/claims/{i:05d}"
        for _ in range(2):
            try:
                fd = os.open(cp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.uname().nodename} {os.getpid()} {time.time():.0f}".encode())
                os.close(fd)
                return True
            except FileExistsError:
                try:
                    stale = time.time() - os.path.getmtime(cp) > args.stale_min * 60
                except OSError:
                    continue
                if not stale:
                    return False
                try:
                    os.remove(cp)
                except OSError:
                    pass
        return False

    mine = (list(range(args.num_designs)) if args.claim
            else [i for i in range(args.num_designs) if i % args.num_workers == args.worker])
    size_gb = dir_size_gb(out)
    t_start, done, stop = time.time(), 0, False
    for _pass in range(1000 if args.claim else 1):
        progress = 0
        for i in mine:
            path = f"{out}/designs/{i:05d}.npz"
            if os.path.exists(path) or os.path.exists(path[:-4] + ".failed"):
                continue
            size_gb = dir_size_gb(out) if done % 20 == 0 else size_gb   # other workers write too
            free_gb = shutil.disk_usage(out).free / 1e9
            if size_gb > args.max_gb or free_gb < args.min_free_gb:
                print(f"[worker {args.worker}] STOP: dataset {size_gb:.1f} GB / limit {args.max_gb}, "
                      f"free {free_gb:.0f} GB", flush=True)
                stop = True
                break
            if args.claim and not claim(i):
                continue
            if os.path.exists(path):
                continue
            dims = {k: float(designs[i][k]) for k in keys}
            t0 = time.time()
            try:
                s, shared = solve_design(dims, bc, rows, F, args.stress_ks_rho_mpa, workers=args.mesh_workers)
            except Exception as e:
                print(f"[worker {args.worker}] design {i} FAILED: {str(e).splitlines()[0][:200]}", flush=True)
                open(path[:-4] + ".failed", "w").write(str(e)[:2000])
                if args.claim:
                    try:
                        os.remove(f"{out}/claims/{i:05d}")
                    except OSError:
                        pass
                continue
            mpath = f"{out}/mesh.npz"
            h = hashlib.sha1(shared["elements"].tobytes() + shared["node_ids"].tobytes()).hexdigest()
            if not os.path.exists(mpath):
                np.savez(f"{mpath}.tmp{os.getpid()}.npz", **shared)
                os.replace(f"{mpath}.tmp{os.getpid()}.npz", mpath)
            ref = np.load(mpath)
            if hashlib.sha1(ref["elements"].tobytes() + ref["node_ids"].tobytes()).hexdigest() != h:
                print(f"[worker {args.worker}] design {i}: mesh connectivity differs from mesh.npz -- skipped", flush=True)
                open(path[:-4] + ".failed", "w").write("mesh connectivity differs from mesh.npz")
                continue
            if np.abs(s["dS"]).max() > 6e4:
                print(f"[worker {args.worker}] design {i}: |dS| {np.abs(s['dS']).max():.3g} overflows float16 -- skipped", flush=True)
                open(path[:-4] + ".failed", "w").write("dS overflows float16")
                continue
            tmp = f"{path}.tmp{os.getpid()}.npz"
            np.savez(tmp, t=np.array([dims[k] for k in keys], np.float32),
                     coords=s["coords"].astype(np.float32), S=s["S"].astype(np.float32),
                     dS=s["dS"].astype(np.float16), ks=s["ks"].astype(np.float32),
                     dks=s["dks"].astype(np.float32), max_vm=s["max_vm"].astype(np.float32))
            os.replace(tmp, path)
            if args.claim:
                try:
                    os.remove(f"{out}/claims/{i:05d}")
                except OSError:
                    pass
            size_gb += os.path.getsize(path) / 1e9
            done += 1
            progress += 1
            print(f"[worker {args.worker}] design {i} ({kinds[i]}): {time.time() - t0:.0f} s, "
                  f"ks max {s['ks'].max():.0f} MPa, done {done}, dataset {size_gb:.1f} GB, "
                  f"{(time.time() - t_start) / done:.0f} s/design", flush=True)
        if stop or not args.claim:
            break
        left = [i for i in mine if not os.path.exists(f"{out}/designs/{i:05d}.npz")
                and not os.path.exists(f"{out}/designs/{i:05d}.failed")]
        if not left:
            break
        if progress == 0:
            time.sleep(120)
    print(f"[worker {args.worker}] finished: {done} designs written", flush=True)


if __name__ == "__main__":
    main()
