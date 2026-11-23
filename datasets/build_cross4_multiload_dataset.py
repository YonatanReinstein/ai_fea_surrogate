"""Multi-load dataset for cross4_11x11: every design is solved for each
free-end tile as the load tile, in two directions, from ONE factorization.

Load cases (per design): for each free-end tile row r = 0..NY-1, the
FEA_LOAD_MODE=tile_nodes load (FEA_LOAD_N over every node of that tile,
weighted 1 - r/r_max by distance from its centre, no rigid tie) acting
    dir 0  "down"    -y
    dir 1  "anchor"  -z, toward the anchored column (z = 0)
-> 2 * NY cases. With no load pad the mesh -- and so the stiffness matrix -- is
the same for every case, so each design is factorized once and the cases are
back-solves (core.fem_adjoint.ShapeAdjoint, rigid_tie=False; it reproduces
ccx's C3D8I).

Designs: datasets.build_tile_load_dataset.sample_designs (scrambled Sobol,
correlated random fields), two_level_frac 0 -- correlated fields only.

Layout of --output_dir:
    mesh.npz            stored ONCE, identical for every design:
        node_ids (N,)          node order of every per-design array
        elements (M,8) int32   connectivity, as indices into node_ids
        elem_tile (M,) int32   IRIT tile index (ix + NX*iy) of each element
        anchored (N,) bool
        cases (C,2) int32      (load row, direction) per case
    config.json         settings, field conventions
    designs.json        thickness of every design (dims keys d1..dK)
    designs/NNNNN.npz   one per design (float32 unless noted):
        t (K,)                 tile thicknesses (d1..dK)
        coords (N,3)           node coordinates
        u (C,N,3)              displacement per case
        S (C,N,6)              nodal stress tensor per case (Pa), averaged over
                               the elements at each node, CalculiX's Voigt order
                               (sxx, syy, szz, sxy, sxz, syz); von Mises follows
                               exactly from it, so it is not stored
        max_vm (C,), ks (C,)   max von Mises / KS (rho 0.25/MPa) over all nodes, Pa
        max_vm_excl (C,)       max von Mises excluding the loaded tile's nodes, Pa
        max_disp (C,)          max |u|
Load vectors are not stored: they follow from (row, direction), the node
coordinates and Mesh._apply_node_load.

Run several workers on disjoint designs (--worker k --num_workers K), or with
--claim: every worker walks the whole list and takes the next unclaimed design
(an O_EXCL claim file in claims/), so late, slow or dead workers leave no holes;
a claim older than --stale_min minutes without its result is taken over.
Designs already written are skipped, so any worker can be restarted. Each worker
stops when the output directory exceeds --max_gb or free space drops below
--min_free_gb.

    python -m datasets.build_cross4_multiload_dataset --output_dir DIR \
        --num_designs 5000 --worker 0 --num_workers 12
Environment (set by the job script): CROSS4_LEG_RULE=avg FEA_LOAD_MODE=tile_nodes
CROSS4_PAD_THICKNESS=0.1 CROSS4_LOAD_PAD_THICKNESS=0 FEA_MESH_RES=6,6,4 FEA_LOAD_N=1e7
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
DIRS = {0: ("down", (0.0, -1.0, 0.0)), 1: ("anchor", (0.0, 0.0, -1.0))}


def dir_size_gb(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1e9


def load_spec(bc, row, direction, F):
    NX, NY, _ = bc.tile_grid()
    tz, ty = bc.LENGTH / NX, bc.HEIGHT / NY
    y0, y1 = row * ty, (row + 1) * ty
    return {"box": ((-1e9, 1e9), (y0, y1), (bc.LENGTH - tz, bc.LENGTH)),
            "center": (0.0, 0.5 * (y0 + y1), bc.LENGTH - 0.5 * tz),
            "total": tuple(F * c for c in DIRS[direction][1])}


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


def solve_design(dims, bc, cases, F, rho_mpa):
    from core.IritModel import IritCModel
    from core.component import Component
    from core.fem_adjoint import ShapeAdjoint
    m = IritCModel(f"data/{GEOMETRY}/CAD_model/model", dims_dict=dims, mesh_threads=os.cpu_count() or 8,
                   fixed_dims=bc.fixed_dims())
    c = Component(CAD_model=m, young=2e11, poisson=0.3)
    c.generate_mesh(*bc.mesh_resolution())
    mesh = c.mesh
    mesh.anchor_nodes_by_condition(bc.anchor_condition)
    N = list(mesh.all_nodes())
    nodes = {n.id: n.coords for n in N}
    elements = {e.id: [n.id for n in e.nodes][:8] for e in mesh.all_elements()}
    anchored = [n.id for n in N if n.anchored]
    mp = json.load(open(f"data/{GEOMETRY}/CAD_model/material_properties.json"))

    forces = []
    for row, direction in cases:
        mesh.clear_forces()
        mesh._apply_node_load(load_spec(bc, row, direction, F))
        f = {n.id: list(n.forces) for n in N if any(n.forces)}
        tot = np.sum(list(f.values()), axis=0)
        assert np.allclose(tot, np.array(DIRS[direction][1]) * F, rtol=1e-6, atol=1e-3), (row, direction, tot)
        forces.append(f)

    sa = ShapeAdjoint(nodes, elements, anchored, [], (0.0, 0.0, 0.0), mp["young_modulus"],
                      mp["poisson_ratio"], rigid_tie=False, nodal_forces=forces[0])
    sa.solve()
    ids = sa.ids
    tile_of_node = {}
    for eid, nl in elements.items():
        for nid in nl:
            tile_of_node.setdefault(nid, set()).add(c.elem_to_tile[eid])
    NXg, NYg, _ = bc.tile_grid()
    C = len(cases)
    U = np.empty((C, len(ids), 3), np.float32)
    S = np.empty((C, len(ids), 6), np.float32)
    max_vm, ks, max_vm_excl, max_disp = (np.empty(C, np.float64) for _ in range(4))
    rho = rho_mpa * 1e-6
    # all C load cases at once: one multi-RHS back-substitution on the single
    # factorization, then one batched stress recovery
    from core.fem_adjoint import _vm
    Ff = np.zeros((3 * sa.N, C))
    for k in range(C):
        for nid, fv in forces[k].items():
            Ff[3 * sa.index[nid]:3 * sa.index[nid] + 3, k] = fv
    Q = sa._lu.solve(sa.T.T @ Ff)
    Uall = np.asarray(sa.T @ Q).T.reshape(C, sa.N, 3)
    Sall = batched_nodal_stress(sa, Uall)
    VMall = _vm(Sall).numpy()
    Sall = Sall.numpy()
    for k, (row, direction) in enumerate(cases):
        u, St, vm = Uall[k], Sall[k], VMall[k]
        load_tile = (NXg - 1) + NXg * row
        excl = np.array([load_tile not in tile_of_node[nid] for nid in ids])
        U[k], S[k] = u, St
        max_vm[k] = vm.max()
        m0 = vm.max()
        ks[k] = m0 + np.log(np.exp(rho * (vm - m0)).sum()) / rho
        max_vm_excl[k] = vm[excl].max()
        max_disp[k] = np.linalg.norm(u, axis=1).max()
    coords = np.array([nodes[i] for i in ids], np.float32)
    eids = sorted(elements)
    index = sa.index
    conn = np.array([[index[n] for n in elements[e]] for e in eids], np.int32)
    elem_tile = np.array([c.elem_to_tile[e] for e in eids], np.int32)
    anch = np.array([i in set(anchored) for i in ids], bool)
    return dict(coords=coords, u=U, S=S, max_vm=max_vm, ks=ks, max_vm_excl=max_vm_excl,
                max_disp=max_disp), dict(node_ids=np.array(ids, np.int64), elements=conn,
                                         elem_tile=elem_tile, anchored=anch)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_designs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amplitude_range", default="0.05,0.5")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--max_gb", type=float, default=420.0)
    p.add_argument("--min_free_gb", type=float, default=50.0)
    p.add_argument("--stress_ks_rho_mpa", type=float, default=0.25)
    p.add_argument("--claim", action="store_true", help="shared work queue instead of a fixed split")
    p.add_argument("--stale_min", type=float, default=30.0)
    args = p.parse_args()
    for k, v in (("FEA_LOAD_MODE", "tile_nodes"), ("CROSS4_LOAD_PAD_THICKNESS", "0")):
        if os.environ.get(k) != v:
            sys.exit(f"set {k}={v} (the multi-load solve assumes no load pad / an untied node load)")
    out = args.output_dir
    os.makedirs(f"{out}/designs", exist_ok=True)
    bc = importlib.import_module(f"data.{GEOMETRY}.boundary_conditions")
    NX, NY, NZ = bc.tile_grid()
    cases = [(r, d) for r in range(NY) for d in (0, 1)]
    F = float(os.environ.get("FEA_LOAD_N") or 5e7)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS") or 8))

    from datasets.build_tile_load_dataset import sample_designs
    amp = tuple(float(v) for v in args.amplitude_range.split(","))
    designs, kinds = sample_designs(GEOMETRY, args.num_designs, args.seed, 0.0, amp)
    keys = [f"d{i + 1}" for i in range(NX * NY * NZ)]
    if args.worker == 0:
        cfg = {"geometry": GEOMETRY, "num_designs": args.num_designs, "seed": args.seed,
               "amplitude_range": amp, "two_level_frac": 0.0, "load_N": F,
               "cases": [{"row": r, "direction": DIRS[d][0], "vector": DIRS[d][1]} for r, d in cases],
               "stress_order": ["sxx", "syy", "szz", "sxy", "sxz", "syz"], "stress_unit": "Pa",
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
                    continue                       # claim vanished: retry the create
                if not stale:
                    return False
                try:
                    os.remove(cp)                  # abandoned claim: take it over
                except OSError:
                    pass
        return False

    mine = (list(range(args.num_designs)) if args.claim
            else [i for i in range(args.num_designs) if i % args.num_workers == args.worker])
    size_gb = dir_size_gb(out)
    t_start, done = time.time(), 0
    stop = False
    # --claim: repeat passes until every design is written or marked failed;
    # a pass with no progress waits for other workers (or for stale claims).
    for _pass in range(1000 if args.claim else 1):
        progress = 0
        for i in mine:
            path = f"{out}/designs/{i:05d}.npz"
            if os.path.exists(path) or os.path.exists(path[:-4] + ".failed"):
                continue
            free_gb = shutil.disk_usage(out).free / 1e9
            if size_gb > args.max_gb or free_gb < args.min_free_gb:
                print(f"[worker {args.worker}] STOP: dataset {size_gb:.1f} GB / limit {args.max_gb}, "
                      f"free {free_gb:.0f} GB", flush=True)
                stop = True
                break
            if args.claim and not claim(i):
                continue
            if os.path.exists(path):                   # finished between the check and the claim
                continue
            dims = {k: float(designs[i][k]) for k in keys}
            t0 = time.time()
            try:
                sample, shared = solve_design(dims, bc, cases, F, args.stress_ks_rho_mpa)
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
                np.savez(f"{mpath}.tmp{os.getpid()}.npz", cases=np.array(cases, np.int32), **shared)
                os.replace(f"{mpath}.tmp{os.getpid()}.npz", mpath)
            ref = np.load(mpath)
            if hashlib.sha1(ref["elements"].tobytes() + ref["node_ids"].tobytes()).hexdigest() != h:
                print(f"[worker {args.worker}] design {i}: mesh connectivity differs from mesh.npz -- skipped", flush=True)
                continue
            tmp = f"{path}.tmp{os.getpid()}.npz"
            np.savez(tmp, t=np.array([dims[k] for k in keys], np.float32), **sample)
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
                  f"max_vm {sample['max_vm'].max() / 1e6:.0f} MPa, done {done}, dataset {size_gb:.1f} GB, "
                  f"{(time.time() - t_start) / done:.0f} s/design", flush=True)
        if stop or not args.claim:
            break
        left = [i for i in mine if not os.path.exists(f"{out}/designs/{i:05d}.npz")
                and not os.path.exists(f"{out}/designs/{i:05d}.failed")]
        if not left:
            break
        if progress == 0:
            time.sleep(120)        # the rest are claimed by others; recheck for stale claims
    print(f"[worker {args.worker}] finished: {done} designs written", flush=True)


if __name__ == "__main__":
    main()
