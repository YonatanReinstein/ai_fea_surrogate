"""Tile-load dataset: every sampled design is solved under a load on EACH tile
of the free-end column, from ONE factorization of its stiffness.

The load case is no longer fixed. For a design, load k is a total force
(0, -F, 0) spread in equal shares over the nodes of the free end face (z = max)
that lie in tile row k's height span -- the production force_pattern's own
patch, slid from the middle row to row k. Equal shares on the patch nodes is
what a distributing coupling with uniform weights applies for a force through
the patch centroid, and unlike the production *RIGID BODY it does not weld the
loaded tile stiff, so the stiffness matrix is the same for every k. Hence:

    factorize K once                        (PARDISO, ~1 s on the 11x11)
    one back-substitution per load          (all NY loads in one call)
    per-element d(K_e)/dt and d(R_e)/dt     (one forward-mode pass along each
                                             element's own tile direction)
    per load: stress, KS, exact dKS/dt      (one more back-substitution + sums)

The per-tile gradient is the ShapeAdjoint adjoint rearranged so nothing that
depends on the load is differentiated again:

    dKS/dt_tile = sum_{e in tile} [ dn_e . (dR_e/dt) u_e  -  lambda_e . (dK_e/dt) u_e ]
    K lambda = R^T dn,   dn = d(KS)/d(nodal stress) of THIS load's field

It relies on every moving node belonging to its own element's tile (no node
moves with two tiles' thicknesses), which is asserted per design.

Anchored tiles (the z = 0 column) never carry load; the free-end column is the
farthest from them. Samples carry the same schema as dataset_builder's adjoint
grad samples (stress in Pa, grad_dks_dx in MPa per unit thickness, dims-key
order) plus the load: load_tile (IRIT tile index), load_dims_idx (dims key
index), load_force (N), and geometry_idx so a train/val split can keep all
loads of one design on the same side. Like dataset_builder's worker output the
samples omit edge_index/tile_idx; graph.pt is copied next to the shards.

KS (--ks_exclude load_tile, the default) leaves out every node of the loaded
tile's elements: the equal-share nodal load makes a local spike there that the
production rigid handle does not, and it dominated KS on thick designs. Each
sample stores the mask as ks_mask (graph rows, True = counts); ks_stress,
max_stress and grad_dks_dx follow it, node_stress is the full field, and
ks_stress_all keeps the all-node KS. Anything that computes KS from a model or
a solve for these samples must apply the same mask.

    python -m training.build_tile_load_dataset --geometry hollow_cube_11x11 \
        --num_geometries 100 --output_dir data/hollow_cube_11x11/dataset_tileload_pilot
"""
import argparse
import importlib
import io
import json
import os
import shutil
import sys
import time
from multiprocessing import get_context

import numpy as np
import torch

FORCE_TOTAL = 5e7          # same total as the production load, applied as -y


# ----------------------------------------------------------------- designs
def sample_designs(geometry, n, seed, two_level_frac, amp_range, t_min=None):
    """Same generator family as dataset_builder's DoE: scrambled Sobol over
    (anchor volume, sigma, amp_up, amp_down, two-level params), correlated
    fields with a per-design anchor drawn uniform in volume, and a share of
    members-on-background two-level designs. t_min raises dims.json's floor
    (and the anchor volume's lower end with it)."""
    from scipy.stats import qmc
    from training.dataset_builder import (_build_or_load_volume_inverse,
                                          _sample_correlated_field, _sample_two_level)
    base = f"data/{geometry}"
    tmpl = json.load(open(f"{base}/CAD_model/dims.json"))
    bc = importlib.import_module(f"data.{geometry}.boundary_conditions")
    NX, NY, NZ = bc.tile_grid(None)
    keys = [f"d{i + 1}" for i in range(NX * NY * NZ)]
    assert list(tmpl) == keys, "dims.json must be exactly one thickness per tile, d1..dK"
    lo = np.array([tmpl[k]["min"] for k in keys])
    hi = np.array([tmpl[k]["max"] for k in keys])
    v_lo, v_hi, v_inv = _build_or_load_volume_inverse(
        base, f"{base}/CAD_model", bc.fixed_dims(), keys, lo[0], hi[0])
    if t_min is not None:
        lo, v_lo = np.maximum(lo, t_min), volume_at(base, t_min)

    m = int(np.ceil(np.log2(max(n, 2))))
    sob = qmc.Sobol(d=6, scramble=True, seed=seed).random_base2(m)[:n]
    rng = np.random.default_rng(seed)
    a_lo, a_hi = amp_range
    loguni = lambda u, a, b: float(np.exp(np.log(a) + u * (np.log(b) - np.log(a))))
    designs, kinds = [], []
    for i in range(n):
        u = sob[i]
        if u[4] < two_level_frac:
            beta = None if u[5] < 0.5 else loguni(u[5], 2.0, 60.0)
            d = _sample_two_level(keys, lo, hi, NX, NY, NZ, rng,
                                  1.0 + u[1] * 5.0, 0.05 + u[2] * 0.35,
                                  lo[0] + (0.25 + 0.75 * u[0]) * (hi[0] - lo[0]),
                                  beta, (u[3] ** 2) * 0.45,
                                  back_texture=0.3 + 0.5 * u[4] / max(two_level_frac, 1e-9))
            kinds.append("two_level")
        else:
            t0 = float(v_inv(v_lo + u[0] * (v_hi - v_lo)))
            d = _sample_correlated_field(keys, lo, hi, np.full(len(keys), t0), NX, NY, NZ, rng,
                                         amplitude_frac_up=loguni(u[2], a_lo, a_hi),
                                         amplitude_frac_down=loguni(u[3], a_lo, a_hi),
                                         sigma=1.0 + u[1] * 11.0)
            kinds.append("field")
        designs.append({k: float(v) for k, v in d.items()})
    return designs, kinds


def volume_at(base, t):
    """Uniform-thickness volume at t, from the cached volume_lookup.npz that
    _build_or_load_volume_inverse writes."""
    npz = np.load(f"{base}/dataset/volume_lookup.npz")
    return float(np.interp(t, npz["thicknesses"], npz["volumes"]))


# ----------------------------------------------------------------- one design
def label_design(job):
    """All free-end-column loads for one design. Returns (idx, bytes, stats);
    the samples come back torch.save'd so no tensor crosses the pool through
    shared memory (see the /dev/shm torch_shm_manager history)."""
    idx, dims, cfg = job
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    from torch.func import jvp
    from core.component import Component
    from core.element import Element
    from core.fem_adjoint import (ShapeAdjoint, _bubble_B, _elastic, _geom, _vm,
                                  element_stiffness, gauss_to_node)
    from core.geometric_jacobian import compute_thickness_jacobian
    from core.mesh import Mesh
    from core.node import Node
    from core.tile_sensitivity import node_to_tile_map, tile_index_to_dims_key

    t_start = time.perf_counter()
    geom, rho_mpa = cfg["geometry"], cfg["rho_mpa"]
    bc = importlib.import_module(f"data.{geom}.boundary_conditions")
    mat = json.load(open(f"data/{geom}/CAD_model/material_properties.json"))
    E, nu = mat["young_modulus"], mat["poisson_ratio"]
    U, V, W = bc.mesh_resolution()
    NX, NY, NZ = bc.tile_grid(None)
    keys = [f"d{i + 1}" for i in range(NX * NY * NZ)]
    key_idx = {k: i for i, k in enumerate(keys)}
    g = torch.load(cfg["graph_path"], weights_only=False)
    edge_index, tile_idx_g = g["edge_index"], g["tile_idx"].view(-1)

    # --- CAD, mesh and dX/dt (the Jacobian builds the nominal mesh itself)
    dX_dt, nodes0, elements0, e2t, model = compute_thickness_jacobian(
        f"data/{geom}/CAD_model/model", dims, U, V, W, fixed_dims=bc.fixed_dims(),
        mesh_threads=cfg["mesh_threads"], return_model=True)
    t_mesh = time.perf_counter() - t_start
    ids = list(nodes0)
    assert ids == list(range(1, len(ids) + 1)), "graph rows assume node ids 1..N in order"
    assert len(ids) == g["num_nodes"]

    comp = Component(CAD_model=model, young=E, poisson=nu)
    nodes = {nid: Node(nid, xyz) for nid, xyz in nodes0.items()}
    comp.mesh = Mesh(nodes, {eid: Element(eid, [nodes[n] for n in nl])
                             for eid, nl in elements0.items()})
    comp.elem_to_tile = e2t
    comp.mesh.anchor_nodes_by_condition(bc.anchor_condition)
    anchored = [n.id for n in comp.mesh.all_nodes() if n.anchored]

    # --- load patches: tile row k's share of the free end face
    xyz = np.array([nodes0[n] for n in ids])
    z_end, y_top = xyz[:, 2].max(), xyz[:, 1].max()
    h, tol = y_top / NY, 1e-6
    on_end = np.abs(xyz[:, 2] - z_end) < tol
    patches = [[ids[i] for i in np.where(on_end & (xyz[:, 1] >= k * h - tol)
                                         & (xyz[:, 1] <= (k + 1) * h + tol))[0]]
               for k in range(NY)]
    n2t = node_to_tile_map(elements0, e2t)
    anchored_tiles = {n2t[n] for n in anchored}
    load_tiles = []
    for k, patch in enumerate(patches):
        assert patch and not any(nodes[n].anchored for n in patch), f"row {k}: empty or anchored patch"
        # the loaded tile = the end-column tile whose elements hold most patch nodes
        votes, pset = {}, set(patch)
        for eid, nl in elements0.items():
            hit = sum(n in pset for n in nl)
            if hit:
                votes[e2t[eid]] = votes.get(e2t[eid], 0) + hit
        t_k = max(votes, key=votes.get)
        assert t_k not in anchored_tiles, f"row {k}: load tile {t_k} holds anchored nodes"
        load_tiles.append(t_k)
    assert len(set(load_tiles)) == NY

    # --- one factorization, all loads
    t0 = time.perf_counter()
    sa = ShapeAdjoint(nodes0, elements0, anchored, [], (0.0, 0.0, 0.0), E, nu, rigid_tie=False)
    sa.solve()
    index = sa.index
    F_full = np.zeros((3 * sa.N, NY))
    for k, patch in enumerate(patches):
        for nid in patch:
            F_full[3 * index[nid] + 1, k] = -FORCE_TOTAL / len(patch)
    Ui = torch.as_tensor(sa.T @ sa._lu.solve(sa.T.T @ F_full))       # (3N, NY)
    t_solve = time.perf_counter() - t0

    # --- per-element operators along each element's own tile direction
    t0 = time.perf_counter()
    conn = torch.as_tensor(sa.conn)
    M = len(sa.conn)
    dof = sa._dof(sa.conn)
    Vn = torch.zeros(sa.N, 3)
    for nid, dx in dX_dt.items():
        Vn[index[nid]] = torch.tensor(dx)
    moving = (Vn.abs().max(dim=1).values > 1e-9).numpy()
    etile = np.array([e2t[e] for e in sa.eids])
    for k_e in range(M):
        for i in sa.conn[k_e]:
            if moving[i]:
                assert n2t[sa.ids[i]] == etile[k_e], "a moving node is shared with another tile"
    etile_k = torch.as_tensor([key_idx[tile_index_to_dims_key(int(t), NX, NY, NZ, keys)]
                               for t in etile])
    Xe, Ve = sa.X.detach()[conn], Vn[conn]
    D6, G2N = _elastic(E, nu), gauss_to_node()

    def R_of(coords):
        B, detJ = _geom(coords)
        Ba = _bubble_B(coords)
        DB = torch.einsum("kl,mplj->mpkj", D6, B)
        DBa = torch.einsum("kl,mplj->mpkj", D6, Ba)
        Kua = torch.einsum("mpki,mpkj,mp->mij", B, DBa, detJ)
        Kaa = torch.einsum("mpki,mpkj,mp->mij", Ba, DBa, detJ)
        A = -torch.linalg.solve(Kaa, Kua.transpose(1, 2))
        return torch.einsum("np,mpki->mnki", G2N, DB + torch.einsum("mpkj,mji->mpki", DBa, A))

    Rop, dRop = jvp(R_of, (Xe,), (Ve,))
    _, dK = jvp(lambda c: element_stiffness(c, E, nu, True), (Xe,), (Ve,))
    cnt = torch.zeros(sa.N)
    cnt.index_add_(0, conn.reshape(-1), torch.ones(M * 8))
    t_pre = time.perf_counter() - t0

    # --- KS exclusion: graph rows of every node of the loaded tile's elements
    excl_rows = [sorted({int(i) for k_e in np.where(etile == t)[0] for i in sa.conn[k_e]})
                 for t in load_tiles]

    # --- per load: stress, KS, gradient, sample
    t0 = time.perf_counter()
    rho = rho_mpa * 1e-6
    out, ks_list, ks_all_list = [], [], []
    for k in range(NY):
        ue = Ui[:, k][dof]                                             # (M,24)
        s = torch.einsum("mnkj,mj->mnk", Rop, ue)
        S = torch.zeros(sa.N, 6)
        S.index_add_(0, conn.reshape(-1), s.reshape(-1, 6))
        S = (S / cnt[:, None].clamp(min=1)).requires_grad_(True)
        vm = _vm(S)
        # KS over the nodes that count: under --ks_exclude load_tile every node
        # of the loaded tile's elements is left out. The equal-share nodal load
        # puts a local stress spike there that the production rigid handle
        # does not have; without it, KS and dKS/dt match the rigid-handle
        # solve (cos 0.96-1.00 on the designs where the spike dominated).
        keep = torch.ones(sa.N, dtype=torch.bool)
        if cfg["ks_exclude"] == "load_tile":
            keep[excl_rows[k]] = False
        vk = vm[keep]
        ks = vk.max() + torch.log(torch.exp(rho * (vk - vk.max())).sum()) / rho
        ks_all = vm.max() + torch.log(torch.exp(rho * (vm - vm.max())).sum()) / rho
        ks.backward()                          # d(KS)/dS: softmax-weighted dvm/dS, 0 on excluded nodes
        dne = (S.grad / cnt[:, None].clamp(min=1))[conn]               # (M,8,6)
        gl = torch.zeros(3 * sa.N)
        gl.index_add_(0, dof.reshape(-1), torch.einsum("mnk,mnkj->mj", dne, Rop).reshape(-1))
        lam = torch.as_tensor(sa.T @ sa._lu.solve(sa.T.T @ gl.numpy()))
        per_e = (torch.einsum("mnkj,mj,mnk->m", dRop, ue, dne)
                 - torch.einsum("mij,mj,mi->m", dK, ue, lam[dof]))
        grad = torch.zeros(len(keys)).index_add_(0, etile_k, per_e)    # Pa per unit t

        vm_np, u_np = vm.detach().numpy(), Ui[:, k].numpy().reshape(-1, 3)
        patch = set(patches[k])
        for nid in ids:
            n = nodes[nid]
            n.forces = [0.0, -FORCE_TOTAL / len(patches[k]), 0.0] if nid in patch else [0.0, 0.0, 0.0]
            n.displacement = u_np[index[nid]].tolist()
            n.stress = float(vm_np[index[nid]])
        comp.mesh.solution_valid = True
        data = comp.to_graph_fixed(edge_index, tile_idx_g, (NX, NY, NZ), with_labels=True,
                                   stress_ks_rho_mpa=rho_mpa)
        del data.edge_index, data.tile_idx            # re-attached from graph.pt at load
        # KS labels follow the mask (Pa, like to_graph_fixed's); node_stress
        # stays the full field. ks_mask rows are graph rows (node ids 1..N).
        data.ks_stress = torch.tensor([[float(ks)]], dtype=torch.float)
        data.max_stress = torch.tensor([[float(vk.max())]], dtype=torch.float)
        data.ks_stress_all = torch.tensor([[float(ks_all)]], dtype=torch.float)
        data.ks_mask = keep.clone()
        data.ks_stress_adjoint = torch.tensor([float(ks) / 1e6], dtype=torch.float)
        data.grad_tile_idx = torch.arange(len(keys), dtype=torch.long)
        data.grad_dks_dx = (grad / 1e6).float()
        data.load_tile = torch.tensor([load_tiles[k]], dtype=torch.long)
        data.load_dims_idx = torch.tensor(
            [key_idx[tile_index_to_dims_key(load_tiles[k], NX, NY, NZ, keys)]], dtype=torch.long)
        data.load_force = torch.tensor([[0.0, -FORCE_TOTAL, 0.0]], dtype=torch.float)
        data.geometry_idx = torch.tensor([idx], dtype=torch.long)
        out.append(data)
        ks_list.append(float(ks) / 1e6)
        ks_all_list.append(float(ks_all) / 1e6)
    t_loads = time.perf_counter() - t0

    buf = io.BytesIO()
    torch.save(out, buf)
    stats = {"idx": idx, "volume": float(comp.get_volume()), "ks_mpa": ks_list,
             "ks_all_mpa": ks_all_list, "n_excluded": [len(e) for e in excl_rows],
             "load_tiles": load_tiles, "patch_nodes": [len(p) for p in patches],
             "t_mesh_jac": t_mesh, "t_factor_solve": t_solve, "t_precompute": t_pre,
             "t_loads": t_loads, "t_total": time.perf_counter() - t_start}
    return idx, buf.getvalue(), stats


# ----------------------------------------------------------------- driver
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--geometry", default="hollow_cube_11x11")
    p.add_argument("--num_geometries", type=int, default=100)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--mesh_threads", type=int, default=1)
    p.add_argument("--stress_ks_rho_mpa", type=float, default=0.25)
    p.add_argument("--ks_exclude", choices=["load_tile", "none"], default="load_tile",
                   help="nodes left out of the KS label and its gradient: load_tile = every "
                        "node of the loaded tile's elements (stored per sample as ks_mask)")
    p.add_argument("--two_level_frac", type=float, default=0.3)
    p.add_argument("--amplitude_range", default="0.05,0.5")
    p.add_argument("--shard_geometries", type=int, default=10,
                   help="designs per shard (each design is NY samples)")
    args = p.parse_args()

    out = args.output_dir
    if os.path.isdir(os.path.join(out, "shards")) and os.listdir(os.path.join(out, "shards")):
        sys.exit(f"{out}/shards already has data; pick a new --output_dir")
    os.makedirs(out, exist_ok=True)
    graph_src = f"data/{args.geometry}/dataset/graph.pt"
    shutil.copy(graph_src, os.path.join(out, "graph.pt"))

    amp = tuple(float(v) for v in args.amplitude_range.split(","))
    designs, kinds = sample_designs(args.geometry, args.num_geometries, args.seed,
                                    args.two_level_frac, amp)
    print(f"{len(designs)} designs: {kinds.count('field')} correlated-field, "
          f"{kinds.count('two_level')} two-level", flush=True)
    cfg = {"geometry": args.geometry, "rho_mpa": args.stress_ks_rho_mpa, "ks_exclude": args.ks_exclude,
           "mesh_threads": args.mesh_threads, "graph_path": graph_src}

    from training import dataset_io
    t0 = time.time()
    results, stats_all, n_written, failures = {}, [], 0, []
    next_idx = 0
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=20) as pool:
        it = pool.imap_unordered(_safe_label, [(i, d, cfg) for i, d in enumerate(designs)])
        for done, (i, blob, stats) in enumerate(it, 1):
            if blob is None:
                failures.append((i, stats))
                print(f"design {i} FAILED: {stats}", flush=True)
            else:
                results[i] = blob
                stats["kind"] = kinds[i]
                stats_all.append(stats)
            # write shards in design order, shard_geometries designs at a time
            while True:
                chunk = list(range(next_idx, min(next_idx + args.shard_geometries, len(designs))))
                if not chunk or not all(j in results or j in dict(failures) for j in chunk):
                    break
                samples = []
                for j in chunk:
                    if j in results:
                        samples += torch.load(io.BytesIO(results.pop(j)), weights_only=False)
                if samples:
                    dataset_io.save_shard(out, n_written, samples)
                    n_written += len(samples)
                next_idx = chunk[-1] + 1
            if done % 10 == 0 or done == len(designs):
                el = time.time() - t0
                print(f"[{done}/{len(designs)}] {n_written} samples written, {el:.0f}s elapsed, "
                      f"{el / done:.1f}s/design wall", flush=True)

    with open(os.path.join(out, "build_summary.json"), "w") as f:
        json.dump({"args": vars(args), "force_total_N": FORCE_TOTAL, "failures": failures,
                   "designs": sorted(stats_all, key=lambda s: s["idx"])}, f)
    ks = np.array([v for s in stats_all for v in s["ks_mpa"]])
    print(f"\ndone: {n_written} samples from {len(stats_all)} designs ({len(failures)} failed) "
          f"in {time.time() - t0:.0f}s")
    if len(ks):
        print(f"KS MPa: min {ks.min():.1f}  p5 {np.percentile(ks, 5):.1f}  median "
              f"{np.median(ks):.1f}  p95 {np.percentile(ks, 95):.1f}  max {ks.max():.1f}")
        for key in ("t_mesh_jac", "t_factor_solve", "t_precompute", "t_loads", "t_total"):
            print(f"  {key:15s} median {np.median([s[key] for s in stats_all]):.2f}s per design")


def _safe_label(job):
    try:
        return label_design(job)
    except Exception as e:  # one bad design must not kill the pilot
        import traceback
        return job[0], None, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"


if __name__ == "__main__":
    main()
