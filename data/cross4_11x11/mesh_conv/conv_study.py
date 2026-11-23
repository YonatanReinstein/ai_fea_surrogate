"""Mesh-convergence study for the cross4_11x11 multiload dataset setup.

For each design and each mesh resolution (U,V,W per IRIT piece), solve all 22
dataset load cases on one factorization (same physics as
datasets.build_cross4_multiload_dataset.solve_design) and save:
  per case: max_vm, max_vm_excl (loaded tile excluded), ks (rho .25/MPa),
            per-tile max von Mises (121), location of the global max
  x = 0 face: node coords + nodal stress (for field comparison vs a reference)
Run from the repo root with the dataset env vars set.
    python conv_study.py --designs 1000,2100 --res 6,6,4 8,8,2 --out DIR
"""
import argparse, importlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from datasets.build_cross4_multiload_dataset import load_spec, batched_nodal_stress, DIRS, GEOMETRY


def solve_res(m, res, bc, cases, F, rho_mpa=0.25):
    from core.component import Component
    from core.fem_adjoint import ShapeAdjoint, _vm
    os.environ["FEA_MESH_RES"] = ",".join(map(str, res))
    c = Component(CAD_model=m, young=2e11, poisson=0.3)
    t0 = time.time()
    c.generate_mesh(*res)
    t_mesh = time.time() - t0
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
        forces.append({n.id: list(n.forces) for n in N if any(n.forces)})
    t0 = time.time()
    sa = ShapeAdjoint(nodes, elements, anchored, [], (0.0, 0.0, 0.0), mp["young_modulus"],
                      mp["poisson_ratio"], rigid_tie=False, nodal_forces=forces[0])
    sa.solve()
    C = len(cases)
    Ff = np.zeros((3 * sa.N, C))
    for k in range(C):
        for nid, fv in forces[k].items():
            Ff[3 * sa.index[nid]:3 * sa.index[nid] + 3, k] = fv
    Q = sa._lu.solve(sa.T.T @ Ff)
    U = np.asarray(sa.T @ Q).T.reshape(C, sa.N, 3)
    S = batched_nodal_stress(sa, U)
    VM = _vm(S).numpy()
    S = S.numpy()
    t_solve = time.time() - t0
    ids = sa.ids
    idx = sa.index
    coords = np.array([nodes[i] for i in ids])
    NX, NY, _ = bc.tile_grid()
    ntile = NX * NY
    # node -> tiles (a node on a tile face belongs to both)
    node_tiles = [[] for _ in ids]
    for eid, nl in elements.items():
        t = c.elem_to_tile[eid]
        for nid in nl:
            node_tiles[idx[nid]].append(t)
    pairs = np.array([(i, t) for i, ts in enumerate(node_tiles) for t in set(ts)])
    tile_max = np.zeros((C, ntile))
    for k in range(C):
        np.maximum.at(tile_max[k], pairs[:, 1], VM[k][pairs[:, 0]])
    rho = rho_mpa * 1e-6
    out = dict(max_vm=[], max_vm_excl=[], ks=[], argmax_xyz=[], argmax_excl_xyz=[])
    for k, (row, direction) in enumerate(cases):
        vm = VM[k]
        lt = (NX - 1) + NX * row
        excl = np.array([lt not in ts for ts in node_tiles])
        m0 = vm.max()
        out["max_vm"].append(m0)
        out["ks"].append(m0 + np.log(np.exp(rho * (vm - m0)).sum()) / rho)
        out["max_vm_excl"].append(vm[excl].max())
        out["argmax_xyz"].append(coords[vm.argmax()])
        ie = np.flatnonzero(excl)[vm[excl].argmax()]
        out["argmax_excl_xyz"].append(coords[ie])
    out = {k: np.array(v) for k, v in out.items()}
    # corner Jacobians of every hex (standard node order): count elements with
    # a non-positive corner determinant, relative to the element's mean
    X = coords[np.array([[idx[n] for n in elements[e]] for e in sorted(elements)])]
    nb = [(1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7), (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3)]
    dets = np.stack([np.linalg.det(np.stack([X[:, a] - X[:, i], X[:, b] - X[:, i], X[:, c] - X[:, i]], -1))
                     for i, (a, b, c) in enumerate(nb)], 1)
    rel = dets / np.abs(dets).mean(1, keepdims=True)
    n_bad = int((rel.min(1) <= 0.0).sum())
    sj_min = float(rel.min())
    face = np.abs(coords[:, 0]) < 1e-6
    out.update(n_bad=n_bad, sj_min=sj_min, tile_max=tile_max, n_nodes=sa.N, n_elem=len(elements), t_mesh=t_mesh, t_solve=t_solve,
               face_xyz=coords[face].astype(np.float32), face_S=S[:, face].astype(np.float32),
               face_ids=np.array(ids)[face])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", required=True)
    p.add_argument("--res", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="data/cross4_11x11/dataset_multiload_5000")
    p.add_argument("--model", default=f"data/{GEOMETRY}/CAD_model/model")
    p.add_argument("--tmap", default=None, help="tmin,tmax: map the design's t linearly from [0.01,0.47] into this range")
    args = p.parse_args()
    from core.IritModel import IritCModel
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS") or 8))
    bc = importlib.import_module(f"data.{GEOMETRY}.boundary_conditions")
    NX, NY, _ = bc.tile_grid()
    cases = [(r, d) for r in range(NY) for d in (0, 1)]
    F = float(os.environ.get("FEA_LOAD_N") or 1e7)
    designs = json.load(open(f"{args.dataset}/designs.json"))["designs"]
    os.makedirs(args.out, exist_ok=True)
    json.dump({k: v for k, v in os.environ.items() if k.startswith(("FEA_", "CROSS4_"))},
              open(f"{args.out}/env.json", "w"), indent=1)
    for di in (int(x) for x in args.designs.split(",")):
        dd = designs[di]
        if args.tmap:
            lo, hi = (float(x) for x in args.tmap.split(","))
            dd = {k: lo + (v - 0.01) / 0.46 * (hi - lo) for k, v in dd.items()}
        m = IritCModel(args.model, dims_dict=dd, mesh_threads=8,
                       fixed_dims=bc.fixed_dims())
        vol = m.get_volume()
        for r in args.res:
            res = tuple(int(x) for x in r.split(","))
            fn = f"{args.out}/d{di:05d}_r{'_'.join(map(str, res))}.npz"
            if os.path.exists(fn):
                continue
            try:
                o = solve_res(m, res, bc, cases, F)
            except Exception as e:
                print(f"design {di} res {res}: FAILED {e!r}", flush=True)
                continue
            np.savez(fn, volume=vol, t=np.array(list(dd.values())), **o)
            print(f"design {di} res {res}: nodes {o['n_nodes']} elem {o['n_elem']} mesh {o['t_mesh']:.1f}s "
                  f"solve {o['t_solve']:.1f}s  max_excl {o['max_vm_excl'].max()/1e6:.2f}  "
                  f"ks {o['ks'].max()/1e6:.2f} MPa  bad_elem {o['n_bad']} relJmin {o['sj_min']:.3f}", flush=True)
        del m


if __name__ == "__main__":
    main()
