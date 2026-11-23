"""cross4_11x11 dataset: every sampled design solved with the load on EACH
free-end tile row, on exactly the pipeline the cross4 topopt runs use.

Unlike hollow_cube's tile-load dataset (datasets/build_tile_load_dataset.py),
the loads cannot share a factorization here: the load pad and its fixed-width
flare sit on the loaded tile only (model_lin.c reads FEA_LOAD_ROW), so every
(design, load row) pair is its own CAD model, mesh and solve. Each task is:

    11 CAD meshes   nominal + 5 colours x +-eps (core.geometric_jacobian.
                    compute_tile_jacobian_colored, meshed in this process)
    1 solve         ShapeAdjoint (C3D8I, PARDISO), rigid-tied load pad
    labels          masked KS (TOPOPT_KS_EXCLUDE, normally "pad"), max stress,
                    exact per-tile dKS/dt and d(mesh volume)/dt

which is what evaluators/ccx_evaluator.py's compute_topopt_gradient returns to
the optimizer with TOPOPT_TILE_JACOBIAN=colored -- same KS rho, same mask, same
mesh volume.

The mesh topology depends on the load row (the pad's boxes and fillet), so the
fixed graph is per row: graph_row{k}.pt (edge_index, tile_idx, x_static),
written from the first sample of that row; every later sample's graph is
checked against it (a mismatch fails that task loudly).

Samples are slimmed to what varies between designs (the full set was 4.3 MB a
sample and filled the /home quota on 2026-10-10):
    coords     (N,3) node coordinates -- the design-dependent part of x
    x_static   (N,7) in graph_row{k}.pt: forces, anchored, normalized tile
               position -- identical for every design at load row k
    rebuild    x = cat([coords, x_static]) (Component.to_graph_fixed's x)
               edge_attr = ||coords[ei[0]] - coords[ei[1]]||, ei = edge_index
Samples also omit edge_index/tile_idx like the other builders. Remaining
schema follows Component.to_graph_fixed (node_disp, node_stress [Pa], volume,
dims, young, poisson, max_stress, ks_stress, max_displacement) plus:

    ks_mask           graph rows, True = counts in KS (pads excluded)
    ks_stress_all     KS over every node (Pa)
    grad_dks_dx       dKS/dt per tile, MPa per unit thickness, dims-key order
    grad_dv_dx        d(mesh volume)/dt per tile, dims-key order
    volume_mesh       hex-mesh volume (the optimizer's objective)
    load_row, load_tile, load_dims_idx, load_force, geometry_idx

Physics settings come from the environment, exactly as for the opts (pass them
all explicitly): FEA_LOAD_N, FEA_MESH_RES, CROSS4_PAD_THICKNESS,
CROSS4_LOAD_PAD_THICKNESS, TOPOPT_KS_EXCLUDE. They are recorded in
build_config.json.

Screening (--screen_row R --screen_max_ks X): run the build once with
--rows R, then again with --rows all and the screen flags: the second pass
only runs the other rows for designs whose row-R KS (from done_tasks.jsonl)
is <= X MPa. Random designs with thin-tile weak links reach 1e3-3e4 MPa
(35% of a 40-design pilot); they would be just as broken under any load row.
Their row-R samples stay in the shards -- filter on ks_stress when training.

Resumable: tasks already in shards (listed in done_tasks.jsonl) are skipped on
restart with the same --output_dir and --seed.

    FEA_LOAD_N=5e6 FEA_MESH_RES=6,6,4 CROSS4_PAD_THICKNESS=0.1 \\
    CROSS4_LOAD_PAD_THICKNESS=0.1 TOPOPT_KS_EXCLUDE=pad \\
    python -m datasets.build_cross4_dataset --num_geometries 10 --workers 8 \\
        --output_dir data/cross4_11x11/dataset_smoke
"""
import argparse
import hashlib
import importlib
import io
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import torch

GEOMETRY = "cross4_11x11"
ENV_KEYS = ("FEA_LOAD_N", "FEA_MESH_RES", "CROSS4_PAD_THICKNESS", "CROSS4_LOAD_PAD_THICKNESS",
            "TOPOPT_KS_EXCLUDE", "CROSS4_PAD_FILLET", "CROSS4_FLARE_CURV", "CROSS4_CNTR_SHAPE",
            "CROSS4_BNDRY_SHAPE", "CROSS4_MAX_CENTRE", "CROSS4_LOAD_FLARE_HALFWIDTH")


def _edges_and_tiles(elements0, e2t, n_nodes):
    """Graph exactly as Component.to_graph_with_labels builds it: every node
    pair within an element, both directions, rows = node id - 1; a node's tile
    is that of the first element (in mesh order) that contains it."""
    conn = np.array([elements0[e] for e in elements0], dtype=np.int64) - 1
    i, j = np.triu_indices(conn.shape[1], 1)
    a, b = conn[:, i].ravel(), conn[:, j].ravel()
    pairs = np.unique(np.concatenate([a * n_nodes + b, b * n_nodes + a]))
    edge_index = torch.as_tensor(np.stack([pairs // n_nodes, pairs % n_nodes]), dtype=torch.long)
    tile = np.full(n_nodes, -1, dtype=np.int64)
    for eid, nl in elements0.items():
        for n in nl:
            if tile[n - 1] < 0:
                tile[n - 1] = e2t[eid]
    tile[tile < 0] = 0
    return edge_index, torch.as_tensor(tile)


def label_task(job):
    """One (design, load row). Returns (task, sample bytes, graph bytes, stats)."""
    (idx, row), dims, cfg = job
    os.environ["FEA_LOAD_ROW"] = str(row)       # read by model_lin.c and boundary_conditions
    torch.set_num_threads(1)
    from core.element import Element
    from core.fem_adjoint import ShapeAdjoint, _geom
    from core.geometric_jacobian import compute_tile_jacobian_colored
    from core.IritModel import IritCModel
    from core.mesh import Mesh
    from core.node import Node
    from core.tile_sensitivity import tile_index_to_dims_key
    from evaluators.ccx_evaluator import _ks_excluded_node_ids

    t0 = time.perf_counter()
    model_dir = f"data/{GEOMETRY}/CAD_model"
    bc = importlib.import_module(f"data.{GEOMETRY}.boundary_conditions")
    mat = json.load(open(f"{model_dir}/material_properties.json"))
    E, nu, yld = mat["young_modulus"], mat["poisson_ratio"], mat["yield_strength"]
    U, V, W = bc.mesh_resolution()
    NX, NY, NZ = bc.tile_grid(None)
    keys = cfg["keys"]
    rho = cfg["rho_mpa"]
    assert bc.load_row() == row

    D_tile, nodes0, elements0, e2t, cad = compute_tile_jacobian_colored(
        f"{model_dir}/model", dims, U, V, W, (NX, NY, NZ), fixed_dims=bc.fixed_dims(),
        model_cls=IritCModel, workers=1)
    t_mesh = time.perf_counter() - t0
    N = len(nodes0)
    assert sorted(nodes0) == list(range(1, N + 1)), "node ids must be 1..N (graph rows)"

    nodes = {k: Node(k, list(v)) for k, v in nodes0.items()}
    mesh = Mesh(nodes, {e: Element(e, [nodes[q] for q in nl]) for e, nl in elements0.items()})
    mesh.anchor_nodes_by_condition(bc.anchor_condition)
    mesh.apply_force_by_pattern(bc.force_pattern)
    ns = mesh.all_nodes()
    sa = ShapeAdjoint(nodes0, elements0, [q.id for q in ns if q.anchored],
                      [q.id for q in ns if any(q.forces)],
                      [sum(q.forces[j] for q in ns) for j in range(3)], E, nu)
    dX = {}
    for d in D_tile.values():
        for n, v in d.items():
            o = dX.get(n, (0.0, 0.0, 0.0))
            dX[n] = tuple(p + q for p, q in zip(o, v))
    sa.assert_tie_is_fixed(dX)
    excl = _ks_excluded_node_ids(model_dir, elements0, e2t, nodes0)
    keep = np.array([i not in excl for i in sa.ids])
    if excl:
        sa.ks_mask = keep.tolist()
    sa.solve()
    _, vm = sa.nodal_stress()
    vm = vm.detach().numpy()                                     # Pa, sa.ids order == rows

    def _ks(v):                                                  # == ShapeAdjoint.ks, Pa
        v = v / 1e6
        return float((v.max() + np.log(np.exp(rho * (v - v.max())).sum()) / rho) * 1e6)
    ks = _ks(vm[keep])                                           # masked, what the opts use
    ks_all = _ks(vm)

    # exact per-tile gradients (same contraction as CCXEvaluator._colored_tile_gradient)
    gX = sa.gradient(rho).detach().numpy()
    X = sa.X.detach().clone().requires_grad_(True)
    _, detJ = _geom(X[torch.as_tensor(sa.conn)])
    vol_mesh = detJ.abs().sum()
    vol_mesh.backward()
    gV = X.grad.numpy()
    rowof = {nid: i for i, nid in enumerate(sa.ids)}
    g_ks = np.zeros(len(keys))
    g_v = np.zeros(len(keys))
    kidx = {k: i for i, k in enumerate(keys)}
    for k, d in D_tile.items():
        for n, v in d.items():
            g_ks[kidx[k]] += float(gX[rowof[n]] @ np.asarray(v))
            g_v[kidx[k]] += float(gV[rowof[n]] @ np.asarray(v))
    t_solve = time.perf_counter() - t0 - t_mesh

    # ---- sample (Component.to_graph_fixed schema)
    from torch_geometric.data import Data
    edge_index, tile_idx = _edges_and_tiles(elements0, e2t, N)
    coords = torch.tensor([nodes0[i] for i in sa.ids], dtype=torch.float)
    forces = torch.tensor([nodes[i].forces or [0.0, 0.0, 0.0] for i in sa.ids], dtype=torch.float)
    anchored = torch.tensor([[float(nodes[i].anchored)] for i in sa.ids], dtype=torch.float)
    t = tile_idx
    pos = torch.stack([(t % NX).float() / max(NX - 1, 1), ((t // NX) % NY).float() / max(NY - 1, 1),
                       (t // (NX * NY)).float() / max(NZ - 1, 1)], dim=1)
    x_static = torch.cat([forces, anchored, pos], dim=1)              # same for every design at this row
    data = Data(coords=coords)                                        # x, edge_attr: see module doc
    data.tile_NX, data.tile_NY, data.tile_NZ = (torch.tensor([v], dtype=torch.long) for v in (NX, NY, NZ))
    u = sa.u.detach().numpy().reshape(-1, 3)
    data.node_disp = torch.tensor(u, dtype=torch.float).unsqueeze(0)
    data.node_stress = torch.tensor(vm, dtype=torch.float).view(1, -1, 1)
    data.volume = torch.tensor([[cad.get_volume()]], dtype=torch.float)
    data.volume_mesh = torch.tensor([[float(vol_mesh)]], dtype=torch.float)
    data.dims = torch.tensor([cad.get_dim_list()], dtype=torch.float)
    data.young = torch.tensor([[E]], dtype=torch.float)
    data.poisson = torch.tensor([[nu]], dtype=torch.float)
    data.max_stress = torch.tensor([[float(vm[keep].max())]], dtype=torch.float)
    data.ks_stress = torch.tensor([[float(ks)]], dtype=torch.float)
    data.ks_stress_all = torch.tensor([[ks_all]], dtype=torch.float)
    data.max_displacement = torch.tensor([[float(np.linalg.norm(u, axis=1).max())]], dtype=torch.float)
    data.ks_mask = torch.as_tensor(keep)
    data.grad_tile_idx = torch.arange(len(keys), dtype=torch.long)
    data.grad_dks_dx = torch.tensor(g_ks / 1e6, dtype=torch.float)     # MPa per unit t
    data.grad_dv_dx = torch.tensor(g_v, dtype=torch.float)
    load_tile = (NX - 1) + NX * row                                    # free-end column, row `row`
    data.load_row = torch.tensor([row], dtype=torch.long)
    data.load_tile = torch.tensor([load_tile], dtype=torch.long)
    data.load_dims_idx = torch.tensor([kidx[tile_index_to_dims_key(load_tile, NX, NY, NZ, keys)]],
                                      dtype=torch.long)
    data.load_force = torch.tensor([list(bc.force_pattern.resultant)], dtype=torch.float)
    data.geometry_idx = torch.tensor([idx], dtype=torch.long)

    gbuf = io.BytesIO()
    torch.save({"edge_index": edge_index, "tile_idx": tile_idx, "x_static": x_static}, gbuf)
    ghash = hashlib.sha1(edge_index.numpy().tobytes() + tile_idx.numpy().tobytes()).hexdigest()
    shash = hashlib.sha1(x_static.numpy().tobytes()).hexdigest()
    sbuf = io.BytesIO()
    torch.save(data, sbuf)
    stats = {"idx": idx, "row": row, "ks_mpa": float(ks) / 1e6, "ks_all_mpa": ks_all / 1e6,
             "max_mpa": float(vm[keep].max()) / 1e6, "volume_mesh": float(vol_mesh),
             "n_nodes": N, "graph_hash": ghash, "static_hash": shash, "t_mesh": t_mesh, "t_solve": t_solve,
             "t_total": time.perf_counter() - t0}
    return (idx, row), sbuf.getvalue(), gbuf.getvalue(), stats


def _safe_label(job):
    try:
        return label_task(job)
    except Exception as e:  # one bad task must not kill the build
        import traceback
        return job[0], None, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--num_geometries", type=int, default=10)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rows", default="all", help="load rows: 'all' or comma list, e.g. 0,5,10")
    p.add_argument("--stress_ks_rho_mpa", type=float, default=0.25)
    p.add_argument("--two_level_frac", type=float, default=0.3)
    p.add_argument("--amplitude_range", default="0.05,0.5")
    p.add_argument("--shard_size", type=int, default=110, help="samples per shard")
    p.add_argument("--screen_row", type=int, default=None,
                   help="with --screen_max_ks: run other rows only for designs that passed on this row")
    p.add_argument("--screen_max_ks", type=float, default=None, help="screen threshold, MPa (masked KS)")
    args = p.parse_args()

    for k in ("FEA_LOAD_N", "FEA_MESH_RES", "CROSS4_PAD_THICKNESS", "CROSS4_LOAD_PAD_THICKNESS",
              "TOPOPT_KS_EXCLUDE"):
        if not os.environ.get(k):
            sys.exit(f"{k} must be set explicitly (same physics as the opts)")
    if os.environ.get("FEA_LOAD_ROW"):
        sys.exit("unset FEA_LOAD_ROW: the builder sets it per task")

    from datasets import dataset_io
    from datasets.build_tile_load_dataset import sample_designs

    out = args.output_dir
    os.makedirs(os.path.join(out, "shards"), exist_ok=True)
    bc = importlib.import_module(f"data.{GEOMETRY}.boundary_conditions")
    NX, NY, NZ = bc.tile_grid(None)
    rows = list(range(NY)) if args.rows == "all" else [int(r) for r in args.rows.split(",")]
    keys = list(json.load(open(f"data/{GEOMETRY}/CAD_model/dims.json")))

    cfg_path = os.path.join(out, "build_config.json")
    config = {"args": vars(args), "env": {k: os.environ.get(k) for k in ENV_KEYS}, "rows": rows}
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        same = {k: old["args"][k] for k in ("seed", "stress_ks_rho_mpa", "two_level_frac", "amplitude_range")} == \
               {k: config["args"][k] for k in ("seed", "stress_ks_rho_mpa", "two_level_frac", "amplitude_range")}
        if not same or old["env"] != config["env"]:
            sys.exit(f"{out} was built with different settings: {old}")
        print(f"resuming {out}", flush=True)
    json.dump(config, open(cfg_path, "w"), indent=1)

    amp = tuple(float(v) for v in args.amplitude_range.split(","))
    designs, kinds = sample_designs(GEOMETRY, args.num_geometries, args.seed, args.two_level_frac, amp)
    if not os.path.exists(os.path.join(out, "designs.json")):
        json.dump({"designs": designs, "kinds": kinds}, open(os.path.join(out, "designs.json"), "w"))
    print(f"{len(designs)} designs ({kinds.count('field')} correlated-field, "
          f"{kinds.count('two_level')} two-level) x {len(rows)} load rows", flush=True)

    done_path = os.path.join(out, "done_tasks.jsonl")
    done = set()
    if os.path.exists(done_path):
        for line in open(done_path):
            r = json.loads(line)
            done.add((r["idx"], r["row"]))
    n_written = len(done)
    graph_hash, static_hash = {}, {}
    for r in rows:
        gp = os.path.join(out, f"graph_row{r}.pt")
        if os.path.exists(gp):
            g = torch.load(gp, weights_only=False)
            graph_hash[r] = hashlib.sha1(g["edge_index"].numpy().tobytes() + g["tile_idx"].numpy().tobytes()).hexdigest()
            if "x_static" in g:
                static_hash[r] = hashlib.sha1(g["x_static"].numpy().tobytes()).hexdigest()

    passed = None
    if args.screen_max_ks is not None:
        if args.screen_row is None:
            sys.exit("--screen_max_ks needs --screen_row")
        ks_screen = {}
        for line in open(done_path):
            r = json.loads(line)
            if r["row"] == args.screen_row:
                ks_screen[r["idx"]] = r["ks_mpa"]
        passed = {i for i, k in ks_screen.items() if k <= args.screen_max_ks}
        print(f"screen: {len(ks_screen)} designs solved on row {args.screen_row}, {len(passed)} with "
              f"KS <= {args.screen_max_ks} MPa ({len(designs) - len(ks_screen)} not screened -> skipped)",
              flush=True)

    cfg = {"keys": keys, "rho_mpa": args.stress_ks_rho_mpa}
    # rows interleaved so every row's graph appears early and a partial build covers all rows
    tasks = [((i, r), designs[i], cfg) for i in range(len(designs)) for r in rows
             if (i, r) not in done and (passed is None or i in passed)]
    print(f"{len(tasks)} tasks to run ({len(done)} already done)", flush=True)

    t0 = time.time()
    pending, pending_stats, failures = [], [], 0
    fail_log = open(os.path.join(out, "failures.jsonl"), "a")

    def flush():
        nonlocal n_written, pending, pending_stats
        if not pending:
            return
        dataset_io.save_shard(out, n_written, pending)
        n_written += len(pending)
        with open(done_path, "a") as f:
            for s in pending_stats:
                f.write(json.dumps(s) + "\n")
        pending, pending_stats = [], []

    ctx = get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=50) as pool:
        for k, (task, blob, gblob, stats) in enumerate(pool.imap_unordered(_safe_label, tasks, chunksize=1), 1):
            if blob is None:
                failures += 1
                fail_log.write(json.dumps({"idx": task[0], "row": task[1], "error": stats}) + "\n")
                fail_log.flush()
                print(f"task {task} FAILED: {stats.splitlines()[0]}", flush=True)
            else:
                r = task[1]
                if r not in graph_hash:
                    with open(os.path.join(out, f"graph_row{r}.pt"), "wb") as f:
                        f.write(gblob)
                    graph_hash[r] = stats["graph_hash"]
                    static_hash[r] = stats["static_hash"]
                    print(f"row {r}: graph {stats['n_nodes']} nodes -> graph_row{r}.pt", flush=True)
                if r not in static_hash:
                    # graph file from before the slim format: add x_static to it
                    gp = os.path.join(out, f"graph_row{r}.pt")
                    g = torch.load(gp, weights_only=False)
                    g["x_static"] = torch.load(io.BytesIO(gblob), weights_only=False)["x_static"]
                    torch.save(g, gp + ".tmp")
                    os.replace(gp + ".tmp", gp)
                    static_hash[r] = stats["static_hash"]
                    print(f"row {r}: x_static added to graph_row{r}.pt", flush=True)
                if stats["graph_hash"] != graph_hash[r] or stats["static_hash"] != static_hash[r]:
                    failures += 1
                    fail_log.write(json.dumps({"idx": task[0], "row": r, "error": "graph mismatch"}) + "\n")
                    print(f"task {task}: graph differs from graph_row{r}.pt, dropped", flush=True)
                else:
                    pending.append(torch.load(io.BytesIO(blob), weights_only=False))
                    stats["kind"] = kinds[task[0]]
                    pending_stats.append(stats)
                    if len(pending) >= args.shard_size:
                        flush()
            if k % 50 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"[{k}/{len(tasks)}] {n_written + len(pending)} samples, {failures} failed, "
                      f"{el:.0f}s, {el / k:.2f}s/task wall, ETA {el / k * (len(tasks) - k) / 3600:.1f} h",
                      flush=True)
    flush()
    fail_log.close()
    print(f"done: {n_written} samples, {failures} failed in this session, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
