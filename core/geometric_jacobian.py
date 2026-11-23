"""Geometric Jacobian of mesh node coordinates w.r.t. per-tile thickness.

For hollow_cube's microstructure (model_lin.c), each tile's geometry is
built from its own scalar ShellThickness alone -- BuildHollowCubeTile only
reads LclData->ShellThicknesses[idx] for its own tile index, and within a
tile only the *inner* hole-wall corners depend on thickness; the *outer*
corners (which get welded to neighboring tiles by irit2inp's point-merge)
are constant. So tiles never share a thickness-sensitive node, and bumping
every tile's thickness by the same eps and remeshing once recovers every
tile's dNode/dThickness in a single extra mesh generation, instead of one
remesh per tile.
"""
from typing import Callable, Dict, Optional, Tuple

from .IritModel import IritCModel, IritModelBase


def _resolve_dims(dims_dict: dict) -> Dict[str, float]:
    return {
        k: (v["default"] if isinstance(v, dict) else v)
        for k, v in dims_dict.items()
    }




_MESH_POOL = None


def _mesh_pool(n):
    """Persistent spawn-context process pool for CAD meshing. Threads do not
    scale here (measured 11 meshes: 13-20 s in threads vs 3.6 s in processes,
    scratch on /dev/shm), and spawn avoids forking a torch-laden parent."""
    global _MESH_POOL
    if _MESH_POOL is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        _MESH_POOL = ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"))
    return _MESH_POOL


def _mesh_worker(args):
    """One CAD mesh in a worker process, returned as compact arrays (cheap to
    pickle): node ids, coords, element ids, connectivity, element tiles."""
    import numpy as np
    script, dims, U, V, W, fixed_dims, mesh_threads, model_cls = args
    nodes, elems, e2t = model_cls(script, dims_dict=dims, fixed_dims=fixed_dims,
                                  mesh_threads=mesh_threads).create_mesh(U=U, V=V, W=W)
    nid = np.fromiter(nodes.keys(), dtype=np.int64, count=len(nodes))
    X = np.array([nodes[k] for k in nid], dtype=np.float64)
    eid = np.fromiter(elems.keys(), dtype=np.int64, count=len(elems))
    conn = np.array([elems[k] for k in eid], dtype=np.int64)
    tiles = np.array([e2t[k] for k in eid], dtype=np.int64)
    return nid, X, eid, conn, tiles


def compute_tile_jacobian_colored(
    irt_script_path: str,
    dims_dict: dict,
    U: int, V: int, W: int,
    tile_grid: Tuple[int, int, int],
    fixed_dims: Optional[dict] = None,
    mesh_threads: int = 1,
    eps: float = 1e-4,
    model_cls: Callable[..., IritModelBase] = IritCModel,
    workers: int = 10,
    move_tol: float = 1e-9,
):
    """Per-tile dX/dt for tiles whose parameter also moves their neighbours'
    nodes (e.g. cross4_11x11, where a leg end on a shared face is
    min(own, neighbour) and the neighbour's leg tapers to it).

    compute_thickness_jacobian bumps every tile at once and credits each
    moving node to one tile, which is exact only when no node moves with two
    tiles' parameters. Here it assumes instead that a tile's parameter moves
    nodes of its own tile and its face neighbours only -- and CHECKS that on
    every call. Tiles are coloured (ix + 2 iy) mod 5 in 2D ((ix + 2 iy + 3 iz)
    mod 7 in 3D), which puts same-coloured tiles at least 3 apart, so their
    influence regions are disjoint: one +-eps remesh pair per colour (10
    meshes in 2D) recovers every tile's node motion separately.

    Central differences: at a min() tie (equal neighbouring parameters) the
    leg end moves with either tile on one side only, so each tile is credited
    half of that motion -- the average of the two one-sided derivatives.

    Returns (D, nodes0, elements0, elem_to_tile, nominal_model) with
    D = {dims_key: {node_id: (dx/dt, dy/dt, dz/dt)}}, nonzero entries only.
    Raises if a moving node lies in the influence region of no tile, or of
    more than one tile, of the bumped colour.
    """
    from .tile_sensitivity import tile_index_to_dims_key

    resolved = _resolve_dims(dims_dict)
    keys = list(resolved)
    NX, NY, NZ = tile_grid
    K = NX * NY * NZ
    key_of = {t: tile_index_to_dims_key(t, NX, NY, NZ, keys) for t in range(K)}
    ijk = {t: (t % NX, (t // NX) % NY, t // (NX * NY)) for t in range(K)}
    ncol = 5 if NZ == 1 else 7
    color = {t: (ijk[t][0] + 2 * ijk[t][1] + 3 * ijk[t][2]) % ncol for t in range(K)}

    nominal = model_cls(irt_script_path, dims_dict=resolved, fixed_dims=fixed_dims,
                        mesh_threads=mesh_threads)
    jobs = []
    for c in range(ncol):
        for sgn in (+1.0, -1.0):
            dd = dict(resolved)
            for t in range(K):
                if color[t] == c:
                    dd[key_of[t]] += sgn * eps
            jobs.append((c, sgn, dd))
    # all bumped meshes in parallel worker processes while the nominal one
    # is meshed here (it is also the CAD model whose volume the caller reads).
    # workers <= 1 meshes everything in this process: for callers that are
    # themselves pool workers (daemonic processes cannot start a pool), e.g.
    # datasets/build_cross4_dataset.py.
    if workers <= 1:
        nodes0, elements0, e2t = nominal.create_mesh(U=U, V=V, W=W)
        results = [_mesh_worker((irt_script_path, j[2], U, V, W, fixed_dims, mesh_threads, model_cls))
                   for j in jobs]
    else:
        pool = _mesh_pool(min(workers, len(jobs)))
        futs = [pool.submit(_mesh_worker, (irt_script_path, j[2], U, V, W, fixed_dims, 1, model_cls))
                for j in jobs]
        nodes0, elements0, e2t = nominal.create_mesh(U=U, V=V, W=W)
        results = [f.result() for f in futs]

    # influence region of each tile: nodes of its own and its face neighbours' elements
    tile_nodes = {t: set() for t in range(K)}
    for eid, nl in elements0.items():
        tile_nodes[e2t[eid]].update(nl)
    def nbrs(t):
        i, j, k = ijk[t]
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = i + di, j + dj, k + dk
            if 0 <= a < NX and 0 <= b < NY and 0 <= c < NZ:
                yield a + NX * b + NX * NY * c
    influence = {t: tile_nodes[t].union(*(tile_nodes[n] for n in nbrs(t))) for t in range(K)}

    import numpy as np
    ids = list(nodes0)
    eids0 = list(elements0)
    conn0 = np.array([elements0[k] for k in eids0], dtype=np.int64)
    tiles0 = np.array([e2t[k] for k in eids0], dtype=np.int64)
    X0ids = np.array(ids, dtype=np.int64)
    def check_and_index(r):
        nid, X, eid, conn, tiles = r
        if (len(nid) != len(ids) or len(eid) != len(eids0) or not np.array_equal(eid, np.array(eids0))
                or not np.array_equal(conn, conn0) or not np.array_equal(tiles, tiles0)
                or set(nid.tolist()) != set(ids)):
            raise RuntimeError("mesh topology changed under the colour bump -- "
                               "the fixed-topology FD Jacobian does not apply")
        order = {k: i for i, k in enumerate(nid.tolist())}
        return X[[order[k] for k in ids]]
    meshes = [check_and_index(r) for r in results]

    D = {key_of[t]: {} for t in range(K)}
    for c in range(ncol):
        Xp = meshes[next(i for i, j in enumerate(jobs) if j[0] == c and j[1] > 0)]
        Xm = meshes[next(i for i, j in enumerate(jobs) if j[0] == c and j[1] < 0)]
        dX = (Xp - Xm) / (2.0 * eps)
        tiles_c = [t for t in range(K) if color[t] == c]
        moved = np.where(np.abs(dX).max(axis=1) > move_tol)[0]
        for r in moved:
            nid = ids[r]
            owners = [t for t in tiles_c if nid in influence[t]]
            if len(owners) != 1:
                raise RuntimeError(
                    f"node {nid} moved under colour {c} but lies in the influence region "
                    f"of {len(owners)} tiles of that colour -- a tile moves nodes beyond "
                    "its face neighbours, so this colouring is not valid for this geometry")
            D[key_of[owners[0]]][nid] = tuple(float(v) for v in dX[r])
    return D, nodes0, elements0, e2t, nominal
