"""Asynchronous CCX-adjoint labeller for designs a running topopt visits.

Why this exists
---------------
Measured 2026-09-10: the surrogate's per-tile dg matches the ccx adjoint at
cosine +0.98 on RANDOM designs matched to a trajectory design's mean/std/total
variation, but only +0.09 on the trajectory design itself. Ten matched-random
controls scored +0.94..+0.996 with no overlap. The failing designs are therefore
not unusual in any measurable way -- they are the designs OC reaches by
descending the surrogate's OWN gradient, i.e. selected by its errors. No i.i.d.
sampler can produce them, so labels have to come from a live trajectory.

Why it is a spool and not a queue
---------------------------------
A ccx adjoint costs ~46s (CPU-bound: one CAD remesh + factorisation + two back
substitutions) against ~4s for a topopt generation (GPU forward/backward plus
one mesh). They overlap, so labelling is free in wall-clock IF the optimiser
never waits on it. The optimiser therefore only writes a ~10KB .npz to a spool
directory and moves on; this process picks them up out of band. A labeller
crash, restart, or slow node cannot stall or corrupt the optimisation.

Usage
-----
    python -m training.trajectory_labeller --spool DIR --labels DIR [--workers 1]

and on the optimiser side set TOPOPT_LABEL_SPOOL (and optionally
TOPOPT_LABEL_EVERY, default 5).
"""
import argparse
import os
import time

import numpy as np


def _gen_of(name):
    try:
        return int(name[4:10])
    except ValueError:
        return -1


def _claim(spool, max_lag, order):
    """Atomically take the NEWEST pending design, and drop anything more than
    max_lag generations behind it.

    The optimiser must never wait on labelling, so it spools faster than the
    labellers can drain. FIFO claiming would then work through an ever-older
    backlog and every label would describe a design the optimiser left long
    ago. LIFO keeps the lag bounded at roughly (labellers x label_seconds) /
    generation_seconds, and the purge keeps the spool from growing without
    bound. os.rename is atomic within a filesystem, so several labellers can
    share one spool with no lock."""
    names = [n for n in os.listdir(spool)
             if n.startswith("gen_") and n.endswith(".npz") and ".claimed" not in n]
    if not names:
        return None, None
    newest = max(_gen_of(n) for n in names)
    for n in names:
        if max_lag and _gen_of(n) < newest - max_lag:
            try:
                os.remove(os.path.join(spool, n))
            except OSError:
                pass
    for name in sorted(names, key=_gen_of, reverse=(order == "lifo")):
        if max_lag and _gen_of(name) < newest - max_lag:
            continue
        src = os.path.join(spool, name)
        # Per-process claim name. Over NFS on a lossy link (newton2's NIC) a
        # retransmitted RENAME can report success to a client whose request
        # lost the race, so with a shared "<name>.claimed.npz" target three
        # labellers each believed they owned gen_000030 (run T), all solved
        # it, and all died at the final os.remove. A unique target means a
        # spurious success is caught by the existence check right after.
        dst = os.path.join(spool, f"{name[:-4]}.claimed.{os.getpid()}.npz")
        try:
            os.rename(src, dst)
        except OSError:
            continue          # someone else got it
        if os.path.exists(dst) and not os.path.exists(src):
            return dst, name[:-4]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--geometry", default="hollow_cube")
    ap.add_argument("--stress_ks_rho_mpa", type=float, default=0.15)
    ap.add_argument("--mesh_threads", type=int, default=4)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4,
                    help="ccx evaluator process pool size")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--claim_order", choices=("fifo", "lifo"), default="fifo",
                    help="fifo: oldest design first (deterministic pipelines); lifo: newest first (non-blocking runs)")
    ap.add_argument("--max_lag", type=int, default=0,
                    help="purge spooled designs more than this many generations behind the newest (0 = keep all)")
    ap.add_argument("--idle_exit", type=float, default=0.0,
                    help="exit after this many seconds with an empty spool (0 = never)")
    a = ap.parse_args()

    os.makedirs(a.spool, exist_ok=True)
    os.makedirs(a.labels, exist_ok=True)

    from evaluators.ccx_evaluator import CCXEvaluator
    from evaluators.gnn_evaluator import _run_sample_worker
    import importlib
    ev = CCXEvaluator(a.geometry, max_workers=a.workers, mesh_threads=a.mesh_threads,
                      nproc=a.nproc, stress_ks_rho_mpa=a.stress_ks_rho_mpa,
                      gradient_source="adjoint")
    # The trainer used to re-mesh every labelled design to build its GNN graph
    # (~14 s of a ~38 s update, on the optimiser's critical path). The graph
    # depends only on the design, so build it here -- the labellers have
    # slack -- and ship it inside the label; the trainer uses it when present.
    _bc = importlib.import_module(f"data.{a.geometry}.boundary_conditions")
    _tile_grid = _bc.tile_grid() if hasattr(_bc, "tile_grid") else None
    _graph_path = f"data/{a.geometry}/dataset/graph.pt"
    _graph_path = _graph_path if os.path.exists(_graph_path) else None
    # CCXEvaluator.model_path is the CAD_model DIRECTORY; the sample worker
    # wants the executable / .irt inside it (GNNEvaluator's convention).
    _mp = os.path.join(ev.model_path, "model.irt")
    _mp = _mp if os.path.exists(_mp) else os.path.join(ev.model_path, "model")
    _fixed_ei = _fixed_ti = None
    if _graph_path is not None:
        from evaluators.gnn_evaluator import _get_fixed_graph
        _fixed_ei, _fixed_ti = (t.numpy() for t in _get_fixed_graph(_graph_path))
    def _build_graph(dims):
        gd, vol, _t, _m = _run_sample_worker((_mp, dims, ev.fixed_dims,
                                              ev.material_properties["young_modulus"],
                                              ev.material_properties["poisson_ratio"],
                                              0, False, False, ev.anchor_condition, ev.force_pattern,
                                              ev.U, ev.V, ev.W, _tile_grid, _graph_path))
        out = {}
        for k, v in gd.items():
            v = np.asarray(v)
            # frozen topology: edge_index / tile_idx are the same for every
            # design and the trainer holds them in graph.pt -- don't ship
            # 14 MB of them per label (19.9 MB -> ~5 MB, compressed ~3 MB)
            if k == "edge_index" and _fixed_ei is not None and v.shape == _fixed_ei.shape and np.array_equal(v, _fixed_ei):
                continue
            if k == "tile_idx" and _fixed_ti is not None and v.shape == _fixed_ti.shape and np.array_equal(v, _fixed_ti):
                continue
            if v.dtype == np.float64:
                v = v.astype(np.float32)
            out[f"graph__{k}"] = v
        return out
    print(f"[labeller] spool={a.spool} labels={a.labels} rho={a.stress_ks_rho_mpa} "
          f"graph_in_label=True", flush=True)

    n_done = 0
    idle_since = None
    while True:
        path, tag = _claim(a.spool, a.max_lag, a.claim_order)
        if path is None:
            if a.idle_exit:
                idle_since = idle_since or time.time()
                if time.time() - idle_since > a.idle_exit:
                    print(f"[labeller] idle {a.idle_exit:.0f}s, exiting after {n_done} labels", flush=True)
                    return
            time.sleep(a.poll)
            continue
        idle_since = None

        t_claim = time.time()
        z = np.load(path, allow_pickle=True)
        names = [str(n) for n in z["free_names"]]
        dims = {n: float(v) for n, v in zip(names, z["x"])}
        t0 = time.time()
        try:
            vol, stress, yld, dV_dx, dG_dx = ev.compute_topopt_gradient(dims)
        except Exception as e:
            # Keep the claimed file so the failure is inspectable rather than lost.
            print(f"[labeller] {tag}: FAILED after {time.time()-t0:.1f}s: {e}", flush=True)
            continue
        dt = time.time() - t0
        try:
            graph = _build_graph(dims)
        except Exception as e:
            print(f"[labeller] {tag}: graph build failed ({e}); label without graph", flush=True)
            graph = {}

        dg = np.array([dG_dx[n] for n in names], dtype=float)
        out = os.path.join(a.labels, f"{tag}.npz")
        # np.savez APPENDS .npz unless the name already ends in it, so a
        # "foo.npz.tmp" temp becomes "foo.npz.tmp.npz" and the rename below
        # fails. Dot-prefix instead -- readers skip dotfiles.
        tmp = os.path.join(a.labels, f".{tag}.npz")
        np.savez_compressed(tmp, free_names=np.array(names), x=z["x"], dg_ccx=dg,
                 ccx_stress=stress, ccx_volume=vol, yield_strength=yld,
                 label_seconds=dt,
                 # pipeline timing (wall clock): spooled by the optimiser ->
                 # claimed here -> solved -> graph built -> written
                 t_spool=float(z["t_spool"]) if "t_spool" in z.files else np.nan,
                 t_claim=t_claim, t_solved=t0 + dt, t_done=time.time(),
                 surrogate_dg=z["dg_raw"] if "dg_raw" in z.files else np.zeros(0),
                 surrogate_stress=z["s0"] if "s0" in z.files else np.nan,
                 gen=z["gen"] if "gen" in z.files else -1, **graph)
        os.replace(tmp, out)          # atomic publish: a reader never sees a partial label
        try:
            os.remove(path)
        except OSError:
            pass                      # never die over cleanup
        n_done += 1

        # If the optimiser spooled its own dg, report agreement immediately --
        # this is the on-trajectory number that held-out probes cannot measure.
        msg = ""
        if "dg_raw" in z.files and np.size(z["dg_raw"]):
            m = np.asarray(z["dg_raw"], float)
            c = float(m @ dg / (np.linalg.norm(m) * np.linalg.norm(dg) + 1e-30))
            ok = np.sign(m) == np.sign(dg)
            wrong = float(np.abs(dg)[~ok].sum() / (np.abs(dg).sum() + 1e-30))
            msg = f"  cosine {c:+.4f}  wrong-way mass {100*wrong:.1f}%"
        print(f"[labeller] {tag}: {dt:.1f}s  ccx stress {stress:.2f} MPa{msg}  "
              f"(total {n_done})", flush=True)


if __name__ == "__main__":
    main()
