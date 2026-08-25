import faulthandler, json, random, torch, os, importlib, signal, sys, threading, time
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import PchipInterpolator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from training import dataset_io
print("Starting dataset_builder.py", flush=True)
import socket
print(f"Running on hostname: {socket.gethostname()}", flush=True)
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
# PyMAPDL defaults to UDS transport on Linux, but MAPDL 2025 R1 only serves
# gRPC over plain TCP — force insecure (TCP) transport so the client connects.
os.environ["PYMAPDL_GRPC_TRANSPORT"] = "insecure"

# Every _ccx_run_single result (a PyG Data object) crosses the ProcessPoolExecutor
# boundary and gets unpickled in the main process. torch's default "file_descriptor"
# sharing strategy leaves one open fd per received tensor (visible as deleted
# /dev/shm/torch_* entries in /proc/<pid>/fd) instead of releasing it -- over tens
# of thousands of samples this exhausts the pool's IPC and hangs the whole run with
# no traceback. "file_system" tracks shared tensors by refcounted /tmp files instead
# of held-open fds, so long-running builds don't accumulate fds.
torch.multiprocessing.set_sharing_strategy("file_system")

_pool_ref = None

# Per-process cache of the frozen-graph constants (edge_index, tile_idx),
# loaded once per worker from graph.pt and reused for every sample -- mirrors
# evaluators/gnn_evaluator.py's _FIXED_GRAPH_CACHE.
_FIXED_GRAPH_CACHE = {}


def _get_fixed_graph(graph_path):
    g = _FIXED_GRAPH_CACHE.get(graph_path)
    if g is None:
        gg = torch.load(graph_path, weights_only=False)
        g = (gg["edge_index"].long(), gg["tile_idx"].long())
        _FIXED_GRAPH_CACHE[graph_path] = g
    return g

def _cleanup_and_exit(signum, frame):
    print(f"Signal {signum} received, shutting down pool...", flush=True)
    if _pool_ref is not None:
        try:
            _pool_ref.exit()
        except Exception:
            pass
    sys.exit(1)

signal.signal(signal.SIGTERM, _cleanup_and_exit)
signal.signal(signal.SIGINT, _cleanup_and_exit)

# On-demand stack dump for diagnosing stalls without needing py-spy/ptrace
# (blocked by this cluster's ptrace_scope). `kill -USR1 <main_pid>` dumps
# every thread's current Python stack (main loop, ProcessPoolExecutor's
# internal queue-management thread, the shard_writer thread) to stderr,
# which lands in this job's %x_%j.log since sbatch points --error there too.
faulthandler.register(signal.SIGUSR1, all_threads=True)


def _ccx_worker_init():
    # ProcessPoolExecutor forks its workers after the SIGTERM/SIGINT handlers
    # above are installed, so every worker inherits _cleanup_and_exit too. Its
    # sys.exit(1) becomes a SystemExit raised *inside* the task currently
    # running in that worker -- but concurrent.futures.process's worker loop
    # wraps task execution in `except BaseException`, so it just reports the
    # SystemExit as a failed task and loops on to the next queued sample
    # instead of actually terminating. Reset to default disposition so a
    # killed job's SIGTERM (broadcast to the whole process group by Slurm)
    # kills workers immediately, the same as it would any other process.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Belt-and-suspenders alongside the OMP_NUM_THREADS/MKL_NUM_THREADS/etc.
    # env vars set in jobs/run_dataset_ccx.slurm: without this, each worker's
    # first CPU tensor op lazily spins up its own PyTorch intra-op thread
    # pool sized to nproc() (the job's full cpuset, e.g. 190) -- confirmed via
    # a running worker having 190 threads all named "pt_main_thread". With
    # pool_size workers each doing this independently, that's pool_size x 190
    # threads fighting over the same real cores. Real parallelism here comes
    # from having many separate worker processes, not from each one also
    # being internally multithreaded.
    torch.set_num_threads(1)

    # Stagger this worker's first shared-memory use across a random window
    # instead of every worker in a freshly-started pool hitting it in the
    # same instant. The "file_system" sharing strategy lazily forks a private
    # torch_shm_manager helper process (+ does a socket handshake with a
    # fairly short built-in timeout) the first time a worker shares a tensor.
    # A synchronized burst of ~90-180 near-simultaneous fork/exec calls right
    # as a fresh pool starts was overrunning that handshake timeout under
    # this cluster's shared-node CPU contention and crashing workers with
    # "Shared memory manager connection has timed out" (see
    # [[project-dataset-builder-ccx-fd-leak]]). Blocking here -- before the
    # worker is handed its first real task -- ensures the stagger actually
    # spreads the fork/exec calls out over the sleep window, rather than
    # racing a real task's own first share() call (which would otherwise
    # still all land in the same instant regardless of this warmup).
    time.sleep(random.uniform(0, 4.0))
    torch.zeros(1).share_memory_()

import ansys.mapdl.core.pool as _mapdl_pool
_orig_version_from_path = _mapdl_pool.version_from_path
def _patched_version_from_path(product, path, *args, **kwargs):
    if path and "ansys" in path.lower():
        return 251
    return _orig_version_from_path(product, path, *args, **kwargs)
_mapdl_pool.version_from_path = _patched_version_from_path

from core.component import Component
from core.IritModel import IritModel, IritCModel
from core.node import Node
from core.element import Element
from core.mesh import Mesh
from core.geometric_jacobian import compute_thickness_jacobian
from core.tile_sensitivity import assemble_tile_thickness_sensitivity
from ansys.mapdl.core import MapdlPool
from ansys.mapdl.core.errors import MapdlRuntimeError


def _sample_correlated_field(tile_keys, lo, hi, default, NX, NY, NZ, rng,
                              amplitude_frac_up=0.10, amplitude_frac_down=0.10):
    # Per-tile i.i.d. uniform sampling (the plain path below) produces
    # spatially uncorrelated "salt-and-pepper" designs -- nothing like the
    # smooth, spatially-correlated material fields a real structure (or a
    # topopt trajectory) actually has, so the surrogate never sees that
    # regime during training. This generates a smooth random field over the
    # tile grid instead: white noise blurred by a random correlation length
    # (short -> textured, long -> near-uniform). .reshape(-1) on the
    # (NX, NY, NZ) array is C-order (last axis fastest =
    # ix*NY*NZ + iy*NZ + iz), matching the X-outermost ShellThicknesses
    # layout model_lin.c reads for d1..dN.
    #
    # Deliberately NOT per-sample min-max normalized: that would force every
    # sample -- even a heavily-blurred, nearly-constant one -- back up to
    # fill the full [lo, hi] range, which was the original bug here (every
    # field sample ended up with some near-zero-thickness pocket somewhere,
    # pushing stress far above what a real topopt trajectory ever sees).
    #
    # Also deliberately normalized by sigma's *expected* per-cell std
    # (1/(2*sqrt(pi)*sigma) for a 2D Gaussian blur of unit white noise --
    # confirmed empirically) rather than left as a raw fixed amplitude: raw
    # amplitude decays local contrast AND the spatial-averaging effect
    # together as sigma grows, so the field's spatial MEAN (i.e. the
    # design's overall volume) stays locked near `default` regardless of
    # sigma -- reproducing the exact same near-zero volume diversity the
    # i.i.d. sampler had (see [[project_gnn_field_sampling]]), just with
    # smooth texture instead of salt-and-pepper. Normalizing out sigma's
    # decay first keeps per-cell contrast comparable across all sigma, so
    # what changes with sigma is purely how many *independent* correlated
    # regions the grid contains -- large sigma means only a few big
    # correlated blobs, so their average (i.e. overall volume) swings much
    # more sample-to-sample than at small sigma, where ~840 near-independent
    # cells average tightly back to `default`. That swinging overall volume
    # at large sigma is exactly the coverage this sampler exists to add.
    sigma = rng.uniform(1.0, 12.0)
    # mode="reflect" on the true (NX, NY, NZ) grid inflates variance near
    # any edge -- up to ~2x the interior value right at a corner, verified
    # empirically for this grid shape across sigma in [1, 12] (see
    # [[project_gnn_anchor_sensitivity_bug]]) -- because reflected padding
    # re-applies kernel weight to mirrored *copies* of already-counted noise
    # instead of fresh independent values, and those copies are perfectly
    # correlated with the original, so their cross-covariance adds to the
    # output variance instead of cancelling the way independent terms would.
    # `expected_std` below is only exact for an infinite (no-boundary)
    # domain, so dividing by it directly would leave edge/corner tiles --
    # including whichever tile happens to sit on a boundary condition --
    # systematically over-amplified and more often clipped to [lo, hi] than
    # interior tiles, distorting the local thickness distribution the
    # surrogate trains on exactly where it matters most.
    #
    # Fix: generate noise on an oversized grid, padded by ~4*sigma_max (the
    # Gaussian kernel's effective reach at scipy's default truncate=4) on
    # each side of every axis with real extent (>1 cell), blur that, then
    # crop back to the true (NX, NY, NZ) region. Every true-grid cell --
    # including corners -- now sits far enough from the padded array's own
    # edge that it's blurred against genuine independent noise on all
    # sides, matching the infinite-domain assumption `expected_std` relies
    # on. Verified empirically (hollow_cube's 40x21x1 grid): corner/edge/
    # interior std all land within ~1% of `expected_std` after this change,
    # vs. up to 2x apart before. Axes with only 1 cell (e.g. hollow_cube's
    # NZ) are left unpadded with mode="reflect": padding a size-1 axis
    # would just repeat independent noise every 2*pad cells for no benefit,
    # and reflect there is already a no-op (the single true cell reflects
    # onto itself, contributing its real variance with no suppression or
    # inflation) -- this generalizes to any geometry's tile_grid() shape,
    # not just hollow_cube's specific dimensions.
    pad = int(np.ceil(4.0 * 12.0))
    dims = (NX, NY, NZ)
    pads = tuple(pad if d > 1 else 0 for d in dims)
    modes = tuple("constant" if d > 1 else "reflect" for d in dims)
    noise = rng.standard_normal(tuple(d + 2 * p for d, p in zip(dims, pads)))
    field = gaussian_filter(noise, sigma=sigma, mode=modes)
    field = field[pads[0]:pads[0] + NX, pads[1]:pads[1] + NY, pads[2]:pads[2] + NZ]
    expected_std = 1.0 / (2.0 * np.sqrt(np.pi) * sigma)
    standardized = field.reshape(-1) / expected_std
    # Asymmetric amplitude: thinning a tile (standardized < 0) drives stress
    # up (and is what actually creates the low-volume/thin-design tail this
    # sampler needs -- see [[project_gnn_field_sampling]]), while thickening
    # it (> 0) has sharply diminishing returns on stress (confirmed
    # empirically -- uniform-baseline stress only drops 93.9->88.5 MPa going
    # from thickness 0.30->0.40) and, with an anchor thickened enough to hit
    # the stress target on its own, mostly just clips at `hi` instead of
    # adding useful diversity. A single symmetric amplitude couples these
    # two very different sensitivities together; splitting them lets the
    # anchor be thickened for the stress target while amplitude_frac_down
    # alone controls how much real thin/low-volume coverage there is.
    amp = np.where(standardized < 0, amplitude_frac_down, amplitude_frac_up)
    values = default + amp * (hi - lo) * standardized
    values = np.clip(values, lo, hi)
    return dict(zip(tile_keys, values.tolist()))


def _build_or_load_volume_inverse(base_path, model_path, fixed_dims, field_tile_keys, lo, hi, n_points=300):
    # field_random_base used to draw its per-sample anchor uniformly in
    # THICKNESS, but thickness->volume is a strongly saturating curve for a
    # fixed-topology geometry like hollow_cube (measured: volume(0.01)=33.3,
    # volume(0.49)=838.7, with the top ~25% of the thickness range
    # contributing well under 10% of that span) -- uniform-in-thickness
    # sampling therefore spends most of its density in the flat/high-volume
    # plateau, which is exactly the "lots of high-volume/low-stress samples"
    # skew this fixes. get_volume() is a CAD-only query (no mesh, no FEA), so
    # sweeping a few hundred uniform-thickness points to trace this curve
    # costs seconds, not the hours a calibration sweep of real ccx runs
    # would -- cheap enough to do once and cache, then invert so the anchor
    # is drawn uniform in VOLUME instead: pick a target volume, look up the
    # thickness that produces it.
    lookup_path = f"{base_path}/dataset/volume_lookup.npz"
    if os.path.exists(lookup_path):
        npz = np.load(lookup_path)
        thicknesses, volumes = npz["thicknesses"], npz["volumes"]
    else:
        exe_path = os.path.join(model_path, "model")
        exe_win_path = os.path.join(model_path, "model.exe")
        thicknesses = np.linspace(lo, hi, n_points)
        volumes = np.empty(n_points)
        for i, t in enumerate(thicknesses):
            dims = {k: float(t) for k in field_tile_keys}
            if os.path.isfile(exe_path):
                m = IritCModel(exe_path, dims_dict=dims, fixed_dims=fixed_dims)
            elif os.path.isfile(exe_win_path):
                m = IritCModel(exe_win_path, dims_dict=dims, fixed_dims=fixed_dims)
            else:
                m = IritModel(os.path.join(model_path, "model.irt"), dims_dict=dims)
            volumes[i] = m.get_volume()
        os.makedirs(os.path.dirname(lookup_path), exist_ok=True)
        np.savez(lookup_path, thicknesses=thicknesses, volumes=volumes)
        print(f"Built thickness->volume lookup ({n_points} pts, cached to {lookup_path})", flush=True)
    # volumes is monotonically non-decreasing in thicknesses (verified for
    # hollow_cube), but genuinely flat -- not just noisy -- at the very top
    # of the range: the design is fully solid there, so volume hits a hard
    # ceiling a few points before hi (measured: last ~6 of 300 points on
    # hollow_cube all landed on the exact same volume). PchipInterpolator
    # requires a strictly increasing x, so collapse any duplicate volumes to
    # their first (thinnest) occurrence before inverting -- that's also the
    # better representative of a saturated plateau for this fix's purpose,
    # since the thinnest design reaching a given volume is the one with the
    # most stress diversity among ties.
    volumes_u, uniq_idx = np.unique(volumes, return_index=True)
    thicknesses_u = thicknesses[uniq_idx]
    inverse = PchipInterpolator(volumes_u, thicknesses_u)
    return float(volumes_u[0]), float(volumes_u[-1]), inverse


def _write_checkpoint(dataset_dir, shard_start, shard_data):
    # Runs on the background shard-writer thread so the disk I/O (torch.save
    # of a shard) never blocks the main process from draining the
    # ProcessPoolExecutor's as_completed queue -- workers keep finishing
    # samples during the write, they just weren't being drained.
    #
    # No longer writes metadata.json here. It used to be a full-array
    # rewrite of the entire cumulative history on every checkpoint (an
    # unused-downstream bookkeeping file -- not read by training or
    # evaluation) -- at ~800 keys/record that rewrite grew to tens of
    # seconds and kept growing, long enough to starve the GIL and stall the
    # main process's result draining for minutes at a time once the dataset
    # passed ~15-20k samples.
    dataset_io.save_shard(dataset_dir, shard_start, shard_data)


def _kill_and_reap_pool(ex):
    # p.kill() alone leaves a zombie plus the parent-side pipe/sentinel fd
    # open until something calls join() on the Process object -- across
    # ~30 retry passes at pool_size=180 that leaked enough fds to blow past
    # the (default 1024) soft `ulimit -n` within a handful of passes, after
    # which even ProcessPoolExecutor()'s own constructor couldn't get an fd
    # for its internal pipe (job 48064, 2026-08-17). join(timeout=...)
    # rather than a bare join() so a process that somehow ignores SIGKILL
    # (observed nowhere so far) can't reintroduce the shutdown(wait=True)
    # hang this manual pool management was built to avoid.
    #
    # A worker's own torch_shm_manager helper (forked lazily the first time
    # that worker shares a tensor -- see _ccx_worker_init) is a *child* of
    # the worker process, not of this main process, so p.kill() (SIGKILL to
    # the worker only) never reaches it. Once the worker dies, the manager
    # is immediately reparented to init and keeps running forever, still
    # holding open -- and therefore never unlinking -- every /dev/shm-backed
    # file it was tracking. Across the many pool-crash-and-retry cycles a
    # long run goes through, these orphans accumulate without bound --
    # confirmed directly on job 48190 (2026-08-20): 189+ orphaned managers,
    # 800k+ leaked files, ~340GB in /dev/shm (a capped tmpfs, independent of
    # ordinary system RAM) -- until new SemLock/shared-tensor allocations
    # fail with "Cannot allocate memory" even though `free` shows hundreds
    # of GB of ordinary RAM still available (see
    # project_dataset_pool_split.md). Read each worker's direct children
    # from /proc *before* killing it -- once the worker is dead its
    # children are reparented to init and this lookup can no longer find
    # them -- so its manager gets killed right alongside it instead of
    # leaking.
    children_by_worker = {}
    for p in ex._processes.values():
        try:
            with open(f"/proc/{p.pid}/task/{p.pid}/children") as f:
                children_by_worker[p.pid] = [int(c) for c in f.read().split()]
        except OSError:
            children_by_worker[p.pid] = []
    for p in ex._processes.values():
        p.kill()
        for cpid in children_by_worker.get(p.pid, []):
            try:
                os.kill(cpid, signal.SIGKILL)
            except OSError:
                pass
    for p in ex._processes.values():
        p.join(timeout=5)


def _ccx_run_single(model_path, mesh_threads, fixed_dims, anchor_condition, force_pattern,
                     U, V, W, young, poisson, ccx_path, nproc, get_tile_grid, dim_idx, dims,
                     graph_path, stress_ks_rho_mpa):
    # Top-level (picklable) worker for ProcessPoolExecutor. Unlike MAPDL, ccx
    # needs no persistent client/pool: each sample is a standalone subprocess
    # in its own tempdir, so a plain process pool is enough (see CCXEvaluator).
    max_retries = 3
    for attempt in range(max_retries):
        try:
            exe_path = os.path.join(model_path, "model")
            exe_win_path = os.path.join(model_path, "model.exe")

            if os.path.isfile(exe_path):
                CAD_model = IritCModel(exe_path, dims_dict=dims, mesh_threads=mesh_threads, fixed_dims=fixed_dims)
            elif os.path.isfile(exe_win_path):
                CAD_model = IritCModel(exe_win_path, dims_dict=dims, mesh_threads=mesh_threads, fixed_dims=fixed_dims)
            else:
                CAD_model = IritModel(os.path.join(model_path, "model.irt"), dims_dict=dims, mesh_threads=mesh_threads)

            comp = Component(CAD_model, young=young, poisson=poisson)
            comp.generate_mesh(U=U, V=V, W=W)
            comp.mesh.anchor_nodes_by_condition(anchor_condition)
            comp.mesh.apply_force_by_pattern(force_pattern)
            comp.ccx_sim(ccx_path=ccx_path, nproc=nproc)

            tg = get_tile_grid(dims) if get_tile_grid is not None else None

            # Frozen-topology fast path: skip the Python edge-set rebuild and
            # node->tile mapping (both constant across dim samples for a
            # fixed-topology geometry) by reusing the precomputed graph.pt
            # artifact. Guard on node count, same as gnn_evaluator.py; fall
            # back to a full rebuild if it doesn't match.
            data = None
            if graph_path is not None and os.path.exists(graph_path):
                edge_index, tile_idx = _get_fixed_graph(graph_path)
                if len(comp.mesh.nodes) == tile_idx.shape[0]:
                    data = comp.to_graph_fixed(edge_index, tile_idx, tg, with_labels=True, stress_ks_rho_mpa=stress_ks_rho_mpa)
                    # edge_index/tile_idx are byte-identical across every sample for a
                    # fixed-topology geometry (that's the whole premise of the fast path
                    # above) -- the main process already has its own copy (see
                    # build_dataset's canonical_edge_index/canonical_tile_idx preload) and
                    # discards whatever we send here anyway. Sending None instead of the
                    # real ~14MB/sample tensor pair avoids paying multiprocessing's
                    # shared-memory cost (a real /tmp file create+mmap+unlink under the
                    # "file_system" sharing strategy) for data the receiver throws away.
                    data.edge_index = None
                    data.tile_idx = None
            if data is None:
                data = comp.to_graph_with_labels(tile_grid=tg, stress_ks_rho_mpa=stress_ks_rho_mpa)
            return dim_idx, data, {
                "id": dim_idx,
                **dims,
                "volume": comp.get_volume(),
                "max_stress": comp.mesh.get_max_stress(),
                "ks_stress": comp.mesh.get_aggregated_stress(rho=stress_ks_rho_mpa * 1e-6),
            }
        except Exception as e:
            print(f"dim_idx={dim_idx} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying...", flush=True)
    print(f"dim_idx={dim_idx} failed after {max_retries} attempts, skipping.", flush=True)
    return dim_idx, None, None


def _ccx_run_grad_sample(model_path, mesh_threads, fixed_dims, anchor_condition, force_pattern,
                          U, V, W, young, poisson, ccx_path, nproc, get_tile_grid, dim_idx, dims,
                          graph_path, stress_ks_rho_mpa, grad_tiles_per_sample, grad_fd_frac,
                          dims_bounds):
    # Ground-truth per-tile sensitivity labels for the gradient-matching
    # training loss: one base ccx run (via _ccx_run_single, unchanged) plus
    # grad_tiles_per_sample single-tile-perturbed reruns of the SAME
    # pipeline, finite-differencing the real solver's ks_stress instead of
    # the surrogate's (topopt_optimizer.py's dg FDs the surrogate itself,
    # which is the thing this loss is meant to correct -- see
    # project_surrogate_sensitivity_vs_value_gap.md).
    #
    # dim_idx also seeds which tiles get perturbed (a fresh local
    # random.Random, not the shared `random` module -- ProcessPoolExecutor
    # workers are long-lived and reused across many samples, so touching the
    # global module's state here would make tile selection depend on task
    # ordering instead of being deterministic per sample).
    base_idx, data, meta = _ccx_run_single(
        model_path, mesh_threads, fixed_dims, anchor_condition, force_pattern,
        U, V, W, young, poisson, ccx_path, nproc, get_tile_grid, dim_idx, dims,
        graph_path, stress_ks_rho_mpa,
    )
    if data is None:
        return dim_idx, None, None

    K = len(dims_bounds)
    rng = random.Random(dim_idx)
    tile_ids = rng.sample(range(K), min(grad_tiles_per_sample, K))

    grad_tile_idx, grad_dks_dx = [], []
    for t in tile_ids:
        key = f"d{t + 1}"
        lo, hi = dims_bounds[key]
        step = grad_fd_frac * (hi - lo)
        x0 = dims[key]
        xp = x0 + step if x0 + step <= hi else x0 - step
        if xp == x0:
            continue
        dims_p = dict(dims)
        dims_p[key] = xp
        _, _, meta_p = _ccx_run_single(
            model_path, mesh_threads, fixed_dims, anchor_condition, force_pattern,
            U, V, W, young, poisson, ccx_path, nproc, get_tile_grid, dim_idx, dims_p,
            graph_path, stress_ks_rho_mpa,
        )
        if meta_p is None:
            continue
        grad_tile_idx.append(t)
        grad_dks_dx.append((meta_p["ks_stress"] - meta["ks_stress"]) / (xp - x0))

    if len(grad_tile_idx) < grad_tiles_per_sample:
        # Every gradient-labeled sample must carry exactly the same number
        # of (tile, sensitivity) pairs -- training batches this dataset by
        # reshaping the concatenated per-sample tensors to
        # [batch_size, grad_tiles_per_sample], which requires a fixed count
        # per sample. A short sample here (a perturbation rerun failed) is
        # treated as a whole-sample failure so the existing pass/retry loop
        # in build_dataset re-attempts it from scratch, same as any other
        # failed dim_idx.
        print(f"dim_idx={dim_idx} only got {len(grad_tile_idx)}/{grad_tiles_per_sample} "
              f"tile perturbations to succeed, treating as failed.", flush=True)
        return dim_idx, None, None

    data.grad_tile_idx = torch.tensor(grad_tile_idx, dtype=torch.long)
    data.grad_dks_dx = torch.tensor(grad_dks_dx, dtype=torch.float)
    return dim_idx, data, meta


def _ccx_run_grad_sample_adjoint(model_path, mesh_threads, fixed_dims, anchor_condition, force_pattern,
                                  U, V, W, young, poisson, ccx_path, nproc, get_tile_grid, dim_idx, dims,
                                  graph_path, stress_ks_rho_mpa, dims_bounds, grad_max_tiles):
    """Adjoint alternative to _ccx_run_grad_sample: one base ccx run (for the
    node/edge training features, unchanged) plus ONE extra mesh regeneration
    and ONE ccx *SENSITIVITY solve -- covering every tile CCX will accept as
    a design node (see core.mesh.Mesh.solve_ccx_sensitivity's docstring) in
    a single pass, instead of grad_tiles_per_sample separate full ccx
    reruns. See core/geometric_jacobian.py, core/tile_sensitivity.py, and
    core/mesh.py's solve_ccx_sensitivity for the chain rule itself.

    Which tiles get a valid sensitivity is a function of mesh topology only
    (CCX's design-node acceptance depends on element connectivity and the
    fixed anchor/force pattern, not on the sampled thickness values -- see
    getdesiinfo3d.f) -- so for a fixed geometry/resolution, the same tile
    set is accepted every sample, and every sample naturally carries the
    same number of (tile, sensitivity) pairs without needing grad_mode's FD
    path random per-sample tile choice.
    """
    base_idx, data, meta = _ccx_run_single(
        model_path, mesh_threads, fixed_dims, anchor_condition, force_pattern,
        U, V, W, young, poisson, ccx_path, nproc, get_tile_grid, dim_idx, dims,
        graph_path, stress_ks_rho_mpa,
    )
    if data is None:
        return dim_idx, None, None

    try:
        exe_path = os.path.join(model_path, "model")
        NX, NY, NZ = get_tile_grid(dims)

        dX_dt, nodes_dict, elements_dict, elem_to_tile = compute_thickness_jacobian(
            exe_path, dims, U, V, W, fixed_dims=fixed_dims, eps=1e-4,
        )
        design_node_ids = [nid for nid, v in dX_dt.items() if any(abs(c) > 1e-9 for c in v)]

        nodes = {nid: Node(nid, xyz) for nid, xyz in nodes_dict.items()}
        elements = {eid: Element(eid, [nodes[nid] for nid in nlist]) for eid, nlist in elements_dict.items()}
        sens_mesh = Mesh(nodes, elements)
        sens_mesh.anchor_nodes_by_condition(anchor_condition)
        sens_mesh.apply_force_by_pattern(force_pattern)

        _, dfdn, normal = sens_mesh.solve_ccx_sensitivity(
            young, poisson, design_node_ids, stress_ks_rho_mpa=stress_ks_rho_mpa,
            ccx_path=ccx_path, nproc=nproc,
        )

        dims_keys = list(dims_bounds.keys())
        per_tile = assemble_tile_thickness_sensitivity(
            dX_dt, dfdn, normal, elements_dict, elem_to_tile, NX, NY, NZ, dims_keys,
        )
    except Exception as e:
        print(f"dim_idx={dim_idx} adjoint sensitivity failed: {e}. Treating sample as failed.", flush=True)
        return dim_idx, None, None

    if not per_tile:
        print(f"dim_idx={dim_idx} adjoint sensitivity accepted zero design nodes, "
              f"treating as failed.", flush=True)
        return dim_idx, None, None

    # Deterministic (sorted by key) so every sample's grad_tile_idx lists the
    # same tiles in the same order -- required since training reshapes the
    # concatenated per-sample tensors to [batch_size, grad_M].
    keys_sorted = sorted(per_tile.keys(), key=lambda k: int(k[1:]))
    if grad_max_tiles is not None and grad_max_tiles > 0:
        keys_sorted = keys_sorted[:grad_max_tiles]

    key_to_idx = {k: i for i, k in enumerate(dims_keys)}
    data.grad_tile_idx = torch.tensor([key_to_idx[k] for k in keys_sorted], dtype=torch.long)
    data.grad_dks_dx = torch.tensor([per_tile[k] for k in keys_sorted], dtype=torch.float)
    return dim_idx, data, meta


def build_dataset(
    geometry: str,
    num_samples: int = 10,
    seed: int = 42,
    pool_size: int = 4,
    nproc: int = 3,
    output_dir: str = None,
    run_location: str = None,
    engine: str = "ansys",
    ccx_path: str = "ccx",
    mesh_threads: int = 4,
    stress_ks_rho_mpa: float = 0.15,
    field_amplitude_frac_up: float = 0.10,
    field_amplitude_frac_down: float = 0.10,
    field_base_thickness: float = None,
    field_random_base: bool = False,
    grad_mode: bool = False,
    grad_tiles_per_sample: int = 6,
    grad_fd_frac: float = 0.02,
    grad_engine: str = "fd",
    grad_max_tiles: int = None,
):
    random.seed(seed)
    torch.manual_seed(seed)
    field_rng = np.random.default_rng(seed)

    base_path = f"data/{geometry}"
    model_path = f"{base_path}/CAD_model"
    dims_json_path = f"{base_path}/CAD_model/dims.json"
    material_props_path = f"{base_path}/CAD_model/material_properties.json"

    dataset_dir = output_dir if output_dir is not None else f"{base_path}/dataset"
    screenshots_dir = f"{dataset_dir}/screenshots"
    if run_location is None:
        run_location = base_path

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)

    dataset, metadata = [], []

    if engine == "ccx":
        # Sharded format: never load the full historical dataset into RAM here
        # (see training/dataset_io.py) -- just a one-time migration of any
        # pre-existing single-file dataset.pt. No metadata.json is read or
        # written on this path (see _write_checkpoint).
        migrated = dataset_io.migrate_legacy_dataset(dataset_dir)
        if migrated:
            print(f"Migrated {migrated} legacy samples into {dataset_dir}/shards/ "
                  f"(old file kept as dataset.pt.bak)", flush=True)
        start_idx = dataset_io.num_existing_samples(dataset_dir)
        print(f"Continuing from sample {start_idx}...")
    # if dataset exists, load and continue
    elif os.path.exists(f"{dataset_dir}/dataset.pt"):
        print(f"Loading existing dataset from {dataset_dir}/dataset.pt")
        dataset = torch.load(f"{dataset_dir}/dataset.pt", weights_only=False)
        with open(f"{dataset_dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        start_idx = len(dataset)
        print(f"Continuing from sample {start_idx}...")
    else:
        start_idx = 0

    with open(material_props_path, "r") as f:
        material_props = json.load(f)
        young = material_props["young_modulus"]
        poisson = material_props["poisson_ratio"]

    with open(dims_json_path, "r") as f:
        dims_template = json.load(f)

    grad_dims_bounds = ({k: (v["min"], v["max"]) for k, v in dims_template.items()}
                         if grad_mode else None)

    module = importlib.import_module(f"data.{geometry}.boundary_conditions")
    anchor_condition = module.anchor_condition
    force_pattern = module.force_pattern
    mesh_resolution = module.mesh_resolution
    get_tile_grid = getattr(module, "tile_grid", None)  # optional, geometry-specific
    # Fixed (non-optimized) dims, e.g. grid topology d1/d2/d3 — not in dims.json,
    # re-injected into every sampled dims dict so the CAD model still gets them.
    fixed_dims = getattr(module, "fixed_dims", lambda: {})()

    # Correlated-field sampling only applies to a fixed-topology geometry
    # whose dims.json is exactly one strut/thickness param per tile
    # (d1..d{NX*NY*NZ}) -- falls back to plain i.i.d. sampling otherwise.
    field_tile_keys = None
    if get_tile_grid is not None:
        NX, NY, NZ = get_tile_grid(None)
        candidate_keys = [f"d{i + 1}" for i in range(NX * NY * NZ)]
        if len(dims_template) == NX * NY * NZ and all(k in dims_template for k in candidate_keys):
            field_tile_keys = candidate_keys
            field_lo = np.array([dims_template[k]["min"] for k in field_tile_keys])
            field_hi = np.array([dims_template[k]["max"] for k in field_tile_keys])
            # field_base_thickness overrides the anchor the field sampler
            # centers on (see _sample_correlated_field's `default` arg)
            # independently of dims.json's own "default" -- that field is
            # also the optimizer's actual starting point and other call
            # sites' fallback, so it's left untouched; this only changes
            # what a *field-sampled* training example is centered around.
            # field_random_base takes priority over both: instead of one
            # fixed anchor shared by every field-sampled example in the run,
            # each example draws its own anchor fresh, uniformly over the
            # tile's real [min, max] range, before the correlated noise is
            # added on top. A fixed default (0.3 here, mid-range) only ever
            # produced training examples clustered around one overall
            # volume -- topopt trajectories walk designs toward the thin
            # end of the range as material gets removed, and the surrogate
            # had ~zero training coverage there (see
            # project_gnn_anchor_sensitivity_bug.md's topopt-vs-ccx
            # validation: -90% stress error by generation 48, growing
            # monotonically as designs got thinner). Sampling the anchor
            # itself across the full range gives every field-sampled example
            # an independent chance at a thin, medium, or thick overall
            # baseline, not just noise around one baseline.
            if field_random_base:
                field_default = None  # drawn fresh per sample below
                volume_lo, volume_hi, volume_inverse = _build_or_load_volume_inverse(
                    base_path, model_path, fixed_dims, field_tile_keys, field_lo[0], field_hi[0]
                )
            elif field_base_thickness is not None:
                field_default = np.full(len(field_tile_keys), field_base_thickness)
            else:
                field_default = np.array([dims_template[k]["default"] for k in field_tile_keys])

    # Generate ALL dims upfront in the main process from the seeded RNG.
    # When resuming, start at dim index start_idx*2 so resumed runs never
    # overlap with dims (including buffer) used by previous runs.
    dim_offset = start_idx * 2
    samples_needed = num_samples - start_idx
    all_dims = []
    num_field_samples = 0
    for _ in range(dim_offset + samples_needed):
        if field_tile_keys is not None:
            if field_random_base:
                # Single scalar anchor per sample (tile bounds are uniform
                # across the grid for hollow_cube -- field_lo/field_hi are
                # constant arrays), broadcast to every tile before the
                # correlated noise is layered on top. Drawn uniform in VOLUME
                # (via the inverted thickness->volume lookup) rather than
                # uniform in thickness -- see _build_or_load_volume_inverse.
                sample_volume = field_rng.uniform(volume_lo, volume_hi)
                sample_base = float(volume_inverse(sample_volume))
                this_default = np.full(len(field_tile_keys), sample_base)
            else:
                this_default = field_default
            all_dims.append(_sample_correlated_field(field_tile_keys, field_lo, field_hi, this_default,
                                                       NX, NY, NZ, field_rng,
                                                       amplitude_frac_up=field_amplitude_frac_up,
                                                       amplitude_frac_down=field_amplitude_frac_down))
            num_field_samples += 1
        else:
            all_dims.append({k: random.uniform(v["min"], v["max"]) for k, v in dims_template.items()})
    print(f"Generated dims [{dim_offset}..{dim_offset + samples_needed - 1}] (offset={dim_offset}), "
          f"{num_field_samples} correlated-field / {len(all_dims) - num_field_samples} i.i.d.", flush=True)

    if engine == "ccx":
        print(f"Running with CalculiX (ccx_path={ccx_path}), pool_size={pool_size}, nproc={nproc}", flush=True)
        U, V, W = mesh_resolution()
        pending = list(range(dim_offset, dim_offset + samples_needed))
        completed = start_idx
        t_start = time.time()
        # A "pass" now also covers recovery from a broken pool (see the
        # ProcessPoolExecutor management below), not just per-sample retries,
        # and a fresh pool has been crashing the torch_shm_manager helper
        # process within ~40-50 samples of each restart under 180-way
        # concurrent load -- 3 was fine when a pass only needed to retry a
        # handful of individually-failed samples, but is nowhere near enough
        # retry budget to grind through thousands of samples 40-50 at a time.
        max_passes = 30
        shard_size = 100
        # Safety net for a failure mode that's neither a clean exception nor
        # a hang from the pool's *workers* (already handled above/below):
        # ProcessPoolExecutor's internal manager thread can die during its
        # own startup (Thread._bootstrap) before this process ever gets a
        # chance to submit anything meaningfully -- Python just prints
        # "Exception ignored in thread started by ..." for that and does NOT
        # propagate it to the main thread, so the executor silently ends up
        # non-functional and as_completed() would otherwise wait forever for
        # results that can never arrive (confirmed repeatedly: job sat with
        # zero progress for 7-46 min despite live worker processes, on a
        # heavily-loaded shared node). idle_timeout is time since the *last*
        # completion, not a fixed deadline for the whole pass -- a large
        # legitimately-slow-but-working backlog must not trip this.
        pass_idle_timeout_sec = 300
        pass_poll_interval_sec = 30

        # Frozen-topology fast path: if data/<geometry>/dataset/graph.pt
        # exists (precomputed edge_index/tile_idx, see core/fixed_graph.py),
        # workers skip the per-sample edge-set rebuild. Absent file -> every
        # sample falls back to the full rebuild in to_graph_with_labels.
        graph_path = f"{base_path}/dataset/graph.pt"
        canonical_edge_index = None
        canonical_tile_idx = None
        if os.path.exists(graph_path):
            print(f"Frozen-graph fast path enabled: {graph_path}", flush=True)
            # Load our own copy once instead of relying on whatever the first
            # worker result happens to carry -- workers on this fast path now
            # send edge_index/tile_idx as None (see _ccx_run_single) since
            # they're byte-identical across every sample and not worth the
            # multiprocessing shared-memory cost to transfer per-sample.
            canonical_edge_index, canonical_tile_idx = _get_fixed_graph(graph_path)
        else:
            print(f"No graph.pt at {graph_path}; rebuilding graph per sample "
                  f"(run `python -m core.fixed_graph --geometry {geometry}` to speed this up)", flush=True)

        # Only the current, not-yet-flushed shard lives in RAM. Once a shard
        # hits shard_size it's saved to its own file and dropped (see
        # training/dataset_io.py) -- the historical dataset is never held
        # resident here, so RAM stays bounded regardless of num_samples.
        current_shard = []
        shard_start = completed
        shard_writer = ThreadPoolExecutor(max_workers=1)
        shard_write_futures = []

        for pass_idx in range(max_passes):
            if not pending:
                break
            if pass_idx > 0:
                # A prior pass's ~pool_size killed worker processes (each
                # holding threads, fds, shared-memory segments) aren't
                # necessarily fully reclaimed by the OS the instant p.kill()
                # returns. Immediately spinning up another full pool_size
                # pool on top of that has been observed hitting transient
                # resource exhaustion itself (e.g. "can't start new thread"
                # starting the executor's own manager thread, before any of
                # the new workers even run their initializer) -- a short
                # pause here gives cleanup a chance to finish first.
                time.sleep(10)
            print(f"Pass {pass_idx+1}/{max_passes}: processing {len(pending)} samples...", flush=True)
            failed_this_pass = []

            # Managed manually instead of `with ProcessPoolExecutor(...) as ex:`
            # -- that form's __exit__ always calls shutdown(wait=True), and if a
            # worker crashes (fut.result() raising BrokenProcessPool) mid-pass,
            # concurrent.futures.process has a known failure mode on Python
            # 3.9-3.11 where shutdown(wait=True) hangs forever joining surviving
            # workers that never get an exit sentinel from a broken pool -- this
            # is indistinguishable from the outside from workers just being slow
            # (they keep running their last assigned task, so ps/CPU usage looks
            # completely healthy) and was the real cause of several "the dataset
            # job looks stuck" incidents that turned out not to be an IPC/fd
            # issue at all. On any exception here we kill the pool's worker
            # processes directly instead, so cleanup can't hang, and resolved
            # is tracked explicitly so unresolved samples still get retried.
            # ProcessPoolExecutor's own constructor (it sets up internal
            # queues/threads immediately, not lazily) has been observed
            # raising the same "can't start new thread" resource-exhaustion
            # error as ex.submit() below -- catch it here too, since `ex`
            # may not even be bound if this raises, so there'd be nothing
            # for the try/finally below to shut down. Skip straight to the
            # next pass (pending is untouched, so everything retries) rather
            # than crashing the whole run.
            try:
                ex = ProcessPoolExecutor(max_workers=pool_size, initializer=_ccx_worker_init)
            except Exception as e:
                import traceback
                print(f"ProcessPoolExecutor construction raised {type(e).__name__}: {e} -- "
                      f"treating this pass as failed, retrying all {len(pending)} "
                      f"pending samples next pass.", flush=True)
                traceback.print_exc()
                continue
            resolved_dim_idxs = set()
            try:
                # Building `futures` itself (not just awaiting fut.result()
                # below) can raise -- e.g. ex.submit()'s first call lazily
                # starts the executor's internal manager thread, which has
                # been observed failing with "can't start new thread" right
                # after a prior pass's pool was force-killed (OS resources
                # from those 190 killed workers not yet fully reclaimed
                # before this pass tries to spin up 190 more). Without this
                # try/except, that exception had no handler at this level
                # (only the narrower one inside the as_completed loop below)
                # and crashed the whole run instead of retrying next pass
                # like every other pool-broke-mid-flight failure mode does.
                # Submitting itself can hang forever with NO exception: if the
                # pool's manager thread died asynchronously during its own
                # bootstrap (see the pool-construction comment above) *after*
                # ex.submit()'s first call already returned successfully, a
                # later ex.submit() in this same loop can block indefinitely
                # inside e.g. self._call_queue.put() waiting for a manager
                # that will never drain it (confirmed: job sat with zero
                # progress for 13+ min with no error at all, after the exact
                # same "Exception ignored in thread ... can't allocate lock"
                # signature seen elsewhere). Neither the try/except below nor
                # the idle-timeout around as_completed() can catch a hang
                # *before* as_completed is even reached, so submission runs
                # in its own daemon thread with a join timeout instead --
                # daemon=True so an actually-permanently-stuck thread (stuck
                # inside a blocking C-level queue put) leaks harmlessly for
                # the rest of this process's life instead of preventing exit.
                submission_timeout_sec = 180
                submit_result = {}
                # grad_mode swaps in a worker that produces ground-truth
                # per-tile sensitivity labels (grad_tile_idx/grad_dks_dx)
                # instead of the plain _ccx_run_single -- same
                # (dim_idx, data, meta) return contract either way, so the
                # pool/retry/shard machinery below needs no other changes.
                # grad_engine="fd" (default): _ccx_run_grad_sample, base ccx
                # run + grad_tiles_per_sample single-tile-perturbed reruns.
                # grad_engine="adjoint": _ccx_run_grad_sample_adjoint, base
                # ccx run + one geometric-Jacobian mesh regen + one ccx
                # *SENSITIVITY solve covering every CCX-accepted tile at
                # once (see core/tile_sensitivity.py).
                if not grad_mode:
                    worker_fn = _ccx_run_single
                    worker_extra_args = ()
                elif grad_engine == "adjoint":
                    worker_fn = _ccx_run_grad_sample_adjoint
                    worker_extra_args = (grad_dims_bounds, grad_max_tiles)
                else:
                    worker_fn = _ccx_run_grad_sample
                    worker_extra_args = (grad_tiles_per_sample, grad_fd_frac, grad_dims_bounds)

                def _do_submit():
                    try:
                        submit_result["futures"] = {
                            ex.submit(
                                worker_fn, model_path, mesh_threads, fixed_dims,
                                anchor_condition, force_pattern, U, V, W,
                                young, poisson, ccx_path, nproc, get_tile_grid,
                                dim_idx, all_dims[dim_idx], graph_path, stress_ks_rho_mpa,
                                *worker_extra_args,
                            ): dim_idx
                            for dim_idx in pending
                        }
                    except Exception as e:
                        submit_result["error"] = e

                submit_thread = threading.Thread(target=_do_submit, daemon=True)
                try:
                    submit_thread.start()
                except RuntimeError as e:
                    # threading.Thread.start() runs in *this* thread, not
                    # inside _do_submit -- its own try/except above can't
                    # catch a failure to create the thread at all (e.g. "can't
                    # start new thread", the same OS resource exhaustion
                    # ex.submit() and the ProcessPoolExecutor constructor
                    # above are already guarded against, from a prior pass's
                    # killed pool_size workers not being fully reclaimed
                    # yet). Without this, that RuntimeError had no handler
                    # anywhere in the call chain (the try above only has a
                    # `finally`) and crashed the whole run instead of
                    # retrying next pass like every other pool-broke
                    # failure mode here does -- confirmed: this exact
                    # traceback killed a live run stone dead with 45487/50000
                    # samples still pending. Reuses the existing
                    # submit_result["error"] branch below instead of a new
                    # one, since the thread never ran and there's nothing to
                    # join.
                    submit_result["error"] = e
                else:
                    submit_thread.join(timeout=submission_timeout_sec)

                if submit_thread.is_alive():
                    print(f"Submitting {len(pending)} tasks did not return within "
                          f"{submission_timeout_sec}s -- pool likely broken (dead "
                          f"manager thread blocking ex.submit() internally, no "
                          f"exception raised). Killing its workers directly and "
                          f"retrying all pending samples next pass.", flush=True)
                    _kill_and_reap_pool(ex)
                    futures = {}
                elif "error" in submit_result:
                    import traceback
                    e = submit_result["error"]
                    print(f"Pool creation/submission raised {type(e).__name__}: {e} -- "
                          f"treating this pass as failed, retrying all {len(pending)} "
                          f"pending samples next pass.", flush=True)
                    traceback.print_exception(type(e), e, e.__traceback__)
                    _kill_and_reap_pool(ex)
                    futures = {}
                else:
                    futures = submit_result["futures"]
                # Wraps as_completed() with an IDLE-time bound rather than a
                # single fixed deadline: as_completed(fs, timeout=N) measures
                # N seconds from when it was *called*, not since the last
                # result -- a single big fixed timeout would false-positive
                # on a large-but-healthy backlog that legitimately takes
                # longer than N to fully drain. Instead, re-arm as_completed
                # in short poll_interval_sec chunks and only declare a hang
                # after idle_timeout_sec of *zero new completions* across
                # chunks -- a pool that's actually producing results keeps
                # resetting the idle clock no matter how large the backlog.
                timed_out = [False]
                def _as_completed_with_idle_timeout(fs, idle_timeout, poll_interval):
                    last_progress = time.time()
                    remaining = set(fs)
                    while remaining:
                        try:
                            for fut in as_completed(remaining, timeout=poll_interval):
                                last_progress = time.time()
                                remaining.discard(fut)
                                yield fut
                        except FuturesTimeoutError:
                            pass
                        if remaining and time.time() - last_progress > idle_timeout:
                            timed_out[0] = True
                            return

                for fut in _as_completed_with_idle_timeout(futures, pass_idle_timeout_sec, pass_poll_interval_sec):
                    dim_idx = futures[fut]
                    try:
                        dim_idx, data, meta = fut.result()
                    except Exception as e:
                        import traceback
                        print(f"Worker for dim_idx={dim_idx} raised {type(e).__name__}: {e} "
                              f"-- pool is likely broken. Killing its workers directly (avoids "
                              f"ProcessPoolExecutor's shutdown(wait=True) hang on a broken pool) "
                              f"and retrying unresolved samples from this pass in a fresh pool "
                              f"next pass.", flush=True)
                        traceback.print_exc()
                        _kill_and_reap_pool(ex)
                        break
                    resolved_dim_idxs.add(dim_idx)
                    if data is None:
                        failed_this_pass.append(dim_idx)
                        print(f"dim_idx={dim_idx} skipped, will retry in next pass.", flush=True)
                        continue

                    # edge_index is identical across samples for this
                    # fixed-topology geometry (verified byte-for-byte); point
                    # every sample at one shared tensor so torch.save's storage
                    # dedup writes it once per shard, not once per sample
                    # (~14MB/sample saved for hollow_cube). On the frozen-graph
                    # fast path, workers send None for this (see
                    # _ccx_run_single) and we already preloaded our own
                    # canonical copy above; the None-check + else-branch below
                    # is only live as a fallback for the no-graph.pt path,
                    # where each worker still computes and sends its own copy.
                    if data.edge_index is None:
                        data.edge_index = canonical_edge_index
                    elif canonical_edge_index is None:
                        canonical_edge_index = data.edge_index
                    else:
                        data.edge_index = canonical_edge_index

                    # Same dedup for tile_idx (also constant across samples).
                    # Unlike edge_index, tile_idx is a dynamic (non-core)
                    # PyG Data field -- setting it to None on the worker side
                    # doesn't leave a None-valued attribute behind, it drops
                    # the attribute entirely, so `hasattr` is False rather
                    # than "present but None" for fast-path samples. Read via
                    # getattr(..., None) instead of gating on hasattr first.
                    received_tile_idx = getattr(data, "tile_idx", None)
                    if received_tile_idx is None:
                        data.tile_idx = canonical_tile_idx
                    elif canonical_tile_idx is None:
                        canonical_tile_idx = received_tile_idx
                    else:
                        data.tile_idx = canonical_tile_idx

                    current_shard.append(data)
                    completed += 1

                    elapsed = time.time() - t_start
                    done_so_far = completed - start_idx
                    remaining_count = num_samples - completed
                    eta_str = ""
                    if done_so_far > 0:
                        eta_sec = elapsed / done_so_far * remaining_count
                        eta_str = f"  ETA {timedelta(seconds=int(eta_sec))}"
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now}] [{completed:04d}/{num_samples}] {geometry}: σmax={meta['max_stress']:.2e}{eta_str}", flush=True)

                    # Non-blocking check: surface a prior checkpoint write's
                    # exception as soon as possible instead of only at the
                    # very end of the run.
                    for wf in shard_write_futures:
                        if wf.done() and wf.exception() is not None:
                            raise wf.exception()

                    if completed % shard_size == 0:
                        shard_write_futures.append(
                            shard_writer.submit(_write_checkpoint, dataset_dir, shard_start, current_shard)
                        )
                        shard_start = completed
                        current_shard = []
                        print(f"--- Checkpoint queued at sample {completed} ---", flush=True)
                if timed_out[0]:
                    print(f"as_completed() saw {pass_idle_timeout_sec}s with zero new "
                          f"completions ({len(resolved_dim_idxs)}/{len(futures)} resolved "
                          f"this pass so far) -- pool's manager thread likely died silently "
                          f"(no exception, just stopped producing results). Killing its "
                          f"workers directly and retrying unresolved samples next pass.", flush=True)
                    _kill_and_reap_pool(ex)
            finally:
                # wait=False: never block here regardless of why we're exiting
                # (normal completion, a broken pool already killed above, or any
                # other exception) -- see the comment where `ex` was created.
                ex.shutdown(wait=False, cancel_futures=True)

            unresolved = [dim_idx for dim_idx in pending if dim_idx not in resolved_dim_idxs]
            if unresolved:
                print(f"{len(unresolved)} samples never got a result this pass "
                      f"(pool broken partway through), retrying next pass.", flush=True)
                failed_this_pass.extend(unresolved)

            pending = failed_this_pass

        if pending:
            print(f"WARNING: {len(pending)} samples failed after {max_passes} passes: {pending}", flush=True)

        if current_shard:
            shard_write_futures.append(
                shard_writer.submit(_write_checkpoint, dataset_dir, shard_start, current_shard)
            )
            current_shard = []

        for wf in shard_write_futures:
            wf.result()
        shard_writer.shutdown()

        print(f"Dataset saved to {dataset_dir}/shards/ ({dataset_io.num_existing_samples(dataset_dir)} samples)")

        return None, metadata

    def run_single_sample(mapdl, dim_idx):
        dims = all_dims[dim_idx]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                exe_path = os.path.join(model_path, "model")
                exe_win_path = os.path.join(model_path, "model.exe")

                if os.path.isfile(exe_path):
                    CAD_model = IritCModel(exe_path, dims_dict=dims, fixed_dims=fixed_dims)
                elif os.path.isfile(exe_win_path):
                    CAD_model = IritCModel(exe_win_path, dims_dict=dims, fixed_dims=fixed_dims)
                else:
                    CAD_model = IritModel(os.path.join(model_path, "model.irt"), dims_dict=dims)

                comp = Component(CAD_model, young=young, poisson=poisson)
                U, V, W = mesh_resolution()
                comp.generate_mesh(U=U, V=V, W=W)
                comp.mesh.anchor_nodes_by_condition(anchor_condition)
                comp.mesh.apply_force_by_pattern(force_pattern)
                comp.ansys_sim(mapdl=mapdl, screenshot_path=None)

                tg = get_tile_grid(dims) if get_tile_grid is not None else None
                data = comp.to_graph_with_labels(tile_grid=tg, stress_ks_rho_mpa=stress_ks_rho_mpa)
                return dim_idx, data, {
                    "id": dim_idx,
                    **dims,
                    "volume": comp.get_volume(),
                    "max_stress": comp.mesh.get_max_stress(),
                    "ks_stress": comp.mesh.get_aggregated_stress(rho=stress_ks_rho_mpa * 1e-6),
                }
            except MapdlRuntimeError as e:
                print(f"dim_idx={dim_idx} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying...", flush=True)
                try:
                    mapdl.clear()
                except Exception as clear_err:
                    # Instance is permanently dead. Kill it so the pool monitor
                    # respawns it, then return a failure marker so the worker
                    # thread stays alive and the sample can be re-queued.
                    print(f"dim_idx={dim_idx} MAPDL instance unrecoverable: {clear_err}", flush=True)
                    try:
                        mapdl.exit()
                    except Exception:
                        pass
                    return dim_idx, None, None
            except Exception as e:
                print(f"dim_idx={dim_idx} unexpected error (attempt {attempt+1}/{max_retries}): {e}. Retrying...", flush=True)
                try:
                    mapdl.clear()
                except Exception as clear_err:
                    print(f"dim_idx={dim_idx} MAPDL instance unrecoverable: {clear_err}", flush=True)
                    try:
                        mapdl.exit()
                    except Exception:
                        pass
                    return dim_idx, None, None
        print(f"dim_idx={dim_idx} failed after {max_retries} attempts, skipping.", flush=True)
        return dim_idx, None, None

    print(f"Creating MapdlPool with pool_size={pool_size}, nproc={nproc}, run_location={base_path}", flush=True)
    try:
        pool = MapdlPool(n_instances=pool_size, nproc=nproc, run_location=run_location, license_server_check=False, start_timeout=120)
        global _pool_ref
        _pool_ref = pool
        print("MapdlPool created successfully, starting pool.map()...", flush=True)
    except Exception as e:
        print(f"ERROR creating MapdlPool: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    try:
        pending = list(range(dim_offset, dim_offset + samples_needed))
        completed = start_idx
        t_start = time.time()
        max_passes = 3

        for pass_idx in range(max_passes):
            if not pending:
                break
            print(f"Pass {pass_idx+1}/{max_passes}: processing {len(pending)} samples...", flush=True)
            failed_this_pass = []

            for submitted_idx, result in zip(pending, pool.map(run_single_sample, pending)):
                # run_single_sample always returns (dim_idx, data_or_None, meta_or_None);
                # None-tuple indicates failure/skip (either out-of-retries or dead instance).
                if result is None or result[1] is None:
                    failed_this_pass.append(submitted_idx)
                    print(f"dim_idx={submitted_idx} skipped, will retry in next pass.", flush=True)
                    continue
                dim_idx, data, meta = result
                dataset.append(data)
                metadata.append(meta)
                completed += 1

                elapsed = time.time() - t_start
                done_so_far = completed - start_idx
                remaining_count = num_samples - completed
                eta_str = ""
                if done_so_far > 0:
                    eta_sec = elapsed / done_so_far * remaining_count
                    eta_str = f"  ETA {timedelta(seconds=int(eta_sec))}"
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] [{completed:04d}/{num_samples}] {geometry}: σmax={meta['max_stress']:.2e}{eta_str}", flush=True)

                if completed % 100 == 0:
                    torch.save(dataset, f"{dataset_dir}/dataset.pt")
                    with open(f"{dataset_dir}/metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)
                    print(f"--- Checkpoint saved at sample {completed} ---", flush=True)

            pending = failed_this_pass

        if pending:
            print(f"WARNING: {len(pending)} samples failed after {max_passes} passes: {pending}", flush=True)

    finally:
        pool.exit()

    torch.save(dataset, f"{dataset_dir}/dataset.pt")
    with open(f"{dataset_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Dataset saved to {dataset_dir}/dataset.pt")

    return dataset, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FEA dataset for given geometry.")
    parser.add_argument("--geometry", type=str, default="arm", help="Geometry name (e.g., 'beam', 'arm').")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--pool_size", type=int, default=4, help="Number of parallel MAPDL instances.")
    parser.add_argument("--nproc", type=int, default=1, help="CPUs per MAPDL instance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save dataset (default: data/<geometry>/dataset).")
    parser.add_argument("--run_location", type=str, default=None, help="Working directory for MAPDL instances (default: data/<geometry>).")
    parser.add_argument("--engine", type=str, default="ansys", choices=["ansys", "ccx"], help="FEA backend: 'ansys' (MAPDL pool) or 'ccx' (CalculiX subprocess pool).")
    parser.add_argument("--ccx_path", type=str, default="ccx", help="Path to the ccx binary (engine=ccx only).")
    parser.add_argument("--mesh_threads", type=int, default=4, help="OMP threads per meshing call (engine=ccx only).")
    parser.add_argument("--stress_ks_rho_mpa", type=float, default=0.15, help="KS aggregation sharpness (1/MPa) for the ks_stress graph target; see core/mesh.py:get_aggregated_stress. Matches ccx_evaluator.py's default.")
    parser.add_argument("--field_amplitude_frac_up", type=float, default=0.10, help="Amplitude for the correlated-field sampler on the thickening (above-default) side only, as a fraction of each tile's (max-min) span, applied after normalizing out sigma's expected per-cell std (so contrast is comparable across all sigma -- see _sample_correlated_field). Thickening has sharply diminishing returns on stress, so this is usually kept modest.")
    parser.add_argument("--field_amplitude_frac_down", type=float, default=0.10, help="Same as --field_amplitude_frac_up but for the thinning (below-default) side. This is what actually drives stress up and creates low-volume/thin-design coverage -- raise this (independently of the up side) to get more of that regime.")
    parser.add_argument("--field_base_thickness", type=float, default=None, help="Overrides the uniform anchor the field sampler centers each tile on (see _sample_correlated_field's `default` arg), independently of dims.json's own per-tile default (which stays untouched -- it's also the optimizer's real starting point). Use to shift the sampled stress distribution: a thinner anchor raises typical stress. Defaults to dims.json's default if unset. Ignored if --field_random_base is set.")
    def _bool(s):
        if isinstance(s, bool):
            return s
        return str(s).lower() in ("1", "true", "yes", "y", "on")
    parser.add_argument("--field_random_base", type=_bool, default=False, help="Instead of one fixed anchor shared by every field-sampled example (dims.json's default, or --field_base_thickness), draw a fresh scalar anchor per sample uniformly over the tile's real [min, max] range, then add the correlated noise on top of that. Gives full coverage of thin/thick overall baselines instead of just noise around one baseline -- use this to cover the low-volume regime a fixed-default dataset misses (see project_gnn_anchor_sensitivity_bug.md).")
    parser.add_argument("--grad_mode", type=_bool, default=False, help="Build a gradient-matching-loss dataset instead of a plain one (engine=ccx only): each sample is a base ccx run plus --grad_tiles_per_sample single-tile-perturbed reruns, finite-differencing the real solver's ks_stress to attach ground-truth per-tile sensitivity labels (grad_tile_idx/grad_dks_dx) used by training/gnn_training.py's gradient-matching loss. Point --output_dir at a directory SEPARATE from the main dataset -- samples from this mode must not be mixed into the same shards as unlabeled samples (PyG's batch collation can't handle a mixed schema).")
    parser.add_argument("--grad_tiles_per_sample", type=int, default=6, help="Number of tiles to finite-difference per sample in --grad_mode (chosen uniformly at random, seeded by dim_idx for reproducibility).")
    parser.add_argument("--grad_fd_frac", type=float, default=0.02, help="FD step size in --grad_mode, as a fraction of each tile's own [min,max] span -- matches optimization/topopt_optimizer.py's --fd_step convention so the two are comparable. Ignored by --grad_engine adjoint.")
    parser.add_argument("--grad_engine", type=str, default="fd", choices=["fd", "adjoint"], help="How --grad_mode computes ground-truth per-tile sensitivity. 'fd' (default): grad_tiles_per_sample separate single-tile-perturbed ccx reruns per sample. 'adjoint': one ccx *SENSITIVITY solve per sample (core/tile_sensitivity.py's chain rule) covering every CCX-accepted tile at once -- far cheaper, but only covers tiles CCX accepts as design nodes (see core/mesh.py's solve_ccx_sensitivity docstring), not every tile.")
    parser.add_argument("--grad_max_tiles", type=int, default=None, help="--grad_engine adjoint only: cap on how many (tile, sensitivity) pairs to keep per sample (sorted by tile index, so the same subset every sample). Unset/<=0 keeps every CCX-accepted tile.")
    args = parser.parse_args()
    build_dataset(
        args.geometry, num_samples=args.num_samples, pool_size=args.pool_size, nproc=args.nproc,
        seed=args.seed, output_dir=args.output_dir, run_location=args.run_location,
        engine=args.engine, ccx_path=args.ccx_path, mesh_threads=args.mesh_threads,
        stress_ks_rho_mpa=args.stress_ks_rho_mpa,
        field_amplitude_frac_up=args.field_amplitude_frac_up, field_amplitude_frac_down=args.field_amplitude_frac_down,
        field_base_thickness=args.field_base_thickness, field_random_base=args.field_random_base,
        grad_mode=args.grad_mode, grad_tiles_per_sample=args.grad_tiles_per_sample, grad_fd_frac=args.grad_fd_frac,
        grad_engine=args.grad_engine, grad_max_tiles=args.grad_max_tiles,
    )
    # By this point every result is already durably written to shards/ on
    # disk -- there's nothing left for normal interpreter shutdown to do.
    # But after a run that went through many pool-crash-and-retry cycles,
    # normal shutdown can hang indefinitely instead of exiting: confirmed
    # directly on job 48190 (2026-08-20), where two of three parallel
    # dataset_builder.py processes sat alive for the rest of the job after
    # printing "Dataset saved", pinned in futex_wait_queue_me with zero CPU
    # progress across repeated checks -- almost certainly multiprocessing's
    # resource_tracker/atexit cleanup trying to coordinate with shared-memory
    # bookkeeping left in a confused state by all those killed workers (see
    # project_dataset_pool_split.md). A hung process here means the whole
    # slurm job (which `wait`s on it) never exits either, burning the rest
    # of its time allocation for nothing. os._exit() skips all of that
    # cleanup and terminates immediately -- safe here specifically because
    # the only thing left to lose is bookkeeping this process doesn't need
    # anymore, not data.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
