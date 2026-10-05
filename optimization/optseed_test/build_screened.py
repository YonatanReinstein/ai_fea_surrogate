"""Two-criterion screened dataset builder. STANDALONE -- imports training.dataset_builder
and training.dataset_io, modifies neither, and writes ONE shard directory so there is
no part-merge at the end (the merge was the slow tail of the 3-part builds).

Pipeline:
  1. generate candidates with dataset_builder's own samplers            free
  2. score each: contrast exact from the design, stress from the fit    free
     W = exp(-0.5((s_hat-T)/sigma_e)^2) * contrast^alpha
  3. accept prop. to W, oversampling by KEEP_FACTOR so the true-stress
     second stage still leaves TARGET_N                                 free
  4. solve accepted designs with _adjoint_run_grad_sample               the cost
  5. keep a sample only if its TRUE ccx stress is in band; shard as we go
"""
import os, sys, json, time, argparse, threading
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError
import importlib
from training import dataset_builder as DB
from training import dataset_io

ap = argparse.ArgumentParser()
ap.add_argument("--geometry", default="hollow_cube_11x11")
ap.add_argument("--target_n", type=int, default=10000)
ap.add_argument("--out", default=None)
ap.add_argument("--pool_size", type=int, default=84)
ap.add_argument("--mesh_threads", type=int, default=2)
ap.add_argument("--nproc", type=int, default=1)
ap.add_argument("--rho", type=float, default=0.25)
ap.add_argument("--T", type=float, default=100.0)
ap.add_argument("--sigma_e", type=float, default=80.0)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--band", default="50,200")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--two_level_frac", type=float, default=0.5)
ap.add_argument("--amp_range", default="0.05,0.5")
ap.add_argument("--coef", default="3.346,-0.819,0.235",
                help="stress predictor log KS = a + b*log(t_med) + c*log(contrast). "
                     "GEOMETRY-SPECIFIC -- the default is hollow_cube_11x11 (R2 0.88). "
                     "hollow_cube (840 tiles) is 3.1838,-0.8339,0.3602 (R2 0.89).")
a = ap.parse_args()
BAND_LO, BAND_HI = (float(v) for v in a.band.split(","))
G, base = a.geometry, f"data/{a.geometry}"
OUT = a.out or f"{base}/dataset_screened{a.target_n//1000}k"
os.makedirs(f"{OUT}/shards", exist_ok=True)

bc = importlib.import_module(f"data.{G}.boundary_conditions")
mat = json.load(open(f"{base}/CAD_model/material_properties.json"))
dims_template = json.load(open(f"{base}/CAD_model/dims.json"))
keys = sorted([k for k in dims_template if k[0] == "d" and k[1:].isdigit()], key=lambda k: int(k[1:]))
lo = np.array([dims_template[k]["min"] for k in keys]); hi = np.array([dims_template[k]["max"] for k in keys])
NX, NY, NZ = bc.tile_grid(); K = NX * NY * NZ
U, V, W_ = bc.mesh_resolution()
model_path, graph_path = f"{base}/CAD_model", f"{base}/dataset/graph.pt"
fixed_dims = bc.fixed_dims()
dims_bounds = {k: (dims_template[k]["min"], dims_template[k]["max"]) for k in keys}
ccx_path = os.environ.get("CCX_PATH", "ccx")
COEF = np.array([float(v) for v in a.coef.split(",")])   # log KS ~ a + b log t_med + c log contrast
print(f"[fit] stress predictor coef = {COEF}", flush=True)
con = lambda d: float(d.max() / max(d.min(), 1e-9))

# ---------------- stages 1-3: candidates, score, accept (no FEA) ----------------
rng = np.random.default_rng(a.seed)
vlo, vhi, vinv = DB._build_or_load_volume_inverse(base, model_path, fixed_dims, keys, lo[0], hi[0])
amp_lo, amp_hi = (float(v) for v in a.amp_range.split(","))

def _loguni(u, aa, bb):
    return float(np.exp(np.log(aa) + u * (np.log(bb) - np.log(aa))))

def make_candidate(u):
    """u: the 6 Sobol coordinates -> one design.

    This reproduces dataset_builder's own axis mapping EXACTLY (its lines
    ~836-890): axis 4 selects two-level vs correlated field, and each branch
    reads the same axes with the same transforms. Inventing a different
    mapping changes the candidate distribution, which silently invalidates the
    screen -- a first version of this file did that and the accepted contrast
    came out at 2.3 instead of ~13.
    """
    if a.two_level_frac > 0 and u[4] < a.two_level_frac:
        _sig = 1.0 + u[1] * 5.0                                    # 1-6 tiles
        _vf  = 0.05 + u[2] * 0.35                                   # 5-40% members
        _tt  = lo[0] + (0.25 + 0.75 * u[0]) * (hi[0] - lo[0])
        _beta = None if u[5] < 0.5 else _loguni(u[5], 2.0, 60.0)    # half rank-map, half soft step
        _tf  = (u[3] ** 2) * 0.45                                   # background height, thin-biased
        d = DB._sample_two_level(keys, lo, hi, NX, NY, NZ, rng, _sig, _vf, _tt, _beta, _tf,
                                 back_texture=0.3 + 0.5 * u[4] / max(a.two_level_frac, 1e-9))
    else:
        anchor = float(vinv(vlo + u[0] * (vhi - vlo)))               # uniform in VOLUME
        a_up = _loguni(u[2], amp_lo, amp_hi)
        a_dn = _loguni(u[3], amp_lo, amp_hi)
        d = DB._sample_correlated_field(keys, lo, hi, np.full(K, anchor), NX, NY, NZ, rng,
                                        amplitude_frac_up=a_up, amplitude_frac_down=a_dn,
                                        sigma=1.0 + u[1] * 11.0)
    return np.array([d[k] for k in keys], float)

from scipy.stats import qmc

def _screen_chunk(args):
    """One Sobol batch, generated + scored in its own process. Returns only the
    ACCEPTED designs, so the pickle back to the parent is small."""
    m_exp, batch_seed = args
    r = np.random.default_rng(batch_seed)
    globals()["rng"] = r                       # the samplers draw from module-level rng
    Um = qmc.Sobol(6, scramble=True, seed=batch_seed).random_base2(m_exp)
    cand = np.array([make_candidate(u) for u in Um])
    c = np.array([con(d) for d in cand])
    s_hat = np.exp(COEF[0] + COEF[1]*np.log(np.median(cand,1)) + COEF[2]*np.log(c))
    Wt = np.exp(-0.5*((s_hat - a.T)/a.sigma_e)**2) * (c**a.alpha)
    keep = r.random(len(cand)) < Wt/max(Wt.max(), 1e-30)
    return cand[keep], len(cand)

KEEP_FACTOR = 1.7                                 # covers the true-stress second stage
need_solve = int(a.target_n * KEEP_FACTOR)
t0 = time.time(); accepted = []; n_cand = 0
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=min(a.pool_size, 24)) as sx:
        rnd = 0
        while len(accepted) < need_solve:
            batch = [(12, a.seed + 1000*rnd + j) for j in range(min(a.pool_size, 24))]
            for accd, nc in sx.map(_screen_chunk, batch):
                accepted.extend(accd); n_cand += nc
            rnd += 1
            print(f"[screen] {n_cand:8d} candidates -> {len(accepted):6d} accepted "
                  f"({100*len(accepted)/n_cand:.2f}%)  {time.time()-t0:.0f}s", flush=True)
    accepted = accepted[:need_solve]
    print(f"[screen] DONE {n_cand} candidates -> {len(accepted)} to solve in {time.time()-t0:.0f}s "
          f"(contrast median {np.median([con(d) for d in accepted]):.1f})", flush=True)

# ---------------- stages 4-5: solve in WAVES, filter on true stress, shard ----------------
# Why waves and not one 17k submit: the first run of this script queued every
# future at once with no watchdog. The pool died silently around sample 3500 and
# as_completed() blocked forever -- 62 min at 0% CPU with 3 surviving processes.
# training/dataset_builder.py never does that: it works in CHUNK_SIZE batches,
# rebuilds the pool between them, and watchdogs stalled waves ("as_completed()
# saw 300s with zero new completions -- pool's manager thread likely died
# silently"). This mirrors that, plus resume from whatever shards already exist.
if __name__ == "__main__":
    SHARD, WAVE, STALL = 100, 1000, 420.0
    kept = dataset_io.num_existing_samples(OUT)
    kept0 = kept
    # shard_start = kept, NOT kept rounded down: if the previous run ended on a
    # PARTIAL shard (say 14 samples in shard_000000_000013), rounding down gives
    # shard_start=0 and the next full shard would be written as
    # shard_000000_000099, clobbering it. Starting at `kept` just appends.
    shard_start = kept
    if kept:
        print(f"[resume] {kept} samples already in {OUT}/shards -- continuing", flush=True)
    shard, n_done, n_drop = [], 0, 0
    writer = ThreadPoolExecutor(max_workers=2)
    t1 = time.time()
    queue = list(accepted)
    rng.shuffle(queue)
    wave_no = 0
    while kept < a.target_n and queue:
        wave, queue = queue[:WAVE], queue[WAVE:]
        wave_no += 1
        ex = ProcessPoolExecutor(max_workers=a.pool_size, initializer=DB._ccx_worker_init)
        try:
            futs = {ex.submit(DB._adjoint_run_grad_sample, model_path, a.mesh_threads, fixed_dims,
                              bc.anchor_condition, bc.force_pattern, U, V, W_,
                              mat["young_modulus"], mat["poisson_ratio"], ccx_path, a.nproc,
                              getattr(bc, "tile_grid", None), j,
                              {k: float(t) for k, t in zip(keys, d)}, graph_path, a.rho,
                              dims_bounds, True): j
                    for j, d in enumerate(wave)}
            last = time.time()
            for f in as_completed(futs, timeout=STALL * 3):
                n_done += 1; last = time.time()
                try:
                    _idx, r, _meta = f.result()
                except Exception:
                    n_drop += 1; r = None
                if r is not None:
                    sv = float(np.asarray(r.ks_stress).ravel()[0]) / 1e6
                    if BAND_LO <= sv <= BAND_HI:
                        shard.append(r); kept += 1
                        if len(shard) == SHARD:
                            writer.submit(dataset_io.save_shard, OUT, shard_start, list(shard))
                            shard_start += SHARD; shard = []
                    else:
                        n_drop += 1
                if n_done % 250 == 0:
                    el = time.time() - t1
                    print(f"[solve] wave {wave_no} | {n_done} solved | kept {kept}/{a.target_n} | "
                          f"dropped {n_drop} ({100*n_drop/max(n_done,1):.0f}%) | "
                          f"{el/max(n_done,1)*a.pool_size:.1f} worker-s/sample | "
                          f"ETA {(a.target_n-kept)*el/max(kept-kept0,1)/60:.0f} min", flush=True)
                if kept >= a.target_n:
                    break
        except TimeoutError:
            print(f"[solve] wave {wave_no} STALLED >{STALL*3:.0f}s with no completions -- "
                  f"rebuilding the pool and requeueing {len(futs)-n_done} designs", flush=True)
            queue = wave + queue
        finally:
            for fu in futs:
                fu.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
            print(f"[wave] {wave_no} closed | kept {kept} | {len(queue)} designs still queued "
                  f"| {time.time()-t1:.0f}s elapsed", flush=True)
    if shard:
        writer.submit(dataset_io.save_shard, OUT, shard_start, list(shard))
    writer.shutdown(wait=True)
    print(f"[done] {dataset_io.num_existing_samples(OUT)} samples in {OUT}/shards "
          f"({n_done} solved this run, {n_drop} off-band, {time.time()-t1:.0f}s)  NO MERGE NEEDED", flush=True)
