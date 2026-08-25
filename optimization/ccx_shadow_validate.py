"""Shadow-validates a running surrogate-driven topopt job against real CCX,
generation by generation, so surrogate drift is caught within a handful of
generations instead of discovered after the fact (see the
project_surrogate_ood_collapse_topopt memory: job 1323938's CNN surrogate
looked fine through gen ~17, then silently drifted to reporting "basically
at yield" for 30+ generations while the real design climbed to 39% over
yield -- a single cheap CCX solve per generation would have caught this by
gen ~20-25 instead of gen 48).

Usage (one-shot, checks every gen not yet in the state file):
    python -m optimization.ccx_shadow_validate --geometry hollow_cube

Usage (watch a live job, polling for new generations):
    python -m optimization.ccx_shadow_validate --geometry hollow_cube --watch --poll_interval 300

Each new generation's base design (topopt_history/gen_*.npz field "x") is
re-solved with real CCX, in small fixed-size batches -- each batch gets its
own CCXEvaluator / ProcessPoolExecutor, since a single large pool spanning
many tasks per worker was observed this session to silently return wrong
results (most likely the known ccx-worker fd-leak issue, see
project_dataset_builder_ccx_fd_leak memory), corrupting a 105-task/6-worker
batch while every <=21-task batch this session came back correct. Each
batch's real CCX volume is cross-checked against the generation's own
recorded volume (a deterministic geometric quantity, not surrogate-derived)
as an integrity guard -- a mismatch means dims weren't solved correctly and
that generation's stress comparison is untrustworthy, not evidence of drift.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from evaluators.ccx_evaluator import CCXEvaluator

CHUNK_SIZE = 16  # safe per-evaluate() batch size (see module docstring)


def load_gen(hist_dir, gen):
    z = np.load(os.path.join(hist_dir, f"gen_{gen:03d}.npz"), allow_pickle=True)
    names = [str(n) for n in z["free_names"]]
    x = z["x"].astype(float)
    return {
        "dims": {n: float(v) for n, v in zip(names, x)},
        "cnn_stress": float(z["s0"]),
        "cnn_vol": float(z["v0"]),
    }


def list_available_gens(hist_dir):
    files = sorted(glob.glob(os.path.join(hist_dir, "gen_*.npz")))
    return sorted(int(os.path.basename(f)[4:-4]) for f in files)


def load_state(state_path):
    if os.path.exists(state_path):
        return json.load(open(state_path))
    return {"checked": {}}  # gen (str) -> record dict


def save_state(state_path, state):
    tmp = state_path + ".tmp"
    json.dump(state, open(tmp, "w"), indent=2)
    os.replace(tmp, state_path)


def check_gens(geometry, hist_dir, gens, yield_mpa, stress_ks_rho_mpa, max_workers, alert_frac):
    records = []
    for start in range(0, len(gens), CHUNK_SIZE):
        chunk = gens[start:start + CHUNK_SIZE]
        loaded = {g: load_gen(hist_dir, g) for g in chunk}
        ev = CCXEvaluator(geometry, max_workers=max_workers, nproc=1, mesh_threads=1,
                           stress_ks_rho_mpa=stress_ks_rho_mpa)
        res = ev.evaluate([loaded[g]["dims"] for g in chunk])
        for j, g in enumerate(chunk):
            ccx_stress = res["stress"][j]
            ccx_vol = res["volume"][j]
            info = loaded[g]
            vol_ok = ccx_vol is not None and abs(ccx_vol - info["cnn_vol"]) < 1e-3
            record = {
                "gen": g,
                "cnn_stress": info["cnn_stress"],
                "ccx_stress": ccx_stress,
                "vol_ok": vol_ok,
            }
            if ccx_stress is not None and vol_ok:
                err = info["cnn_stress"] - ccx_stress
                rel = abs(err) / max(ccx_stress, 1e-9)
                record["error"] = err
                record["rel_error"] = rel
                record["ccx_feasible"] = ccx_stress <= yield_mpa
                record["cnn_feasible"] = info["cnn_stress"] <= yield_mpa
                alert = rel > alert_frac or record["ccx_feasible"] != record["cnn_feasible"]
                record["alert"] = alert
                tag = "[ALERT]" if alert else "        "
                print(f"{tag} gen {g:3d} | cnn={info['cnn_stress']:8.3f} | ccx={ccx_stress:8.3f} | "
                      f"err={err:+8.3f} ({100*rel:5.1f}%) | ccx_feasible={record['ccx_feasible']}",
                      flush=True)
                if record["ccx_feasible"] != record["cnn_feasible"]:
                    print(f"         ^ feasibility DISAGREEMENT at gen {g}: surrogate says "
                          f"{'feasible' if record['cnn_feasible'] else 'infeasible'}, "
                          f"real CCX says {'feasible' if record['ccx_feasible'] else 'infeasible'}",
                          flush=True)
            else:
                record["alert"] = True
                reason = "ccx solve failed" if ccx_stress is None else "volume mismatch (dims corrupted?)"
                print(f"[ALERT] gen {g:3d} | {reason}", flush=True)
            records.append(record)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="hollow_cube")
    ap.add_argument("--history_dir", default=None, help="Default: optimization/artifacts/topopt_history")
    ap.add_argument("--state", default=None, help="Default: optimization/artifacts/ccx_shadow_validate_state.json")
    ap.add_argument("--stress_ks_rho_mpa", type=float, default=0.15)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--alert_frac", type=float, default=0.03,
                     help="Relative |error| fraction (of ccx stress) above which a generation is flagged.")
    ap.add_argument("--watch", action="store_true", help="Keep polling for new generations instead of exiting.")
    ap.add_argument("--poll_interval", type=float, default=300.0, help="Seconds between polls in --watch mode.")
    ap.add_argument("--recheck_all", action="store_true", help="Ignore state file, re-check every generation found.")
    args = ap.parse_args()

    hist_dir = args.history_dir or f"optimization/artifacts/topopt_history"
    state_path = args.state or "optimization/artifacts/ccx_shadow_validate_state.json"
    material_props = json.load(open(f"data/{args.geometry}/CAD_model/material_properties.json"))
    yield_mpa = material_props["yield_strength"]

    state = {"checked": {}} if args.recheck_all else load_state(state_path)

    while True:
        available = list_available_gens(hist_dir)
        already = {int(g) for g in state["checked"].keys()}
        new_gens = [g for g in available if g not in already]

        if new_gens:
            print(f"[shadow] {len(new_gens)} new generation(s) to check: "
                  f"{new_gens[0]}..{new_gens[-1]}", flush=True)
            records = check_gens(args.geometry, hist_dir, new_gens, yield_mpa,
                                  args.stress_ks_rho_mpa, args.max_workers, args.alert_frac)
            for r in records:
                state["checked"][str(r["gen"])] = r
            save_state(state_path, state)

            alerts = [r for r in records if r.get("alert")]
            if alerts:
                print(f"[shadow] {len(alerts)}/{len(records)} new generation(s) ALERTED "
                      f"(relative error > {100*args.alert_frac:.0f}% or feasibility disagreement): "
                      f"gens {[r['gen'] for r in alerts]}", flush=True)
        else:
            print("[shadow] no new generations", flush=True)

        if not args.watch:
            break
        time.sleep(args.poll_interval)

    n_checked = len(state["checked"])
    n_alerts = sum(1 for r in state["checked"].values() if r.get("alert"))
    print(f"[shadow] done. {n_checked} generation(s) checked total, {n_alerts} alerted. "
          f"state: {state_path}", flush=True)


if __name__ == "__main__":
    main()
