"""Train the hierarchical CNN surrogate on the multi-load cross4 dataset.

Predicts, per node, displacement (3) and stress tensor (6) for one (design,
load case) at a time; von Mises / peak / KS are derived from the predicted
tensor. Data: training.multiload_data (one design file serves all its kept
cases; cases with peak (excluding the loaded tile) above --peak_max_mpa are
left out). Train / validation split by design.

    python -m training.train_multiload --run_dir training/runs/multiload_v1

Loss: MSE over the 9 standardized targets (per-component mean/std from the
training set). Validation reports physical errors: relative L2 of the
displacement and of von Mises, and the relative error of each case's peak von
Mises (excluding the loaded tile) and KS.
LR: AdamW, linear warmup, then x0.99 per epoch and x0.5 when the epoch's
training loss stops improving (no cosine).
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from training.multiload_data import MultiLoadStatic, DesignDataset, case_mask
from utils.gnn_hierarchical_cnn import HierarchicalGNNCNN


def von_mises(S):
    sxx, syy, szz, sxy, sxz, syz = (S[..., i] for i in range(6))
    return torch.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                      + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2) + 1e-12)


def compute_stats(ds, n_designs, rng):
    xs, es, ts, ys = [], [], [], []
    for i in rng.sample(range(len(ds)), min(n_designs, len(ds))):
        for d in ds[i][:4]:
            xs.append(d.x); ts.append(d.tile_x); ys.append(d.y)
            es.append(d.edge_attr[::7])
    cat = lambda L: torch.cat(L, 0).double()
    st = {}
    for name, L in (("x", xs), ("edge", es), ("tile", ts), ("y", ys)):
        a = cat(L)
        st[name + "_mean"] = a.mean(0).float()
        st[name + "_std"] = a.std(0).clamp_min(1e-12).float()
    # load columns are almost all zero: standardize by their nonzero spread
    xl = cat(xs)[:, 3:6]
    nz = xl.abs().sum(1) > 0
    st["x_mean"][3:6] = 0.0
    st["x_std"][3:6] = xl[nz].abs().mean(0).clamp_min(1e-12).float()
    st["tile_mean"][1:] = 0.0
    st["tile_std"][1:] = 1.0
    return st


Y_COLS = slice(0, 9)          # set in main(): 3:9 = stress tensor only (--stress_only)


def to_device_batch(cases, st, dev):
    b = Batch.from_data_list(cases).to(dev, non_blocking=True)
    b.y = b.y[:, Y_COLS]
    b.x = (b.x - st["x_mean"]) / st["x_std"]
    b.edge_attr = (b.edge_attr - st["edge_mean"]) / st["edge_std"]
    b.tile_x = (b.tile_x - st["tile_mean"]) / st["tile_std"]
    return b


def forward(model, b):
    _, pred = model(b.x, b.edge_index, b.edge_attr, b.batch, tile_idx=b.tile_idx,
                    tile_NX=11, tile_NY=11, tile_NZ=1, tile_x=b.tile_x,
                    num_graphs=b.num_graphs, node_id=b.node_id)
    return pred


@torch.no_grad()
def validate(model, loader, st, dev, static, max_designs, rho_mpa=0.25):
    model.eval()
    agg = {k: [] for k in ("u_rel", "vm_rel", "peak_rel", "ks_rel", "loss")}
    has_u = Y_COLS.start == 0
    nd = 0
    load_node_sets = {r: torch.as_tensor(v, device=dev) for r, v in static.load_nodes.items()}
    for cases in loader:
        if nd >= max_designs:
            break
        nd += 1
        for j in range(0, len(cases), 4):
            chunk = cases[j:j + 4]
            b = to_device_batch(chunk, st, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = forward(model, b)
            p = p.float()
            yn = (b.y - st["y_mean"]) / st["y_std"]
            agg["loss"].append(float(((p - yn) ** 2).mean()))
            P = p * st["y_std"] + st["y_mean"]
            for g in range(b.num_graphs):
                m = b.batch == g
                Ps, Ts = (P[m, 3:], b.y[m, 3:]) if has_u else (P[m], b.y[m])
                vp, vt = von_mises(Ps), von_mises(Ts)
                if has_u:
                    Pu, Tu = P[m, :3], b.y[m, :3]
                    agg["u_rel"].append(float((Pu - Tu).norm() / Tu.norm().clamp_min(1e-30)))
                agg["vm_rel"].append(float((vp - vt).norm() / vt.norm().clamp_min(1e-30)))
                row = static.cases[int(b.case[g])][0]
                keep = torch.ones(int(m.sum()), dtype=torch.bool, device=dev)
                keep[load_node_sets[row]] = False
                pk, tk = vp[keep].max(), vt[keep].max()
                agg["peak_rel"].append(float((pk - tk) / tk))
                r = rho_mpa * 1e-6
                ksf = lambda v: v.max() + torch.log(torch.exp(r * (v - v.max())).sum()) / r
                kp, kt = ksf(vp.double()), ksf(vt.double())
                agg["ks_rel"].append(float((kp - kt) / kt))
    model.train()
    out = {k: float(np.median(np.abs(v))) if k != "loss" else float(np.mean(v)) for k, v in agg.items() if v}
    out.update({k + "_p90": float(np.percentile(np.abs(v), 90)) for k, v in agg.items() if v and k != "loss"})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", default="data/cross4_11x11/dataset_multiload_5000")
    p.add_argument("--run_dir", default="training/runs/multiload_v1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--graphs_per_step", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--coarse_dim", type=int, default=128)
    p.add_argument("--cnn_depth", type=int, default=2)
    p.add_argument("--peak_max_mpa", type=float, default=300.0)
    p.add_argument("--val_designs", type=int, default=250)
    p.add_argument("--val_every", type=int, default=2000, help="steps between validations")
    p.add_argument("--val_max_designs", type=int, default=60)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_checkpoint", type=int, default=0, help="activation checkpointing (saves memory)")
    p.add_argument("--max_steps", type=int, default=0, help="stop after this many steps (smoke tests)")
    p.add_argument("--stress_only", type=int, default=1,
                   help="1: predict the 6 stress components only (displacement left out of the loss)")
    args = p.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    static = MultiLoadStatic(args.data)
    n_designs = json.load(open(os.path.join(args.data, "config.json")))["num_designs"]
    ids = list(range(n_designs))
    rng.shuffle(ids)
    val_ids, train_ids = sorted(ids[:args.val_designs]), sorted(ids[args.val_designs:])
    mask = case_mask(args.data, train_ids, args.peak_max_mpa * 1e6)
    n_cases = int(mask.sum())
    print(f"[data] {len(train_ids)} train / {len(val_ids)} val designs; "
          f"{n_cases} train cases kept of {mask.size} ({mask.mean():.1%}) at peak <= {args.peak_max_mpa} MPa", flush=True)

    train_ds = DesignDataset(args.data, train_ids, static, args.peak_max_mpa * 1e6)
    val_ds = DesignDataset(args.data, val_ids, static, args.peak_max_mpa * 1e6)
    stats_path = os.path.join(args.run_dir, "stats.pt")
    if os.path.exists(stats_path):
        st = torch.load(stats_path)
    else:
        st = compute_stats(train_ds, 60, rng)
        torch.save(st, stats_path)
    st = {k: v.to(dev) for k, v in st.items()}
    global Y_COLS
    Y_COLS = slice(3, 9) if args.stress_only else slice(0, 9)
    st["y_mean"], st["y_std"] = st["y_mean"][Y_COLS], st["y_std"][Y_COLS]
    out_dim = Y_COLS.stop - Y_COLS.start

    model = HierarchicalGNNCNN(node_in_dim=8, edge_in_dim=4, hidden_dim=args.hidden_dim, tile_in_dim=4,
                               num_pos_nodes=static.N, coarse_dim=args.coarse_dim, cnn_depth=args.cnn_depth,
                               node_out_dim=out_dim, use_checkpoint=bool(args.use_checkpoint)).to(dev)
    model.set_tile_grid(11, 11, 1)
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ck_path = os.path.join(args.run_dir, "last.pt")
    start_epoch, step, best, prev_epoch_loss, lr_scale = 0, 0, float("inf"), float("inf"), 1.0
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, step, best = ck["epoch"], ck["step"], ck["best"]
        prev_epoch_loss, lr_scale = ck.get("prev_epoch_loss", float("inf")), ck.get("lr_scale", 1.0)
        print(f"[resume] epoch {start_epoch}, step {step}, best val vm_rel {best:.4f}", flush=True)
    json.dump(vars(args), open(os.path.join(args.run_dir, "args.json"), "w"), indent=1)

    def save(path, epoch, extra=None):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step,
                    "best": best, "prev_epoch_loss": prev_epoch_loss, "lr_scale": lr_scale,
                    "args": vars(args), "stats": {k: v.cpu() for k, v in st.items()},
                    "model_kwargs": dict(node_in_dim=8, edge_in_dim=4, hidden_dim=args.hidden_dim, tile_in_dim=4,
                                         num_pos_nodes=static.N, coarse_dim=args.coarse_dim,
                                         cnn_depth=args.cnn_depth, node_out_dim=out_dim),
                    **(extra or {})}, path + ".tmp")
        os.replace(path + ".tmp", path)

    mk_loader = lambda ds, shuffle: torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=shuffle, num_workers=args.workers, collate_fn=lambda x: x[0],
        persistent_workers=True, prefetch_factor=2)
    train_loader, val_loader = mk_loader(train_ds, True), mk_loader(val_ds, False)
    log = open(os.path.join(args.run_dir, "log.jsonl"), "a")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        t_ep, ep_loss, ep_n, t_log, run_loss, run_n = time.time(), 0.0, 0, time.time(), 0.0, 0
        for cases in train_loader:
            rng.shuffle(cases)
            for j in range(0, len(cases), args.graphs_per_step):
                b = to_device_batch(cases[j:j + args.graphs_per_step], st, dev)
                yn = (b.y - st["y_mean"]) / st["y_std"]
                lr = args.lr * lr_scale * min(1.0, (step + 1) / args.warmup_steps)
                for g in opt.param_groups:
                    g["lr"] = lr
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = forward(model, b)
                loss = ((pred.float() - yn) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                lv = float(loss)
                ep_loss += lv; ep_n += 1; run_loss += lv; run_n += 1
                if step % 100 == 0:
                    dt = time.time() - t_log
                    print(f"[train] epoch {epoch} step {step} loss {run_loss / run_n:.4f} lr {lr:.2e} "
                          f"{run_n / dt:.2f} steps/s  mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
                    log.write(json.dumps({"step": step, "epoch": epoch, "loss": run_loss / run_n, "lr": lr}) + "\n"); log.flush()
                    t_log, run_loss, run_n = time.time(), 0.0, 0
                if args.max_steps and step >= args.max_steps:
                    v = validate(model, val_loader, st, dev, static, args.val_max_designs)
                    print(f"[val] step {step}: " + ", ".join(f"{k} {x:.4f}" for k, x in sorted(v.items())), flush=True)
                    print(f"[smoke] stopping at step {step}; peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
                    return
                if step % args.val_every == 0:
                    v = validate(model, val_loader, st, dev, static, args.val_max_designs)
                    print(f"[val] step {step}: " + ", ".join(f"{k} {x:.4f}" for k, x in sorted(v.items())), flush=True)
                    log.write(json.dumps({"step": step, "epoch": epoch, "val": v}) + "\n"); log.flush()
                    if v["vm_rel"] < best:
                        best = v["vm_rel"]
                        save(os.path.join(args.run_dir, "best.pt"), epoch, {"val": v})
                    save(ck_path, epoch)
        ep_loss /= max(ep_n, 1)
        if ep_loss > prev_epoch_loss * 0.995:
            lr_scale *= 0.5
            print(f"[lr] epoch loss {ep_loss:.4f} did not improve on {prev_epoch_loss:.4f}: lr x0.5", flush=True)
        lr_scale *= 0.99
        prev_epoch_loss = min(prev_epoch_loss, ep_loss)
        print(f"[epoch] {epoch} done: loss {ep_loss:.4f}, {ep_n} steps, {(time.time() - t_ep) / 60:.1f} min", flush=True)
        save(ck_path, epoch + 1)


if __name__ == "__main__":
    main()
