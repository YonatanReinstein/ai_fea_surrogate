"""Fine-tune the surrogate DURING a topopt run, from ccx-adjoint labels of the
designs that run actually visits.

Why on-policy is the only option here
-------------------------------------
Measured 2026-09-10: ten random designs matched to a trajectory design's
mean/std/total-variation scored cosine +0.94..+0.996 against the ccx adjoint;
the trajectory design itself scored +0.09, with no overlap. The failing designs
are not unusual in any measurable way -- they are selected by the surrogate's
OWN errors, because OC descends its gradient. No i.i.d. sampler can generate
them, so the labels have to come from a live trajectory.

Safety
------
The base model scores cosine ~+0.98 against ccx on in-distribution designs. Two
guards keep fine-tuning from trading that away:
  * REPLAY -- most of every batch is original dataset_gradadj samples, so the
    update cannot specialise onto the narrow stream of trajectory designs.
  * A HELD-OUT GATE -- after each update the new weights are scored on a fixed
    held-out set; if the |stress bias| there exceeds the gate (--max_stress_abs,
    or the base bias + --max_stress_drift), the update is DISCARDED and the
    previous weights stay live. (A cosine gate, --max_regression, was removed
    2026-10-01 along with the cosine loss; the cosine is still logged.)
Neither guard is optional: the failure mode being defended against
(catastrophic forgetting onto a self-selected distribution) is exactly what an
unguarded on-policy loop produces.
"""
import argparse, io, os, time, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch


def grad_rel_err(pred, true, power, sat):
    """Same per-tile gradient loss as training/gnn_training.py (--grad_err_mode rel):
    saturated percent error r/(1+r/sat), r = |pred-true|/|true|, weighted by
    (|true|/sum|true|)^power normalised to mean 1. One sample, 1-D tensors."""
    r = (pred - true).abs() / (true.abs() + 1e-12)
    e = r / (1.0 + r / sat)
    w = (true.abs() / true.abs().sum().clamp_min(1e-30)) ** power
    return (w / w.mean() * e).mean()


def grad_sign_err(pred, true, delta):
    """Same per-tile smooth wrong-sign loss as gnn_training (--grad_sign_loss_weight):
    0.5*(1 - z/(delta+|z|)), z = pred*sign(true)/|true|, mean over tiles."""
    z = pred * torch.sign(true) / (true.abs() + 1e-12)
    return (0.5 * (1.0 - z / (delta + z.abs()))).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--live_ckpt", required=True, help="weights the optimiser hot-reloads")
    ap.add_argument("--geometry", default="hollow_cube")
    ap.add_argument("--replay_dataset", default="data/hollow_cube/grad_val_mid/dataset.pt")
    ap.add_argument("--holdout_dataset", default="data/hollow_cube/grad_audit/dataset.pt")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--update_every", type=int, default=3, help="new labels per update")
    ap.add_argument("--steps", type=int, default=20, help="samples drawn per update (steps = samples / batch_size)")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="graphs per optimizer step. The model batches natively (tile_local = batch*K + tile_idx), "
                         "so B graphs cost about one forward/double-backward instead of B; --steps keeps meaning "
                         "samples seen, i.e. the number of optimizer steps is steps/batch_size.")
    ap.add_argument("--replay_frac", type=float, default=0.8)
    ap.add_argument("--holdout_n", type=int, default=24)
    ap.add_argument("--max_stress_drift", type=float, default=0.01,
                    help="discard an update whose |held-out stress bias| grows by more than this "
                         "(relative to the base's bias; see --max_stress_abs)")
    ap.add_argument("--max_stress_abs", type=float, default=None,
                    help="ABSOLUTE stress gate: discard an update whose |held-out stress bias| exceeds "
                         "this, regardless of the base. The relative gate punished a well-calibrated "
                         "base: rho-0.25 epoch 80 started at -0.2%% bias, so the gate was 1.2%% and 42%% "
                         "of updates -- the ones correcting on-trajectory optimism -- were discarded "
                         "(job 1399058); every earlier run had 4-14%% of headroom just from a worse base.")
    # Loss terms, matching training/gnn_training.py. Defaults = the 11x11
    # screened run (jobs/train_11x11_screened.slurm, job 1430006). Each term is
    # divided by its know-nothing baseline as in the trainer. There is no node
    # term: trajectory labels carry no per-node stress.
    ap.add_argument("--graph_loss_weight", type=float, default=0.25,
                    help="KS stress term: |KS_pred - KS_true| / KS_true")
    ap.add_argument("--grad_loss_weight", type=float, default=0.25,
                    help="per-tile saturated percent error of dKS/dt")
    ap.add_argument("--grad_sign_loss_weight", type=float, default=0.25,
                    help="per-tile smooth wrong-sign loss")
    ap.add_argument("--grad_weight_power", type=float, default=0.15)
    ap.add_argument("--grad_rel_sat", type=float, default=2.0)
    ap.add_argument("--grad_sign_delta", type=float, default=0.05)
    ap.add_argument("--no_amp", dest="amp", action="store_false",
                    help="train in fp32. Default: bf16 autocast on the training forward, as "
                         "gnn_training pretrains -- 25%% faster per step, dg cosine vs fp32 >= 0.99995 "
                         "(L40S, hollow_cube 840 tiles). Held-out scoring/gating stays fp32.")
    ap.add_argument("--mesh_workers", type=int, default=8,
                    help="parallel graph rebuilds per update (the update's critical path)")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--idle_exit", type=float, default=300.0)
    ap.add_argument("--ema", type=float, default=0.0,
                    help="publish an exponential moving average of the trained weights: "
                         "live = (1-ema)*live + ema*trained per accepted update (0 = publish "
                         "the trained weights directly, the old behaviour)")
    ap.add_argument("--traj_buffer", type=int, default=0,
                    help="Keep a FIFO of the last N labelled TRAJECTORY designs and draw the on-policy "
                         "fraction uniformly from it, instead of only the newest bundle. Without it every "
                         "update is gradient descent on 5 near-identical designs (one iterate + four 5%% "
                         "jitter siblings) and consecutive updates see near-identical data -- correlated "
                         "online data, whose expected failure mode is memorising the current neighbourhood. "
                         "0 = old behaviour (newest bundle only).")
    ap.add_argument("--val_every", type=int, default=0,
                    help="Hold every Nth generation's labels OUT of training and score on them instead: "
                         "an ON-trajectory validation set, same distribution, never trained on. The default "
                         "gate holdout is off-trajectory (grad_audit) and reads a flat 0.999 while the "
                         "on-trajectory cosine swings 0.67-0.97, so it cannot see memorisation. 0 = off.")
    ap.add_argument("--val_gate", action="store_true",
                    help="Gate updates on the on-trajectory validation score (--val_every) instead of the "
                         "off-trajectory holdout. Needs enough validation designs to be stable.")
    ap.add_argument("--node_emb_lr_scale", type=float, default=1.0,
                    help="Learning-rate multiplier for node_emb, the per-NODE lookup table that is "
                         "independent of the design and absorbed 96.5%% of all weight movement in run T. "
                         "<1 lets it adapt where it genuinely helps without soaking up every update; "
                         "0 is equivalent to --freeze node_emb.")
    ap.add_argument("--anchor_l2", type=float, default=0.0,
                    help="Elastic penalty lambda*||theta - theta_base||^2 pulling the fine-tuned weights "
                         "back toward the pretrained checkpoint (applied to the node_emb group only, or to "
                         "all trainable params with --anchor_all). A gentler alternative to freezing.")
    ap.add_argument("--anchor_all", action="store_true",
                    help="Apply --anchor_l2 to every trainable parameter, not just node_emb.")
    ap.add_argument("--freeze", default="",
                    help="Comma-separated module prefixes to hold fixed during online fine-tuning "
                         "(e.g. 'node_emb', or 'node_emb,encoder,fine_convs,decode_conv,head' for the "
                         "whole fine level). Aliases: 'fine' = the node-level stack, 'coarse' = the "
                         "tile U-Net + tile encoders. Measured on run T after ~400 updates: node_emb "
                         "(1.54M params, a per-NODE lookup table that is independent of the design) "
                         "absorbs 96.5%% of all weight movement, the U-Net 3.0%%, everything else <0.5%%. "
                         "Freezing it keeps every design-dependent pathway trainable while removing the "
                         "biggest surface for memorising the current trajectory. NOTE: the backward still "
                         "reaches the thickness leaf (the loss is on d(stress)/d(thickness)), so this is "
                         "an accuracy lever, not a speed one.")
    ap.add_argument("--state_path", default=None,
                    help="trainer_state.pt next to the live ckpt: Adam moments, update count, EMA "
                         "state and the ORIGINAL base gates. Written on every publish; loaded at start "
                         "if it exists, so a resumed trainer continues rather than restarting with fresh "
                         "optimizer moments and gates re-anchored to the current (already drifted) weights.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="resume: treat labels already on disk as consumed (do not retrain on them)")
    a = ap.parse_args()

    from torch_geometric.data import Data, Batch
    from evaluators.gnn_evaluator import GNNEvaluator, _run_sample_worker
    from evaluators.probe_gradloss_sensitivity import load_model
    from training.gnn_training import gnn_input_fn
    from training.dataset_io import load_dataset
    from core.tile_sensitivity import dims_key_order_permutation
    import importlib
    bc = importlib.import_module(f"data.{a.geometry}.boundary_conditions")

    # Data-parallel over GPUs under torchrun (WORLD_SIZE > 1). The update is
    # GPU-bound and linear in graphs (91 ms/graph fp32 on an L40S, no gain
    # from bigger batches), so splitting each batch across GPUs is the only
    # way to cut it without training on less. Rank 0 does everything the
    # single-GPU trainer did (poll, score, gate, publish); every rank builds
    # the same on-policy samples from the label files rank 0 broadcasts,
    # draws the same batches from the shared seed, runs its share of each
    # batch, and the gradients are summed -- so each optimizer step sees the
    # whole batch's gradient and the ranks' weights stay identical.
    _world = int(os.environ.get("WORLD_SIZE", "1"))
    _rank = int(os.environ.get("RANK", "0"))
    if _world > 1:
        import torch.distributed as dist
        # ONLINE_FT_DIST_BACKEND=gloo lets ranks share one GPU (testing only)
        _lr = int(os.environ.get("LOCAL_RANK", "0")) % max(1, torch.cuda.device_count())
        torch.cuda.set_device(_lr)
        dist.init_process_group(os.environ.get("ONLINE_FT_DIST_BACKEND", "nccl"))
        dev = torch.device("cuda", _lr)
        if _rank != 0:
            import builtins
            builtins.print = lambda *_a, **_k: None   # one log; tracebacks still reach stderr
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _bcast(obj):
        """rank 0's value of obj, on every rank"""
        if _world == 1:
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _allreduce_grads(params):
        """Sum gradients over ranks. A param left without a gradient on EVERY
        rank stays None, as it would on one GPU (Adam skips it)."""
        if _world == 1:
            return
        has = torch.tensor([p.grad is not None for p in params], dtype=torch.float, device=dev)
        dist.all_reduce(has, op=dist.ReduceOp.MAX)
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in params])
        dist.all_reduce(flat)
        o = 0
        for p, h in zip(params, has.tolist()):
            n = p.numel()
            p.grad = flat[o:o + n].view_as(p).clone() if h else None
            o += n

    NX, NY, NZ = bc.tile_grid(); K = NX * NY * NZ
    permL = dims_key_order_permutation(NX, NY, NZ)
    perm = torch.tensor(permL, dtype=torch.long, device=dev)
    perm_inv = torch.argsort(perm)

    gev = GNNEvaluator(a.geometry, ckpt_path=a.base_ckpt)
    model, stats = load_model(a.base_ckpt, a.geometry, dev)
    x_mean, x_std, e_mean, e_std, nt_mean, nt_std = stats
    _ALIASES = {"fine": "node_emb,encoder,fine_convs,decode_conv,head",
                "coarse": "down_blocks,up_blocks,mid,bottleneck,tile_input_encoder,tile_emb"}
    _prefixes = []
    for tok in (t.strip() for t in a.freeze.split(",")):
        if tok:
            _prefixes.extend(_ALIASES.get(tok, tok).split(","))
    if _prefixes:
        n_frozen = sum(p.numel() for n, p in model.named_parameters()
                       if any(n.startswith(q) for q in _prefixes))
        for n, p in model.named_parameters():
            if any(n.startswith(q) for q in _prefixes):
                p.requires_grad_(False)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[trainer] frozen: {','.join(_prefixes)} -- {n_frozen}/{n_all} params "
              f"({100*n_frozen/n_all:.1f}%) held fixed", flush=True)
    _ne_params = [p for n, p in model.named_parameters() if n.startswith("node_emb") and p.requires_grad]
    _rest = [p for n, p in model.named_parameters() if not n.startswith("node_emb") and p.requires_grad]
    if _ne_params and a.node_emb_lr_scale != 1.0:
        opt = torch.optim.Adam([{"params": _rest, "lr": a.lr},
                                {"params": _ne_params, "lr": a.lr * a.node_emb_lr_scale}])
        print(f"[trainer] node_emb lr scaled x{a.node_emb_lr_scale} "
              f"({a.lr:.1e} -> {a.lr*a.node_emb_lr_scale:.1e})", flush=True)
    else:
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    _anchor = {}
    if a.anchor_l2 > 0:
        for n, p in model.named_parameters():
            if p.requires_grad and (a.anchor_all or n.startswith("node_emb")):
                _anchor[n] = p.detach().clone()
        print(f"[trainer] elastic anchor: lambda={a.anchor_l2:g} on "
              f"{'all trainable params' if a.anchor_all else 'node_emb'} "
              f"({sum(v.numel() for v in _anchor.values())} params)", flush=True)

    replay = load_dataset(a.replay_dataset)
    holdout = load_dataset(a.holdout_dataset)[:a.holdout_n]
    # Fixed-mesh datasets can ship without per-sample edge_index/tile_idx
    # (e.g. data/hollow_cube_11x11/dataset_screened30k); take them from the
    # geometry's graph.pt, as gnn_training does.
    if any(getattr(_s, "edge_index", None) is None or getattr(_s, "tile_idx", None) is None
           for _s in list(replay) + list(holdout)):
        from evaluators.gnn_evaluator import _get_fixed_graph
        assert gev.graph_path is not None, "replay/holdout samples lack edge_index/tile_idx and there is no graph.pt"
        _ei, _ti = _get_fixed_graph(gev.graph_path)
        for _s in list(replay) + list(holdout):
            if getattr(_s, "edge_index", None) is None:
                assert _s.edge_attr.shape[0] == _ei.shape[1] and _s.x.shape[0] == _ti.shape[0]
                _s.edge_index = _ei
            if getattr(_s, "tile_idx", None) is None:
                _s.tile_idx = _ti
    if getattr(model, "node_emb", None) is not None:
        _ne = model.node_emb.num_embeddings
        for _s in list(replay) + list(holdout):
            if not hasattr(_s, "node_id") and _s.x.shape[0] == _ne:
                _s.node_id = torch.arange(_ne, dtype=torch.long)
    print(f"[trainer] replay={len(replay)} holdout={len(holdout)} "
          f"lr={a.lr} replay_frac={a.replay_frac} amp={'bf16' if a.amp else 'off'}", flush=True)

    def dg_from_sample(s, train=True):
        """model's own autograd dg for a dataset sample, in dims-key order."""
        b = Batch.from_data_list([s])
        x, ei, ea, bidx, tidx, tNX, tNY, tNZ, tx, nid = gnn_input_fn(b)
        t_leaf = s.dims[0, 0:K].float().to(dev)[perm].view(-1, 1).clone().requires_grad_(True)
        bidx, tidx = bidx.to(dev), tidx.to(dev)
        ns = t_leaf.view(1, K)[bidx, tidx].unsqueeze(1)
        x = x.float().to(dev); x[:, 3:6] = x[:, 3:6] / 1e6
        # Two graph sources, two layouts: _run_sample_worker (on-policy designs)
        # already emits the per-node strut column, while dataset samples do not
        # and need it appended. Replace vs append -- getting this wrong is a
        # silent width mismatch, or worse, a column offset.
        xg = torch.cat([x[:, :-1], ns], dim=1) if x.shape[1] == x_mean.shape[-1] \
             else torch.cat([x, ns], dim=1)
        xn = (xg - x_mean) / x_std
        # dataset samples carry no node_id; the model was trained with one
        # (positional node_emb) -- same fix as evaluators/probe_dg_alignment.py
        if nid is None and getattr(model, "node_emb", None) is not None \
                and xg.shape[0] == model.node_emb.num_embeddings:
            nid = torch.arange(xg.shape[0], dtype=torch.long)
        gp, _ = model(xn, ei.to(dev), (ea.float().to(dev) - e_mean) / e_std, bidx,
                      tile_idx=tidx, tile_NX=tNX, tile_NY=tNY, tile_NZ=tNZ,
                      tile_x=t_leaf, num_graphs=1,
                      node_id=nid.to(dev) if nid is not None else None)
        s_mpa = (gp * nt_std + nt_mean).clamp_min(1e-6).squeeze()
        d = torch.autograd.grad(s_mpa, t_leaf, create_graph=train)[0].view(-1)
        return d[perm_inv[s.grad_tile_idx.to(dev)]], s_mpa

    # GPU-resident inputs for the fixed-topology fast path. Re-collating 4
    # graphs (866k edges each) on the CPU and copying them host->device every
    # step was ~8% of a step; every graph shares one edge_index, so each
    # sample's x / edge_attr is moved once and the batch is a GPU concat.
    _topo = {}

    def _gpu_inputs(s):
        c = s.__dict__.get("_ft_gpu")
        if c is not None:
            return c
        if not _topo:
            _topo.update(ei=s.edge_index.to(dev), ei_cpu=s.edge_index, tidx=s.tile_idx.to(dev),
                         tidx_cpu=s.tile_idx, N=int(s.x.shape[0]), nxyz=(int(s.tile_NX[0]),
                         int(s.tile_NY[0]), int(s.tile_NZ[0])), eiB={})
        same = (s.edge_index.shape == _topo["ei_cpu"].shape and torch.equal(s.edge_index, _topo["ei_cpu"])
                and s.tile_idx.shape == _topo["tidx_cpu"].shape and torch.equal(s.tile_idx, _topo["tidx_cpu"]))
        x = s.x.float().to(dev); x[:, 3:6] = x[:, 3:6] / 1e6
        if x.shape[1] == x_mean.shape[-1]:
            x = x[:, :-1]            # strut column is re-derived from the thickness leaf
        nid = s.node_id if "node_id" in s else None
        if nid is None and getattr(model, "node_emb", None) is not None \
                and x.shape[0] == model.node_emb.num_embeddings:
            nid = torch.arange(x.shape[0], dtype=torch.long)
        c = {"same": same, "x": x, "ea": (s.edge_attr.float().to(dev) - e_mean) / e_std,
             "nid": nid.to(dev) if nid is not None else None,
             "t": s.dims[0, 0:K].float().to(dev)[perm]}
        object.__setattr__(s, "_ft_gpu", c)   # instance attr, not a PyG data key
        return c

    def dg_from_samples(samples, train=True):
        """Batched dg_from_sample: one forward + one double-backward for B
        graphs. Every sample must carry node_id if the model uses node_emb
        (replay samples get one at load time), or none must. Training steps
        run under bf16 autocast unless --no_amp, as gnn_training pretrains."""
        B = len(samples)
        cs = [_gpu_inputs(s) for s in samples]
        if all(c["same"] for c in cs) and (all(c["nid"] is None for c in cs)
                                           or all(c["nid"] is not None for c in cs)):
            N = _topo["N"]
            if B not in _topo["eiB"]:
                _topo["eiB"][B] = (torch.cat([_topo["ei"] + i * N for i in range(B)], dim=1),
                                   torch.arange(B, device=dev).repeat_interleave(N),
                                   _topo["tidx"].repeat(B))
            ei, bidx, tidx = _topo["eiB"][B]
            t_leaf = torch.stack([c["t"] for c in cs]).clone().requires_grad_(True)
            ns = t_leaf[bidx, tidx].unsqueeze(1)
            xn = (torch.cat([torch.cat([c["x"] for c in cs]), ns], dim=1) - x_mean) / x_std
            nid = torch.cat([c["nid"] for c in cs]) if cs[0]["nid"] is not None else None
            tNX, tNY, tNZ = _topo["nxyz"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=train and a.amp and dev.type == "cuda"):
                gp, _ = model(xn, ei, torch.cat([c["ea"] for c in cs]), bidx,
                              tile_idx=tidx, tile_NX=tNX, tile_NY=tNY, tile_NZ=tNZ,
                              tile_x=t_leaf.view(-1, 1), num_graphs=B, node_id=nid)
            s_mpa = (gp.float() * nt_std + nt_mean).clamp_min(1e-6).view(B)
            d = torch.autograd.grad(s_mpa.sum(), t_leaf, create_graph=train)[0]
            return [d[i][perm_inv[samples[i].grad_tile_idx.to(dev)]] for i in range(B)], s_mpa
        if not _topo.get("warned"):
            _topo["warned"] = True
            print("[trainer] mixed topologies in a batch: using the CPU-collated path", flush=True)
        # collate only what the forward needs: replay samples carry extra
        # per-node label fields (node_disp, node_stress, ...) that on-policy
        # samples don't, and PyG's collate demands identical key sets
        _keys = ("x", "edge_index", "edge_attr", "tile_idx", "tile_NX", "tile_NY", "tile_NZ", "node_id")
        b = Batch.from_data_list([Data(**{k: s[k] for k in _keys if k in s}) for s in samples])
        x, ei, ea, bidx, tidx, tNX, tNY, tNZ, tx, nid = gnn_input_fn(b)
        t_leaf = torch.stack([s.dims[0, 0:K].float()[perm.cpu()] for s in samples]).to(dev).clone().requires_grad_(True)
        bidx, tidx = bidx.to(dev), tidx.to(dev)
        ns = t_leaf[bidx, tidx].unsqueeze(1)
        x = x.float().to(dev); x[:, 3:6] = x[:, 3:6] / 1e6
        xg = torch.cat([x[:, :-1], ns], dim=1) if x.shape[1] == x_mean.shape[-1] \
             else torch.cat([x, ns], dim=1)
        xn = (xg - x_mean) / x_std
        if nid is None and getattr(model, "node_emb", None) is not None \
                and xg.shape[0] == B * model.node_emb.num_embeddings:
            nid = torch.arange(model.node_emb.num_embeddings, dtype=torch.long).repeat(B)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=train and a.amp and dev.type == "cuda"):
            gp, _ = model(xn, ei.to(dev), (ea.float().to(dev) - e_mean) / e_std, bidx,
                          tile_idx=tidx, tile_NX=tNX, tile_NY=tNY, tile_NZ=tNZ,
                          tile_x=t_leaf.view(-1, 1), num_graphs=B,
                          node_id=nid.to(dev) if nid is not None else None)
        s_mpa = (gp.float() * nt_std + nt_mean).clamp_min(1e-6).view(B)
        # graphs are independent, so d(sum)/dt_leaf gives each row its own gradient
        d = torch.autograd.grad(s_mpa.sum(), t_leaf, create_graph=train)[0]
        return [d[i][perm_inv[samples[i].grad_tile_idx.to(dev)]] for i in range(B)], s_mpa

    def true_stress_mpa(s):
        """Labels and dataset samples store this differently: the adjoint
        builder writes ks_stress_adjoint in MPa while ks_stress stays Pa."""
        if hasattr(s, 'ks_stress_adjoint'):
            return float(torch.as_tensor(s.ks_stress_adjoint).reshape(-1)[0])
        return float(torch.as_tensor(s.ks_stress).reshape(-1)[0]) / 1e6

    # Know-nothing baselines, as in gnn_training: each term is divided by the
    # score of a predictor that knows nothing, so the weights mean equal
    # importance relative to difficulty. KS: always predict the median replay
    # KS. Gradient: predict 0 for every tile, exactly 1/(1+1/sat). Sign: 0.5.
    _ks_rep = np.array([true_stress_mpa(s_) for s_ in replay])
    base_ks = float(np.mean(np.abs(np.median(_ks_rep) - _ks_rep) / np.maximum(_ks_rep, 1e-6)))
    base_grad = 1.0 / (1.0 + 1.0 / a.grad_rel_sat)
    base_sign = 0.5
    print(f"[trainer] loss: {a.graph_loss_weight} ks/{100*base_ks:.1f}% + "
          f"{a.grad_loss_weight} grad(p={a.grad_weight_power}, sat={a.grad_rel_sat})/{100*base_grad:.1f}% + "
          f"{a.grad_sign_loss_weight} sign(delta={a.grad_sign_delta})/{base_sign:.2f}", flush=True)

    def loss_for(pred, true):
        """Gradient terms, as in gnn_training: percent error + sign loss. The
        cosine is no longer a loss (removed 2026-10-01); it is still used
        below as an update GATE (a metric)."""
        return (a.grad_loss_weight * grad_rel_err(pred, true, a.grad_weight_power, a.grad_rel_sat) / base_grad
                + a.grad_sign_loss_weight * grad_sign_err(pred, true, a.grad_sign_delta) / base_sign)

    def stress_loss(pred_mpa, true_mpa):
        """KS term, as in gnn_training: |KS_pred - KS_true| / KS_true, weighted
        and baseline-normalised. Without a stress term the trainer optimises
        dg alone and the VALUE head drifts unwatched -- measured on the first
        online run, bias went -12.83%% -> -14.66%% while dg cosine improved,
        and the resulting design was MORE overstressed (112.82 MPa vs 109.93
        for the unlearned baseline)."""
        t = torch.as_tensor(true_mpa, dtype=torch.float, device=dev).clamp_min(1e-6)
        return a.graph_loss_weight * (pred_mpa - t).abs() / t / base_ks

    @torch.enable_grad()
    def holdout_scores():
        """(median dg cosine, mean signed stress error). BOTH are gated: an
        update that sharpens the gradient while pushing stress further into
        under-prediction makes topopt worse, not better."""
        cs, es = [], []
        for s in holdout:
            # GPU-resident inputs; B=1 so no batch cross-talk, fp32 (train=False)
            (p,), sp = dg_from_samples([s], train=False)
            p = p.detach()
            t = s.grad_dks_dx.float().to(dev)
            cs.append(float(torch.nn.functional.cosine_similarity(
                p.view(1, -1), t.view(1, -1), dim=1)))
            tv = true_stress_mpa(s)
            es.append((float(sp.detach()) - tv) / max(tv, 1e-6))
        return float(np.median(cs)), float(np.mean(es))

    if a.batch_size > 1 and len(holdout) >= 2:
        # batched path must reproduce the per-sample path exactly (up to fp);
        # eval mode, or dropout makes the two calls differ by construction
        model.eval()
        with torch.enable_grad():
            pb, sb = dg_from_samples(holdout[:2], train=False)
            p0, s0 = dg_from_sample(holdout[0], train=False)
            p1, s1 = dg_from_sample(holdout[1], train=False)
        err = max(float((pb[0] - p0).abs().max() / (p0.abs().max() + 1e-9)),
                  float((pb[1] - p1).abs().max() / (p1.abs().max() + 1e-9)),
                  float(abs(sb[0] - s0) / s0), float(abs(sb[1] - s1) / s1))
        if os.environ.get("ONLINE_FT_DEBUG"):
            with torch.enable_grad():
                pq, sq = dg_from_samples(holdout[:1], train=False)
                pr, sr = dg_from_samples([holdout[0], holdout[0]], train=False)
            with torch.enable_grad():
                p2, s2 = dg_from_sample(holdout[2], train=False)
                pu, su = dg_from_samples([holdout[0], holdout[2]], train=False)
                pv, sv = dg_from_samples([holdout[2], holdout[0]], train=False)
            print(f"[trainer] self-test pairs: s2 single {float(s2):.4f}; [0,2] -> {float(su[0]):.4f} {float(su[1]):.4f}; [2,0] -> {float(sv[0]):.4f} {float(sv[1]):.4f}; "
                  f"dg rel: {float((pu[0]-p0).abs().max()/p0.abs().max()):.2e} {float((pu[1]-p2).abs().max()/p2.abs().max()):.2e} {float((pv[1]-p0).abs().max()/p0.abs().max()):.2e}; edges {[int(h.edge_index.shape[1]) for h in holdout[:3]]}", flush=True)
            print(f"[trainer] self-test B=1 via batched: stress {float(sq[0]):.4f} (single {float(s0):.4f}) dg max|diff| {float((pq[0]-p0).abs().max()):.3e}; "
                  f"B=2 same sample twice: {float(sr[0]):.4f} {float(sr[1]):.4f} dg max|diff| {float((pr[0]-p0).abs().max()):.3e} {float((pr[1]-p0).abs().max()):.3e}", flush=True)
            print(f"[trainer] self-test detail: stress single {float(s0):.4f} {float(s1):.4f} batched {float(sb[0]):.4f} {float(sb[1]):.4f}; "
                  f"dg0 max|diff| {float((pb[0]-p0).abs().max()):.4e} of max|p0| {float(p0.abs().max()):.4e}, "
                  f"dg1 max|diff| {float((pb[1]-p1).abs().max()):.4e} of max|p1| {float(p1.abs().max()):.4e}; "
                  f"nodes/graph {[int(h.x.shape[0]) for h in holdout[:2]]} node_id {[hasattr(h,'node_id') for h in holdout[:2]]}", flush=True)
        # Expect ~1e-2, not 1e-6: the conv MLPs use PyG LayerNorm in its
        # default mode="graph", which inside message() (no batch vector)
        # normalises over ALL edges in the batch, so graphs interact
        # slightly (measured: stress 78.388 -> 78.347 when paired, dg rel
        # 7e-3). gnn_training pretrained with batches of graphs, so batched
        # fine-tuning is the regime the weights came from; batch-1 was the
        # outlier. Anything much larger than that would be a real bug.
        print(f"[trainer] batched-vs-single self-test: max rel diff {err:.2e} "
              f"({'OK (graph-mode LayerNorm cross-talk expected ~1e-2)' if err < 5e-2 else 'MISMATCH -- falling back to batch_size 1'})", flush=True)
        if err >= 5e-2:
            a.batch_size = 1
    a.batch_size = _bcast(a.batch_size)
    base_cos, base_bias = holdout_scores() if _rank == 0 else (float("nan"), float("nan"))
    print(f"[trainer] base held-out: cosine {base_cos:+.4f}  stress bias {100*base_bias:+.2f}%", flush=True)
    _state0 = None
    if a.state_path and os.path.exists(a.state_path):
        try:
            _state0 = torch.load(a.state_path, map_location="cpu", weights_only=False)
            opt.load_state_dict(_state0["optimizer"])
            base_cos, base_bias = float(_state0["base_cos"]), float(_state0["base_bias"])
            print(f"[trainer] resume: restored optimizer + gates from {a.state_path} "
                  f"(update {_state0.get('n_upd', 0)}, original base cosine {base_cos:+.4f} "
                  f"bias {100*base_bias:+.2f}%)", flush=True)
        except Exception as e:
            print(f"[trainer] resume: could not load {a.state_path} ({e}); starting fresh", flush=True)
            _state0 = None
    stress_gate = a.max_stress_abs if a.max_stress_abs is not None else abs(base_bias) + a.max_stress_drift
    print(f"[trainer] gate: |stress bias| <= {100*stress_gate:.2f}% "
          f"({'absolute' if a.max_stress_abs is not None else 'base-relative'}); cosine is logged, not gated", flush=True)

    # EMA of the published weights. Every accepted update moved the surrogate's
    # stress at the SAME design by +/-1-3 MPa (re-anchor lines in the lag-4
    # runs: g -0.0095 -> -0.030 across one reload), so the optimiser chased a
    # constraint that jumped more per generation than its step could move it:
    # lambda bouncing 130 <-> 4400 at full steps, saturated at the bisection
    # bounds at 0.005 steps, volume stuck at ~420-455 either way while the
    # ccx-driven run (a constraint that never moves between gens) reached 250.
    # The trained model keeps learning at full speed; only what the optimiser
    # consumes is smoothed. Lags the trained weights by ~1/ema updates.
    ema_state = ({k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
                 if a.ema > 0 else None)
    if _state0 is not None and _state0.get("ema_state") is not None and ema_state is not None:
        ema_state = {k: v.clone() for k, v in _state0["ema_state"].items()}

    def ema_update():
        if ema_state is None or _rank != 0:
            return
        for k, v in model.state_dict().items():
            v = v.detach().cpu()
            if v.is_floating_point():
                ema_state[k].mul_(1.0 - a.ema).add_(v, alpha=a.ema)
            else:
                ema_state[k].copy_(v)

    # Publishing was 2 s of a ~21 s update on newton4: it re-read the base
    # checkpoint from NFS and wrote both files there synchronously. The base
    # is now read once, and the resume state (15 MB, only needed if the job
    # dies) is written by a background thread from a CPU snapshot, so the
    # optimiser only waits for the live file.
    import threading
    _base_ck = torch.load(a.base_ckpt, map_location="cpu", weights_only=False)
    _state_thread = [None]

    def _write_state(st):
        tmp = a.state_path + ".writing.pt"
        torch.save(st, tmp)
        os.replace(tmp, a.state_path)

    def publish():
        if _rank != 0:
            return
        ck = dict(_base_ck)
        ck["model_state"] = ({k: v.clone() for k, v in ema_state.items()} if ema_state is not None
                             else {k: v.detach().cpu() for k, v in model.state_dict().items()})
        tmp = a.live_ckpt + ".writing.pt"
        torch.save(ck, tmp)
        os.replace(tmp, a.live_ckpt)
        if a.state_path:
            if _state_thread[0] is not None:
                _state_thread[0].join()
            # deep CPU copy: the next opt.step() mutates the moments in place
            st = {"optimizer": _cpu_copy(opt.state_dict()), "n_upd": n_upd, "base_cos": base_cos,
                  "base_bias": base_bias, "stress_gate": stress_gate,
                  "ema_state": ({k: v.clone() for k, v in ema_state.items()} if ema_state is not None else None)}
            _state_thread[0] = threading.Thread(target=_write_state, args=(st,))
            _state_thread[0].start()

    def _cpu_copy(o):
        if torch.is_tensor(o):
            return o.detach().to("cpu", copy=True)
        if isinstance(o, dict):
            return {k: _cpu_copy(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(_cpu_copy(v) for v in o)
        return o

    traj_buf, val_set = [], []          # FIFO of past on-policy designs / on-trajectory validation

    def val_scores():
        """(median dg cosine, mean signed stress error) on the ON-trajectory
        validation designs -- same distribution as what the optimiser is
        actually walking through, never trained on."""
        if not val_set:
            return float("nan"), float("nan")
        cs, es = [], []
        for s_ in val_set:
            (p_,), sp = dg_from_samples([s_], train=False)
            p_ = p_.detach(); t_ = s_.grad_dks_dx.float().to(dev)
            cs.append(float(torch.nn.functional.cosine_similarity(p_.view(1, -1), t_.view(1, -1), dim=1)))
            tv = true_stress_mpa(s_)
            es.append((float(sp.detach()) - tv) / max(tv, 1e-6))
        return float(np.median(cs)), float(np.mean(es))

    seen, pend, n_upd, idle = set(), [], 0, None
    if _state0 is not None:
        n_upd = int(_state0.get("n_upd", 0))
    publish()
    if a.skip_existing:
        # resume: labels up to .consumed_upto are already in the live weights;
        # anything newer (labelled after the previous trainer died) still has
        # to be trained on, or the optimiser's lagged wait deadlocks on it.
        mk = os.path.join(a.labels, '.consumed_upto')
        upto = int(open(mk).read().strip() or -1) if os.path.exists(mk) else -1
        seen = {f for f in glob.glob(os.path.join(a.labels, "gen_*.npz"))
                if int(os.path.basename(f)[4:10]) <= upto}
        print(f"[trainer] resume: skipping {len(seen)} labels up to gen {upto}", flush=True)
    _trainable = [p for p in model.parameters() if p.requires_grad]
    backlog = []                        # rank 0: labels seen but not yet trained on

    def _next_bundle():
        """rank 0: block until a full bundle of new labels is available and
        return it (oldest first), or None after --idle_exit with nothing."""
        nonlocal idle
        while True:
            new = [f for f in sorted(glob.glob(os.path.join(a.labels, "gen_*.npz"))) if f not in seen]
            # A capped bundle leaves the rest of a burst in the backlog; it must
            # be drained even when no NEW label arrives, or the lagged optimiser
            # (blocked on those very labels) never spools another design and the
            # pipeline deadlocks. Seen live with 1 label/gen: gens 5-7 arrived
            # together, gen 7 stayed pending, topopt waited on it forever.
            if not new and len(backlog) < a.update_every:
                if a.idle_exit:
                    idle = idle or time.time()
                    if time.time() - idle > a.idle_exit:
                        print(f"[trainer] idle, exiting after {n_upd} updates", flush=True)
                        return None
                time.sleep(a.poll); continue
            idle = None
            for f in new:
                seen.add(f); backlog.append(f)
            if len(backlog) < a.update_every:
                continue
            # Cap the bundle at one update's worth. With ~38 labellers delivering
            # in bursts, taking everything pending produced updates of 55-78
            # designs; each on-policy design needs its mesh rebuilt (~3s) before
            # training, so one update could stall for ~4 minutes and the lagged
            # optimiser blocked on it (waits of 137s / 211s at lag 8). Taking
            # update_every at a time keeps every update ~10 meshes + steps and
            # drains a backlog over successive updates instead of one giant one.
            bundle = backlog[:a.update_every]
            del backlog[:a.update_every]
            return bundle

    while True:
        # rank 0 decides which labels make up the update; the other ranks follow
        pend = _bcast(_next_bundle() if _rank == 0 else None)
        if pend is None:
            return
        if _world > 1:
            # the ranks must hold identical weights (same steps, summed grads,
            # same gate decision); a float64 checksum per rank catches a drift
            ck = torch.tensor([sum(float(p.double().sum()) for p in model.parameters())],
                              dtype=torch.float64, device=dev)
            cks = [torch.zeros_like(ck) for _ in range(_world)]
            dist.all_gather(cks, ck)
            if len({float(c) for c in cks}) != 1:
                raise RuntimeError(f"[trainer] rank weights diverged: {[float(c) for c in cks]}")
        later = []
        used = list(pend)
        _t_start = time.time()
        _tl = []                        # per-label (t_spool, t_claim, t_solved, t_done)

        # build on-policy samples: rebuild the graph for each labelled design.
        # The labeller's ccx mesh is a different pipeline from the GNN graph, so
        # this rebuild is unavoidable -- but the ten builds are independent, and
        # done serially (~3s each) they were ~30s of the ~40s update, i.e. the
        # whole pipeline's critical path once the lagged wait is correct.
        _jobs = []; _ready = []
        for f in pend:
            z = np.load(f, allow_pickle=True)
            if "t_done" in z.files:
                _tl.append(tuple(float(z[k]) for k in ("t_spool", "t_claim", "t_solved", "t_done")))
            names = [str(n) for n in z["free_names"]]
            dims = {n: float(v) for n, v in zip(names, z["x"])}
            gkeys = [k for k in z.files if k.startswith("graph__")]
            if gkeys:
                # labeller shipped the graph: no re-mesh needed. It omits the
                # frozen-topology edge_index / tile_idx; take them from graph.pt.
                gd = {k[len("graph__"):]: z[k] for k in gkeys}
                if ("edge_index" not in gd or "tile_idx" not in gd) and gev.graph_path is not None:
                    from evaluators.gnn_evaluator import _get_fixed_graph
                    _ei, _ti = _get_fixed_graph(gev.graph_path)
                    gd.setdefault("edge_index", _ei.numpy()); gd.setdefault("tile_idx", _ti.numpy())
                _ready.append((f, z, names, gd))
            else:
                _jobs.append((f, z, names, dims))
        _t_mesh = time.time()
        with ProcessPoolExecutor(max_workers=max(1, min(len(_jobs), a.mesh_workers))) as ex:
            futs = {ex.submit(_run_sample_worker, (gev.model_path, dims, gev.fixed_dims,
                        gev.young, gev.poisson, 0, False, False, gev.anchor_condition,
                        gev.force_pattern, gev.U, gev.V, gev.W, (NX, NY, NZ), gev.graph_path)): (f, z, names)
                    for (f, z, names, dims) in _jobs}
            onp = []
            results = [(f, z, names, gd) for (f, z, names, gd) in _ready]
            for fut, (f, z, names) in futs.items():
                try:
                    gd, vol, _t, _m = fut.result()
                    results.append((f, z, names, gd))
                except Exception as e:
                    print(f"[trainer] mesh failed for {os.path.basename(f)}: {e}", flush=True); continue
            for (f, z, names, gd) in results:
                d = Data(**{k: torch.from_numpy(np.asarray(v)) for k, v in gd.items()})
                # _run_sample_worker pre-appends the per-node strut column;
                # replay samples don't have it. dg_from_sample(s) re-derive it
                # from the thickness leaf either way, so strip it here and
                # every sample takes the same append path -- a mixed batch
                # otherwise fails to collate (10 vs 11 columns).
                if d.x.shape[1] == x_mean.shape[-1]:
                    d.x = d.x[:, :-1]
                if gev.node_id is not None and d.x.shape[0] == gev.node_id.shape[0]:
                    d.node_id = gev.node_id
                d.dims = torch.tensor([[float(z["x"][i]) for i in range(len(names))]], dtype=torch.float)
                d.grad_tile_idx = torch.arange(K)
                # label is dG_dx (divided by yield); training works in MPa/unit
                d.grad_dks_dx = torch.tensor(np.asarray(z["dg_ccx"], float) * float(z["yield_strength"]),
                                             dtype=torch.float)
                d.ks_stress_adjoint = torch.tensor([float(z["ccx_stress"])], dtype=torch.float)
                _gen = int(z["gen"]) if "gen" in z.files else -1
                _is_base = "_p" not in os.path.basename(f)
                if a.val_every and _gen >= 0 and _gen % a.val_every == 0 and _is_base:
                    # ON-trajectory validation: never trained on
                    val_set.append(d)
                    if len(val_set) > 24:
                        val_set.pop(0)
                    continue
                onp.append(d)
                if a.traj_buffer and _is_base:
                    traj_buf.append(d)
                    if len(traj_buf) > a.traj_buffer:
                        traj_buf.pop(0)
        pend = later
        if not onp:
            continue

        _t_mesh = time.time() - _t_mesh
        _t_train = time.time()
        snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.train()
        rng = np.random.default_rng(n_upd)
        Bsz = max(1, a.batch_size)
        for _ in range(max(1, a.steps // Bsz)):
            batch = []
            opt.zero_grad(set_to_none=True)
            # On-policy draws come from the trajectory FIFO when it is enabled:
            # uniform over the last --traj_buffer labelled iterates instead of
            # only the 5 near-identical designs of this bundle. The newest
            # bundle is always in the pool, so nothing is lost -- it is just
            # no longer the only thing being fitted.
            _pool = (traj_buf if (a.traj_buffer and traj_buf) else onp)
            for _b in range(Bsz):
                use_replay = rng.random() < a.replay_frac
                batch.append(replay[int(rng.integers(len(replay)))] if use_replay
                             else _pool[int(rng.integers(len(_pool)))])
            # this rank's share; / Bsz (the WHOLE batch) so the summed
            # gradient is exactly the single-GPU mean over the batch
            mine = batch[_rank::_world]
            L = None
            if mine:
                preds, pred_s = dg_from_samples(mine, train=True)
                L = sum(loss_for(pr, s.grad_dks_dx.float().to(dev))
                        + stress_loss(pred_s[i], true_stress_mpa(s))
                        for i, (pr, s) in enumerate(zip(preds, mine))) / Bsz
            if _anchor and _rank == 0:
                _pen = sum(((p - _anchor[n]) ** 2).sum()
                           for n, p in model.named_parameters() if n in _anchor)
                L = (L if L is not None else 0.0) + a.anchor_l2 * _pen
            if L is not None:
                L.backward()
            _allreduce_grads(_trainable)
            torch.nn.utils.clip_grad_norm_(_trainable, 1.0)
            opt.step()
        model.eval()
        _t_train = time.time() - _t_train
        if _rank != 0:
            # follower: take rank 0's gate decision and nothing else
            if _bcast(None):
                model.load_state_dict(snapshot)
            else:
                n_upd += 1
            continue
        _t_score = time.time()
        cos_now, bias_now = holdout_scores()
        vcos, vbias = val_scores()
        if a.val_gate and len(val_set) >= 8 and vcos == vcos:
            # Switch the gate to the on-trajectory set -- and RE-BASELINE the
            # stress bias with it (base-relative gate only).
            if not globals().get("_VAL_BASELINED"):
                globals()["_VAL_BASELINED"] = True
                base_cos, base_bias = vcos, vbias
                stress_gate = a.max_stress_abs if a.max_stress_abs is not None else abs(base_bias) + a.max_stress_drift
                print(f"[trainer] gate switched to the on-trajectory set: base bias {100*base_bias:+.2f}%, "
                      f"|stress bias| <= {100*stress_gate:.2f}%", flush=True)
            cos_now, bias_now = vcos, vbias
        _t_score = time.time() - _t_score
        bad_stress = abs(bias_now) > stress_gate
        _bcast(bool(bad_stress))
        def _advance_marker():
            # Record the highest generation whose labels have been CONSUMED --
            # published or discarded. The optimiser's lagged wait keys on this;
            # a discard used to leave it behind, so the optimiser waited 1800s
            # per generation for a label bundle that had already been eaten,
            # and the trainer, seeing no new labels, idle-exited (run E, gen 70).
            try:
                gens = [int(np.load(f, allow_pickle=True)['gen']) for f in used]
                hi = max(gens) if gens else -1
                mk = os.path.join(a.labels, '.consumed_upto')
                prev = int(open(mk).read().strip() or -1) if os.path.exists(mk) else -1
                if hi > prev:
                    tmp = mk + '.tmp'
                    open(tmp, 'w').write(str(hi)); os.replace(tmp, mk)
            except Exception as e:
                print(f'[trainer] consumed marker failed: {e}', flush=True)

        if bad_stress:
            model.load_state_dict(snapshot)          # roll back, keep serving old weights
            print(f"[trainer] update {n_upd}: cosine {cos_now:+.4f} bias {100*bias_now:+.2f}% "
                  f"-- DISCARDED (stress)", flush=True)
            _advance_marker()
        else:
            n_upd += 1
            ema_note = ""
            if ema_state is not None:
                ema_update()
                # score what the optimiser will actually see
                trained = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict({k: v.to(dev) for k, v in ema_state.items()})
                cos_ema, bias_ema = holdout_scores()
                model.load_state_dict(trained)
                ema_note = f" | EMA({a.ema}) cosine {cos_ema:+.4f} bias {100*bias_ema:+.2f}%"
            _t_pub = time.time()
            publish()
            _advance_marker()
            _t_pub = time.time() - _t_pub
            _vnote = (f" | on-traj val cosine {vcos:+.4f} bias {100*vbias:+.2f}% (n={len(val_set)})"
                      if vcos == vcos else "")
            _bnote = f" | pool {len(_pool)}" if a.traj_buffer else ""
            if _tl:
                # where a bundle's time went, wall clock: spooled -> claimed by a
                # labeller (queue) -> solved -> graph built -> trainer picked it up
                # (after the LAST label of the bundle) -> published
                T = np.array(_tl)
                _bnote += (f" | pipe: spool->all labelled {T[:,3].max()-T[:,0].min():.1f}s "
                           f"(queue max {np.nanmax(T[:,1]-T[:,0]):.1f}s, solve max {np.max(T[:,2]-T[:,1]):.1f}s, "
                           f"graph max {np.max(T[:,3]-T[:,2]):.1f}s), labelled->trainer {_t_start-T[:,3].max():.1f}s, "
                           f"trainer {time.time()-_t_start:.1f}s")
            print(f"[trainer] update {n_upd}: {len(onp)} designs, cosine {cos_now:+.4f} "
                  f"(base {base_cos:+.4f}), stress bias {100*bias_now:+.2f}% "
                  f"(base {100*base_bias:+.2f}%){ema_note}{_vnote}{_bnote} -- published "
                  f"[mesh {_t_mesh:.1f}s train {_t_train:.1f}s score {_t_score:.1f}s publish {_t_pub:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
