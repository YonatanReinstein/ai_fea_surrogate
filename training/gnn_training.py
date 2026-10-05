import torch
import json
import os
import re
import glob
import time
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from utils.gnn_surrogate import GNN, HierarchicalGNN
from utils.gnn_hierarchical_mp import HierarchicalGNNMP
from utils.gnn_hierarchical_cnn import HierarchicalGNNCNN
from training.dataset_io import load_dataset, load_shard, shard_paths
from core.tile_sensitivity import dims_key_order_permutation


def weighted_node_grad_loss(pred, true, batch_idx, scale, power=1.0, cos_weight=0.0):
    """Match the adjoint NODAL field dKS/dX, weighted toward the nodes the
    geometric composition actually sums.

    An UNWEIGHTED mean over this target does not work, and job 1374298 is the
    proof: 1.93M node targets whose mean |slog| is 0.049 against a std of
    0.291 (97.2% of the field's mass sits in one decile of nodes), so plain
    MSE is minimised by predicting ~0 everywhere. That run drove its loss from
    0.0959 to 0.0248 -- below the 0.0848 zero-prediction floor -- while the
    predicted magnitudes collapsed to 87x too small and the COMPOSED per-tile
    dg went anti-correlated (cosine -0.360, 82.8% wrong-way mass).

    Two corrections, mirroring what already works elsewhere in this file:
      power     -- per-graph magnitude weighting, exactly weighted_node_rel_err's
                   scheme but keyed on |dKS/dX| instead of von Mises stress.
                   The composition sums <g_n, dX_n/dt>, so mass-carrying
                   nodes are the ones worth fitting.
      cos_weight-- per-sample direction term on the raw node field. Scale-
                   invariant, so it cannot be satisfied by shrinking toward
                   zero -- the exact failure above. The best dg model this
                   project has produced (run 2 early, median cosine +0.683 vs
                   a +0.558 constant-direction control) came from adding the
                   tile-level analogue of this term.
    """
    def _sl(v):
        return torch.sign(v) * torch.log1p(v.abs() / scale)
    mag = true.norm(dim=1)
    w_pow = mag.pow(power)
    totals = scatter(w_pow, batch_idx, dim=0, reduce="sum")
    w = w_pow / (totals[batch_idx] + 1e-12)
    sq = ((_sl(pred) - _sl(true)) ** 2).sum(dim=1)
    loss = scatter(w * sq, batch_idx, dim=0, reduce="sum").mean()

    cos = None
    if cos_weight > 0.0:
        num = scatter((pred * true).sum(dim=1), batch_idx, dim=0, reduce="sum")
        pn  = scatter((pred * pred).sum(dim=1), batch_idx, dim=0, reduce="sum").clamp_min(1e-24).sqrt()
        tn  = scatter((true * true).sum(dim=1), batch_idx, dim=0, reduce="sum").clamp_min(1e-24).sqrt()
        cos = (num / (pn * tn)).clamp(-1.0, 1.0)
        loss = loss + cos_weight * (1.0 - cos).mean()
    return loss, cos


# ======================================================
# Helper functions
# ======================================================
def gnn_input_fn(data):
    tile_idx  = getattr(data, 'tile_idx',  None)
    tile_NX   = getattr(data, 'tile_NX',   None)
    tile_NY   = getattr(data, 'tile_NY',   None)
    tile_NZ   = getattr(data, 'tile_NZ',   None)
    # Fixed-topology: NX/NY/NZ are identical for every sample of a geometry,
    # so pull them out to plain Python ints here (once, on CPU) instead of
    # passing GPU tensors into model.forward() -- int(tensor[0]) inside the
    # (compiled) forward is a torch.compile graph break / recompile source.
    tile_NX = int(tile_NX[0]) if tile_NX is not None else None
    tile_NY = int(tile_NY[0]) if tile_NY is not None else None
    tile_NZ = int(tile_NZ[0]) if tile_NZ is not None else None
    tile_x    = getattr(data, 'tile_x',    None)
    node_id   = getattr(data, 'node_id',   None)  # stable per-node index
    return (data.x, data.edge_index, data.edge_attr, data.batch,
            tile_idx, tile_NX, tile_NY, tile_NZ, tile_x, node_id)


def gnn_target_fn(data):
    # returns [batch_size, 1] tensor (graph-level). KS-aggregated stress
    # (core/mesh.py:get_aggregated_stress), not the hard max -- see
    # utils/gnn_surrogate.py:ks_pool for why the model's graph_pred matches
    # this instead of a plain global_max_pool.
    return torch.cat([data.ks_stress], dim=-1)


# Graph-level ks_stress is heavily right-skewed (raw z-score: skew~2.6,
# kurtosis~6.4 on a 10k-sample hollow_cube dataset) -- a few rare thin/weak
# designs spike into the GPa range and dominate a linear std, squeezing the
# other 99% of samples' normalized range. log-then-z-score fixes that
# (skew~1.1, kurtosis~0.3, zero samples beyond +-3 vs ~3% for raw) without
# discarding those samples. Deliberately graph-level only: node-level stress
# (gnn_node_target_fn below) feeds ks_pool's KS-aggregation math, which is
# calibrated in raw physical MPa (rho_mpa matches core/mesh.py's units) and
# must stay linear for that physics to be correct -- see ks_pool's docstring.
LOG_STRESS_EPS = 1e-6


def gnn_log_target_fn(data):
    return torch.log(gnn_target_fn(data).clamp_min(LOG_STRESS_EPS))


def _shard_count(path):
    """Samples in shard_{start:06d}_{end:06d}.pt, read from the name alone."""
    a_, b_ = os.path.basename(path)[len("shard_"):-len(".pt")].split("_")
    return int(b_) - int(a_) + 1


def gnn_node_target_fn(data):
    # returns [N_total, 1] tensor (node-level von Mises stress)
    return data.node_stress.view(-1, 1)


def ks_rel_err(graph_pred_raw_mpa, ks_true_mpa):
    """KS stress loss: mean relative error |KS_pred - KS_true| / KS_true over the
    batch, so the logged value IS the mean KS stress error (0.043 = 4.3%).
    Replaced a normalized log-space MSE on 2026-10-01: same percent-error
    meaning, but its value was not directly readable."""
    p = graph_pred_raw_mpa.float().view(-1)
    t = ks_true_mpa.float().view(-1)
    return ((p - t).abs() / t.clamp_min(LOG_STRESS_EPS)).mean()


def weighted_node_rel_err(node_pred_raw_mpa, node_targ_raw, ks_true_mpa, batch_idx, stress_weight_power=1.0):
    """Node stress loss: per design, the stress-weighted mean node error as a
    fraction of that design's true KS (peak) stress, then the mean over designs:

        w_i  = sigma_i^alpha / sum_j sigma_j^alpha        (per design, sums to 1)
        loss = mean_designs  sum_i w_i * |sigma_pred,i - sigma_true,i| / KS_true

    0.02 = nodes are off by 2% of the peak stress on average. Dividing by the
    design's KS rather than each node's own stress avoids the blow-up on lightly
    loaded nodes (down to ~1 MPa). alpha = --stress_weight_power: 0 = every node
    equal, 1 = weight proportional to true stress (on dataset_screened30k the
    top 10% of nodes then hold 24% of the weight), 2 = 45%. Replaced a weighted
    MSE in normalized units on 2026-10-01; same weights, readable value."""
    raw = node_targ_raw.float().squeeze(-1).clamp_min(0.0)
    raw_pow = raw.pow(stress_weight_power)
    graph_totals = scatter(raw_pow, batch_idx, dim=0, reduce="sum")
    weights = raw_pow / (graph_totals[batch_idx] + 1e-8)
    ks = ks_true_mpa.float().view(-1).clamp_min(LOG_STRESS_EPS)
    err = (node_pred_raw_mpa.float().squeeze(-1) - raw).abs() / ks[batch_idx]
    per_graph = scatter(weights * err, batch_idx, dim=0, reduce="sum")
    return per_graph.mean()


# ======================================================
# Main Training Function
# ======================================================
def train_gnn_model(
    geometry: str,
    num_samples: int,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 4,
    hidden_dim: int = 128,
    conv_layers: int = 6,
    node_loss_weight: float = 1.0,
    graph_loss_weight: float = 1.0,
    stress_weight_power: float = 1.0,
    weight_decay: float = 1e-4,
    dataset_path: str = None,
    use_checkpoint: bool = False,
    checkpoint_fine_only: bool = False,
    use_node_emb: bool = True,
    coarse_arch: str = "cnn",
    transformer_heads: int = 4,
    transformer_ff_mult: int = 2,
    transformer_dropout: float = 0.1,
    mp_dropout: float = 0.0,
    coarse_dim: int = None,
    cnn_depth: int = 3,
    checkpoint_dir: str = None,
    run_dir: str = "training/runs/train",
    stress_ks_rho_mpa: float = 0.15,
    use_compile: bool = True,
    resume_lr_scale: float = None,
    lr_decay_per_epoch: float = 0.99,
    lr_plateau_patience: int = 5,
    lr_plateau_factor: float = 0.5,
    seed_offset: int = 42,
    grad_dataset_path: str = None,
    grad_from_train: bool = False,
    node_grad_weight: float = 0.0,
    node_grad_power: float = 1.0,
    node_grad_cos_weight: float = 0.0,
    grad_loss_weight: float = 0.1,
    grad_sign_loss_weight: float = 0.0,
    grad_err_mode: str = "slog",
    grad_sign_delta: float = 0.05,
    grad_weight_power: float = 0.15,
    grad_rel_sat: float = 2.0,
    grad_sign_weight: float = 0.0,
    grad_mass_weight: float = 0.0,
    grad_batch_size: int = None,
):
    torch.manual_seed(42)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))

    # ----------------------------------------------------
    # Paths
    # ----------------------------------------------------
    dataset_path = dataset_path or f"data/{geometry}/dataset/dataset.pt"

    save_dir       = checkpoint_dir or f"data/{geometry}/checkpoints/"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    # ----------------------------------------------------
    # Device / DDP
    # ----------------------------------------------------
    # torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; plain `python -m` leaves them
    # unset, so ddp=False and everything below is identical to single-GPU.
    # Checked via LOCAL_RANK (not "WORLD_SIZE > 1"): a `torchrun
    # --nproc_per_node=1` launch also sets WORLD_SIZE=1, and that single-process
    # case still needs the ddp branch's shard-based dataset loader, which reads
    # only the shard files --num_samples needs -- the non-ddp branch's
    # load_dataset() loads the ENTIRE dataset unconditionally before subsetting,
    # which OOM'd a real run against the ~295GB 50k-sample hollow_cube dataset
    # even with --num_samples 4000 (see project_gradient_matching_loss.md).
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    if not is_main:
        # Silence non-main ranks -- both processes otherwise interleave
        # identical dataset/model-setup logs into one unreadable stream.
        import builtins
        builtins.print = lambda *a, **kw: None

    use_amp = torch.cuda.is_available()
    print(f"Using device: {device}")
    print(f"[amp] bf16 autocast enabled={use_amp}")
    print(f"[gpu] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    print(f"[ddp] enabled={ddp} world_size={world_size}")
    if torch.cuda.is_available():
        print(f"[gpu] device_name={torch.cuda.get_device_name(device)}")
        print(f"[gpu] capability={torch.cuda.get_device_capability(device)}")
        print(f"[gpu] torch.version.cuda={torch.version.cuda}")
        print(f"[gpu] arch_list={torch.cuda.get_arch_list()}")

    # Where the per-tile strut thicknesses start in the dims vector depends on
    # how the geometry encodes its grid topology:
    #   - grid in fixed_dims (hollow_cube, bistable): ALL dims are struts -> offset 0
    #   - grid in dims[d1,d2,d3] (tile, variable grid): struts follow -> offset 3
    import importlib
    _bc = importlib.import_module(f"data.{geometry}.boundary_conditions")
    _fixed_dims = getattr(_bc, "fixed_dims", lambda: {})()
    strut_offset = 0 if _fixed_dims else 3
    print(f"[strut] geometry={geometry} fixed_grid={bool(_fixed_dims)} strut_offset={strut_offset}")

    graph_path = os.path.join(os.path.dirname(dataset_path), "graph.pt")
    _graph_node_id, _num_pos_nodes_from_graph = None, 0
    _graph_edge_index, _graph_tile_idx = None, None
    if os.path.exists(graph_path):
        _g = torch.load(graph_path, weights_only=False)
        _graph_edge_index = _g["edge_index"].long()
        _graph_tile_idx = _g["tile_idx"].view(-1).long()
        _graph_node_id = torch.arange(_g["num_nodes"], dtype=torch.long)
        _num_pos_nodes_from_graph = _g["num_nodes"]

    # dims-key order (d1..dK) is NOT tile_idx's own (IRIT-native) order
    # whenever NX != NY -- see core/tile_sensitivity.py's
    # dims_key_order_permutation. Precomputed once here (NX/NY/NZ are fixed
    # for the whole run) and reused by _preprocess below via a single
    # vectorized gather, instead of recomputing per sample.
    _strut_perm = None
    _strut_perm_inv = None
    _get_tile_grid = getattr(_bc, "tile_grid", None)
    if _get_tile_grid is not None:
        _NX, _NY, _NZ = _get_tile_grid(None)
        _strut_perm = torch.tensor(dims_key_order_permutation(_NX, _NY, _NZ), dtype=torch.long)
        _strut_perm_inv = torch.argsort(_strut_perm).to(device)

    def _preprocess(sample):
        # Fixed-mesh datasets can ship without per-sample edge_index/tile_idx
        # (dataset_builder's workers drop them and only the main process
        # re-attaches; data/hollow_cube_11x11/dataset_screened30k was saved
        # without them). Both are identical for every sample, so take them
        # from the graph.pt next to the dataset.
        if getattr(sample, "edge_index", None) is None or getattr(sample, "tile_idx", None) is None:
            assert _graph_edge_index is not None and sample.x.shape[0] == _num_pos_nodes_from_graph, (
                f"sample has no edge_index/tile_idx and no matching graph.pt at {graph_path}")
            assert sample.edge_attr.shape[0] == _graph_edge_index.shape[1], (
                f"edge_attr has {sample.edge_attr.shape[0]} edges but graph.pt has {_graph_edge_index.shape[1]}")
            sample.edge_index = _graph_edge_index
            sample.tile_idx = _graph_tile_idx
        # convert force to MN / stresses to MPa
        sample.x[:, 3:6] = sample.x[:, 3:6] / 1e+6
        sample.max_stress = sample.max_stress / 1e+6
        sample.ks_stress = sample.ks_stress / 1e+6
        sample.node_stress = sample.node_stress / 1e+6
        # rho consistency check. The stored ks_stress target was built at the
        # dataset builder's rho; the model pools its node map with THIS run's
        # --stress_ks_rho_mpa, and the two must agree or the graph loss fits a
        # rho-A pool to a rho-B target (found 2026-09-16: dataset at 0.25,
        # every training run pooled at the 0.15 default -- the model flattened
        # its stress peak to reconcile them, and every optimizer run inherited
        # a wrong constraint). Checked on the first few samples only.
        if not getattr(_preprocess, "_rho_checked", 0) >= 3:
            _preprocess._rho_checked = getattr(_preprocess, "_rho_checked", 0) + 1
            _ns = sample.node_stress.float().view(-1); _m = _ns.max()
            _ks = float(_m + torch.log(torch.exp(stress_ks_rho_mpa * (_ns - _m)).sum()) / stress_ks_rho_mpa)
            _tgt = float(sample.ks_stress.float().view(-1)[0])
            _dev = abs(_ks - _tgt) / max(abs(_tgt), 1e-6)
            tag = "OK" if _dev < 0.02 else "MISMATCH -- dataset was built at a different rho"
            print(f"[rho-check] rho={stress_ks_rho_mpa}: KS(node_stress)={_ks:.3f} vs stored ks_stress={_tgt:.3f} "
                  f"({100*_dev:.1f}% off) {tag}", flush=True)
        if hasattr(sample, "grad_dks_dx"):
            # The two --grad_mode builders store this label in DIFFERENT
            # units, so key off which one wrote the sample:
            #   fd      -- dataset_builder._ccx_run_grad_sample finite-
            #              differences raw ccx ks_stress, giving Pa/unit, so
            #              it needs the /1e6 like every other stress field.
            #   adjoint -- dataset_builder._adjoint_run_grad_sample ALREADY
            #              divides by 1e6 when storing the label, and stores
            #              ks_stress_adjoint in MPa while ks_stress stays Pa
            #              (that 1e6 is exactly the agreement-guard ratio).
            # Applying the conversion twice put the adjoint labels 1e6 too
            # small -- measured mean|dks/dx| 109.75 -> 1.098e-04 MPa/unit --
            # which parks the signed-log grad loss in log1p's linear regime
            # and trains the surrogate toward a ZERO gradient field, i.e. the
            # exact defect this dataset was built to fix. ks_stress_adjoint
            # is written only by the adjoint path, so it is the discriminator.
            if not hasattr(sample, "ks_stress_adjoint"):
                sample.grad_dks_dx = sample.grad_dks_dx / 1e+6
        # inject per-node strut dimension: dims[strut_offset + tile_idx],
        # after reordering from dims-key order to tile_idx-native order
        # (see _strut_perm above -- indexing dims-key order directly by
        # tile_idx silently fed most nodes another tile's thickness).
        K = int(sample.tile_NX) * int(sample.tile_NY) * int(sample.tile_NZ)
        tile_struts_dims_key_order = sample.dims[0, strut_offset:strut_offset+K].float()
        tile_struts = (tile_struts_dims_key_order[_strut_perm] if _strut_perm is not None
                       else tile_struts_dims_key_order)
        strut_param = tile_struts[sample.tile_idx].unsqueeze(1)
        sample.x = torch.cat([sample.x, strut_param], dim=1)
        sample.tile_x = tile_struts.view(-1, 1)
        if _graph_node_id is not None and sample.x.shape[0] == _num_pos_nodes_from_graph:
            sample.node_id = _graph_node_id
        return sample

    if ddp:
        # ----------------------------------------------------
        # DDP-safe loading: each rank reads only its OWN shard files off
        # disk, never the full dataset. The naive approach (every rank calls
        # load_dataset() then subsets in-memory) was measured at ~575GB RSS
        # per process for the full 50k-sample hollow_cube set -- 3 ranks
        # would need ~1.7TB, more than any node on this cluster has. Val
        # stays on rank 0 only (matches the val_loader design above, which
        # already only runs validation on rank 0).
        # ----------------------------------------------------
        dataset_dir = os.path.dirname(dataset_path)
        all_shards = shard_paths(dataset_dir)
        assert all_shards, f"no shards found under {dataset_dir}/shards -- DDP path requires a sharded dataset"

        if num_samples is not None:
            # Shards are named shard_{start:06d}_{end:06d}.pt, so the sample
            # count per shard is readable from the filename without opening
            # it. Trim the shard pool down to ~num_samples worth *before*
            # the train/val split below, so --num_samples is honored here
            # the same way the single-process path honors it (line ~293) --
            # previously this branch always used every shard regardless of
            # --num_samples, silently training on the full dataset.
            # Sizes come from EVERY shard's name, not the first one's: a
            # filtered dataset (training/filter_dataset.py) has variable-size
            # shards, and the first one need not be typical.
            _cum = np.cumsum([_shard_count(p) for p in all_shards])
            n_shards_needed = min(len(all_shards), int(np.searchsorted(_cum, num_samples)) + 1)
            all_shards = all_shards[:n_shards_needed]

        n_val_shards = max(1, int(len(all_shards) * 0.2))
        val_shards = all_shards[-n_val_shards:]
        train_shards_all = all_shards[:-n_val_shards]
        my_train_shards = train_shards_all[rank::world_size]

        train_set = []
        for p in my_train_shards:
            for sample in load_shard(p):
                _preprocess(sample)  # converts max_stress to MPa -- must run before the threshold check
                if sample.max_stress < 3000:
                    train_set.append(sample)

        if is_main:
            val_set = []
            for p in val_shards:
                for sample in load_shard(p):
                    _preprocess(sample)
                    if sample.max_stress < 3000:
                        val_set.append(sample)
        else:
            val_set = []

        num_pos_nodes = _num_pos_nodes_from_graph
        print(f"[ddp-shards] rank={rank} train_shards={len(my_train_shards)}/{len(train_shards_all)} "
              f"local_train={len(train_set)} val_shards={len(val_shards) if is_main else 0}")

        # Approx sample count across all ranks, for the printed summary only
        # (exact global count would need an all_reduce; not worth it here).
        num_samples = len(train_set) * world_size + sum(_shard_count(p) for p in val_shards)
        print(f"Total samples used (approx): {num_samples}")
        print(f"Train (this rank): {len(train_set)}, Val (rank 0 only): {len(val_set) if is_main else 'n/a'}")

        # Normalization stats: computed from rank 0's local train shard slice
        # (a large, effectively-random ~1/world_size subsample -- shard order
        # has no known correlation with sample characteristics) and broadcast
        # to every rank, since all ranks must normalize identically.
        if is_main:
            import time as _time
            _t_stats0 = _time.time()
            stat_source = val_set if val_set else train_set
            # Per-node/per-edge cats below are the memory-heaviest step in the
            # whole pipeline -- concatenating ALL of train_set's x/edge_attr
            # (48249 nodes / 865720 edges per sample) peaks at ~111GB temp
            # RSS for a 19857-sample rank-local slice, on top of the ~287GB
            # already held just loading train_set+val_set. That's what
            # actually stalled job 1323092 for 4+ hours (confirmed via: zero
            # new torch.compile cache files since job start, ruling out a
            # compile hang; both DDP ranks in R state, consistent with rank0
            # thrashing under memory pressure while rank1 NCCL-spins on the
            # broadcast below waiting for it) -- this same code had only ever
            # run before at smaller per-rank shard counts (3-way DDP splits),
            # never at this 2-way split's ~50% larger per-rank slice. A
            # bounded random subsample gives a statistically indistinguishable
            # mean/std at a fraction of the memory.
            _stats_cap = 2000
            _stats_train = (
                [train_set[i] for i in torch.randperm(len(train_set))[:_stats_cap].tolist()]
                if len(train_set) > _stats_cap else train_set
            )
            print(f"[stats] computing normalization stats from {len(_stats_train)}/{len(train_set)} "
                  f"train samples (capped at {_stats_cap})", flush=True)
            # log-space now (see gnn_log_target_fn) -- graph_target_std_full
            # feeds the "RMSE" reported below, which is therefore RMSE in
            # log(MPa) units, not raw MPa.
            graph_target_std_full = (torch.cat([gnn_log_target_fn(d) for d in stat_source], dim=0).float().std() + 1e-8)
            all_targets = torch.cat([gnn_log_target_fn(d) for d in _stats_train], dim=0).float()
            all_node_targets = torch.cat([gnn_node_target_fn(d) for d in _stats_train], dim=0).float()
            all_x = torch.cat([d.x for d in _stats_train], dim=0).float()
            all_edge_attr = torch.cat([d.edge_attr for d in _stats_train], dim=0).float()
            print(f"[stats] done in {_time.time() - _t_stats0:.1f}s", flush=True)
            stats = torch.cat([
                all_targets.mean(dim=0), all_targets.std(dim=0) + 1e-8,
                all_node_targets.mean(dim=0), all_node_targets.std(dim=0) + 1e-8,
                all_x.mean(dim=0), all_x.std(dim=0) + 1e-8,
                all_edge_attr.mean(dim=0), all_edge_attr.std(dim=0) + 1e-8,
                graph_target_std_full.view(1),
            ]).to(device)
            stats_len = torch.tensor([stats.numel()], device=device)
        else:
            stats_len = torch.tensor([0], device=device)
        dist.broadcast(stats_len, src=0)
        if not is_main:
            stats = torch.zeros(int(stats_len.item()), device=device)
        dist.broadcast(stats, src=0)

        # target_mean/std and node_target_mean/std are width-1 (both gnn_target_fn
        # and gnn_node_target_fn always return a single scalar per node/graph).
        # Every slice below is .clone()'d so each stat is its own independent
        # tensor/storage/autograd-version-counter -- without this they're all
        # views sharing `stats`'s single storage, so an in-place write to ANY
        # of them (or to `stats` itself) bumps every other slice's version
        # counter too. That's invisible with one forward/backward per step,
        # but the grad-matching loss's second forward (see grad_iter below)
        # reads x_mean/x_std in a second autograd graph before the first
        # graph's backward runs, and DDP's own per-step buffer bookkeeping
        # was enough to trip a "modified by an inplace operation" version
        # mismatch on the shared storage (confirmed via a 2-GPU smoke test:
        # error persisted even with static_graph=True, which rules out a
        # pure DDP-bucketing explanation and points at this aliasing).
        i = 0
        target_mean = stats[i:i+1].clone(); i += 1
        target_std  = stats[i:i+1].clone(); i += 1
        node_target_mean = stats[i:i+1].clone(); i += 1
        node_target_std  = stats[i:i+1].clone(); i += 1
        node_in_dim_probe = train_set[0].x.shape[1] if train_set else val_set[0].x.shape[1]
        x_mean = stats[i:i+node_in_dim_probe].clone(); i += node_in_dim_probe
        x_std  = stats[i:i+node_in_dim_probe].clone(); i += node_in_dim_probe
        edge_in_dim_probe = train_set[0].edge_attr.shape[1] if train_set else val_set[0].edge_attr.shape[1]
        edge_mean = stats[i:i+edge_in_dim_probe].clone(); i += edge_in_dim_probe
        edge_std  = stats[i:i+edge_in_dim_probe].clone(); i += edge_in_dim_probe
        graph_target_std_full = stats[i:i+1].clone(); i += 1  # kept as a tensor -- .item() called at each use site, matching the non-ddp path

        print("Target mean (log MPa):", target_mean.detach().cpu().numpy())
        print("Target std (log MPa):", target_std.detach().cpu().numpy())
        print("Node target mean (linear MPa):", node_target_mean.detach().cpu().numpy())
        print("Node target std (linear MPa):", node_target_std.detach().cpu().numpy())
    else:
        # ----------------------------------------------------
        # Single-process loading: unchanged from before.
        # ----------------------------------------------------
        dataset = load_dataset(dataset_path)
        for sample in dataset:
            _preprocess(sample)
        num_pos_nodes = _num_pos_nodes_from_graph
        if num_pos_nodes:
            attached = sum(1 for d in dataset if hasattr(d, "node_id"))
            print(f"[fixed-graph] graph.pt loaded: nodes={num_pos_nodes}, "
                  f"attached to {attached}/{len(dataset)} samples")
        else:
            print(f"[fixed-graph] no graph.pt at {graph_path}; using generic GNN")

        cleaned_dataset = [d for d in dataset if d.max_stress < 3000]
        dataset = cleaned_dataset

        num_samples = min(num_samples, len(dataset))
        print(f"Dataset size after cleaning: {len(dataset)}")

        # log-space (see gnn_log_target_fn) -- graph_target_std_full feeds the
        # "RMSE" reported below, which is therefore RMSE in log(MPa), not raw MPa.
        all_graph_targets_full = torch.cat([gnn_log_target_fn(d) for d in dataset], dim=0).float()
        graph_target_mean_full = all_graph_targets_full.mean()
        graph_target_std_full  = all_graph_targets_full.std() + 1e-8
        print(f"Dataset ks_stress mean: {graph_target_mean_full.item():.4f} (log MPa)")
        print(f"Dataset ks_stress std:  {graph_target_std_full.item():.4f} (log MPa)")

        total_samples = len(dataset)
        if num_samples is None or num_samples > total_samples:
            num_samples = total_samples

        # Random train/val split (instead of slicing in order)
        indices = torch.randperm(total_samples)[:num_samples]
        n_train = int(num_samples * 0.8)
        train_idx = indices[:n_train]
        val_idx   = indices[n_train:]
        train_set = [dataset[i] for i in train_idx]
        val_set   = [dataset[i] for i in val_idx]

        print(f"Total samples used: {num_samples}")
        print(f"Train: {len(train_set)}, Val: {len(val_set)}")

        # Compute normalization stats on TRAIN ONLY (log-space for the graph
        # target -- see gnn_log_target_fn)
        all_targets = torch.cat([gnn_log_target_fn(d) for d in train_set], dim=0).float()
        target_mean = all_targets.mean(dim=0).to(device)
        target_std  = (all_targets.std(dim=0) + 1e-8).to(device)

        all_node_targets = torch.cat([gnn_node_target_fn(d) for d in train_set], dim=0).float()
        node_target_mean = all_node_targets.mean(dim=0).to(device)
        node_target_std  = (all_node_targets.std(dim=0) + 1e-8).to(device)

        all_x = torch.cat([d.x for d in train_set], dim=0).float()
        x_mean = all_x.mean(dim=0).to(device)
        x_std  = (all_x.std(dim=0) + 1e-8).to(device)

        all_edge_attr = torch.cat([d.edge_attr for d in train_set], dim=0).float()
        edge_mean = all_edge_attr.mean(dim=0).to(device)
        edge_std  = (all_edge_attr.std(dim=0) + 1e-8).to(device)

        print("Target mean (log MPa):", target_mean.detach().cpu().numpy())
        print("Target std (log MPa):", target_std.detach().cpu().numpy())
        print("Node target mean (linear MPa):", node_target_mean.detach().cpu().numpy())
        print("Node target std (linear MPa):", node_target_std.detach().cpu().numpy())
        print("X mean:", x_mean.detach().cpu().numpy())
        print("X std:", x_std.detach().cpu().numpy())
        print("Edge mean:", edge_mean.detach().cpu().numpy())
        print("Edge std:", edge_std.detach().cpu().numpy())

    # ----------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------
    if ddp:
        # train_set is already THIS rank's own shard slice (see the ddp
        # loading branch above) -- no DistributedSampler here, that would
        # re-partition an already-rank-local list by rank/world_size again
        # (net effect: each rank training on ~1/world_size^2 of the data).
        # Per-epoch reshuffling instead comes from shuffle=True picking up
        # the torch.manual_seed(seed_offset+epoch) reseed already done at the
        # top of the epoch loop, same mechanism the non-ddp path relies on.
        #
        # Shard counts don't divide evenly across ranks (e.g. 400 shards /
        # 3 ranks = 134/133/133), so local sample counts can differ slightly.
        # DDP's backward-time gradient all-reduce is a collective op -- if
        # ranks ran different step counts, the rank that finishes first would
        # block forever waiting on a collective the others never issue
        # (deadlock). Truncate every rank to the global minimum so step
        # counts match exactly.
        local_n = torch.tensor([len(train_set)], device=device)
        dist.all_reduce(local_n, op=dist.ReduceOp.MIN)
        train_set = train_set[:int(local_n.item())]
        train_sampler = None
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=4, persistent_workers=True, pin_memory=True,
                                  prefetch_factor=4)
    else:
        train_sampler = None
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=8, persistent_workers=True, pin_memory=True,
                                  prefetch_factor=4)
    # Validation only runs on rank 0, against the full (unsharded) val_set --
    # it's a forward-only pass (no .backward()), so DDP's collective gradient
    # sync never triggers and other ranks don't need to participate at all.
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, persistent_workers=True, pin_memory=True,
                            prefetch_factor=4) if is_main else None

    # ----------------------------------------------------
    # Gradient-matching dataset (optional) -- ground-truth per-tile FD
    # sensitivity labels from training/dataset_builder.py's --grad_mode
    # (base ccx run + N single-tile-perturbed reruns, FD'd against the real
    # solver). Kept in its own dataset/DataLoader, never merged into
    # train_set/val_set: PyG's Batch collation decides which attrs to batch
    # from only the FIRST sample in a batch, so a shuffled loader mixing
    # labeled/unlabeled samples either crashes or silently drops the field
    # depending on shuffle order. Loaded in full on every rank (DDP or not)
    # -- it's small by construction (each sample costs M+1 real FEA solves
    # to build), so no sharding is needed the way the (much larger) main
    # dataset requires under DDP.
    # ----------------------------------------------------
    def _slog_scaled(v, scale):
        """Signed log1p compression, shared by the tile-gradient loss and
        the node-gradient head. Both targets span many decades, so a plain
        MSE is dominated by the largest-magnitude entries."""
        return torch.sign(v) * torch.log1p(v.abs() / scale)

    grad_enabled = (grad_dataset_path is not None) or grad_from_train
    # Scale for the node-gradient signed-log loss. Same role as
    # grad_target_scale below: it only sets where log1p's compression
    # starts, so a mean of |dKS/dX| over a subsample is enough.
    node_grad_scale = None
    if node_grad_weight > 0.0:
        _ng = [s.grad_dks_dnode.abs().mean() for s in train_set[:512]
               if hasattr(s, "grad_dks_dnode")]
        assert _ng, ("--node_grad_weight set but no train sample has "
                     "grad_dks_dnode; rebuild with --grad_store_nodes true")
        node_grad_scale = (torch.stack(_ng).mean() + 1e-12).to(device)
        print(f"[node-grad] rank={rank} scale={node_grad_scale.item():.6g} MPa/length "
              f"from {len(_ng)} samples")
    grad_loader = None
    grad_M = 0
    grad_target_scale = None
    K_tiles = 0
    if grad_dataset_path or grad_from_train:
        if grad_from_train:
            # Draw gradient supervision from THIS rank's already-loaded train
            # shards instead of a separate file. Every dataset_gradadj sample
            # carries grad_dks_dx/grad_tile_idx already -- data/hollow_cube/
            # gradsub was only a 1-in-10 SYMLINK subsample of the very same
            # shards (41 of 501, stride 10), so the old default trained the
            # grad loss on 4,100 of 50,001 available gradients.
            #
            # That starved it: grad_iter below cycles, so the number of grad
            # steps per epoch is set by the MAIN loader (2496 steps), not by
            # the pool size. With 4,100 samples at grad_batch_size 4 that is
            # ~2.4 passes over the same gradients EVERY epoch (~41 passes by
            # epoch 17). Job 1373261's held-out dg peaked at epoch ~4
            # (median cosine +0.715, beating the +0.613 constant-direction
            # control) and collapsed to +0.062 by epoch 15 while TRAIN cos
            # rose to +0.9585 -- textbook memorisation of the small pool.
            #
            # train_set is this rank's own shard slice and is already
            # _preprocess()ed, so do NOT re-run it (that would re-convert
            # max_stress to MPa a second time). Costs no extra memory (same
            # objects) and no extra compute (the iterator cycles either way).
            # Val shards are excluded by construction, so grad_audit stays
            # clean as a held-out set.
            grad_dataset = [s for s in train_set if hasattr(s, "grad_tile_idx")]
            assert grad_dataset, (
                "--grad_from_train set but no train sample has grad_tile_idx; "
                "the main dataset must be built with --grad_mode adjoint")
        else:
            grad_dataset = load_dataset(grad_dataset_path)
            for sample in grad_dataset:
                _preprocess(sample)
        assert grad_dataset and hasattr(grad_dataset[0], "grad_tile_idx"), (
            f"{grad_dataset_path} has no grad_tile_idx/grad_dks_dx samples -- build it with "
            f"training/dataset_builder.py --grad_mode true (or jobs/build_grad_dataset_hollow_cube.slurm)"
        )
        grad_M = int(grad_dataset[0].grad_tile_idx.shape[0])
        K_tiles = int(grad_dataset[0].tile_NX) * int(grad_dataset[0].tile_NY) * int(grad_dataset[0].tile_NZ)
        all_grad_dks_dx = torch.cat([d.grad_dks_dx for d in grad_dataset]).float()
        # Scale for the signed-log loss below -- mean |true sensitivity|,
        # not std, since dg/dx is already known to be extremely heavy-tailed
        # (see project_surrogate_sensitivity_vs_value_gap.md) and this value
        # only sets where log1p's compression kicks in, not a normalization
        # that needs to be unbiased.
        grad_target_scale = (all_grad_dks_dx.abs().mean() + 1e-8).to(device)
        print(f"[grad-loss] rank={rank} {len(grad_dataset)} samples, M={grad_M} tiles/sample, "
              f"K_tiles={K_tiles}, mean|dks/dx| scale={grad_target_scale.item():.4f} MPa/unit")
        print(f"[grad-loss] mode={grad_err_mode}" + (f" weight_power={grad_weight_power} rel_sat={grad_rel_sat}"
              if grad_err_mode == "rel" else f" mass_weight={grad_mass_weight}") + (f" sign_loss_weight={grad_sign_loss_weight} sign_delta={grad_sign_delta}"
                 if grad_sign_loss_weight > 0.0 else " sign_loss=off"))
        grad_loader = DataLoader(grad_dataset, batch_size=(grad_batch_size or batch_size),
                                 shuffle=True, num_workers=2, persistent_workers=True,
                                 pin_memory=True)

    def _cycle(loader):
        # Plain re-iteration (fresh shuffle each pass since shuffle=True)
        # instead of itertools.cycle, which would cache every yielded batch
        # forever just to replay it -- needless duplication of an already
        # in-memory dataset.
        while True:
            for b in loader:
                yield b

    grad_iter = _cycle(grad_loader) if grad_loader is not None else None

    # ----------------------------------------------------
    # Model setup
    # ----------------------------------------------------
    example = train_set[0]
    node_in_dim = example.x.shape[1]
    edge_in_dim = example.edge_attr.shape[1]
    out_dim     = gnn_target_fn(example).shape[1]

    hierarchical = hasattr(train_set[0], 'tile_idx')
    assert coarse_arch in ("transformer", "gnn", "cnn"), coarse_arch
    assert not (node_grad_weight > 0.0 and coarse_arch != "cnn"), (
        "--node_grad_weight is only implemented for coarse_arch=cnn "
        "(HierarchicalGNNCNN.grad_head); it would silently do nothing otherwise")
    if hierarchical:
        ModelClass = {"transformer": HierarchicalGNN, "gnn": HierarchicalGNNMP,
                      "cnn": HierarchicalGNNCNN}[coarse_arch]
    else:
        ModelClass = GNN
    num_pos_nodes = num_pos_nodes if use_node_emb else 0
    print(f"[model] use_node_emb={use_node_emb} num_pos_nodes={num_pos_nodes} coarse_arch={coarse_arch if hierarchical else 'N/A'}")
    model_kwargs = dict(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=hidden_dim,
        num_layers=conv_layers,
        use_checkpoint=use_checkpoint,
        num_pos_nodes=num_pos_nodes,
        node_target_mean=node_target_mean,
        node_target_std=node_target_std,
        stress_ks_rho_mpa=stress_ks_rho_mpa,
    )
    if hierarchical:
        model_kwargs["checkpoint_fine_only"] = checkpoint_fine_only
        if coarse_arch == "transformer":
            model_kwargs["transformer_heads"] = transformer_heads
            model_kwargs["transformer_ff_mult"] = transformer_ff_mult
            model_kwargs["transformer_dropout"] = transformer_dropout
        elif coarse_arch == "gnn":
            model_kwargs["dropout"] = mp_dropout
            model_kwargs["coarse_dim"] = coarse_dim
        else:  # cnn
            model_kwargs["dropout"] = mp_dropout
            model_kwargs["coarse_dim"] = coarse_dim
            model_kwargs["cnn_depth"] = cnn_depth
            model_kwargs["predict_node_grad"] = node_grad_weight > 0.0
    raw_model = ModelClass(**model_kwargs).to(device)
    if hierarchical:
        # Fixed-topology: the tile grid never changes during a training run,
        # so precompute it once here rather than lazily inside forward().
        raw_model.set_tile_grid(
            int(example.tile_NX), int(example.tile_NY), int(example.tile_NZ)
        )
    print(f"[model] use_checkpoint={use_checkpoint} "
          f"checkpoint_fine_only={checkpoint_fine_only if hierarchical else 'N/A'}")

    # torch.compile: fuses the many small gather/cat/elementwise ops around
    # each EdgeAttrConv into fewer kernels. Compiled through `model` (used for
    # the forward calls below); checkpoint/resume always save and load via
    # raw_model.state_dict(), since OptimizedModule.state_dict() prefixes keys
    # with "_orig_mod." and would otherwise break gnn_evaluator.py / resume,
    # which load into a plain (uncompiled) module.
    # Compile first, then DDP-wrap the compiled module (the reverse order
    # doesn't work: DDP's parameter hooks need to see the real params, not a
    # compiled wrapper). --use_compile true remains the operator's call on
    # whether the coarse_arch in use is compile-safe (transformer's coarse
    # self-attention crashes under torch.compile independent of ddp -- see
    # jobs/train_model.slurm; gnn coarse arch is unaffected).
    compiled = use_compile and torch.cuda.is_available()
    if compiled and ddp:
        # Dynamo's DDPOptimizer graph-splitting pass can't handle the
        # higher-order op torch.utils.checkpoint introduces (use_checkpoint
        # True here) -- crashes every rank with "Found a higher order op in
        # the graph". Falls back to one compiled bucket for the whole graph
        # instead of DDP-bucket-aligned splitting; costs some overlap between
        # backward and gradient all-reduce, but is required for compile to
        # work at all under DDP+checkpointing.
        #
        # Tried instead: wrapping each checkpoint() *call site* in
        # utils/gnn_hierarchical_mp.py with @torch._dynamo.disable, so the
        # higher-order op never enters the traced graph and DDPOptimizer can
        # bucket normally. Confirmed it doesn't crash and VRAM stays in
        # budget (39/41GB peak/reserved on a 2-GPU probe) -- but it's a real
        # throughput *regression*: ~4.3-4.4s/step vs ~1.2-1.5s/step baseline
        # (2-GPU decoy probe, jobs/train_model_gnn_probe2gpu.slurm), roughly
        # 3x slower. The graph breaks apparently cost far more (recompile/
        # dispatch overhead at each of the 4 break points, every step, every
        # rank) than the bucketing they were meant to restore. Reverted --
        # don't retry this approach without a different mechanism.
        torch._dynamo.config.optimize_ddp = False
    model = torch.compile(raw_model) if compiled else raw_model
    if ddp:
        # find_unused_parameters=True is needed for HierarchicalGNN
        # (transformer coarse_arch): tile_input_encoder is only exercised
        # when tile_x is not None, so it's not guaranteed to appear in every
        # step's graph, which without this trips DDP's "did not receive
        # grad" desync check. HierarchicalGNNMP (gnn coarse_arch) has no such
        # conditionally-used params -- confirmed by DDP's own runtime check
        # ("did not find any unused parameters") on every gnn-arch run so
        # far -- so leaving it True there only pays for an extra full
        # autograd-graph traversal every step with no benefit. HierarchicalGNNCNN
        # (cnn coarse_arch) has its own always-dead params on fixed-tile-grid
        # geometries (global_emb/tile_global_edge_attr/global_proj_fallback,
        # only exercised on the no-tile-grid fallback path), which crashed a
        # 2-GPU cnn run (job 1318900) with "Expected to have finished
        # reduction..." until this was added -- also needs True.
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=(coarse_arch in ("transformer", "cnn")),
                    # static_graph=True whenever the grad loss is on (either
                    # grad-loss block calls model(...) a SECOND time (grad
                    # batch) before the single combined loss.backward(), and
                    # PyTorch's own DDP docs list "multiple forward calls in
                    # one iteration" as a static_graph use case (same ops run
                    # every step here, so it's safe). Left off by default
                    # (only one forward/step normally -- no need to pay the
                    # first-few-iteration graph-capture cost).
                    static_graph=grad_enabled,
                    # broadcast_buffers=False whenever the grad loss is on
                    # set: DDP's default (True) re-copies every registered
                    # buffer -- including the model's node_target_mean/
                    # node_target_std (utils/gnn_surrogate.py's
                    # register_buffer(..., persistent=False)) -- in place, at
                    # the START of every forward() call. Those buffers
                    # participate in the model's own autograd graph (they
                    # denormalize node_pred internally), so calling forward()
                    # twice per iteration means the grad batch's forward
                    # re-copies the buffer in place *after* the main batch's
                    # forward already recorded a dependency on its
                    # pre-copy version -- autograd then refuses the later
                    # combined backward() with "modified by an inplace
                    # operation" (confirmed via a 2-GPU smoke test: this,
                    # not the multi-forward bucketing static_graph addresses,
                    # was the actual cause). Safe to disable here: these
                    # buffers are set once at model construction from stats
                    # already broadcast-synced across ranks and never change
                    # during training, so the redundant per-forward re-copy
                    # was doing nothing useful even before this bug.
                    broadcast_buffers=(not grad_enabled))
    print(f"[compile] torch.compile enabled={compiled}")
    print(f"[ddp] model wrapped={ddp}")

    # ----------------------------------------------------
    # Baseline normalization of the loss terms. Each term is divided by the
    # score of a predictor that knows nothing, so every term starts near 1
    # and equal weights mean equal importance relative to how hard each task
    # is:
    #   ks    always predict the median train KS  -> mean |median - KS| / KS
    #   node  predict 0 at every node             -> mean sum_i w_i sigma_i / KS
    #   grad  predict 0 for every tile (rel mode) -> exactly 1 / (1 + 1/sat)
    #   sign  predict 0 / coin flip               -> exactly 0.5
    # Only the trained total is normalized. The log prints the raw %. Computed
    # from the training data, not hard-coded, so the same code works on any
    # dataset. On hollow_cube_11x11/dataset_screened30k (500 samples):
    # ks 26.1%, node 26.7%, grad 66.7%, sign 0.5.
    # ----------------------------------------------------
    _bl_ks = torch.stack([gnn_target_fn(d).float().view(-1)[0] for d in train_set]).to(device)
    _bl_n, _bl_sum = torch.zeros((), device=device), torch.zeros((), device=device)
    for d in train_set[:2000]:
        _n = gnn_node_target_fn(d).float().view(-1).clamp_min(0.0)
        _k = gnn_target_fn(d).float().view(-1)[0].clamp_min(LOG_STRESS_EPS)
        _w = _n.pow(stress_weight_power); _w = _w / (_w.sum() + 1e-8)
        _bl_n += ((_w * _n).sum() / _k).to(device); _bl_sum += 1
    _bl_med = _bl_ks.median()
    _bl_k_err = ((_bl_med - _bl_ks).abs() / _bl_ks.clamp_min(LOG_STRESS_EPS)).sum()
    _bl_k_cnt = torch.tensor(float(_bl_ks.numel()), device=device)
    if ddp:
        for _t in (_bl_k_err, _bl_k_cnt, _bl_n, _bl_sum):
            dist.all_reduce(_t)
    base_ks = float(_bl_k_err / _bl_k_cnt)
    base_node = float(_bl_n / _bl_sum)
    base_grad = (1.0 / (1.0 + 1.0 / grad_rel_sat)) if grad_err_mode == "rel" else 1.0
    base_sign = 0.5
    print(f"[loss-norm] baselines: ks {100*base_ks:.1f}%  node {100*base_node:.1f}%  "
          f"grad {100*base_grad:.1f}%{'' if grad_err_mode == 'rel' else ' (slog mode: not normalized)'}  "
          f"sign {base_sign:.3f}  -> each loss term is divided by its baseline")

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = torch.nn.MSELoss()

    train_losses = []
    val_losses   = []
    train_graph_losses = []
    val_graph_losses   = []
    train_node_losses  = []
    val_node_losses    = []
    train_grad_losses  = []  # gradient-matching loss (train-only, see grad_iter above)
    train_gcos_losses  = []  # 1-cos(pred_dg,true_dg); the OC-relevant direction metric

    # ----------------------------------------------------
    # Resume from latest checkpoint if available
    # ----------------------------------------------------
    start_epoch = 0
    ckpt_files = glob.glob(os.path.join(save_dir, "*_epochs.pt"))
    epoch_re = re.compile(r"(\d+)_epochs\.pt$")
    parsed = [(int(m.group(1)), p) for p in ckpt_files for m in [epoch_re.search(p)] if m]
    if parsed:
        start_epoch, latest_ckpt = max(parsed, key=lambda t: t[0])
        print(f"Resuming from checkpoint: {latest_ckpt} (epoch {start_epoch})")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print("[resume] checkpoint has no optimizer_state (pre-fix checkpoint) -- "
                  "AdamW moments restart from zero, expect a small loss jump")

        if resume_lr_scale is not None:
            # optimizer.load_state_dict() above restored the checkpoint's own
            # lr, so a plain --lr on the CLI would otherwise be silently
            # discarded. This multiplies the restored lr by resume_lr_scale
            # once, on top of which the per-epoch decay/plateau-halving below
            # continues to apply as normal (no cosine schedule/T_max involved).
            resumed_lr = optimizer.param_groups[0]["lr"]
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * resume_lr_scale
            print(f"[resume] lr scale requested -> x{resume_lr_scale}, "
                  f"{resumed_lr:.2e} -> {optimizer.param_groups[0]['lr']:.2e}")

        losses_path = os.path.join(run_dir, "losses.json")
        if os.path.exists(losses_path):
            with open(losses_path, "r") as f:
                prev = json.load(f)
            train_losses = prev.get("train_losses", [])[:start_epoch]
            val_losses   = prev.get("val_losses", [])[:start_epoch]
            train_graph_losses = prev.get("train_graph_losses", [])[:start_epoch]
            val_graph_losses   = prev.get("val_graph_losses", [])[:start_epoch]
            train_node_losses  = prev.get("train_node_losses", [])[:start_epoch]
            val_node_losses    = prev.get("val_node_losses", [])[:start_epoch]
            train_grad_losses  = prev.get("train_grad_losses", [])[:start_epoch]
            train_gcos_losses  = prev.get("train_gcos_losses", [])[:start_epoch]

    # Best-checkpoint tracking (by val ks_stress/graph loss) -- recovered from
    # the resumed loss history so a resumed run doesn't reset "best" and
    # clobber gnn_surrogate.pt with a worse checkpoint before catching back up.
    best_val_graph_loss = min(val_graph_losses) if val_graph_losses else float("inf")
    # LR-plateau signal uses *training* loss, not val graph loss -- the latter
    # is noisy epoch-to-epoch (small val-batch KS/max-stress estimate), which
    # made the plateau trigger fire on noise rather than genuine stalling.
    best_train_loss = min(train_losses) if train_losses else float("inf")
    # Epochs since the last training-loss improvement, recovered from history
    # so a resumed run doesn't reset the plateau-halving countdown to 0.
    epochs_since_improve = (
        (len(train_losses) - 1 - train_losses.index(best_train_loss))
        if train_losses else 0
    )
    surrogate_path = f"data/{geometry}/gnn_surrogate.pt"

    def _build_ckpt(include_optimizer=False):
        ckpt = {
            "model_state": raw_model.state_dict(),
            "node_in_dim": node_in_dim,
            "edge_in_dim": edge_in_dim,
            "hidden_dim": hidden_dim,
            "coarse_dim": raw_model.coarse_dim if hierarchical and coarse_arch in ("gnn", "cnn") else None,
            "cnn_depth": cnn_depth if hierarchical and coarse_arch == "cnn" else None,
            "conv_layers": conv_layers,
            "out_dim": out_dim,
            "hierarchical": hierarchical,
            "num_pos_nodes": num_pos_nodes,
            "coarse_arch": coarse_arch,
            # Needed to rebuild the model with its dKS/dX head; without it a
            # loader constructs predict_node_grad=False and silently drops
            # every grad_head.* weight. Loaders written before this key
            # existed can infer it from the state_dict key prefix instead.
            "predict_node_grad": node_grad_weight > 0.0,
            "transformer_heads": transformer_heads,
            "transformer_ff_mult": transformer_ff_mult,
            "stress_ks_rho_mpa": stress_ks_rho_mpa,
            # target_mean/std are log-space (see gnn_log_target_fn) -- used
            # ONLY inside this file's graph_loss computation, to weight
            # errors evenly across magnitude scales instead of a few extreme
            # samples dominating a linear MSE. They do NOT change what
            # graph_pred represents at inference: ks_pool always denormalizes
            # graph_pred to raw linear MPa via node_target_mean/std (which
            # stay linear -- required for ks_pool's physics), and log is
            # monotonic, so the optimum graph_pred is still the true raw
            # ks_stress either way. Evaluators (gnn_evaluator.py) correctly
            # never read target_mean/std at all -- purely a training-time
            # loss-weighting artifact, kept here just for inspection.
            "graph_target_log_normalized": True,
            "target_mean": target_mean.detach().cpu(),
            "target_std": target_std.detach().cpu(),
            "node_target_mean": node_target_mean.detach().cpu(),
            "node_target_std": node_target_std.detach().cpu(),
            "x_mean": x_mean.detach().cpu(),
            "x_std": x_std.detach().cpu(),
            "edge_mean": edge_mean.detach().cpu(),
            "edge_std": edge_std.detach().cpu(),
        }
        if include_optimizer:
            ckpt["optimizer_state"] = optimizer.state_dict()
        return ckpt


    # ======================================================
    # Training Loop
    # ======================================================
    for epoch in range(start_epoch + 1, epochs + 1):
        # re-seed per epoch so resumed runs don't replay the same batch order
        # (also what reshuffles the ddp path's plain shuffle=True loader --
        # no DistributedSampler.set_epoch() needed since there isn't one)
        torch.manual_seed(seed_offset + epoch)
        epoch_t0 = time.time()

        # -----------------------------
        # TRAIN
        # -----------------------------
        model.train()
        total_train = 0.0
        total_train_graph = 0.0
        total_train_node  = 0.0
        total_train_grad  = 0.0
        total_train_gcos  = 0.0
        total_train_sgnerr = 0.0
        total_train_sgnl  = 0.0
        total_train_ngrad = 0.0
        total_train_ncos  = 0.0

        # Per-section timing (only collected on epoch 1 to keep overhead off the steady state).
        profile_this_epoch = (epoch == start_epoch + 1)
        t_data = t_h2d = t_fwd = t_bwd = t_step = 0.0
        n_steps = 0

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        loader_iter = iter(train_loader)
        if profile_this_epoch: _sync()
        t_mark = time.perf_counter()

        while True:
            try:
                batch_data = next(loader_iter)
            except StopIteration:
                break
            if profile_this_epoch:
                _sync(); t_now = time.perf_counter(); t_data += t_now - t_mark; t_mark = t_now

            optimizer.zero_grad()

            x, edge_index, edge_attr, batch_idx, tile_idx, tile_NX, tile_NY, tile_NZ, tile_x, node_id = gnn_input_fn(batch_data)
            num_graphs = batch_data.num_graphs
            x = x.float().to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            edge_attr  = edge_attr.float().to(device, non_blocking=True)
            batch_idx  = batch_idx.to(device, non_blocking=True)
            if tile_idx is not None:
                tile_idx = tile_idx.to(device, non_blocking=True)
                # tile_NX/NY/NZ are already plain Python ints (see gnn_input_fn)
            if tile_x is not None:
                tile_x = tile_x.float().to(device, non_blocking=True)
            if node_id is not None:
                node_id = node_id.to(device, non_blocking=True)

            # normalize features
            x_norm = (x - x_mean) / x_std
            edge_attr_norm = (edge_attr - edge_mean) / edge_std

            targ_mpa = gnn_target_fn(batch_data).float().to(device, non_blocking=True)
            node_targ = gnn_node_target_fn(batch_data).float().to(device, non_blocking=True)
            if profile_this_epoch:
                _sync(); t_now = time.perf_counter(); t_h2d += t_now - t_mark; t_mark = t_now

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                _mo = model(
                    x_norm, edge_index, edge_attr_norm, batch_idx,
                    tile_idx=tile_idx, tile_NX=tile_NX, tile_NY=tile_NY, tile_NZ=tile_NZ,
                    tile_x=tile_x, num_graphs=num_graphs, node_id=node_id,
                    **({"return_node_grad": True} if node_grad_weight > 0.0 else {}),
                )
                if node_grad_weight > 0.0:
                    graph_pred, node_pred, node_grad_pred = _mo
                else:
                    graph_pred, node_pred = _mo
                # graph_pred and node_pred live in node-normalized space (see
                # ks_pool) -- de-normalize both to raw linear MPa, then score
                # them as relative errors (see ks_rel_err / weighted_node_rel_err).
                graph_pred_raw_mpa = (graph_pred * node_target_std + node_target_mean).clamp_min(LOG_STRESS_EPS)
                node_pred_raw_mpa = node_pred * node_target_std + node_target_mean
                graph_loss = ks_rel_err(graph_pred_raw_mpa, targ_mpa)
                node_loss  = weighted_node_rel_err(node_pred_raw_mpa, node_targ, targ_mpa, batch_idx, stress_weight_power)
                loss = graph_loss_weight * graph_loss / base_ks + node_loss_weight * node_loss / base_node

                # Node-gradient head: match the ADJOINT NODAL field
                # dKS/dX directly, then let core/geometric_jacobian.py's
                # exact dX/dt do the chain rule at inference instead of
                # asking the network to learn a closed-form Jacobian.
                #
                # Rides the MAIN batch: every dataset_gradadj sample
                # carries grad_dks_dnode, so this needs no second forward,
                # no create_graph=True double backward, and no separate
                # pool that can be cycled to memorisation (the defect that
                # killed job 1373261 -- 4,100 gradients re-visited ~41x).
                #
                # Signed-log is the right transform HERE even though it was
                # the wrong one for the per-tile loss. Per tile the metric
                # is absolute and mass-weighted, so log1p discarded exactly
                # the structure OC follows. Per node, slog is effectively a
                # uniform RELATIVE-error objective, and the composition was
                # measured to tolerate ~100% relative node error while still
                # beating the constant-direction control (20% error ->
                # tile cosine +0.997, 0.0% wrong-way mass).
                if node_grad_weight > 0.0:
                    ng_true = batch_data.grad_dks_dnode.float().to(device, non_blocking=True)
                    node_grad_loss, node_grad_cos = weighted_node_grad_loss(
                        node_grad_pred.float(), ng_true, batch_idx, node_grad_scale,
                        power=node_grad_power, cos_weight=node_grad_cos_weight)
                    loss = loss + node_grad_weight * node_grad_loss

            grad_loss = None
            grad_cos = None
            if grad_iter is not None:
                # Gradient-matching (Sobolev) loss: penalize the surrogate's
                # own autodiff sensitivity d(ks_stress)/d(tile thickness) for
                # deviating from the ground-truth FD sensitivity computed on
                # the real solver (see training/dataset_builder.py's
                # --grad_mode). Folded into the SAME combined `loss` below,
                # backed by a single outer loss.backward() call -- this
                # "inner autograd.grad(create_graph=True), then one outer
                # backward()" is the standard DDP-safe Sobolev/gradient-
                # penalty pattern: DDP's gradient-sync hooks only fire on the
                # actual .backward() call, not on the plain autograd.grad()
                # call below, so no extra DDP handling is needed.
                gbatch = next(grad_iter)
                (g_x, g_edge_index, g_edge_attr, g_batch_idx, g_tile_idx, g_tile_NX,
                 g_tile_NY, g_tile_NZ, g_tile_x, g_node_id) = gnn_input_fn(gbatch)
                g_num_graphs = gbatch.num_graphs
                g_x = g_x.float().to(device, non_blocking=True)
                g_edge_index = g_edge_index.to(device, non_blocking=True)
                g_edge_attr  = g_edge_attr.float().to(device, non_blocking=True)
                g_batch_idx  = g_batch_idx.to(device, non_blocking=True)
                g_tile_idx   = g_tile_idx.to(device, non_blocking=True)
                g_tile_x     = g_tile_x.float().to(device, non_blocking=True)
                if g_node_id is not None:
                    g_node_id = g_node_id.to(device, non_blocking=True)
                # grad_tile_idx is stored in dims-key order by
                # dataset_builder.py's _ccx_run_grad_sample (it's built from
                # "d{t+1}" keys), but d_pred's columns below are tile_idx-native
                # order (t_leaf/tile_x, per _preprocess) -- convert before
                # gathering, or this silently reads another tile's gradient.
                g_grad_tile_idx = gbatch.grad_tile_idx.to(device, non_blocking=True).view(g_num_graphs, grad_M)
                g_grad_tile_idx = _strut_perm_inv[g_grad_tile_idx]
                g_grad_dks_dx = gbatch.grad_dks_dx.float().to(device, non_blocking=True).view(g_num_graphs, grad_M)

                # Shared differentiable leaf: strut thickness enters the
                # model via BOTH the appended x column (fine mesh, see
                # _preprocess) and tile_x (coarse embedding) -- rebuilding
                # both from one leaf is required so the sensitivity captures
                # both paths, not just tile_x's.
                t_leaf = g_tile_x.detach().clone().requires_grad_(True)  # [g_num_graphs*K_tiles, 1]
                node_strut = t_leaf.view(g_num_graphs, K_tiles)[g_batch_idx, g_tile_idx].unsqueeze(1)
                g_x_grad = torch.cat([g_x[:, :-1], node_strut], dim=1)
                g_x_norm = (g_x_grad - x_mean) / x_std
                g_edge_attr_norm = (g_edge_attr - edge_mean) / edge_std

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    g_graph_pred, _ = model(
                        g_x_norm, g_edge_index, g_edge_attr_norm, g_batch_idx,
                        tile_idx=g_tile_idx, tile_NX=g_tile_NX, tile_NY=g_tile_NY, tile_NZ=g_tile_NZ,
                        tile_x=t_leaf, num_graphs=g_num_graphs, node_id=g_node_id,
                    )
                    g_graph_pred_raw_mpa = (g_graph_pred * node_target_std + node_target_mean).clamp_min(LOG_STRESS_EPS)

                d_pred = torch.autograd.grad(
                    g_graph_pred_raw_mpa.float().sum(), t_leaf, create_graph=True, retain_graph=True,
                )[0].view(g_num_graphs, K_tiles)
                pred_grad_at_tiles = torch.gather(d_pred, 1, g_grad_tile_idx)

                # Signed-log MSE: robust to the extreme dynamic range and
                # sign flips already observed in FD sensitivities (see
                # project_surrogate_sensitivity_vs_value_gap.md) -- a plain
                # MSE would be dominated by the largest-magnitude tiles.
                def _slog(v):
                    return torch.sign(v) * torch.log1p(v.abs() / grad_target_scale)
                # Mass weighting of the slog MSE. Equal weight per tile means
                # ~60% of the loss is the near-zero tiles: they carry <0.01% of
                # the |dg| mass, sit at chance-level sign agreement on TRAINING
                # designs at every epoch (never learnable from random designs),
                # and OC never uses them. Meanwhile the top decile (96% of the
                # mass, the needle OC follows) got 10% of the loss and stayed at
                # 77-80% train sign agreement from epoch 8 to 43 while the
                # total grad loss halved -- the epochs were spent polishing
                # log-magnitudes of tiles that do not matter. w_i mixes uniform
                # with each tile's share of its sample's |true dg| mass:
                #   w_i = (1-b) + b * M * |t_i| / sum_j |t_j|      (mean w = 1)
                # b=1 gives pure mass weighting (top-decile tiles ~10x, near-zero
                # tiles ~0); the slog transform is kept for its dynamic range.
                _se = (_slog(pred_grad_at_tiles) - _slog(g_grad_dks_dx)) ** 2
                if grad_err_mode == "rel":
                    # Percent-error mode, no log. Per tile:
                    #   r_i = |pred_i - true_i| / |true_i|        (relative error)
                    #   e_i = r_i / (1 + r_i / sat)               (smooth saturation)
                    #   w_i = (|true_i| / sum_j |true_j|) ** p, normalised to mean 1 per sample
                    # The saturation is there because r explodes on near-zero
                    # tiles: on the 11x11 trajectory, unbounded r gave the worst
                    # 1% of tiles 47% of the loss (errors up to 1e4 = 1,000,000%).
                    # e ~ r for small r and -> sat for large r, with a gradient
                    # that shrinks but never reaches zero (0.028 of full at
                    # r=10 for sat=2; a hard cap gives 0, tanh 0.0002). sat=2 is
                    # exactly an equal-size sign flip. p sets the tile balance:
                    # at p=0.15, sat=2 the |ccx dg| bands bottom-50% / 50-75 /
                    # 75-90 / top-10% took 58 / 19 / 12 / 11% of the loss, vs
                    # 0 / 0 / 0.4 / 99.5% for the old slog + mass-weight 1.0.
                    _t = g_grad_dks_dx.float()
                    _r = (pred_grad_at_tiles.float() - _t).abs() / (_t.abs() + 1e-12)
                    _e = _r / (1.0 + _r / grad_rel_sat)
                    _w = (_t.abs() / _t.abs().sum(dim=1, keepdim=True).clamp_min(1e-30)) ** grad_weight_power
                    _w = _w / _w.mean(dim=1, keepdim=True)
                    grad_loss = (_w * _e).mean()
                elif grad_mass_weight > 0.0:
                    _t = g_grad_dks_dx.float().abs()
                    _w = _t.shape[1] * _t / _t.sum(dim=1, keepdim=True).clamp_min(1e-30)
                    _w = (1.0 - grad_mass_weight) + grad_mass_weight * _w
                    grad_loss = (_w * _se).mean()
                else:
                    grad_loss = _se.mean()
                loss = loss + grad_loss_weight * grad_loss / base_grad

                # Metrics, no gradient: the raw cosine (logged as "cos", kept so
                # runs stay comparable with earlier ones) and the exact 0/1
                # wrong-sign rate over every tile (logged as "sgnerr"). The
                # cosine is NOT a loss any more: every tile's pull on it goes as
                # |dg|^2, so on hollow_cube_11x11 the single largest tile
                # carried 73% of it (removed 2026-10-01).
                with torch.no_grad():
                    grad_cos = (1.0 - torch.nn.functional.cosine_similarity(
                        pred_grad_at_tiles.float(), g_grad_dks_dx.float(), dim=1, eps=1e-12,
                    )).mean()
                    grad_sgnerr = (torch.sign(pred_grad_at_tiles.float())
                                   != torch.sign(g_grad_dks_dx.float())).float().mean()

                # Per-tile smooth SIGN loss, averaged over the tiles of each
                # sample and then over the batch, every tile weighted equally:
                #   z_i    = p_i * sign(t_i) / |t_i|   (signed fraction of the truth)
                #   loss_i = 0.5 * (1 - z_i / (delta + |z_i|))
                # ~0 for a correct sign (0.024 at z=1, delta=0.05), ~1 for a
                # wrong one (0.976 at z=-1): a trainable version of sgnerr. A
                # literal 0/1 step has zero gradient everywhere; this rational
                # form keeps delta/(delta+|z|)^2, which never reaches zero.
                if grad_sign_loss_weight > 0.0:
                    _t = g_grad_dks_dx.float()
                    _z = pred_grad_at_tiles.float() * torch.sign(_t) / (_t.abs() + 1e-12)
                    grad_sign_loss = (0.5 * (1.0 - _z / (grad_sign_delta + _z.abs()))).mean()
                    loss = loss + grad_sign_loss_weight * grad_sign_loss / base_sign

                # Mass-weighted SIGN term. Cosine and the slog MSE both leave
                # the metric OC is most exposed to untouched: the fraction of
                # total |dg| MASS whose sign is wrong. Measured across every
                # run this session, that number sits at 15-22% while a
                # constant-direction control achieves 9.0% -- and it does NOT
                # improve as cosine does (the low-LR run reached cosine +0.828
                # with 18.7% wrong-way mass). Cosine is dominated by the few
                # largest tiles, so a model can align globally while flipping
                # the sign on a fifth of the mass; OC then drives exactly
                # those tiles the wrong way.
                #
                # Hinge on the wrong side only: relu(-pred*sign(true)) is 0
                # wherever the sign already agrees, so this adds no pressure
                # on tiles that are already correct and cannot fight the
                # magnitude term there. Weighted by |true| and normalised per
                # sample, it is a differentiable surrogate for exactly the
                # wrong-way-mass fraction the probe reports.
                if grad_sign_weight > 0.0:
                    _t = g_grad_dks_dx.float()
                    _p = pred_grad_at_tiles.float()
                    _w = _t.abs() / (_t.abs().sum(dim=1, keepdim=True) + 1e-12)
                    # MARGIN hinge, not a plain sign hinge. relu(-p*sign(t)) is
                    # exactly 0 at p=0, so a plain version has a degenerate minimum
                    # at 'predict nothing' -- the same collapse that destroyed the
                    # node-gradient head (job 1374298, predictions 87x too small).
                    # Measured: a zero predictor scored a PERFECT 0.000 under the
                    # plain hinge while the real model scored 0.556, and dividing by
                    # mean|pred| does NOT fix it (0/eps is still 0).
                    #
                    # Requiring pred*sign(true) >= m * mean|true| instead means a
                    # zero prediction costs the full margin m on every tile, while a
                    # correct, confidently-signed one costs nothing. Small tiles sit
                    # below the margin but carry ~zero weight, so they do not drive it.
                    _m = 0.5
                    _s = _t.abs().mean(dim=1, keepdim=True) + 1e-12
                    _margin = torch.relu(_m - _p * torch.sign(_t) / _s)
                    grad_sign = (_w * _margin).sum(dim=1).mean()
                    loss = loss + grad_sign_weight * grad_sign

            if profile_this_epoch:
                _sync(); t_now = time.perf_counter(); t_fwd += t_now - t_mark; t_mark = t_now

            loss.backward()
            if profile_this_epoch:
                _sync(); t_now = time.perf_counter(); t_bwd += t_now - t_mark; t_mark = t_now

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if profile_this_epoch:
                _sync(); t_now = time.perf_counter(); t_step += t_now - t_mark; t_mark = t_now

            total_train       += loss.item()
            total_train_graph += graph_loss.item()
            total_train_node  += node_loss.item()
            if grad_loss is not None:
                total_train_grad += grad_loss.item()
            if grad_cos is not None:
                total_train_gcos += grad_cos.item()
                total_train_sgnerr += grad_sgnerr.item()
                if grad_sign_loss_weight > 0.0:
                    total_train_sgnl += grad_sign_loss.item()
            if node_grad_weight > 0.0:
                total_train_ngrad += node_grad_loss.item()
                if node_grad_cos is not None:
                    total_train_ncos += (1.0 - node_grad_cos).mean().item()
            n_steps += 1

        if profile_this_epoch and n_steps > 0:
            total = t_data + t_h2d + t_fwd + t_bwd + t_step
            print(
                f"[profile] steps={n_steps}  "
                f"data={t_data:6.1f}s ({100*t_data/total:4.1f}%)  "
                f"h2d={t_h2d:6.1f}s ({100*t_h2d/total:4.1f}%)  "
                f"fwd={t_fwd:6.1f}s ({100*t_fwd/total:4.1f}%)  "
                f"bwd={t_bwd:6.1f}s ({100*t_bwd/total:4.1f}%)  "
                f"opt={t_step:6.1f}s ({100*t_step/total:4.1f}%)"
            )

        n_train_batches = max(len(train_loader), 1)
        avg_train       = total_train       / n_train_batches
        avg_train_graph = total_train_graph / n_train_batches
        avg_train_node  = total_train_node  / n_train_batches
        avg_train_grad  = total_train_grad  / n_train_batches
        avg_train_gcos  = total_train_gcos  / n_train_batches
        avg_train_sgnerr = total_train_sgnerr / n_train_batches
        avg_train_sgnl  = total_train_sgnl  / n_train_batches
        avg_train_ngrad = total_train_ngrad / n_train_batches
        avg_train_ncos  = total_train_ncos  / n_train_batches
        train_losses.append(avg_train)
        train_graph_losses.append(avg_train_graph)
        train_node_losses.append(avg_train_node)
        train_grad_losses.append(avg_train_grad)
        train_gcos_losses.append(avg_train_gcos)

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[vram] peak={peak:.2f} GB | reserved={reserved:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        # -----------------------------
        # VALIDATION (rank 0 only). Forward-only, so no *gradient* sync is
        # needed -- but calling the DDP-wrapped `model` here would still be
        # wrong: DDP.forward() unconditionally does its own internal buffer-
        # sync broadcast collective whenever require_forward_param_sync is
        # set (which it is, left True by the training loop's last step),
        # regardless of torch.no_grad(). Since only rank 0 reaches this
        # forward, that stray broadcast has no matching call on the other
        # ranks and collides with whatever collective they call next (was
        # silently corrupting the LR-decay broadcast below on ranks>0, then
        # cascading into a real hang by epoch 2 -- see
        # TORCH_DISTRIBUTED_DEBUG=DETAIL trace, job 1299063). Call the bare
        # module (bypassing DDP.forward()) to avoid triggering it.
        # -----------------------------
        eval_model = model.module if ddp else model
        if is_main:
            eval_model.eval()
            total_val = 0.0
            total_val_graph = 0.0
            total_val_node  = 0.0

            with torch.no_grad():
                for batch_data in val_loader:
                    x, edge_index, edge_attr, batch_idx, tile_idx, tile_NX, tile_NY, tile_NZ, tile_x, node_id = gnn_input_fn(batch_data)
                    num_graphs = batch_data.num_graphs
                    x = x.float().to(device, non_blocking=True)
                    edge_index = edge_index.to(device, non_blocking=True)
                    edge_attr  = edge_attr.float().to(device, non_blocking=True)
                    batch_idx  = batch_idx.to(device, non_blocking=True)
                    if tile_idx is not None:
                        tile_idx = tile_idx.to(device, non_blocking=True)
                        # tile_NX/NY/NZ are already plain Python ints (see gnn_input_fn)
                    if tile_x is not None:
                        tile_x = tile_x.float().to(device, non_blocking=True)
                    if node_id is not None:
                        node_id = node_id.to(device, non_blocking=True)

                    x_norm = (x - x_mean) / x_std
                    edge_attr_norm = (edge_attr - edge_mean) / edge_std

                    targ_mpa = gnn_target_fn(batch_data).float().to(device, non_blocking=True)
                    node_targ = gnn_node_target_fn(batch_data).float().to(device, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        graph_pred, node_pred = eval_model(
                            x_norm, edge_index, edge_attr_norm, batch_idx,
                            tile_idx=tile_idx, tile_NX=tile_NX, tile_NY=tile_NY, tile_NZ=tile_NZ,
                            tile_x=tile_x, num_graphs=num_graphs, node_id=node_id,
                        )
                        graph_pred_raw_mpa = (graph_pred * node_target_std + node_target_mean).clamp_min(LOG_STRESS_EPS)
                        node_pred_raw_mpa = node_pred * node_target_std + node_target_mean
                        graph_loss = ks_rel_err(graph_pred_raw_mpa, targ_mpa)
                        node_loss  = weighted_node_rel_err(node_pred_raw_mpa, node_targ, targ_mpa, batch_idx, stress_weight_power)
                        loss = graph_loss_weight * graph_loss / base_ks + node_loss_weight * node_loss / base_node

                    total_val       += loss.item()
                    total_val_graph += graph_loss.item()
                    total_val_node  += node_loss.item()

            n_val_batches = max(len(val_loader), 1)
            avg_val       = total_val       / n_val_batches
            avg_val_graph = total_val_graph / n_val_batches
            avg_val_node  = total_val_node  / n_val_batches
            val_losses.append(avg_val)
            val_graph_losses.append(avg_val_graph)
            val_node_losses.append(avg_val_node)
        else:
            avg_val = avg_val_graph = avg_val_node = 0.0

        # LR schedule: every epoch decays by lr_decay_per_epoch, plus an extra
        # lr_plateau_factor on top once lr_plateau_patience epochs pass with no
        # *training*-loss improvement (val graph loss is too noisy epoch-to-epoch
        # to gate on -- see comment at best_train_loss init). Open-ended -- no
        # T_max/epoch-horizon needed, so extending a run's --epochs and resuming
        # never requires recomputing a fixed-length schedule. avg_train differs
        # per rank (each rank trains on its own data shard), so the decision is
        # made from rank 0's value and broadcast to the other ranks -- every
        # rank's optimizer must apply the same lr or DDP's gradient sync no
        # longer corresponds to identical updates.
        if is_main:
            if avg_train < best_train_loss:
                best_train_loss = avg_train
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            decay_mult = lr_decay_per_epoch
            if epochs_since_improve >= lr_plateau_patience:
                decay_mult *= lr_plateau_factor
                epochs_since_improve = 0
                print(f"[lr] no train-loss improvement for {lr_plateau_patience} epochs -> "
                      f"extra x{lr_plateau_factor} on top of the x{lr_decay_per_epoch} decay")
        else:
            decay_mult = 0.0  # overwritten by the broadcast below

        if ddp:
            decay_t = torch.tensor([decay_mult], device=device, dtype=torch.float64)
            dist.broadcast(decay_t, src=0)
            decay_mult = decay_t.item()

        for pg in optimizer.param_groups:
            pg["lr"] *= decay_mult

        # All checkpoint/metric writes are rank-0-only: avg_val_graph is a
        # dummy 0.0 on other ranks (they skip validation entirely above), so
        # letting them through here would both race on the same files and
        # corrupt best-checkpoint tracking with a fake zero loss.
        if is_main:
            # -----------------------------
            # Best-checkpoint tracking: whenever val ks_stress/graph loss improves,
            # save straight to data/{geometry}/gnn_surrogate.pt -- the exact path
            # evaluators/gnn_evaluator.py loads -- so "the surrogate" always is
            # the best-val checkpoint, independent of the every-10-epoch cadence
            # below (best epoch isn't necessarily a multiple of 10).
            # -----------------------------
            if avg_val_graph < best_val_graph_loss:
                best_val_graph_loss = avg_val_graph
                best_ckpt = _build_ckpt()
                best_ckpt["best_epoch"] = epoch
                best_ckpt["best_val_graph_loss"] = best_val_graph_loss
                torch.save(best_ckpt, surrogate_path)
                print(f"[best] new best val ks err {100*best_val_graph_loss:.2f}% "
                      f"at epoch {epoch} -> {surrogate_path}")

            # -----------------------------
            # Save checkpoint every 10 epochs
            # -----------------------------
            # Per-epoch model-only snapshot. best-val selection above keys
            # on val ks_stress, which job 1373261 showed is the WRONG
            # criterion for dg: its stress kept improving (0.0508 log MPa at
            # epoch 15) while held-out dg fell below a constant-direction
            # predictor. Snapshotting every epoch lets
            # evaluators/probe_dg_alignment.py pick the best-dg epoch offline,
            # without adding a double-backward validation pass to the loop.
            torch.save(_build_ckpt(), os.path.join(save_dir, f"epoch_{epoch:03d}.pt"))

            if epoch % 10 == 0 or epoch == epochs:
                ckpt = _build_ckpt(include_optimizer=True)
                torch.save(ckpt, os.path.join(save_dir, f"{epoch}_epochs.pt"))

                with open(os.path.join(run_dir, "losses.json"), "w") as f:
                    json.dump({
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "train_graph_losses": train_graph_losses,
                        "val_graph_losses": val_graph_losses,
                        "train_node_losses": train_node_losses,
                        "val_node_losses": val_node_losses,
                        "train_grad_losses": train_grad_losses,
                        "train_gcos_losses": train_gcos_losses,
                    }, f, indent=4)

            epoch_secs = time.time() - epoch_t0
            grad_str = ((f" / grad {100*avg_train_grad:.1f}%" if grad_err_mode == "rel" else f" / grad {avg_train_grad:.6f}")
                        if grad_iter is not None else "")
            if node_grad_weight > 0.0:
                grad_str += f" / ngrad {avg_train_ngrad:.6f}"
                if node_grad_cos_weight > 0.0:
                    grad_str += f" / ncos {1.0 - avg_train_ncos:+.4f}"
            if grad_iter is not None:
                grad_str += f" / cos {1.0 - avg_train_gcos:+.4f} / sgnerr {100*avg_train_sgnerr:.1f}%"
                if grad_sign_loss_weight > 0.0:
                    grad_str += f" / sgnl {avg_train_sgnl:.4f}"
            print(
                f"Epoch {epoch:03d}/{epochs} | {epoch_secs:6.1f}s | "
                f"Train: {avg_train:.4f} (ks {100*avg_train_graph:.2f}% / node {100*avg_train_node:.2f}%{grad_str}) | "
                f"Val: {avg_val:.4f} (ks {100*avg_val_graph:.2f}% / node {100*avg_val_node:.2f}%) | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    if ddp:
        dist.destroy_process_group()


# ======================================================
# Command-Line
# ======================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default="arm", type=str)
    parser.add_argument("--num_samples", default=3000, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--conv_layers", default=6, type=int)
    parser.add_argument("--node_loss_weight", default=1.0, type=float)
    parser.add_argument("--graph_loss_weight", default=1.0, type=float)
    parser.add_argument("--stress_weight_power", default=1.0, type=float,
                        help="Exponent applied to raw node stress before normalizing into "
                             "per-node loss weights (alpha). 0=uniform weighting (plain MSE), "
                             "1=weight proportional to stress (previous default), "
                             ">1=high-stress nodes dominate more.")
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--dataset", default=None, type=str)
    def _bool(s):
        if isinstance(s, bool):
            return s
        return str(s).lower() in ("1", "true", "yes", "y", "on")
    parser.add_argument("--use_checkpoint", type=_bool, default=False,
                        help="Enable gradient checkpointing (trades compute for memory). "
                             "Pass true/false (e.g. --use_checkpoint true).")
    parser.add_argument("--checkpoint_fine_only", type=_bool, default=False,
                        help="If use_checkpoint=true, checkpoint only the fine-mesh "
                             "(encode/decode) layers in HierarchicalGNN. Coarse layers "
                             "run without checkpoint (their activations are tiny).")
    parser.add_argument("--use_node_emb", type=_bool, default=True,
                        help="Enable the per-node learned embedding table for fixed-topology "
                             "geometries. Pass false to ablate it (e.g. --use_node_emb false).")
    parser.add_argument("--coarse_arch", default="cnn", choices=["transformer", "gnn", "cnn"],
                        help="Coarse-stage architecture for the hierarchical model: "
                             "'transformer' = full self-attention over tile+global tokens "
                             "(HierarchicalGNN), 'gnn' = multi-hop message passing over the "
                             "6-connected tile grid (HierarchicalGNNMP, pre-transformer arch), "
                             "'cnn' = a 3D U-Net over the dense tile grid (HierarchicalGNNCNN).")
    parser.add_argument("--transformer_heads", default=4, type=int,
                        help="Attention heads in the coarse tile transformer (HierarchicalGNN only).")
    parser.add_argument("--transformer_ff_mult", default=2, type=int,
                        help="Feedforward width multiplier in the coarse tile transformer.")
    parser.add_argument("--transformer_dropout", default=0.1, type=float,
                        help="Dropout in the coarse tile transformer (attention + FF).")
    parser.add_argument("--mp_dropout", default=0.0, type=float,
                        help="Dropout applied after encoder, each coarse-stage conv "
                             "(before the residual add), and in the head MLP "
                             "(--coarse_arch gnn or cnn only).")
    parser.add_argument("--coarse_dim", default=None, type=int,
                        help="Width of the tile virtual-node embeddings / U-Net channels "
                             "(--coarse_arch gnn or cnn only). Decoupled from "
                             "--hidden_dim so sweeping hidden_dim doesn't shrink coarse-scale "
                             "capacity; projected to hidden_dim at the fine/coarse boundary. "
                             "Defaults to --hidden_dim when unset.")
    parser.add_argument("--cnn_depth", default=3, type=int,
                        help="Number of U-Net down/up levels in the coarse stage "
                             "(--coarse_arch cnn only). Each level halves any tile-grid axis "
                             "that isn't already size 1 (e.g. hollow_cube's NZ=1 stays put), "
                             "so this degrades gracefully to a 2D/1D U-Net on thin geometries.")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="Override the checkpoint save/resume directory "
                             "(default: data/{geometry}/checkpoints/).")
    parser.add_argument("--run_dir", default="training/runs/train", type=str,
                        help="Directory to write losses.json (and read it back on resume).")
    parser.add_argument("--stress_ks_rho_mpa", default=0.15, type=float,
                        help="KS aggregation sharpness (1/MPa) for the graph target/prediction. "
                             "Must match the rho the dataset's ks_stress was built with "
                             "(see training/dataset_builder.py --stress_ks_rho_mpa).")
    parser.add_argument("--use_compile", type=_bool, default=True,
                        help="Wrap the model in torch.compile (fuses the gather/cat/elementwise "
                             "ops around each EdgeAttrConv). No-op on CPU. Pass false to disable "
                             "if compilation fails or is slower for a given config.")
    parser.add_argument("--resume_lr_scale", default=None, type=float,
                        help="Only takes effect when resuming from a checkpoint: multiplies the "
                             "checkpoint's restored lr by this factor once, e.g. 0.5 to halve it. "
                             "Plain --lr on resume otherwise has no effect, since "
                             "optimizer.load_state_dict() restores the checkpoint's own lr. "
                             "No effect on a fresh (non-resumed) run.")
    parser.add_argument("--lr_decay_per_epoch", default=0.99, type=float,
                        help="Multiplicative lr decay applied every epoch (no T_max/epoch-horizon "
                             "needed, so extending --epochs on a resumed run just continues "
                             "decaying from wherever lr left off).")
    parser.add_argument("--lr_plateau_patience", default=5, type=int,
                        help="Epochs of no training-loss improvement before an extra "
                             "--lr_plateau_factor decay is applied on top of the per-epoch decay.")
    parser.add_argument("--lr_plateau_factor", default=0.5, type=float,
                        help="Extra multiplicative lr decay applied once --lr_plateau_patience "
                             "epochs pass with no training-loss improvement.")
    parser.add_argument("--seed_offset", default=42, type=int,
                        help="Per-epoch reshuffle seed is seed_offset+epoch (torch.manual_seed). "
                             "Change this to get a different deterministic batch order/shuffle "
                             "sequence across all epochs -- e.g. to dodge a specific batch shape "
                             "that trips a torch.compile guard/specialization bug at some epoch, "
                             "without disabling --use_compile.")
    parser.add_argument("--node_grad_power", default=1.0, type=float,
                        help="Magnitude-weighting exponent for the node-gradient loss (weight ~ |dKS/dX|^power, normalised per graph, same scheme as --stress_weight_power). 0 = unweighted, which collapses to a zero predictor on this sparse field.")
    parser.add_argument("--node_grad_cos_weight", default=0.0, type=float,
                        help="Weight on 1-cos(pred,true) over each sample's node-gradient field. Scale-invariant, so unlike the MSE term it cannot be satisfied by shrinking predictions toward zero.")
    parser.add_argument("--node_grad_weight", default=0.0, type=float,
                        help="Weight on the per-node dKS/dX head (signed-log MSE). Composed to per-tile dg at inference through the EXACT CAD Jacobian, so the net never learns geometry. Rides the main batch: no double backward, no separate grad pool. coarse_arch=cnn only. 0 disables it.")
    parser.add_argument("--grad_from_train", default=False, type=_bool,
                        help="Draw the gradient-loss pool from the loaded train shards "
                             "(~40k gradients) instead of --grad_dataset_path. The old "
                             "gradsub file is a 1-in-10 symlink subsample of the SAME "
                             "shards (4,100 samples), which memorised by ~epoch 5.")
    parser.add_argument("--grad_dataset_path", default=None, type=str,
                        help="Path to a gradient-matching dataset built with "
                             "training/dataset_builder.py --grad_mode true, e.g. "
                             "data/hollow_cube/dataset_grad/dataset.pt (nominal filename -- "
                             "actual sharded data lives under .../dataset_grad/shards/, same "
                             "convention as --dataset, see training/dataset_io.py). Adds a "
                             "Sobolev-style loss term "
                             "that matches the surrogate's own autodiff sensitivity "
                             "d(ks_stress)/d(tile thickness) against ground-truth FD sensitivities "
                             "from the real solver. Unset (default) -> feature fully disabled, "
                             "no behavior change.")
    parser.add_argument("--grad_sign_weight", default=0.0, type=float,
                        help="Weight on a mass-weighted sign hinge over the per-tile dg: relu(-pred*sign(true)) weighted by |true| and normalised per sample. A differentiable surrogate for the wrong-way-|dg|-mass fraction, the one metric that has not improved while cosine did. 0 disables it.")
    parser.add_argument("--grad_mass_weight", default=0.0, type=float,
                        help="Blend the per-tile slog gradient MSE toward |true dg| mass weighting: "
                             "0 = uniform (old behaviour), 1 = each tile weighted by its share of the "
                             "sample's gradient mass (near-zero tiles ~0, top decile ~10x).")
    parser.add_argument("--grad_sign_loss_weight", default=0.0, type=float,
                        help="Weight on the per-tile smooth wrong-sign loss 0.5*(1 - z/(delta+|z|)), "
                             "z = pred*sign(true)/|true|, averaged over all tiles (~0 right sign, ~1 wrong). "
                             "0 disables it. Not the old mass-weighted hinge, which is --grad_sign_weight.")
    parser.add_argument("--grad_sign_delta", default=0.05, type=float,
                        help="Sign loss sharpness. Smaller = closer to a 0/1 step (0.024 at a perfect "
                             "tile and 0.976 at an exact flip for 0.05).")
    parser.add_argument("--grad_err_mode", default="slog", choices=["slog", "rel"],
                        help="Per-tile gradient error. slog: signed-log MSE (+ --grad_mass_weight). "
                             "rel: saturated percent error r/(1+r/--grad_rel_sat), r=|pred-true|/|true|, "
                             "weighted by (|true|/sum|true|)^--grad_weight_power. No log.")
    parser.add_argument("--grad_weight_power", default=0.15, type=float,
                        help="rel mode: exponent p on each tile's share of the sample's |dg| mass. "
                             "0 = every tile equal, 1 = pure mass weighting.")
    parser.add_argument("--grad_rel_sat", default=2.0, type=float,
                        help="rel mode: percent error saturates toward this (2.0 = 200%%, an equal-size sign flip).")
    parser.add_argument("--grad_loss_weight", default=0.1, type=float,
                        help="Weight on the gradient-matching loss term (only used when "
                             "--grad_dataset_path is set). Start conservative -- this is a new, "
                             "previously-untrained signal.")
    parser.add_argument("--grad_batch_size", default=None, type=int,
                        help="Batch size for the gradient-matching dataset loader (only used when "
                             "--grad_dataset_path is set). Defaults to --batch_size.")
    args = parser.parse_args()

    if int(os.environ.get("RANK", "0")) == 0:
        print("[args] " + json.dumps(vars(args), indent=2, sort_keys=True))

    train_gnn_model(
        geometry=args.geometry,
        num_samples=args.num_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        conv_layers=args.conv_layers,
        node_loss_weight=args.node_loss_weight,
        graph_loss_weight=args.graph_loss_weight,
        stress_weight_power=args.stress_weight_power,
        weight_decay=args.weight_decay,
        dataset_path=args.dataset,
        use_node_emb=args.use_node_emb,
        use_checkpoint=args.use_checkpoint,
        checkpoint_fine_only=args.checkpoint_fine_only,
        coarse_arch=args.coarse_arch,
        transformer_heads=args.transformer_heads,
        transformer_ff_mult=args.transformer_ff_mult,
        transformer_dropout=args.transformer_dropout,
        mp_dropout=args.mp_dropout,
        coarse_dim=args.coarse_dim,
        cnn_depth=args.cnn_depth,
        checkpoint_dir=args.checkpoint_dir,
        run_dir=args.run_dir,
        stress_ks_rho_mpa=args.stress_ks_rho_mpa,
        use_compile=args.use_compile,
        resume_lr_scale=args.resume_lr_scale,
        lr_decay_per_epoch=args.lr_decay_per_epoch,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        seed_offset=args.seed_offset,
        grad_dataset_path=args.grad_dataset_path,
        grad_from_train=args.grad_from_train,
        node_grad_weight=args.node_grad_weight,
        node_grad_power=args.node_grad_power,
        node_grad_cos_weight=args.node_grad_cos_weight,
        grad_loss_weight=args.grad_loss_weight,
        grad_sign_loss_weight=args.grad_sign_loss_weight,
        grad_err_mode=args.grad_err_mode,
        grad_sign_delta=args.grad_sign_delta,
        grad_weight_power=args.grad_weight_power,
        grad_rel_sat=args.grad_rel_sat,
        grad_sign_weight=args.grad_sign_weight,
        grad_mass_weight=args.grad_mass_weight,
        grad_batch_size=args.grad_batch_size,
    )
