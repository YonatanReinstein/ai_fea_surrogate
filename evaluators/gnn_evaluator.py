import os
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from abc import ABC
from utils.gnn_surrogate import GNN, HierarchicalGNN
from utils.gnn_hierarchical_mp import HierarchicalGNNMP
from utils.gnn_hierarchical_cnn import HierarchicalGNNCNN
from evaluators.base_evaluator import BaseEvaluator
from core.IritModel import IritCModel
from core.component import Component
from training.gnn_training import gnn_input_fn, gnn_target_fn
import json
from multiprocessing import Pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import importlib
import threading
from time import sleep, perf_counter
import socket

STOP = False

# Per-process cache of the frozen-graph constants (edge_index, tile_idx). Loaded
# once per persistent worker from graph.pt and reused for every sample, so the
# 206k-edge rebuild in to_graph_with_labels is skipped on fixed-topology runs.
_FIXED_GRAPH_CACHE = {}


def _get_fixed_graph(graph_path):
    g = _FIXED_GRAPH_CACHE.get(graph_path)
    if g is None:
        gg = torch.load(graph_path, weights_only=False)
        g = (gg["edge_index"].long(), gg["tile_idx"].long())
        _FIXED_GRAPH_CACHE[graph_path] = g
    return g


def listener():
    global STOP
    s = socket.socket()
    s.bind(("127.0.0.1", 5001))
    s.listen(1)
    conn, addr = s.accept()
    msg = conn.recv(16)
    if msg == b"STOP":
        STOP = True
    conn.close()
    s.close()





def _run_sample_worker(args):
    (
        model_path,
        dims,
        fixed_dims,
        young,
        poisson,
        sample_index,
        screenshot,
        heatmap,
        anchor_condition,
        force_pattern,
        U,
        V,
        W,
        tile_grid,      # (NX, NY, NZ) tuple or None
        graph_path,     # path to frozen graph.pt, or None for generic rebuild
    ) = args
    import random
    timings = {}
    t = perf_counter()
    cad_model = IritCModel(model_path, dims_dict=dims, fixed_dims=fixed_dims)
    component = Component(cad_model, young, poisson)
    timings["cad_model"] = perf_counter() - t

    t = perf_counter()
    component.generate_mesh(U=U, V=V, W=W)
    timings["generate_mesh"] = perf_counter() - t

    t = perf_counter()
    component.mesh.anchor_nodes_by_condition(anchor_condition)
    component.mesh.apply_force_by_pattern(force_pattern)
    timings["boundary_conditions"] = perf_counter() - t

    t = perf_counter()
    fixed = None
    if graph_path is not None and os.path.exists(graph_path):
        edge_index, tile_idx = _get_fixed_graph(graph_path)
        # Guard: only take the fast path if this sample's mesh matches the
        # frozen topology; otherwise fall back to a full rebuild.
        if len(component.mesh.nodes) == tile_idx.shape[0]:
            fixed = component.to_graph_fixed(edge_index, tile_idx, tile_grid)
    data = fixed if fixed is not None else \
        component.to_graph_with_labels(with_labels=False, tile_grid=tile_grid)
    timings["to_graph"] = perf_counter() - t
    if tile_grid is not None and hasattr(data, 'tile_idx'):
        NX, NY, NZ = tile_grid
        K = NX * NY * NZ
        strut_params = torch.tensor(
            [dims.get(f"d{i + 1}", 0.0) for i in range(K)], dtype=torch.float
        )
        strut_feat = strut_params[data.tile_idx].unsqueeze(1)
        data.x = torch.cat([data.x, strut_feat], dim=1)
        data.tile_x = strut_params.view(-1, 1)
    if screenshot:
        save_path = f"screenshots/mesh_{sample_index+1}.png"
        component.mesh.plot_mesh(save_path=save_path)
    t = perf_counter()
    volume = component.get_volume()
    timings["get_volume"] = perf_counter() - t
    data_dict = {k: v.numpy() for k, v in data.items() if hasattr(v, 'numpy')}
    # Only pay the pickling cost of the mesh (nodes/elements) when a heatmap
    # was actually requested; node prediction happens later in the main
    # process (batched inference), so the mesh has to travel back with it.
    mesh_obj = component.mesh if heatmap else None
    return data_dict, volume, timings, mesh_obj


class GNNEvaluator(BaseEvaluator):
    def __init__(self, geometry_name: str, screenshots: bool = False, processes: int = None, batch_size: int = 128,
                 heatmap: bool = False, heatmap_dir: str = "screenshots", heatmap_base_only: bool = False,
                 ckpt_path: str = None):
        super().__init__(geometry_name)
        self.processes = processes
        self.screenshots = screenshots
        self.batch_size = batch_size
        # When enabled, evaluate() also renders a per-sample stress heatmap
        # (Ansys-style nodal von Mises coloring) driven by the GNN's
        # node-level predictions instead of a real FEA solve.
        self.heatmap = heatmap
        self.heatmap_dir = heatmap_dir
        # When enabled (without the full per-sample `heatmap` above), evaluate()
        # returns the mesh + node-stress dict for dims_list[0] only -- callers
        # like TopologyOptimizer always put the current/base design at index 0,
        # so this rides along on the evaluation that already happens every
        # generation instead of triggering a second solve/inference just for a picture.
        self.heatmap_base_only = heatmap_base_only
        if self.heatmap:
            os.makedirs(self.heatmap_dir, exist_ok=True)

        ckpt_path = ckpt_path or f"data/{geometry_name}/gnn_surrogate.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        if self.device.type == "cpu" and self.processes:
            # OMP_NUM_THREADS=1 in the job env keeps each meshing worker
            # single-threaded so `processes`-many of them don't oversubscribe
            # the node. That same env var would otherwise also pin this (main)
            # process's own torch calls to 1 thread -- but inference here runs
            # after the worker pool is done for the generation, so the workers
            # are idle and this process can safely claim their cores instead.
            torch.set_num_threads(self.processes)

        # graph_pred = KS-aggregate(node_pred) (see utils/gnn_surrogate.py:ks_pool),
        # matching core/mesh.py:get_aggregated_stress -- not the hard max.
        self.target_mean = ckpt["node_target_mean"].to(self.device)
        self.target_std  = ckpt["node_target_std"].to(self.device)
        self.stress_ks_rho_mpa = ckpt.get("stress_ks_rho_mpa", 0.15)
        self.x_mean = ckpt["x_mean"].to(self.device)
        self.x_std  = ckpt["x_std"].to(self.device)
        self.edge_mean = ckpt["edge_mean"].to(self.device)
        self.edge_std  = ckpt["edge_std"].to(self.device)

        self.node_in_dim = ckpt["node_in_dim"]
        self.edge_in_dim = ckpt.get("edge_in_dim", 1)
        num_layers = ckpt["conv_layers"]
        hidden_dim = ckpt.get("hidden_dim") or ckpt["model_state"]["encoder.lins.0.bias"].shape[0]

        # Fixed-topology exploit: per-node embedding (0 -> disabled).
        num_pos_nodes = ckpt.get("num_pos_nodes", 0)

        hierarchical = ckpt.get("hierarchical", False)
        coarse_arch = ckpt.get("coarse_arch", "transformer")
        if hierarchical:
            ModelClass = {"transformer": HierarchicalGNN, "gnn": HierarchicalGNNMP,
                          "cnn": HierarchicalGNNCNN}[coarse_arch]
        else:
            ModelClass = GNN
        model_kwargs = dict(
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            node_target_mean=self.target_mean,
            node_target_std=self.target_std,
            stress_ks_rho_mpa=self.stress_ks_rho_mpa,
        )
        if hierarchical:
            model_kwargs["num_pos_nodes"] = num_pos_nodes
            if coarse_arch == "transformer":
                model_kwargs["transformer_heads"] = ckpt["transformer_heads"]
                model_kwargs["transformer_ff_mult"] = ckpt["transformer_ff_mult"]
            elif coarse_arch == "gnn":
                model_kwargs["coarse_dim"] = ckpt.get("coarse_dim")
            else:  # cnn
                model_kwargs["coarse_dim"] = ckpt.get("coarse_dim")
                model_kwargs["cnn_depth"] = ckpt.get("cnn_depth", 3)
        self.model = ModelClass(**model_kwargs).to(self.device)

        # Frozen-graph fast path: if graph.pt exists, workers reuse its constant
        # edge_index/tile_idx instead of rebuilding the graph per sample.
        graph_path = f"data/{self.geometry_name}/dataset/graph.pt"
        self.graph_path = graph_path if os.path.exists(graph_path) else None

        # Shared frozen-graph artifact: stable per-node id for the node embedding.
        self.node_id = None
        if num_pos_nodes and self.graph_path is not None:
            g = torch.load(graph_path, weights_only=False)
            self.node_id = torch.arange(g["num_nodes"], dtype=torch.long)

        self.sample_counter = 0

        material_properties_path = f"data/{self.geometry_name}/CAD_model/material_properties.json"
        material_properties = json.loads(open(material_properties_path, "r").read())
        self.young = material_properties["young_modulus"]
        self.poisson = material_properties["poisson_ratio"]
        self.yield_strength = material_properties["yield_strength"]
        model_dir = f"data/{self.geometry_name}/CAD_model"
        if os.path.exists(f"{model_dir}/model.irt"):
            self.model_path = f"{model_dir}/model.irt"
        else:
            self.model_path = f"{model_dir}/model"

        module = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
        self.anchor_condition = module.anchor_condition
        self.force_pattern = module.force_pattern
        self.mesh_resolution = module.mesh_resolution
        self.U, self.V, self.W = self.mesh_resolution()
        self._get_tile_grid = getattr(module, "tile_grid", None)
        self.fixed_dims = getattr(module, "fixed_dims", lambda: {})()

        if hierarchical:
            # Fixed-topology: precompute the tile grid once (see gnn_training.py).
            # Must happen before load_state_dict: HierarchicalGNNCNN lazily builds
            # its U-Net down/bottleneck/up submodules inside set_tile_grid (their
            # shapes depend on the grid dims), so those keys don't exist in the
            # freshly-constructed model until this runs.
            NX, NY, NZ = self._get_tile_grid(None)
            self.model.set_tile_grid(NX, NY, NZ)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.listener_thread = threading.Thread(target=listener, daemon=True).start()

        # Persistent worker pool: spawn the workers (and pay the heavy torch/irit
        # re-import cost under 'spawn') exactly once, then reuse across every
        # generation. Re-creating the pool per evaluate() dominated wall time.
        self.pool = Pool(processes=self.processes)



    def evaluate(self, dims_list: list[dict]):
        #print("start evaluation of batch size:", len(dims_list))
        batch_size = len(dims_list)
        t_eval_start = perf_counter()


        # Unique screenshot indexes
        indexes = list(range(self.sample_counter, self.sample_counter + batch_size))
        self.sample_counter += batch_size

        # Build args
        t = perf_counter()
        all_args = [
            (
                self.model_path,
                dims,
                self.fixed_dims,
                self.young,
                self.poisson,
                idx,
                self.screenshots,
                self.heatmap or (self.heatmap_base_only and i == 0),
                self.anchor_condition,
                self.force_pattern,
                self.U,
                self.V,
                self.W,
                self._get_tile_grid(dims) if self._get_tile_grid is not None else None,
                self.graph_path,
            )
            for i, (dims, idx) in enumerate(zip(dims_list, indexes))
        ]
        t_build_args = perf_counter() - t

        # Pipelined pool+inference: fire GPU inference on each mini-batch of
        # self.batch_size samples as soon as the pool has meshed that many,
        # instead of waiting for all `batch_size` samples to finish meshing
        # first. pool.imap keeps worker processes pulling the next task
        # immediately after finishing the previous one regardless of how fast
        # this loop drains completed results, so CPU meshing for the next
        # mini-batch proceeds in the background while the GPU forward pass
        # below runs on the current one -- wall time collapses toward
        # max(mesh_time, inference_time) instead of their sum.
        volume_list = [None] * batch_size
        worker_timings = [None] * batch_size
        all_stress = [None] * batch_size
        base_heatmap = None
        t_inference_total = 0.0

        def run_inference_chunk(chunk):
            nonlocal base_heatmap, t_inference_total
            t_chunk = perf_counter()
            graph_list = []
            for _local_idx, graph_dict, _mesh in chunk:
                g = Data(**{k: torch.from_numpy(v) for k, v in graph_dict.items()})
                if self.node_id is not None and g.x.shape[0] == self.node_id.shape[0]:
                    g.node_id = self.node_id
                graph_list.append(g)
            batch_data = Batch.from_data_list(graph_list)

            x, edge_index, edge_attr, batch, _ti, _nx, _ny, _nz, _tx, node_id = gnn_input_fn(batch_data)
            x[:, 3:6] = x[:, 3:6] / 1e+6   # scale force vector (Fx,Fy,Fz) to match training (gnn_training.py:95)

            x = x.to(self.device)
            edge_attr = edge_attr.float().to(self.device)

            x = (x - self.x_mean) / self.x_std
            edge_attr = (edge_attr - self.edge_mean) / self.edge_std

            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long)

            edge_index = edge_index.to(self.device)
            batch = batch.to(self.device)

            # Predict
            tile_idx = getattr(batch_data, 'tile_idx', None)
            tile_NX  = getattr(batch_data, 'tile_NX',  None)
            tile_NY  = getattr(batch_data, 'tile_NY',  None)
            tile_NZ  = getattr(batch_data, 'tile_NZ',  None)
            # Fixed-topology: NX/NY/NZ are identical for every sample, so pass
            # plain Python ints rather than GPU tensors (see gnn_training.py's
            # gnn_input_fn / gnn_hierarchical_mp.py for why).
            tile_NX = int(tile_NX[0]) if tile_NX is not None else None
            tile_NY = int(tile_NY[0]) if tile_NY is not None else None
            tile_NZ = int(tile_NZ[0]) if tile_NZ is not None else None
            tile_x   = getattr(batch_data, 'tile_x',   None)
            if tile_idx is not None:
                tile_idx = tile_idx.to(self.device)
            if tile_x is not None:
                tile_x = tile_x.float().to(self.device)
            if node_id is not None:
                node_id = node_id.to(self.device)

            with torch.inference_mode():
                graph_pred, node_pred = self.model(
                    x, edge_index, edge_attr, batch,
                    tile_idx=tile_idx, tile_NX=tile_NX, tile_NY=tile_NY, tile_NZ=tile_NZ,
                    tile_x=tile_x, node_id=node_id,
                )

            # Denormalize
            stress = graph_pred * self.target_std + self.target_mean
            stress = stress.squeeze(-1)
            chunk_stress = stress.detach().cpu().numpy().tolist()
            for (local_idx, _graph_dict, _mesh), s in zip(chunk, chunk_stress):
                all_stress[local_idx] = s

            if self.heatmap or self.heatmap_base_only:
                # node_pred lives in node-normalized space (the same space
                # graph_pred is computed FROM before ks_pool denormalizes
                # internally -- see utils/gnn_surrogate.py:ks_pool), so it
                # denormalizes the same way via target_mean/target_std here.
                node_stress = (node_pred * self.target_std + self.target_mean).squeeze(-1).detach().cpu()
                batch_cpu = batch.detach().cpu()
                num_graphs = int(batch_data.num_graphs)
                counts = torch.bincount(batch_cpu, minlength=num_graphs).tolist()
                start = 0
                for g in range(num_graphs):
                    n = counts[g]
                    local_stress = node_stress[start:start + n]
                    start += n
                    local_idx, _graph_dict, mesh = chunk[g]
                    sample_idx = indexes[local_idx]
                    # Node i in the graph (0-based) is mesh node id i+1 — see
                    # to_graph_with_labels / to_graph_fixed node ordering.
                    stress_dict = {i + 1: float(local_stress[i]) for i in range(n)}
                    if self.heatmap:
                        save_path = os.path.join(self.heatmap_dir, f"heatmap_{sample_idx + 1}.png")
                        mesh.plot_stress_heatmap(stress_dict, save_path=save_path)
                    if self.heatmap_base_only and local_idx == 0:
                        base_heatmap = {"mesh": mesh, "stress": stress_dict}
            t_inference_total += perf_counter() - t_chunk

        t = perf_counter()
        it = self.pool.imap(_run_sample_worker, all_args)
        chunk = []
        local_idx = 0
        try:
            for result in it:
                if STOP:
                    print("Graceful stop requested.")

                    # -------- KEY PART --------
                    self.pool.close()
                    sleep(3)      # do not accept new tasks
                    self.pool.terminate()    # kill worker process
                    # -------------------------
                    print("Stopped during evaluation.")
                    self.pool.join()
                    self.pool = None

                    break

                graph_dict, volume, worker_timing, mesh = result
                volume_list[local_idx] = volume
                worker_timings[local_idx] = worker_timing
                chunk.append((local_idx, graph_dict, mesh))
                local_idx += 1

                if len(chunk) >= self.batch_size:
                    run_inference_chunk(chunk)
                    chunk = []
            if chunk:
                run_inference_chunk(chunk)
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Terminating pool.")
            self.pool.terminate()
            self.pool = None
        t_pipeline = perf_counter() - t

        # ---- timing report ----
        worker_timings = [wt for wt in worker_timings if wt is not None]
        n = len(worker_timings)
        worker_avg = {
            k: sum(wt[k] for wt in worker_timings) / n
            for k in worker_timings[0]
        }
        worker_total = sum(sum(wt.values()) for wt in worker_timings)
        t_eval_total = perf_counter() - t_eval_start
        print(
            f"[GNNEvaluator] batch={batch_size} processes={self.processes} "
            f"total={t_eval_total:.2f}s | "
            f"build_args={t_build_args:.2f}s pipeline={t_pipeline:.2f}s "
            f"(inference={t_inference_total:.2f}s of that, overlapped with meshing)",
            flush=True,
        )
        print(
            f"[GNNEvaluator]   per-sample avg (in worker): "
            + " ".join(f"{k}={v:.3f}s" for k, v in worker_avg.items())
            + f" | cpu-sum={worker_total:.2f}s",
            flush=True,
        )

        return {
            "stress": all_stress,        # shape [batch_size]
            "volume": volume_list,    # list of floats
            "yield_strength": self.yield_strength,
            "base_heatmap": base_heatmap,  # {"mesh", "stress"} for dims_list[0], or None
        }



if __name__ == "__main__":
    evaluator = GNNEvaluator("arm")
    dims_example = {
        "d1": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d2": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d3": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d4": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d5": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d6": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d7": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d8": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d9": {"default": 3.0, "min": 0.0, "max": 3.0},
        "d10": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d11": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d12": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d13": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d14": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d15": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d16": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d17": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d18": {"default": 3.0, "min": 0.0, "max": 3.0},
        "d19": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d20": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d21": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d22": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d23": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d24": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d25": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d26": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d27": {"default": 3.0, "min": 0.0, "max": 3.0},
        "d28": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d29": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d30": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d31": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d32": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d33": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d34": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d35": {"default": 1.5, "min": 0.0, "max": 1.5},
        "d36": {"default": 3.0, "min": 0.0, "max": 3.0}
    }
    results = evaluator.evaluate([dims_example])#, dims_example])
    print(results)