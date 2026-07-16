import os
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from abc import ABC
from utils.gnn_surrogate import GNN, HierarchicalGNN
from evaluators.base_evaluator import BaseEvaluator
from core.IritModel import IritCModel
from core.component import Component
from training.gnn_training import gnn_input_fn, gnn_target_fn
import json
from multiprocessing import Pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
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
                 heatmap: bool = False, heatmap_dir: str = "screenshots"):
        super().__init__(geometry_name)
        self.processes = processes
        self.screenshots = screenshots
        self.batch_size = batch_size
        # When enabled, evaluate() also renders a per-sample stress heatmap
        # (Ansys-style nodal von Mises coloring) driven by the GNN's
        # node-level predictions instead of a real FEA solve.
        self.heatmap = heatmap
        self.heatmap_dir = heatmap_dir
        if self.heatmap:
            os.makedirs(self.heatmap_dir, exist_ok=True)

        ckpt_path = f"data/{geometry_name}/gnn_surrogate.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # graph_pred = max(node_pred) lives in node-normalized space
        self.target_mean = ckpt["node_target_mean"].to(self.device)
        self.target_std  = ckpt["node_target_std"].to(self.device)
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
        ModelClass = HierarchicalGNN if hierarchical else GNN
        model_kwargs = dict(
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        if hierarchical:
            model_kwargs["num_pos_nodes"] = num_pos_nodes
            model_kwargs["transformer_heads"] = ckpt["transformer_heads"]
            model_kwargs["transformer_ff_mult"] = ckpt["transformer_ff_mult"]
        self.model = ModelClass(**model_kwargs).to(self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

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
                self.heatmap,
                self.anchor_condition,
                self.force_pattern,
                self.U,
                self.V,
                self.W,
                self._get_tile_grid(dims) if self._get_tile_grid is not None else None,
                self.graph_path,
            )
            for dims, idx in zip(dims_list, indexes)
        ]
        t_build_args = perf_counter() - t
        results = []


        t = perf_counter()
        it = self.pool.imap(_run_sample_worker, all_args)
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

                results.append(result)
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Terminating pool.")
            self.pool.terminate()
            self.pool = None
        t_pool = perf_counter() - t

        graph_dicts, volume_list, worker_timings, mesh_list = zip(*results)
        t = perf_counter()
        graph_list = [
            Data(**{k: torch.from_numpy(v) for k, v in d.items()})
            for d in graph_dicts
        ]
        # Attach shared frozen-graph stable node id (broadcast onto each sample).
        if self.node_id is not None:
            for g in graph_list:
                if g.x.shape[0] == self.node_id.shape[0]:
                    g.node_id = self.node_id
        t_graph_build = perf_counter() - t

        # Build DataLoader
        all_stress = []
        sample_offset = 0  # running count of graphs consumed, to index into indexes/mesh_list

        t = perf_counter()
        loader = DataLoader(graph_list, batch_size=self.batch_size, shuffle=False)
        for batch_data in loader:
            # Prepare inputs
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
            tile_x   = getattr(batch_data, 'tile_x',   None)
            if tile_idx is not None:
                tile_idx = tile_idx.to(self.device)
                tile_NX  = tile_NX.to(self.device)
                tile_NY  = tile_NY.to(self.device)
                tile_NZ  = tile_NZ.to(self.device)
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
            stress = stress.squeeze()
            all_stress.extend(stress.detach().cpu().numpy().tolist())

            if self.heatmap:
                # node_pred lives in the same normalized space as graph_pred
                # (graph_pred = max(node_pred)), so it denormalizes the same way.
                node_stress = (node_pred * self.target_std + self.target_mean).squeeze(-1).detach().cpu()
                batch_cpu = batch.detach().cpu()
                num_graphs = int(batch_data.num_graphs)
                counts = torch.bincount(batch_cpu, minlength=num_graphs).tolist()
                start = 0
                for g in range(num_graphs):
                    n = counts[g]
                    local_stress = node_stress[start:start + n]
                    start += n
                    sample_idx = indexes[sample_offset + g]
                    mesh = mesh_list[sample_offset + g]
                    # Node i in the graph (0-based) is mesh node id i+1 — see
                    # to_graph_with_labels / to_graph_fixed node ordering.
                    stress_dict = {i + 1: float(local_stress[i]) for i in range(n)}
                    save_path = os.path.join(self.heatmap_dir, f"heatmap_{sample_idx + 1}.png")
                    mesh.plot_stress_heatmap(stress_dict, save_path=save_path)
                sample_offset += num_graphs
        t_inference = perf_counter() - t

        # ---- timing report ----
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
            f"build_args={t_build_args:.2f}s pool={t_pool:.2f}s "
            f"graph_build={t_graph_build:.2f}s inference={t_inference:.2f}s",
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
            "yield_strength": self.yield_strength
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