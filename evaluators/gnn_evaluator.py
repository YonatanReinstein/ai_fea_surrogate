import torch
from abc import ABC
from utils.gnn_surrogate import GNN
from evaluators.base_evaluator import BaseEvaluator
from core.IritModel import IIritModel
from core.component import Component
from training.gnn_training import gnn_input_fn, gnn_target_fn
import json
from multiprocessing import Pool
from torch_geometric.loader import DataLoader



def _run_sample_worker(args):
    (
        geometry_name,
        model_path,
        dims,
        young,
        poisson,
        sample_index
    ) = args
    cad_model = IIritModel(model_path, dims_dict=dims)
    component = Component(cad_model, young, poisson)
    import importlib
    module = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
    anchor_condition = module.anchor_condition
    force_pattern = module.force_pattern
    mesh_resolution = module.mesh_resolution
    U, V, W = mesh_resolution()
    component.generate_mesh(U=U, V=V, W=W)
    component.mesh.anchor_nodes_by_condition(anchor_condition)
    component.mesh.apply_force_by_pattern(force_pattern)
    data = component.to_graph_with_labels(with_labels=False)  # No labels during evaluation
    save_path=f"screenshots/mesh_{sample_index+1}.png"
    component.mesh.plot_mesh(save_path=save_path)
    return data


def _run_sample_worker(args):
    (
        geometry_name,
        model_path,
        dims,
        young,
        poisson,
        sample_index,
        screenshot
    ) = args
    cad_model = IIritModel(model_path, dims_dict=dims)
    component = Component(cad_model, young, poisson)
    import importlib
    module = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
    anchor_condition = module.anchor_condition
    force_pattern = module.force_pattern
    mesh_resolution = module.mesh_resolution
    U, V, W = mesh_resolution()
    component.generate_mesh(U=U, V=V, W=W)
    component.mesh.anchor_nodes_by_condition(anchor_condition)
    component.mesh.apply_force_by_pattern(force_pattern)
    data = component.to_graph_with_labels(with_labels=False)  
    if screenshot:
        save_path=f"screenshots/mesh_{sample_index+1}.png"
        component.mesh.plot_mesh(save_path=save_path)
    volume = component.get_volume()
    return data, volume


class GNNEvaluator(BaseEvaluator):
    def __init__(self, geometry_name: str,screenshots: bool = False, processes: int = None):
        super().__init__(geometry_name)
        self.processes = processes
        self.screenshots = screenshots

        ckpt_path = f"data/{geometry_name}/gnn_surrogate.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.glob_mean = ckpt["targets_mean"].float()
        self.glob_std  = ckpt["targets_std"].float()

        self.node_in_dim = ckpt["node_in_dim"]

        self.model = GNN(
            node_in_dim=self.node_in_dim,
            hidden_dim=128,
            num_layers=6
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.sample_counter = 0


    def evaluate(self, dims_list: list[dict]):
        batch_size = len(dims_list)

        model_path = f"data/{self.geometry_name}/CAD_model/model.irt"
        props = json.load(open(f"data/{self.geometry_name}/CAD_model/material_properties.json"))

        young = props["young_modulus"]
        poisson = props["poisson_ratio"]

        # Unique screenshot indexes
        indexes = list(range(self.sample_counter, self.sample_counter + batch_size))
        self.sample_counter += batch_size

        # Build args
        all_args = [
            (self.geometry_name, model_path, dims, young, poisson, idx , self.screenshots)
            for dims, idx in zip(dims_list, indexes)
        ]

        # Run workers
        if self.processes is None:
            with Pool(processes=1) as pool:
                results = pool.map(_run_sample_worker, all_args)
        else:
            with Pool(processes=self.processes) as pool:
                results = pool.map(_run_sample_worker, all_args)

        # Unpack
        graph_list, volume_list = zip(*results)

        # Build DataLoader
        loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False)
        batch_data = next(iter(loader))

        # Prepare inputs
        x, edge_index, batch = gnn_input_fn(batch_data)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)

        # Predict
        with torch.no_grad():
            pred_norm = self.model(x, edge_index, batch)

        # Denormalize
        stress = pred_norm * self.glob_std + self.glob_mean
        stress = stress.squeeze()

        return {
            "stress": stress,        # shape [batch_size]
            "volume": volume_list    # list of floats
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
    results = evaluator.evaluate([dims_example, dims_example])
    print(results)