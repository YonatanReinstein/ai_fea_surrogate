import torch
from abc import ABC
from utils.gnn_surrogate import GNN
from evaluators.base_evaluator import BaseEvaluator
from core.IritModel import IIritModel
from core.component import Component
from training.gnn_training import gnn_input_fn, gnn_target_fn
import json
        
class GNNEvaluator(BaseEvaluator):


    def __init__(self, geometry_name: str):
        super().__init__(geometry_name)
        checkpoint_path = f"data/{self.geometry_name}/surrogates/gnn_surrogate_1000.pt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Normalization
        self.glob_mean = ckpt["glob_mean"].float()
        self.glob_std  = ckpt["glob_std"].float()
        self.i = 0

        # Architecture info
        self.node_in_dim = ckpt["node_in_dim"]
        self.out_features_global = ckpt["out_global"]

        # Build GNN
        self.model = GNN(
            node_in_dim=self.node_in_dim,
            hidden_dim=256,
            num_layers=4
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()


    def evaluate(self, dims: dict):

        model_path = f"data/{self.geometry_name}/CAD_model/model.irt"
        cad_model = IIritModel(model_path, dims_dict=dims)

        material_props_path = f"data/{self.geometry_name}/CAD_model/material_properties.json"
        with open(material_props_path, "r") as f:
            material_properties = json.load(f)
        component = Component(cad_model, young=material_properties["young_modulus"], poisson=material_properties["poisson_ratio"])

        import importlib
        module = importlib.import_module(f"data.{self.geometry_name}.boundary_conditions")
        anchor_condition = module.anchor_condition
        force_pattern = module.force_pattern
        mesh_resolution = module.mesh_resolution
        U, V, W = mesh_resolution()
        component.generate_mesh(U=U, V=V, W=W)
        component.mesh.anchor_nodes_by_condition(anchor_condition)
        component.mesh.apply_force_by_pattern(force_pattern)
        data = component.to_graph_with_labels(with_labels=False)  # No labels during evaluation
        save_path=f"screenshots/mesh_{self.i+1}.png"
        component.mesh.plot_mesh(save_path=save_path)
        self.i += 1
        # Prepare GNN inputs
        x, edge_index, batch = gnn_input_fn(data)
        # Forward pass
        with torch.no_grad():
            pred_norm = self.model(x, edge_index, batch)   # [1,3]
        # Denormalize
        stress = pred_norm * self.glob_std + self.glob_mean  # [1,3]
        stress.squeeze_(0)
        volume = component.get_volume()
        
        return {
            "stress": stress,
            "volume": volume
        }
