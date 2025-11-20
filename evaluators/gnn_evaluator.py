import torch
from abc import ABC
from utils.gnn_surrogate import GNN
from evaluators.base_evaluator import BaseEvaluator
from core.IritModel import IIritModel
from core.component import Component
import json
        
class GNNEvaluator(BaseEvaluator):
    """
    Global-only GNN surrogate evaluator.
    Predicts: volume, max_stress, max_displacement.
    """


    def __init__(self, geometry_name: str):
        super().__init__(geometry_name)
        checkpoint_path = f"data/{self.geometry_name}/surrogates/gnn_surrogate_100.pt"

        ckpt = torch.load(checkpoint_path, weights_only=False)

        # Normalization
        self.glob_mean = ckpt["glob_mean"].float()
        self.glob_std  = ckpt["glob_std"].float()

        # Architecture info
        self.in_features = ckpt["in_features"]
        self.out_features_global = ckpt["out_global"]

        # Build GNN
        self.model = GNN(
            in_features=self.in_features,
            out_features_global=self.out_features_global
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
        # Prepare GNN inputs
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Forward pass
        with torch.no_grad():
            pred_norm = self.model(x, edge_index, batch)   # [1,3]
        # Denormalize
        pred = pred_norm * self.glob_std + self.glob_mean  # [1,3]
        pred = pred.squeeze(0)                             # [3]
        print(pred.tolist())
        volume, stress, disp = pred.tolist()
        
        return {
            "volume": volume,
            "stress": stress,
            "disp": disp
        }
