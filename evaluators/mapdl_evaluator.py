import importlib

from data.beam.boundary_conditions import mesh_resolution
from .base_evaluator import BaseEvaluator
from core.component import Component
from core.IritModel import IIritModel
import json

     
class MAPDLEvaluator(BaseEvaluator):
    def __init__(self, geometry_name):
        super().__init__(geometry_name)
        self.geometry = geometry_name
        self.model_path = f"data/{geometry_name}/CAD_model/model.irt"
        module = importlib.import_module(f"data.{geometry_name}.boundary_conditions")
        self.anchor_condition = module.anchor_condition
        self.force_pattern = module.force_pattern
        self.mesh_resolution = module.mesh_resolution
        self.U, self.V, self.W = self.mesh_resolution()
        with open(f"data/{geometry_name}/CAD_model/material_properties.json", "r") as f:
            self.material_properties = json.load(f)
        self.i = 0


    def evaluate(self, dims: dict, young: float = 2.1e11, poisson: float = 0.3):
        CAD_model = IIritModel(self.model_path, dims_dict=dims)
        self.comp = Component(CAD_model=CAD_model, young=self.material_properties["young_modulus"], poisson=self.material_properties["poisson_ratio"])

        self.comp.generate_mesh(U=self.U, V=self.V, W=self.W)
        self.comp.mesh.anchor_nodes_by_condition(self.anchor_condition)
        self.comp.mesh.apply_force_by_pattern(self.force_pattern)
        save_path_=f"screenshots/mesh_{self.i+1}"
        self.comp.ansys_sim(screenshot_path=save_path_ + f"_ansys.png")
        self.comp.mesh.plot_mesh(save_path=save_path_ + ".png")
        self.i += 1

        result = {
            "volume": self.comp.get_volume(),
            "stress": self.comp.mesh.get_max_stress(),
            "disp": self.comp.mesh.get_max_displacement(),
        }
        return result
    
    
    

    