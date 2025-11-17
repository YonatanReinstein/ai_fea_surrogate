import importlib
from .base_evaluator import BaseEvaluator
from core.component import Component
from core.IritModel import IIritModel

     
class MAPDLEvaluator(BaseEvaluator):
    def __init__(self, geometry_name):
        super().__init__(geometry_name)
        self.geometry = geometry_name
        self.model_path = f"data/{geometry_name}/CAD_model/model.irt"


    def evaluate(self, dims: dict, young: float = 2.1e11, poisson: float = 0.3):
        CAD_model = IIritModel(self.model_path, dims_dict=dims)
        comp = Component(CAD_model=CAD_model, young=young, poisson=poisson)
        comp.generate_mesh(U=11, V=2, W=2)
        module = importlib.import_module(f"data.{self.geometry}.boundary_conditions")
        anchor_condition = module.anchor_condition
        force_pattern = module.force_pattern
        comp.mesh.anchor_nodes_by_condition(anchor_condition)
        comp.mesh.apply_force_by_pattern(force_pattern)
        comp.mesh.solve(young=young, poisson=poisson)
        result = {
            "volume": comp.get_volume(),
            "stress": comp.mesh.get_max_stress(),
            "disp": comp.mesh.get_max_displacement(),
        }
        return result
    