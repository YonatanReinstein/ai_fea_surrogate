from abc import ABC, abstractmethod
import json

class BaseEvaluator(ABC):
    def __init__(self, geometry_name):
        self.geometry_name = geometry_name
        with open(f"data/{geometry_name}/CAD_model/material_properties.json", "r") as f:
            self.material_properties = json.load(f)
    @abstractmethod
    def evaluate(self, dims: dict):
        pass
