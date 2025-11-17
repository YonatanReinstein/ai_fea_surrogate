from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    def __init__(self, geometry_name):
        self.geometry_name = geometry_name

    @abstractmethod
    def evaluate(self, dims: dict):
        pass
