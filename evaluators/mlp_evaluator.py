import torch
import json
from utils.mlp_surrogate import MLPSurrogate
from .base_evaluator import BaseEvaluator

class MLPEvaluator(BaseEvaluator):
    def __init__(self, geometry_name: str):
        super().__init__(geometry_name)
        dims_path = f"data/{geometry_name}/CAD_model/dims.json"
        with open (dims_path, "r") as f:
            dims_data = json.load(f)
        in_dim = len(dims_data)
        self.model = MLPSurrogate(in_dim=in_dim)
        model_state_path = f"data/{geometry_name}/surrogates/mlp_surrogate.pt"
        self.model.load_state_dict(torch.load(model_state_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def evaluate(self, dims):
        x = torch.tensor([dims], dtype=torch.float32)
        with torch.no_grad():
            y = self.model(x).numpy()[0]
        return {"stress": float(y[0]), "disp": float(y[1])}


if __name__ == "__main__":
    evaluator = MLPEvaluator(
        geometry_name="beam",
        in_dim=3,
        model_state_path="data/beam/surrogates/mlp_surrogate.pt"
    )

    test_dims = [0.5, 0.5, 0.5]  # Example input dimensions
    results = evaluator.evaluate(test_dims)
    print(f"Predicted Stress: {results['stress']}, Predicted Displacement: {results['disp']}")