import torch
from utils.mlp_surrogate import MLP
from .base_evaluator import BaseEvaluator
from training.gnn_training import mlp_input_fn, mlp_target_fn

class MLPEvaluator(BaseEvaluator):
    def __init__(self, geometry_name: str):
        super().__init__(geometry_name)
        dataset_path = f"data/{geometry_name}/dataset/dataset_100.pt"
        checkpoint_path = f"data/{geometry_name}/surrogates/mlp_surrogate_100.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.in_mean = checkpoint["input_mean"]
        self.in_std  = checkpoint["input_std"]
        self.out_mean = checkpoint["target_mean"]
        self.out_std  = checkpoint["target_std"]
        dataset = torch.load(dataset_path, weights_only=False)
        sample = dataset[0]
        in_features  = mlp_input_fn(sample).numel()
        out_features = mlp_target_fn(sample).numel()
        self.model = MLP(in_features=in_features, out_features=out_features)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def evaluate(self, dims_dict: dict):
        print("Evaluating dims:", list(dims_dict.values()))
        dims = torch.tensor(list(dims_dict.values()), dtype=torch.float).unsqueeze(0)
        print("Input dims to MLP:", dims)
        x = (dims - self.in_mean) / self.in_std
        print("Normalized input to MLP:", x)
        with torch.no_grad():
            pred_norm = self.model(x)[0]
        y = pred_norm * self.out_std + self.out_mean
        print("mean_out:", self.out_mean)
        print("std_out:", self.out_std)
        y = y[0]
        #("Denormalized output from MLP:", y)
        return {"volume": float(y[0]), "stress": float(y[1]), "disp": float(y[2])}


if __name__ == "__main__":
    evaluator = MLPEvaluator(
        geometry_name="beam",
    )
    test_dims = {"width": 1, "height": 2} 
    results = evaluator.evaluate(test_dims)
    print(results)


