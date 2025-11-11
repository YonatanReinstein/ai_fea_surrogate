from .base_evaluator import BaseEvaluator


class GNNEvaluator(BaseEvaluator):
    def __init__(self, geometry_name, model_dir="data", *args, **kwargs):
        super().__init__(geometry_name)
        self.model_dir = model_dir
        self.warned = False

    def evaluate(self, graph_or_dims):
        """
        Dummy implementation until GNNSurrogate is integrated.
        Currently just prints a warning and returns a fake result.
        """
        if not self.warned:
            print(f"⚠️  [GNNEvaluator] GNN surrogate not yet implemented. "
                  f"Returning dummy values for '{self.geometry_name}'.")
            self.warned = True

        # Return placeholder predictions
        return {"stress": 1e8, "disp": 1e-3}
