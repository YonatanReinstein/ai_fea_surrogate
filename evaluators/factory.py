from .mlp_evaluator import MLPEvaluator
from .gnn_evaluator import GNNEvaluator
from .mapdl_evaluator import MAPDLEvaluator
import json

def get_evaluator(geometry_name, arch="mlp"):
    if arch == "mlp":
        return MLPEvaluator(geometry_name)
    elif arch == "gnn":
        return GNNEvaluator(geometry_name)
    elif arch == "mapdl":
        return MAPDLEvaluator(geometry_name)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
