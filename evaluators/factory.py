from .mlp_evaluator import MLPEvaluator
from .gnn_evaluator import GNNEvaluator
from .mapdl_evaluator import MAPDLEvaluator
from .ccx_evaluator import CCXEvaluator


def get_evaluator(geometry_name, arch="mlp", screenshots: bool = False, processes: int = None, batch_size: int = 128):
    if arch == "mlp":
        return MLPEvaluator(geometry_name)
    elif arch == "gnn":
        return GNNEvaluator(geometry_name, screenshots=screenshots, processes=processes, batch_size=batch_size)
    elif arch == "mapdl":
        return MAPDLEvaluator(geometry_name)
    elif arch == "ccx":
        return CCXEvaluator(geometry_name)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
