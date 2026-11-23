from .mlp_evaluator import MLPEvaluator
from .gnn_evaluator import GNNEvaluator
from .mapdl_evaluator import MAPDLEvaluator
from .ccx_evaluator import CCXEvaluator


def get_evaluator(geometry_name, arch="mlp", screenshots: bool = False, processes: int = None, batch_size: int = 128,
                   mesh_threads: int = 4, nproc: int = 1, heatmap_base_only: bool = False, stress_ks_rho_mpa: float = 0.15,
                   gnn_ckpt_path: str = None, ccx_axis_path: str = None, gradient_nproc: int = None,
                   ccx_gradient: str = "adjoint"):
    if arch == "mlp":
        return MLPEvaluator(geometry_name)
    elif arch == "gnn":
        return GNNEvaluator(geometry_name, screenshots=screenshots, processes=processes, batch_size=batch_size,
                             heatmap_base_only=heatmap_base_only, ckpt_path=gnn_ckpt_path,
                             stress_ks_rho_mpa=stress_ks_rho_mpa)
    elif arch == "gnn_dsig":
        from .gnn_dsig_evaluator import GNNDsigEvaluator
        return GNNDsigEvaluator(geometry_name, ckpt_path=gnn_ckpt_path,
                                stress_ks_rho_mpa=stress_ks_rho_mpa, mesh_threads=mesh_threads)
    elif arch == "mapdl":
        return MAPDLEvaluator(geometry_name)
    elif arch == "ccx":
        return CCXEvaluator(geometry_name, max_workers=processes or 8, mesh_threads=mesh_threads, nproc=nproc,
                             stress_ks_rho_mpa=stress_ks_rho_mpa, ccx_axis_path=ccx_axis_path,
                             gradient_nproc=gradient_nproc, gradient_source=ccx_gradient)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
