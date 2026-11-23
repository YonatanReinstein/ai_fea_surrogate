from .fem_evaluator import FEMEvaluator


def get_evaluator(geometry_name, arch="gnn_dsig", processes: int = None, mesh_threads: int = 4, nproc: int = 1,
                  stress_ks_rho_mpa: float = 0.15, gnn_ckpt_path: str = None):
    if arch == "gnn_dsig":
        from .gnn_dsig_evaluator import GNNDsigEvaluator
        return GNNDsigEvaluator(geometry_name, ckpt_path=gnn_ckpt_path,
                                stress_ks_rho_mpa=stress_ks_rho_mpa, mesh_threads=mesh_threads)
    if arch == "fem":
        return FEMEvaluator(geometry_name, max_workers=processes or 8, mesh_threads=mesh_threads, nproc=nproc,
                            stress_ks_rho_mpa=stress_ks_rho_mpa)
    raise ValueError(f"Unknown architecture: {arch} (use gnn_dsig or fem)")
