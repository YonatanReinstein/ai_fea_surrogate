from .base_evaluator import BaseEvaluator
from utils.component import Component

     
class MAPDLEvaluator(BaseEvaluator):
    def __init__(self, geometry_name, irt_dir, dims: dict):
        super().__init__(geometry_name)
        comp = Component(
            irt_script_path=f"{irt_dir}/model.irt",
            dims_path=None
        )
        self.workdir = os.path.abspath(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.mapdl = launch_mapdl(run_location=self.workdir, override=True)
        self.mapdl.clear()
        self.mapdl.prep7()
        print(f"[MAPDLEvaluator] MAPDL launched in {self.workdir}")

    def _build_model(self, dims):
        """Create geometry/mesh based on the input dimensions."""
        # Example: create a simple block geometry
        Lx, Ly, Lz = dims
        self.mapdl.block(0, Lx, 0, Ly, 0, Lz)
        self.mapdl.et(1, "SOLID186")
        self.mapdl.mp("EX", 1, 2.1e11)  # Young's modulus
        self.mapdl.mp("NUXY", 1, 0.3)   # Poisson's ratio
        self.mapdl.esize(Lx / 10)
        self.mapdl.vmesh("ALL")

    def _apply_boundary_conditions(self):
        """Fix one face and apply load on the opposite face."""
        self.mapdl.nsel("S", "LOC", "X", 0)
        self.mapdl.d("ALL", "ALL", 0)
        self.mapdl.nsel("S", "LOC", "X", None)
        self.mapdl.sfe("ALL", 1, "PRES", 1e6)  # example pressure

    def evaluate(self, dims):
        """Run the full simulation and return stress and displacement."""
        self.mapdl.finish()
        self.mapdl.clear()
        self.mapdl.prep7()
        self._build_model(dims)
        self._apply_boundary_conditions()
        self.mapdl.finish()

        self.mapdl.run("/SOLU")
        self.mapdl.antype("STATIC")
        self.mapdl.solve()
        self.mapdl.finish()

        self.mapdl.post1()
        max_stress = self.mapdl.get("Smax", "ELEM", 0, "S", "EQV", "MAX")
        max_disp = self.mapdl.get("Umax", "NODE", 0, "U", "SUM", "MAX")
        result = {"stress": float(max_stress), "disp": float(max_disp)}
        self.record(dims, result)
        return result

    def __del__(self):
        """Close MAPDL when done."""
        try:
            self.mapdl.exit()
        except:
            pass
