from .IritModel import IritModelBase
from .node import Node
from .element import Element
from .mesh import Mesh


class Component:
    def __init__(self, CAD_model: IritModelBase, young: float, poisson: float):
        self.CAD_model = CAD_model
        self.young = young
        self.poisson = poisson
        self.mesh = None

    def generate_mesh(self, U: int =10, V: int =10, W: int =10):
        nodes_dict, elements_dict, elem_to_tile = self.CAD_model.create_mesh(U=U, V=V, W=W)
        nodes = {nid: Node(nid, xyz) for nid, xyz in nodes_dict.items()}
        elements = {
            eid: Element(eid, [nodes[nid] for nid in nlist])
            for eid, nlist in elements_dict.items()
        }
        self.mesh = Mesh(nodes, elements)
        self.elem_to_tile = elem_to_tile  # elem_id -> 0-based tile index
        # TOPOPT_STIFF_PENALTY: soften thin tiles (see core.mesh.element_stiffness_scales)
        from .mesh import element_stiffness_scales
        scales = element_stiffness_scales(self.CAD_model.dims_template, elem_to_tile,
                                          getattr(self.CAD_model, "_fixed_dims", {}) or {})
        if scales:
            self.mesh.elem_scale = scales

    def fem_sim(self):
        """Linear static solve with core.fem_adjoint (no external solver)."""
        if self.mesh is None:
            raise ValueError("Mesh has not been generated yet.")
        self.mesh.solve_fem(self.young, self.poisson)

    def get_volume(self):
        return self.CAD_model.get_volume()
