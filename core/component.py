from torch_geometric.data import Data
import torch
from .IritModel import IIritModel
from .node import Node
from .element import Element
from .mesh import Mesh


class Component:
    def __init__(self, CAD_model: IIritModel, young: float, poisson: float):
        self.CAD_model = CAD_model
        self.young = young
        self.poisson = poisson
        self.mesh = None

    def clear_boundaries(self):
        if self.mesh is not None:
            self.mesh.clear_anchors()
            self.mesh.clear_forces()

    def generate_mesh(self, U: int =10, V: int =10, W: int =10):
        nodes_dict, elements_dict = self.CAD_model.create_mesh(U=U, V=V, W=W)
        nodes = {nid: Node(nid, xyz) for nid, xyz in nodes_dict.items()}    # Create Node objects
        elements = {
            eid: Element(eid, [nodes[nid] for nid in nlist])                # Create Element objects
            for eid, nlist in elements_dict.items()
        }
        self.mesh = Mesh(nodes, elements)
        
    def ansys_sim(self):
        if self.mesh is None:
            raise ValueError("Mesh has not been generated yet.")
        self.mesh.solve(self.young, self.poisson) 

    def get_volume(self):
        return self.CAD_model.get_volume()

    def to_graph_with_labels(self, with_labels: bool = True) -> Data:
#        if self.mesh is None:
#            raise ValueError("Mesh has not been generated yet.")
        
        node_feats = []
        node_disp = []
        node_stress = []

        # --- Node-wise features and labels ---
        for node in self.mesh.nodes.values():
            # === Input features ===
            feats = []
            feats.extend(node.coords)             # (x, y, z)
            feats.extend(node.forces or [0, 0, 0])# (Fx, Fy, Fz)
            feats.append(float(node.anchored))    # anchored flag
            node_feats.append(feats)

            # === Targets ===
            ux, uy, uz = node.displacement        # displacement vector
            sigma_vm = node.stress                # scalar von Mises stress
            node_disp.append([ux, uy, uz])
            node_stress.append([sigma_vm])

        x = torch.tensor(node_feats, dtype=torch.float)           
        node_disp = torch.tensor(node_disp, dtype=torch.float).unsqueeze(0)   
        node_stress = torch.tensor(node_stress, dtype=torch.float).unsqueeze(0)

        # --- Build edges (bidirectional) ---
        edges = set()
        for elem in self.mesh.elements.values():
            ids = [n.id - 1 for n in elem.nodes]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    edges.add((ids[i], ids[j]))
                    edges.add((ids[j], ids[i]))

        if edges:
            edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # --- Global labels ---
        volume = torch.tensor([self.get_volume()], dtype=torch.float).unsqueeze(0)
        dims = torch.tensor(list(self.CAD_model.get_dim_list()), dtype=torch.float).unsqueeze(0)      
        poisson = torch.tensor([self.poisson], dtype=torch.float).unsqueeze(0)
        young = torch.tensor([self.young], dtype=torch.float).unsqueeze(0)
        if with_labels:
            max_stress=torch.tensor([self.mesh.get_max_stress()], dtype=torch.float).unsqueeze(0)
            max_displacement=torch.tensor([self.mesh.get_max_displacement()], dtype=torch.float).unsqueeze(0)
        else:
            max_stress=torch.tensor([0.0], dtype=torch.float).unsqueeze(0)
            max_displacement=torch.tensor([0.0], dtype=torch.float).unsqueeze(0)

        # --- Assemble Data object ---
        data = Data(
            x=x,
            edge_index=edge_index,
            node_disp=node_disp,     
            node_stress=node_stress, 
            volume=volume,              
            dims = dims,      
            max_stress=max_stress,
            max_displacement=max_displacement,
            poisson = poisson,
            young = young
        )
        return data


if __name__ == "__main__":

    model_path = "data/arm/CAD_model/model.irt"
    json_path = "data/arm/CAD_model/dims.json"
    cad_model = IIritModel(model_path, json_path)
    component = Component(cad_model, young=2.1e11, poisson=0.3)
    geometry = "arm"

    import importlib
    module = importlib.import_module(f"data.{geometry}.boundary_conditions")
    anchor_condition = module.anchor_condition
    force_pattern = module.force_pattern
    mesh_resolution = module.mesh_resolution
    U, V, W = mesh_resolution()
    component.generate_mesh(U=U, V=V, W=W)
    component.mesh.anchor_nodes_by_condition(anchor_condition)
    component.mesh.apply_force_by_pattern(force_pattern)
    component.mesh.solve(young=2e11, poisson=0.3)
    component.mesh.plot_mesh()
    print("Max stress:", component.mesh.get_max_stress())



