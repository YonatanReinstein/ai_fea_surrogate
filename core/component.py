from torch_geometric.data import Data
import torch
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

    def clear_boundaries(self):
        if self.mesh is not None:
            self.mesh.clear_anchors()
            self.mesh.clear_forces()

    def generate_mesh(self, U: int =10, V: int =10, W: int =10):
        nodes_dict, elements_dict, elem_to_tile = self.CAD_model.create_mesh(U=U, V=V, W=W)
        nodes = {nid: Node(nid, xyz) for nid, xyz in nodes_dict.items()}
        elements = {
            eid: Element(eid, [nodes[nid] for nid in nlist])
            for eid, nlist in elements_dict.items()
        }
        self.mesh = Mesh(nodes, elements)
        self.elem_to_tile = elem_to_tile  # elem_id -> 0-based tile index
        
    def ansys_sim(self,  mapdl = None, screenshot_path: str = None):
        if self.mesh is None:
            raise ValueError("Mesh has not been generated yet.")
        self.mesh.solve(self.young, self.poisson, mapdl = mapdl, screenshot_path=screenshot_path) 

    def get_volume(self):
        return self.CAD_model.get_volume()

    def to_graph_with_labels(self, with_labels: bool = True, tile_grid=None) -> Data:
        # tile_grid: (NX, NY, NZ) tuple to add per-node tile features, or None.
        # Requires self.elem_to_tile populated by generate_mesh().

        # Pre-compute node -> tile index mapping (first element wins for boundary nodes)
        node_to_tile = {}
        if tile_grid is not None and getattr(self, 'elem_to_tile', {}):
            for elem in self.mesh.elements.values():
                t = self.elem_to_tile.get(elem.id, -1)
                if t >= 0:
                    for n in elem.nodes:
                        if n.id not in node_to_tile:
                            node_to_tile[n.id] = t

        node_feats = []
        node_disp = []
        node_stress = []
        tile_indices = []  # per-node 0-based tile index (empty when tile_grid is None)

        # --- Node-wise features and labels ---
        for node in self.mesh.nodes.values():
            # === Input features ===
            feats = []
            feats.extend(node.coords)              # (x, y, z)
            feats.extend(node.forces or [0, 0, 0]) # (Fx, Fy, Fz)
            feats.append(float(node.anchored))     # anchored flag

            if tile_grid is not None:
                NX, NY, NZ = tile_grid
                t = node_to_tile.get(node.id, 0)
                iz = t % NZ
                iy = (t // NZ) % NY
                ix = t // (NY * NZ)
                # Normalized tile coordinates in [0, 1]
                feats.append(ix / max(NX - 1, 1))
                feats.append(iy / max(NY - 1, 1))
                feats.append(iz / max(NZ - 1, 1))
                tile_indices.append(t)

            node_feats.append(feats)

            # === Targets ===
            ux, uy, uz = node.displacement        # displacement vector
            sigma_vm = node.stress                # scalar von Mises stress
            node_disp.append([ux, uy, uz])
            node_stress.append([sigma_vm])

        x = torch.tensor(node_feats, dtype=torch.float)
        node_disp = torch.tensor(node_disp, dtype=torch.float).unsqueeze(0)
        node_stress = torch.tensor(node_stress, dtype=torch.float).unsqueeze(0)
        tile_idx = torch.tensor(tile_indices, dtype=torch.long) if tile_indices else None

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
            coords = x[:, :3]
            edge_attr = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], dim=1, keepdim=True)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

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
            edge_attr=edge_attr,
            node_disp=node_disp,
            node_stress=node_stress,
            volume=volume,
            dims=dims,
            max_stress=max_stress,
            max_displacement=max_displacement,
            poisson=poisson,
            young=young,
        )
        if tile_idx is not None:
            NX, NY, NZ = tile_grid
            data.tile_idx = tile_idx                                  # [N] per-node tile index
            data.tile_NX = torch.tensor([NX], dtype=torch.long)       # [1] → [G] when batched
            data.tile_NY = torch.tensor([NY], dtype=torch.long)
            data.tile_NZ = torch.tensor([NZ], dtype=torch.long)
        return data

    def to_graph_fixed(self, edge_index, tile_idx, tile_grid) -> Data:
        """Fast graph build for frozen-topology geometries (inference only).

        edge_index and tile_idx are constant across dim samples and supplied
        precomputed (data/<geometry>/dataset/graph.pt), so the expensive
        Python edge-set rebuild and node->tile mapping in
        to_graph_with_labels are skipped. Only the coordinate-dependent
        node features and edge_attr (distances) are recomputed here. Node
        ordering matches to_graph_with_labels (mesh.nodes.values(), n.id-1),
        which is what edge_index/tile_idx were built against.
        """
        nodes = list(self.mesh.nodes.values())

        coords   = torch.tensor([list(n.coords) for n in nodes], dtype=torch.float)        # [N,3]
        forces   = torch.tensor([list(n.forces) if n.forces else [0.0, 0.0, 0.0]
                                 for n in nodes], dtype=torch.float)                        # [N,3]
        anchored = torch.tensor([[float(n.anchored)] for n in nodes], dtype=torch.float)   # [N,1]

        feats = [coords, forces, anchored]
        if tile_grid is not None:
            NX, NY, NZ = tile_grid
            t  = tile_idx.long()
            ix = (t // (NY * NZ)).float() / max(NX - 1, 1)
            iy = ((t // NZ) % NY).float() / max(NY - 1, 1)
            iz = (t % NZ).float() / max(NZ - 1, 1)
            feats.append(torch.stack([ix, iy, iz], dim=1))                                  # [N,3]
        x = torch.cat(feats, dim=1)

        edge_attr = torch.norm(coords[edge_index[0]] - coords[edge_index[1]],
                               dim=1, keepdim=True)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if tile_grid is not None:
            NX, NY, NZ = tile_grid
            data.tile_idx = tile_idx.long()
            data.tile_NX  = torch.tensor([NX], dtype=torch.long)
            data.tile_NY  = torch.tensor([NY], dtype=torch.long)
            data.tile_NZ  = torch.tensor([NZ], dtype=torch.long)
        return data


if __name__ == "__main__":
    from core.IritModel import IritCModel
    import importlib
    geometry = "hollow_cube"  # Change to "bistable" to test the bistable component
    module = importlib.import_module(f"data.{geometry}.boundary_conditions")
    model_path = "data/hollow_cube/CAD_model/model"
    json_path = "data/hollow_cube/CAD_model/dims.json"
    cad_model = IritCModel(model_path, json_path, debug=True,
                           fixed_dims=module.fixed_dims())
    component = Component(cad_model, young=2.1e11, poisson=0.3)
    anchor_condition = module.anchor_condition
    force_pattern = module.force_pattern
    mesh_resolution = module.mesh_resolution
    U, V, W = mesh_resolution()
    component.generate_mesh(U=U, V=V, W=W)
    component.mesh.anchor_nodes_by_condition(anchor_condition)
    component.mesh.apply_force_by_pattern(force_pattern)
    #component.mesh.solve(young=2e11, poisson=0.3, screenshot_path=f"mapdl.png")
    print("volume:", component.get_volume())

    component.mesh.plot_mesh(save_path=f"mesh.png")

    ## --- Tile connectivity visualization ---
    ## Use to_graph_with_labels so tile_idx/NX/NY/NZ are identical to what the GNN receives.
    #def _resolve(v):
    #    return v["default"] if isinstance(v, dict) else v
#
    #resolved_dims = {k: _resolve(v) for k, v in cad_model.dims_template.items()}
    #NX, NY, NZ = module.tile_grid(resolved_dims)
    #data = component.to_graph_with_labels(with_labels=False, tile_grid=(NX, NY, NZ))
    #tile_idx = data.tile_idx.numpy()  # [N] — same tensor the GNN uses
#
    #node_list = list(component.mesh.nodes.values())  # same iteration order as to_graph_with_labels
    #K = NX * NY * NZ
    #from collections import defaultdict
    #tile_node_coords = defaultdict(list)
    #node_tile_map = {}
    #for node, t in zip(node_list, tile_idx):
    #    tile_node_coords[int(t)].append(node.coords)
    #    node_tile_map[node.id] = int(t)
#
    #tile_centers = []
    #for t in range(K):
    #    coords = tile_node_coords[t]
    #    if coords:
    #        tile_centers.append([sum(c[i] for c in coords) / len(coords) for i in range(3)])
    #    else:
    #        tile_centers.append([0.0, 0.0, 0.0])
#
    #tile_edges = []
    #for ix in range(NX):
    #    for iy in range(NY):
    #        for iz in range(NZ):
    #            flat = ix * NY * NZ + iy * NZ + iz
    #            for dix, diy, diz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
    #                jx, jy, jz = ix + dix, iy + diy, iz + diz
    #                if jx < NX and jy < NY and jz < NZ:
    #                    nb = jx * NY * NZ + jy * NZ + jz
    #                    tile_edges.append((flat, nb))
#
    #component.mesh.plot_mesh_with_tiles(tile_centers, tile_edges, node_tile_map=node_tile_map, save_path="mesh_tiles.png")
    ##print("Max stress:", component.mesh.get_max_stress())



