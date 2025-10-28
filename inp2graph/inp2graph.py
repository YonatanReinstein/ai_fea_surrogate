import torch
from torch_geometric.data import Data
from itertools import combinations

def read_inp_c3d8(path):
    nodes = {}
    elements = []
    reading_nodes = False
    reading_elements = False

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("**"):
                continue

            if line.startswith("*NODE"):
                reading_nodes = True
                reading_elements = False
                continue
            elif line.startswith("*ELEMENT") and "C3D8" in line:
                reading_nodes = False
                reading_elements = True
                continue
            elif line.startswith("*"):
                # other sections, ignore
                reading_nodes = False
                reading_elements = False
                continue

            if reading_nodes:
                # node_id, x, y, z
                parts = [p.strip() for p in line.split(",")]
                node_id = int(parts[0])
                coords = list(map(float, parts[1:4]))
                nodes[node_id] = coords
            elif reading_elements:
                # element_id, 8 node ids
                parts = [p.strip() for p in line.split(",")]
                elem_nodes = list(map(int, parts[1:]))  # skip elem ID
                if len(elem_nodes) == 8:
                    elements.append(elem_nodes)

    # build node tensor
    node_ids = sorted(nodes.keys())
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    x = torch.tensor([nodes[nid] for nid in node_ids], dtype=torch.float)

    # build edges (undirected)
    edges = set()
    for elem in elements:
        for a, b in combinations(elem, 2):  # all pairs of 8 nodes
            i, j = node_id_to_idx[a], node_id_to_idx[b]
            edges.add((i, j))
            edges.add((j, i))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

if __name__ == "__main__":
    data = read_inp_c3d8("two_cubes.inp")
    print(data)
    # optional: inspect
    print("Nodes:", data.num_nodes, "Edges:", data.num_edges)
    # Convert edge_index tensor to a list of (source, target) tuples
    edges = data.edge_index.t().tolist()  # transpose: shape [num_edges, 2]
    print("\nEdges (source -> target):")
    for src, dst in edges:
        print(f"{src} -> {dst}")