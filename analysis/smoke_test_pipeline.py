"""End-to-end batching smoke test: build Data objects with shared node_id like
the training loader does, batch them with PyG DataLoader, run gnn_input_fn + model.
Verifies node_id is NOT auto-incremented by PyG (must repeat 0..N-1 per graph).
Run: python -m analysis.smoke_test_pipeline
"""
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from training.gnn_training import gnn_input_fn
from utils.gnn_surrogate import HierarchicalGNN

g = torch.load("data/hollow_cube/dataset/graph.pt", weights_only=False)
ei, tile1, N = g["edge_index"], g["tile_idx"], g["num_nodes"]
NX, NY, NZ = g["tile_grid"]; K = NX * NY * NZ
node_id = torch.arange(N, dtype=torch.long)
node_in_dim = 11

def make_sample():
    d = Data(
        x=torch.randn(N, node_in_dim),
        edge_index=ei.clone(),
        edge_attr=torch.randn(ei.shape[1], 1),
        node_stress=torch.randn(1, N, 1),
        max_stress=torch.randn(1, 1),
        tile_idx=tile1.clone(),
        tile_NX=torch.tensor([NX]), tile_NY=torch.tensor([NY]), tile_NZ=torch.tensor([NZ]),
        tile_x=torch.randn(K, 1),
        node_id=node_id,
    )
    return d

dataset = [make_sample() for _ in range(3)]
loader = DataLoader(dataset, batch_size=2, shuffle=False)
batch = next(iter(loader))
x, edge_index, edge_attr, b, ti, nx, ny, nz, tx, nid_b = gnn_input_fn(batch)

print(f"batch: x={tuple(x.shape)} node_id={tuple(nid_b.shape)}")
# critical: node_id must repeat 0..N-1 per graph (PyG must NOT increment it)
assert nid_b.max().item() == N - 1, f"node_id was incremented! max={nid_b.max().item()}"
assert torch.equal(nid_b[:N], torch.arange(N)) and torch.equal(nid_b[N:2*N], torch.arange(N))
print("node_id stable across graphs (not incremented) ✅")

model = HierarchicalGNN(node_in_dim=node_in_dim, edge_in_dim=1, hidden_dim=64,
                        num_layers=6, num_pos_nodes=N)
gp, npd = model(x, edge_index, edge_attr, b, tile_idx=ti, tile_NX=nx, tile_NY=ny,
                tile_NZ=nz, tile_x=tx, num_graphs=batch.num_graphs, node_id=nid_b)
print(f"forward: graph_pred={tuple(gp.shape)} node_pred={tuple(npd.shape)} "
      f"finite={torch.isfinite(npd).all().item()}")
assert gp.shape == (2, 1) and npd.shape == (2 * N, 1)
print("OK")
