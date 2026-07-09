"""Smoke test: HierarchicalGNN forward with fixed-topology node embeddings,
using the real frozen graph from graph.pt. Builds a 2-graph batch of synthetic
features and checks output shapes in both train and eval mode.
Run: python -m analysis.smoke_test_model
"""
import torch
from utils.gnn_surrogate import HierarchicalGNN

g = torch.load("data/hollow_cube/dataset/graph.pt", weights_only=False)
ei1 = g["edge_index"]
tile1 = g["tile_idx"]
N = g["num_nodes"]
NX, NY, NZ = g["tile_grid"]
K = NX * NY * NZ
E = ei1.shape[1]
node_in_dim = 11

# Build a 2-graph batch
B = 2
x = torch.randn(B * N, node_in_dim)
edge_index = torch.cat([ei1 + i * N for i in range(B)], dim=1)
edge_attr = torch.randn(edge_index.shape[1], 1)
batch = torch.repeat_interleave(torch.arange(B), N)
tile_idx = tile1.repeat(B)
node_id = torch.arange(N).repeat(B)
tile_x = torch.randn(B * K, 1)
tNX, tNY, tNZ = torch.tensor([NX]), torch.tensor([NY]), torch.tensor([NZ])

model = HierarchicalGNN(
    node_in_dim=node_in_dim, edge_in_dim=1, hidden_dim=64, num_layers=6,
    num_pos_nodes=N,
)
print(f"params: {sum(p.numel() for p in model.parameters()):,}  "
      f"(node_emb: {model.node_emb.weight.numel():,})")

for mode in ("train", "eval"):
    model.train(mode == "train")
    gp, npd = model(x, edge_index, edge_attr, batch,
                    tile_idx=tile_idx, tile_NX=tNX, tile_NY=tNY, tile_NZ=tNZ,
                    tile_x=tile_x, num_graphs=B, node_id=node_id)
    print(f"[{mode}] graph_pred={tuple(gp.shape)} node_pred={tuple(npd.shape)} "
          f"finite={torch.isfinite(npd).all().item()}")
    assert gp.shape == (B, 1) and npd.shape == (B * N, 1)

# Backward works (per-node embedding receives grad)
model.train()
gp, npd = model(x, edge_index, edge_attr, batch, tile_idx=tile_idx, tile_NX=tNX,
                tile_NY=tNY, tile_NZ=tNZ, tile_x=tile_x, num_graphs=B, node_id=node_id)
npd.sum().backward()
print("node_emb grad present:", model.node_emb.weight.grad is not None
      and model.node_emb.weight.grad.abs().sum().item() > 0)
print("OK")
