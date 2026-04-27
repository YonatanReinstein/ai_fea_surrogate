# Run Architecture

## Model
GNN with `EdgeAttrConv` message passing (`utils/gnn_surrogate.py`):
- Encoder: MLP `[node_in_dim, hidden_dim, hidden_dim]`
- 4 × `EdgeAttrConv` layers, message MLP `[2*hidden_dim + edge_dim, hidden_dim, hidden_dim]`, residual
- Head: MLP `[hidden_dim, hidden_dim, 1]` → per-node prediction
- Graph output: `global_max_pool` over node predictions

## Changes vs previous run
- **Edge features**: added edge **length** as an edge attribute (`edge_attr`)
- **Aggregation**: changed `aggr` from `max` → **`mean`** in `EdgeAttrConv`

## Training config (from `jobs/train_model.slurm`)
| param | value |
|---|---|
| geometry | tile |
| num_samples | 5000 |
| epochs | 100 |
| lr | 4e-4 |
| batch_size | 8 |
| hidden_dim | 128 |
| conv_layers | 4 |
| node_loss_weight | 3.0 |
| weight_decay | 1e-3 |

Optimizer: AdamW · Scheduler: StepLR (step=10, γ=0.75) · Loss: MSE on normalized targets · grad-clip 1.0

## Best loss
- Best **val** loss: **2.9107** (epoch 69)
- Best **train** loss: **2.5121** (epoch 80)
