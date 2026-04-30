# Run Architecture

## Model
GNN with `EdgeAttrConv` message passing (`utils/gnn_surrogate.py`):
- Encoder: MLP `[node_in_dim, hidden_dim, hidden_dim]`
- 4 × `EdgeAttrConv` layers, message MLP `[2*hidden_dim + edge_dim, hidden_dim, hidden_dim]`, residual
- Head: MLP `[hidden_dim, hidden_dim, 1]` → per-node prediction
- Graph output: `global_max_pool` over node predictions
- **Virtual node**: one learnable node per graph, bidirectionally connected to all real nodes via learnable edge attributes — provides global context during message passing

## Changes vs previous run
- **Virtual node**: added a global virtual node (learned embedding `[hidden_dim]` + learned edge attr `[edge_dim]`) connected to every real node in both directions; stripped before the head so output shapes are unchanged

## Ablation result
Running with mesh edges removed (virtual node only, no real edges): val node loss 28.17 vs 0.06, val graph loss 316.94 vs 0.36 — **GNN carries the model**, virtual node alone is useless without local message passing.

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

## Val set performance (physical units)
| metric | value |
|---|---|
| MAE | 2,031 MPa |
| RMSE | 2,708 MPa |
| Mean relative error | 4.5% |
| Median relative error | 3.5% |
| 90th pct relative error | 9.6% |
| Spearman rank correlation | 0.972 |

## Best loss
- Best **val** loss: **0.6524** (epoch 100)
- Best **train** loss: **0.4911** (epoch 100)
