import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.utils import scatter


def ks_pool(node_pred, batch, num_graphs, node_target_mean, node_target_std, rho_mpa, ks_mask=None):
    """KS (Kreisselmeier-Steinhauser) smooth-max pool over per-graph node
    predictions, matching core/mesh.py:get_aggregated_stress.

    node_pred lives in node-normalized space, but unlike a hard max, KS
    aggregation is nonlinear and does not commute with that normalization --
    pooling it directly would apply the wrong effective smoothing bandwidth
    (off by a factor of node_target_std). So: denormalize to raw MPa, KS
    aggregate with rho_mpa (the same sharpness the dataset's ks_stress
    target was built with), then renormalize the scalar result back to
    node-normalized space, so downstream denorm logic is unaffected -- it
    just sees a different graph_pred.
    """
    raw = node_pred.squeeze(-1) * node_target_std.squeeze() + node_target_mean.squeeze()
    if ks_mask is not None:
        # nodes outside the mask (tile-load datasets: the loaded tile) do not
        # count toward KS -- the same rule the labels were built with
        raw = torch.where(ks_mask.view(-1).bool(), raw, torch.full_like(raw, -1e30))
    per_graph_max = scatter(raw, batch, dim=0, dim_size=num_graphs, reduce="max")
    shifted = raw - per_graph_max[batch]
    sum_exp = scatter(torch.exp(rho_mpa * shifted), batch, dim=0, dim_size=num_graphs, reduce="sum")
    ks_raw = per_graph_max + torch.log(sum_exp) / rho_mpa
    return ((ks_raw - node_target_mean.squeeze()) / node_target_std.squeeze()).unsqueeze(-1)


class EdgeAttrConv(MessagePassing):
    # norm_mode: PyG's MLP "layer_norm" is torch_geometric's LayerNorm, whose
    # default mode="graph" -- called without a batch vector it normalizes with
    # ONE mean/std over every edge of every graph in the batch, so a graph's
    # output depends on the others it is batched with (~1% on the dsig model).
    # "node" is the standard per-edge LayerNorm; "graph" kept for old checkpoints.
    def __init__(self, hidden_dim, edge_dim, norm_mode="graph"):
        super().__init__(aggr="mean")
        self.mlp = MLP([2 * hidden_dim + edge_dim, hidden_dim, hidden_dim], norm="layer_norm",
                       norm_kwargs={"mode": norm_mode})

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))


class BipartiteEdgeAttrConv(MessagePassing):
    """EdgeAttrConv variant for edges crossing two differently-sized node
    spaces (e.g. the fine/coarse boundary of a hierarchical GNN). Source
    features are projected to the target width before the message MLP, so
    the MLP itself -- and the target update -- live entirely at dst_dim;
    src_dim is never upsampled beyond this one projection."""
    def __init__(self, src_dim, dst_dim, edge_dim, norm_mode="graph"):
        super().__init__(aggr="mean")
        self.src_proj = nn.Linear(src_dim, dst_dim)
        self.mlp = MLP([2 * dst_dim + edge_dim, dst_dim, dst_dim], norm="layer_norm",
                       norm_kwargs={"mode": norm_mode})        # see EdgeAttrConv

    def forward(self, x_src, x_dst, edge_index, edge_attr):
        return self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr,
                               size=(x_src.size(0), x_dst.size(0)))

    def message(self, x_i, x_j, edge_attr):
        x_j = self.src_proj(x_j)
        return self.mlp(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))




