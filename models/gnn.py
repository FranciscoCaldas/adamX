from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GNN(nn.Module):
    """PyTorch Geometric GIN-style network for graph property prediction."""

    def __init__(
        self,
        num_node_features: int,
        num_outputs: int,
        hidden_dim: int = 256,
        num_layers: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.node_encoder = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Linear(hidden_dim, num_outputs)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.node_encoder(x)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.head(x)
