"""Clean-room STPGNN from the AAAI 2024 equations and architecture diagram."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _row_normalize(graph: torch.Tensor) -> torch.Tensor:
    graph = graph.clamp_min(0)
    return graph / graph.sum(-1, keepdim=True).clamp_min(1e-6)


class PivotalNodeIdentification(nn.Module):
    """Combine physical degree and learned node affinity to identify pivots."""
    def __init__(self, nodes, width, topk, graph):
        super().__init__()
        self.topk = min(topk, nodes)
        self.source = nn.Parameter(torch.randn(nodes, width) * 0.1)
        self.target = nn.Parameter(torch.randn(nodes, width) * 0.1)
        self.register_buffer("physical_degree", graph.sum(0) + graph.sum(1))
        self.last_scores = None
        self.last_indices = None

    def forward(self):
        affinity = torch.sigmoid(self.source @ self.target.T)
        learned_degree = affinity.sum(0) + affinity.sum(1)
        scores = learned_degree + self.physical_degree
        self.last_scores = scores
        self.last_indices = scores.topk(self.topk).indices
        # A smooth pivotal membership retains gradient for every node while the
        # exact top-k pivotal set remains inspectable.
        membership = torch.sigmoid(scores - scores.mean())
        return _row_normalize(affinity), membership


class PivotalGraphConvolution(nn.Module):
    """Equation 7: temporal-window aggregation centred on pivotal nodes."""
    def __init__(self, width, temporal_span):
        super().__init__()
        self.temporal_span = temporal_span
        self.temporal_weights = nn.Parameter(torch.ones(temporal_span) / temporal_span)
        self.projection = nn.Linear(width, width)
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x, graph, pivotal):
        # Left padding keeps the sequence contract while making each output use
        # the previous d observations as in the paper's sliding formulation.
        padded = F.pad(x.permute(0, 2, 3, 1), (self.temporal_span - 1, 0))
        windows = padded.unfold(-1, self.temporal_span, 1).permute(0, 3, 1, 2, 4)
        temporal = (windows * self.temporal_weights).sum(-1)
        spatial = torch.einsum("nm,btmh->btnh", graph, temporal)
        return torch.tanh(self.projection(spatial) + self.bias) * pivotal[None, None, :, None]


class ParallelSTLayer(nn.Module):
    """Pivotal graph, ordinary graph, and temporal linear units in parallel."""
    def __init__(self, width, temporal_span, dropout):
        super().__init__()
        self.pivotal = PivotalGraphConvolution(width, temporal_span)
        self.graph_projection = nn.Linear(width, width)
        self.temporal = nn.Conv2d(width, width, (3, 1), padding=(1, 0))
        self.fusion = nn.Linear(3 * width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, graph, pivotal):
        pivotal_path = self.pivotal(x, graph, pivotal)
        ordinary = self.graph_projection(torch.einsum("nm,btmh->btnh", graph, x))
        temporal = self.temporal(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        merged = self.fusion(torch.cat((pivotal_path, ordinary, temporal), -1))
        return self.norm(x + self.dropout(torch.relu(merged)))


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, num_nodes, adj_mx=None, dropout=0.1,
                 topk=4, residual_channels=16, end_channels=64,
                 kernel_size=2, blocks=2, layers=2, dims=16):
        super().__init__()
        if min(seq_len, pred_len, num_nodes, residual_channels, blocks, layers) < 1:
            raise ValueError("invalid non-positive STPGNN dimension")
        if adj_mx is None:
            graph = torch.eye(num_nodes)
        else:
            graph = torch.as_tensor(np.asarray(adj_mx), dtype=torch.float32)
            if graph.shape != (num_nodes, num_nodes):
                raise ValueError(f"adjacency must have shape {(num_nodes, num_nodes)}")
        graph = _row_normalize(graph + torch.eye(num_nodes))
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.register_buffer("physical_graph", graph)
        self.input_projection = nn.Linear(1 + 4, residual_channels)
        self.identifier = PivotalNodeIdentification(num_nodes, dims, topk, graph)
        self.st_layers = nn.ModuleList(
            ParallelSTLayer(residual_channels, max(1, kernel_size), dropout)
            for _ in range(blocks * layers)
        )
        self.readout = nn.Sequential(nn.Linear(seq_len * residual_channels, end_channels), nn.ReLU(), nn.Linear(end_channels, pred_len))

    def _mark_features(self, x, marks):
        """Four calendar features per node and step, shape ``[B, T, N, 4]``.

        Raw six-column stamps ``[B, T, 6]`` give (month, day, weekday, hour)
        fractions shared by every node. Spatiotemporal node covariates
        ``[B, T, N, F]`` (``[time_in_day, day_in_week]`` for traffic data) are
        used per node; fewer than four are zero-padded.
        """
        batch, length = x.shape[:2]
        if marks is None:
            return x.new_zeros(batch, length, self.num_nodes, 4)
        if marks.shape[:2] != x.shape[:2]:
            raise ValueError("encoder marks must match batch and sequence")
        if marks.ndim == 4:
            if marks.shape[2] != self.num_nodes:
                raise ValueError("node covariates must match num_nodes")
            take = marks[..., :4]
            return F.pad(take, (0, 4 - take.shape[-1]))
        if marks.shape[-1] >= 6:
            shared = torch.stack((marks[..., 1] / 12, marks[..., 2] / 31, marks[..., 3] / 7, marks[..., 4] / 24), -1)
        else:
            take = marks[..., :4]
            shared = F.pad(take, (0, 4 - take.shape[-1]))
        return shared.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"expected (*,{self.seq_len},{self.num_nodes})")
        marks = self._mark_features(x_enc, x_mark_enc)
        features = torch.cat((x_enc.unsqueeze(-1), marks), -1)
        hidden = self.input_projection(features)
        adaptive, pivotal = self.identifier()
        graph = _row_normalize(self.physical_graph + adaptive)
        for layer in self.st_layers:
            hidden = layer(hidden, graph, pivotal)
        flattened = hidden.permute(0, 2, 1, 3).flatten(-2)
        return self.readout(flattened).transpose(1, 2)


STPGNN = Model
