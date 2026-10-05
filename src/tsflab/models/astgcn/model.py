"""Paper-derived ASTGCN forecasting branch.

This module is an independent implementation of the attention, Chebyshev graph
convolution, temporal convolution, and horizon projection described by Guo et
al. It deliberately implements one recent-history branch: TSFLab's batch
contract does not provide the paper's separately sampled daily and weekly
windows required by the three-branch fusion module.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.graph_spectral import chebyshev_supports
from tsflab.models._components.marks import to_spatiotemporal


def _coerce_features(values: torch.Tensor, width: int) -> torch.Tensor:
    """Keep the value channel and deterministically fit covariates to ``width``."""
    if values.shape[-1] >= width:
        return values[..., :width]
    padding = values.new_zeros((*values.shape[:-1], width - values.shape[-1]))
    return torch.cat((values, padding), dim=-1)


class SpatialTemporalAttention(nn.Module):
    """Learn the paper's dense spatial and temporal attention matrices."""

    def __init__(self, seq_len: int, num_nodes: int, features: int) -> None:
        super().__init__()
        descriptor = seq_len * features
        temporal_descriptor = num_nodes * features
        width = max(4, min(32, descriptor, temporal_descriptor))
        self.spatial_query = nn.Linear(descriptor, width)
        self.spatial_key = nn.Linear(descriptor, width)
        self.temporal_query = nn.Linear(temporal_descriptor, width)
        self.temporal_key = nn.Linear(temporal_descriptor, width)
        self.spatial_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.temporal_bias = nn.Parameter(torch.zeros(seq_len, seq_len))
        self.scale = math.sqrt(width)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, nodes, features = x.shape
        node_view = x.permute(0, 2, 1, 3).reshape(batch, nodes, steps * features)
        spatial = torch.matmul(
            self.spatial_query(node_view), self.spatial_key(node_view).transpose(-1, -2)
        ) / self.scale
        spatial = torch.softmax(spatial + self.spatial_bias, dim=-1)

        time_view = x.reshape(batch, steps, nodes * features)
        temporal = torch.matmul(
            self.temporal_query(time_view), self.temporal_key(time_view).transpose(-1, -2)
        ) / self.scale
        temporal = torch.softmax(temporal + self.temporal_bias, dim=-1)
        return spatial, temporal


class AttentionChebyshevConvolution(nn.Module):
    """Apply ``sum_k Theta_k (T_k(L) elementwise S) X`` at every time step."""

    def __init__(self, supports: torch.Tensor, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.register_buffer("chebyshev_supports", supports)
        self.weight = nn.Parameter(torch.empty(supports.shape[0], input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor) -> torch.Tensor:
        outputs = []
        for order, support in enumerate(self.chebyshev_supports):
            attended = support.unsqueeze(0) * spatial_attention
            propagated = torch.einsum("bij,btjf->btif", attended, x)
            outputs.append(torch.einsum("btif,fo->btio", propagated, self.weight[order]))
        return F.relu(torch.stack(outputs).sum(0) + self.bias)


class ASTGCNBlock(nn.Module):
    """Spatial-temporal attention, Chebyshev convolution, and gated time filter."""

    def __init__(
        self,
        seq_len: int,
        num_nodes: int,
        input_dim: int,
        graph_dim: int,
        time_dim: int,
        supports: torch.Tensor,
    ) -> None:
        super().__init__()
        self.attention = SpatialTemporalAttention(seq_len, num_nodes, input_dim)
        self.graph_convolution = AttentionChebyshevConvolution(
            supports, input_dim, graph_dim
        )
        self.time_filter = nn.Conv2d(graph_dim, time_dim, (1, 3), padding=(0, 1))
        self.time_gate = nn.Conv2d(graph_dim, time_dim, (1, 3), padding=(0, 1))
        self.residual = nn.Linear(input_dim, time_dim)
        self.normalization = nn.LayerNorm(time_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial, temporal = self.attention(x)
        temporally_attended = torch.einsum("bts,bsnf->btnf", temporal, x)
        graph = self.graph_convolution(temporally_attended, spatial)
        graph = graph.permute(0, 3, 2, 1)
        filtered = torch.tanh(self.time_filter(graph))
        gated = torch.sigmoid(self.time_gate(graph))
        temporal_features = (filtered * gated).permute(0, 3, 2, 1)
        return self.normalization(temporal_features + self.residual(x))


class Model(nn.Module):
    """One-branch ASTGCN with explicit graph and covariate contracts."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        nb_block: int = 2,
        K: int = 3,
        nb_chev_filter: int = 64,
        nb_time_filter: int = 64,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, nb_block, K, nb_chev_filter, nb_time_filter) < 1:
            raise ValueError("lengths, nodes, blocks, graph order, and widths must be positive")
        if cov_dim < 0:
            raise ValueError("cov_dim must be non-negative")
        if adj_mx is None:
            raise ValueError(
                "ASTGCN needs the road-network adjacency adj_mx [enc_in, enc_in] for its Chebyshev "
                "supports; use a graph dataset that ships adj_mx.npy (for example a PEMS preset)"
            )
        adjacency = np.asarray(adj_mx, dtype=np.float32)
        if adjacency.shape != (enc_in, enc_in):
            raise ValueError(f"adj_mx must have shape {(enc_in, enc_in)}")
        supports = chebyshev_supports(adjacency, K)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = enc_in
        self.input_dim = 1 + cov_dim
        blocks = []
        width = self.input_dim
        for _ in range(nb_block):
            blocks.append(
                ASTGCNBlock(
                    seq_len, enc_in, width, nb_chev_filter,
                    nb_time_filter, supports,
                )
            )
            width = nb_time_filter
        self.blocks = nn.ModuleList(blocks)
        # The paper's final convolution treats history positions as channels.
        self.forecast = nn.Conv2d(seq_len, pred_len, kernel_size=(1, width))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(
                f"x_enc must have shape [batch, {self.seq_len}, {self.num_nodes}]"
            )
        history = _coerce_features(to_spatiotemporal(x_enc, x_mark_enc), self.input_dim)
        for block in self.blocks:
            history = block(history)
        return self.forecast(history).squeeze(-1)
