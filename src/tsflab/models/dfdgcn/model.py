"""Local DFDGCN implementation from paper and official-code review."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.adaptive_node_embedding_adjacency import (
    adaptive_node_embedding_adjacency,
)
from tsflab.models._components.gated_dilated_conv import gated_dilated_conv
from tsflab.models._components.graph_utils import adj_to_supports
from tsflab.models._components.marks import to_spatiotemporal


class DynamicGraphMix(nn.Module):
    """Mix static, adaptive, and per-sample frequency-domain neighborhoods (paper Eq. 7)."""

    def __init__(self, channels: int, out_channels: int, supports: int = 4, order: int = 2) -> None:
        super().__init__()
        self.order = order
        self.projection = nn.Conv2d(channels * (1 + supports * order), out_channels, 1)

    @staticmethod
    def _propagate(x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        # Not named ``_apply``: that would shadow ``nn.Module._apply`` and break ``.to(device)``.
        if graph.ndim == 2:
            return torch.einsum("bcnt,nm->bcmt", x, graph)
        return torch.einsum("bcnt,bnm->bcmt", x, graph)

    def forward(self, x: torch.Tensor, graphs: list[torch.Tensor]) -> torch.Tensor:
        terms = [x]
        for graph in graphs:
            value = self._propagate(x, graph)
            terms.append(value)
            for _ in range(2, self.order + 1):
                value = self._propagate(value, graph)
                terms.append(value)
        return self.projection(torch.cat(terms, dim=1))


class FrequencyGraph(nn.Module):
    """Dynamic frequency-domain graph (paper Eqs. 4-6, official ``DFDGCN.forward``).

    FFT magnitudes of each node's window are L2-normalized over nodes and then over
    frequencies and scaled by ``a``, embedded, and concatenated with a node identity
    embedding and time-of-day / day-of-week embeddings of the latest step. A per-node
    projection, LayerNorm, dropout and a directed bilinear form give ``ReLU`` scores;
    each row keeps its ``subgraph`` largest scores (top-k mask) before the softmax.
    """

    def __init__(self, seq_len: int, nodes: int, fft_dim: int, identity_dim: int, hidden: int,
                 steps_per_day: int, a: float, subgraph: int, dropout: float) -> None:
        super().__init__()
        self.a, self.subgraph, self.steps_per_day = a, min(subgraph, nodes), steps_per_day
        self.spectrum = nn.Parameter(torch.randn(seq_len // 2 + 1, fft_dim))
        self.identity = nn.Parameter(torch.randn(nodes, identity_dim))
        self.time_of_day = nn.Parameter(torch.empty(steps_per_day, seq_len))
        self.day_of_week = nn.Parameter(torch.empty(7, seq_len))
        self.node_projection = nn.Parameter(torch.randn(nodes, fft_dim + identity_dim + 2 * seq_len, hidden))
        self.direction = nn.Parameter(torch.randn(hidden, hidden))
        self.norm = nn.LayerNorm([nodes, hidden], eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.time_of_day)
        nn.init.xavier_uniform_(self.day_of_week)

    def forward(self, values: torch.Tensor, time_in_day: torch.Tensor, day_in_week: torch.Tensor) -> torch.Tensor:
        """``values`` ``[B, T, N]``; latest-step calendar fractions ``[B, N]`` in ``[0, 1)``."""
        magnitude = torch.fft.rfft(values.transpose(1, 2), dim=-1).abs()
        magnitude = F.normalize(magnitude, p=2.0, dim=1, eps=1e-12)
        magnitude = F.normalize(magnitude, p=2.0, dim=2, eps=1e-12) * self.a
        tod = (time_in_day * self.steps_per_day).round().long().clamp(0, self.steps_per_day - 1)
        dow = (day_in_week * 7).round().long().clamp(0, 6)
        features = torch.cat((
            magnitude @ self.spectrum,
            self.identity.unsqueeze(0).expand(values.shape[0], -1, -1),
            self.time_of_day[tod],
            self.day_of_week[dow],
        ), dim=-1)
        hidden = torch.relu(torch.einsum("bnd,ndh->bnh", features, self.node_projection))
        directed = self.dropout(self.norm(hidden)) @ self.direction
        scores = torch.relu(directed @ hidden.transpose(1, 2))
        # Top-k per row with a small random tie-break, as the official dy_mask_graph.
        keep = (scores + torch.rand_like(scores) * 0.01).topk(self.subgraph, dim=-1).indices
        mask = torch.zeros_like(scores).scatter_(-1, keep, 1.0)
        return torch.softmax(scores * mask, dim=-1)


class Model(nn.Module):
    """Dilated temporal forecaster with frequency-derived dynamic graphs."""

    def __init__(self, seq_len: int, pred_len: int, num_nodes: int, adj_mx: np.ndarray | None = None, dropout: float = 0.3, residual_channels: int = 16, dilation_channels: int = 16, skip_channels: int = 64, end_channels: int = 128, kernel_size: int = 2, blocks: int = 2, layers: int = 2, a: float = 1.0, fft_emb: int = 10, identity_emb: int = 10, hidden_emb: int = 30, subgraph: int = 20, steps_per_day: int = 288) -> None:
        super().__init__()
        if residual_channels != dilation_channels:
            raise ValueError("local DFDGCN requires equal residual and dilation widths")
        if min(subgraph, steps_per_day) < 1:
            raise ValueError("subgraph and steps_per_day must be positive")
        if adj_mx is None:
            raise ValueError(
                "DFDGCN needs the dataset adjacency adj_mx [N, N] for its predefined graphs; "
                "use a graph dataset that ships adj_mx.npy (for example a PEMS preset)"
            )
        adjacency = np.asarray(adj_mx, dtype=np.float32)
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("adj_mx shape must match num_nodes")
        static = adj_to_supports(adjacency)
        self.register_buffer("forward_support", static[0])
        self.register_buffer("reverse_support", static[1])
        self.adaptive_source = nn.Parameter(torch.randn(num_nodes, fft_emb))
        self.adaptive_target = nn.Parameter(torch.randn(fft_emb, num_nodes))
        self.frequency_graph = FrequencyGraph(seq_len, num_nodes, fft_emb, identity_emb, hidden_emb,
                                              steps_per_day, a, subgraph, dropout)
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, num_nodes
        self.input_projection = nn.Conv2d(2, residual_channels, 1)  # value + time of day (official in_dim = 2)
        count = blocks * layers
        self.filters = nn.ModuleList(nn.Conv2d(residual_channels, residual_channels, (1, kernel_size), dilation=(1, 2 ** (index % layers))) for index in range(count))
        self.gates = nn.ModuleList(nn.Conv2d(residual_channels, residual_channels, (1, kernel_size), dilation=(1, 2 ** (index % layers))) for index in range(count))
        self.graph_layers = nn.ModuleList(DynamicGraphMix(residual_channels, residual_channels) for _ in range(count))
        self.skip_layers = nn.ModuleList(nn.Conv2d(residual_channels, skip_channels, 1) for _ in range(count))
        self.norms = nn.ModuleList(nn.BatchNorm2d(residual_channels) for _ in range(count))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(nn.ReLU(), nn.Conv2d(skip_channels, end_channels, 1), nn.ReLU(), nn.Conv2d(end_channels, pred_len, 1))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):

        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"DFDGCN expects (B, {self.seq_len}, {self.num_nodes}) values")
        data = to_spatiotemporal(x_enc, x_mark_enc)[..., :3]
        if data.shape[-1] < 3:
            raise ValueError("DFDGCN needs time-of-day and day-of-week covariates")
        hidden = self.input_projection(data[..., :2].permute(0, 3, 2, 1))
        adaptive = adaptive_node_embedding_adjacency(self.adaptive_source, self.adaptive_target)
        latest = data[:, -1]
        dynamic = self.frequency_graph(x_enc, latest[..., 1], latest[..., 2])
        graphs = [self.forward_support, self.reverse_support, adaptive, dynamic]
        skips = None
        for filter_layer, gate_layer, graph_layer, skip_layer, norm in zip(
            self.filters, self.gates, self.graph_layers, self.skip_layers, self.norms
        ):
            gated = gated_dilated_conv(hidden, filter_layer, gate_layer)
            skips = skip_layer(gated) if skips is None else skips + skip_layer(gated)
            hidden = norm(hidden + self.dropout(graph_layer(gated, graphs)))
        assert skips is not None
        return self.output(skips)[..., -1]


__all__ = ["Model", "DynamicGraphMix", "FrequencyGraph"]
