"""Local STDN implementation from paper and official-code review."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from tsflab.models._components.marks import normalized_time_features


def _laplacian_positions(adjacency: np.ndarray, width: int) -> torch.Tensor:
    degree = adjacency.sum(1)
    inverse = np.zeros_like(degree, dtype=np.float64)
    inverse[degree > 0] = degree[degree > 0] ** -0.5
    laplacian = np.eye(adjacency.shape[0]) - inverse[:, None] * adjacency * inverse[None]
    _, vectors = np.linalg.eigh((laplacian + laplacian.T) / 2)
    positions = np.zeros((adjacency.shape[0], width), dtype=np.float32)
    usable = vectors[:, 1 : 1 + min(width, max(0, adjacency.shape[0] - 1))]
    positions[:, : usable.shape[1]] = usable
    return torch.from_numpy(positions)


class DynamicDiffusion(nn.Module):
    """Per-batch dynamic graph propagation used by the trend branch."""

    def __init__(self, width: int, order: int) -> None:
        super().__init__()
        self.order = order
        self.projection = nn.Linear(width * (order + 1), width)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        terms = [x]
        value = x
        for _ in range(self.order):
            value = torch.einsum("bnc,bnm->bmc", value, graph)
            terms.append(value)
        return self.projection(torch.cat(terms, -1))


class Model(nn.Module):
    """Spatial-temporal decomposition with attention and dynamic diffusion."""

    def __init__(self, seq_len: int, pred_len: int, num_nodes: int, adj_mx: np.ndarray | None = None, time_slice_size: int = 60, K: int = 4, d: int = 8, L: int = 1, order: int = 2, reference: int = 4, out_channels: int = 1) -> None:
        super().__init__()
        if out_channels != 1:
            raise ValueError("TSFLab STDN exposes one value per node")
        width = K * d
        slots = 1440 // time_slice_size
        adjacency = np.eye(num_nodes, dtype=np.float32) if adj_mx is None else np.asarray(adj_mx, dtype=np.float32)
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("adj_mx shape must match num_nodes")
        self.register_buffer("spatial_positions", _laplacian_positions(adjacency, width))
        self.seq_len, self.pred_len, self.num_nodes, self.slots = seq_len, pred_len, num_nodes, slots
        self.value_projection = nn.Linear(1, width)
        self.time_embedding = nn.Embedding(slots, width)
        self.weekday_embedding = nn.Embedding(7, width)
        self.spatial_projection = nn.Linear(width, width)
        self.trend_gate = nn.Linear(width, width)
        self.history_attention = nn.ModuleList(nn.MultiheadAttention(width, K, batch_first=True) for _ in range(L))
        self.attention_norm = nn.ModuleList(nn.LayerNorm(width) for _ in range(L))
        self.trend_encoder = nn.GRU(width, width, batch_first=True)
        self.graph_query = nn.Linear(width, width)
        self.graph_key = nn.Linear(width, width)
        self.dynamic_diffusion = DynamicDiffusion(width, order)
        self.horizon_embedding = nn.Parameter(torch.empty(pred_len, width))
        self.output = nn.Linear(2 * width, 1)
        self.reference = min(reference, num_nodes)
        nn.init.xavier_uniform_(self.horizon_embedding)

    def _calendar(self, marks: torch.Tensor | None, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if marks is None:
            tod = torch.arange(length, device=device).view(1, length) % self.slots
            return tod, torch.zeros_like(tod)
        if marks.ndim == 4:
            # Spatiotemporal batches carry node covariates (B, T, N, F) whose
            # first two channels are [time_in_day, day_in_week] in [0, 1). The
            # calendar is shared by the nodes, so node 0 is read, as the
            # official runner does (``TE[:, :, 0, :]``).
            if marks.shape[-1] < 2:
                raise ValueError("STDN needs [time_in_day, day_in_week] node covariates")
            normalized = marks[:, -length:, 0, :2]
        else:
            normalized = normalized_time_features(marks[:, -length:])
        tod = (normalized[..., 0] * self.slots).long() % self.slots
        dow = (normalized[..., 1] * 7).long() % 7
        return tod, dow

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"STDN expects (B, {self.seq_len}, {self.num_nodes}) values")
        tod, dow = self._calendar(x_mark_enc, self.seq_len, x_enc.device)
        if tod.shape[0] == 1 and x_enc.shape[0] != 1:
            tod, dow = tod.expand(x_enc.shape[0], -1), dow.expand(x_enc.shape[0], -1)
        spatial = self.spatial_projection(self.spatial_positions).view(1, 1, self.num_nodes, -1)
        temporal = (self.time_embedding(tod) + self.weekday_embedding(dow)).unsqueeze(2)
        embedding = spatial + temporal
        values = self.value_projection(x_enc.unsqueeze(-1))
        gate = torch.sigmoid(self.trend_gate(embedding))
        trend, seasonal = values * gate, values * (1 - gate)
        batch, length, nodes, width = values.shape
        encoded_trend, _ = self.trend_encoder(trend.transpose(1, 2).reshape(batch * nodes, length, width))
        summary = encoded_trend[:, -1].reshape(batch, nodes, width)
        scores = self.graph_query(summary) @ self.graph_key(summary).transpose(-1, -2) * width**-0.5
        if self.reference < nodes:
            threshold = scores.topk(self.reference, dim=-1).values[..., -1:]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
        graph = torch.softmax(scores, -1)
        trend_future = self.dynamic_diffusion(summary, graph).unsqueeze(1).expand(-1, self.pred_len, -1, -1)
        future_tod, future_dow = self._calendar(x_mark_dec, self.pred_len, x_enc.device)
        if future_tod.shape[0] == 1 and batch != 1:
            future_tod, future_dow = future_tod.expand(batch, -1), future_dow.expand(batch, -1)
        query = self.time_embedding(future_tod) + self.weekday_embedding(future_dow) + self.horizon_embedding.unsqueeze(0)
        query = (query.unsqueeze(2) + spatial).transpose(1, 2).reshape(batch * nodes, self.pred_len, width)
        memory = seasonal.transpose(1, 2).reshape(batch * nodes, length, width)
        for attention, norm in zip(self.history_attention, self.attention_norm):
            update, _ = attention(query, memory, memory, need_weights=False)
            query = norm(query + update)
        seasonal_future = query.reshape(batch, nodes, self.pred_len, width).transpose(1, 2)
        return self.output(torch.cat((trend_future, seasonal_future), -1)).squeeze(-1)


__all__ = ["Model", "DynamicDiffusion"]
