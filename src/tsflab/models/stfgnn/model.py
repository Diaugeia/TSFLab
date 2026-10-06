"""STFGNN: spatial-temporal fusion graph neural network.

Independent implementation of Li and Zhu (AAAI 2021), checked against the
official MXNet code at revision a02feee5 (no license; nothing copied). It is the
STSGCN pipeline with three changes:

* a temporal graph ``A_TG`` from banded ("fast") DTW between nodes' training
  series (Alg. 1-2), fitted once on the training split by ``fit_temporal_graph``;
* a 4-step fusion graph (Fig. 3b): ``A_TG`` on the outer diagonal blocks and the
  corners, the spatial graph ``A_SG`` on the two inner blocks, identity links
  between the same node in every other pair of steps, self-loops;
* in every layer a gated dilated convolution ``tanh(conv) * sigmoid(conv)``
  (kernel 2, dilation ``K - 1 = 3``, Eq. 7) added to the concatenated outputs of
  the ``T - 3`` fusion-graph modules.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch import nn

from tsflab.models._components.synchronous_graph_conv import (
    SynchronousGraphModule,
    mxnet_xavier_uniform_,
)

_WINDOW = 4


def _binary(matrix) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return ((values > 0) | (values.T > 0)).astype(np.float32)


def fusion_adjacency(spatial, temporal) -> torch.Tensor:
    """Official 4N fusion graph ``[[T, I, I, T], [I, S, I, I], [I, I, S, I], [T, I, I, T]]`` plus self-loops."""
    spatial, temporal = _binary(spatial), _binary(temporal)
    nodes = spatial.shape[0]
    identity = np.eye(nodes, dtype=np.float32)
    blocks = [
        [temporal, identity, identity, temporal],
        [identity, spatial, identity, identity],
        [identity, identity, spatial, identity],
        [temporal, identity, identity, temporal],
    ]
    fused = np.block(blocks)
    np.fill_diagonal(fused, 1.0)
    return torch.from_numpy(fused)


def day_segments(series: torch.Tensor, steps_per_day: int) -> torch.Tensor:
    """``[T, N]`` -> ``[N, L, D]``: whole periods of length ``L``, each z-scored over its own steps.

    With less than one full period the whole series is a single segment.
    """
    length = series.shape[0]
    days = length // steps_per_day
    if days >= 1:
        segments = series[: days * steps_per_day].reshape(days, steps_per_day, -1)
    else:
        segments = series.unsqueeze(0)
    mean = segments.mean(dim=1, keepdim=True)
    std = segments.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
    return ((segments - mean) / std).permute(2, 1, 0).contiguous()


def banded_dtw(profiles: torch.Tensor, band: int, order: int = 1, chunk: int = 32768) -> torch.Tensor:
    """Pairwise DTW ``[N, N]`` restricted to ``|i - j| <= band`` (Alg. 2).

    ``profiles`` is ``[N, L, D]``: a length-``L`` sequence of ``D``-vectors per
    node. The step cost is the order-``order`` norm of the vector difference
    raised to ``order``; the distance is the cumulative cost to the end raised to
    ``1 / order``.
    """
    nodes, length, _ = profiles.shape
    rows, cols = torch.triu_indices(nodes, nodes, offset=1, device=profiles.device)
    distances = torch.zeros(nodes, nodes, dtype=profiles.dtype, device=profiles.device)
    offsets = torch.arange(-band, band + 1, device=profiles.device)
    width = offsets.numel()
    for begin in range(0, rows.numel(), chunk):
        left, right = rows[begin : begin + chunk], cols[begin : begin + chunk]
        a, b = profiles[left], profiles[right]
        pairs = left.numel()
        inf = torch.full((pairs, 1), float("inf"), dtype=profiles.dtype, device=profiles.device)
        previous = torch.full((pairs, width), float("inf"), dtype=profiles.dtype, device=profiles.device)
        for i in range(length):
            columns = i + offsets
            valid = (columns >= 0) & (columns < length)
            difference = a[:, columns.clamp(0, length - 1)] - b[:, i : i + 1]
            cost = torch.linalg.vector_norm(difference, ord=order, dim=-1).pow(order)
            cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))
            # Predecessors in band coordinates: (i-1, j-1) -> same offset, (i-1, j) -> offset + 1.
            diagonal = previous
            above = torch.cat((previous[:, 1:], inf), dim=1)
            vertical = torch.minimum(diagonal, above)
            if i == 0:
                vertical = torch.where(columns == 0, torch.zeros_like(vertical), vertical)
            current = torch.empty_like(cost)
            running = inf.squeeze(1)
            for k in range(width):
                running = cost[:, k] + torch.minimum(vertical[:, k], running)
                current[:, k] = running
            previous = current
        final = previous[:, band].pow(1.0 / order)
        distances[left, right] = final
        distances[right, left] = final
    return distances


def temporal_graph(distances: torch.Tensor, neighbors: int) -> np.ndarray:
    """``neighbors`` smallest-distance other nodes per row, symmetrized, with self-loops."""
    nodes = distances.shape[0]
    scored = distances.detach().double().cpu().clone()
    scored.fill_diagonal_(float("inf"))
    scored = torch.nan_to_num(scored, nan=float("inf"))
    graph = np.eye(nodes, dtype=np.float32)
    count = min(neighbors, nodes - 1)
    if count > 0:
        nearest = torch.argsort(scored, dim=1, stable=True)[:, :count].numpy()
        graph[np.repeat(np.arange(nodes), count), nearest.reshape(-1)] = 1.0
    return np.maximum(graph, graph.T)


class GatedDilatedConv(nn.Module):
    """Eq. 7: ``tanh(Theta_1 * X + a) * sigmoid(Theta_2 * X + b)`` along time, kernel 2, dilation ``K - 1``."""

    def __init__(self, channels: int, magnitude: float) -> None:
        super().__init__()
        self.filter = nn.Conv2d(channels, channels, (1, 2), dilation=(1, _WINDOW - 1))
        self.gate = nn.Conv2d(channels, channels, (1, 2), dilation=(1, _WINDOW - 1))
        for conv in (self.filter, self.gate):
            # MXNet Xavier fans for an (out, in, 1, 2) kernel.
            mxnet_xavier_uniform_(conv.weight, channels * 2, channels * 2, magnitude)
            nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.permute(0, 3, 2, 1)  # [B, C, N, T]
        output = torch.tanh(self.filter(hidden)) * torch.sigmoid(self.gate(hidden))
        return output.permute(0, 3, 2, 1)  # [B, T - 3, N, C]


class STFGNL(nn.Module):
    """Position embeddings, ``T - 3`` fusion-graph modules over 4-step windows, plus the gated conv."""

    def __init__(
        self,
        seq_len: int,
        num_nodes: int,
        width: int,
        filters: list[int],
        temporal_emb: bool,
        spatial_emb: bool,
        gated_conv: bool,
        magnitude: float,
    ) -> None:
        super().__init__()
        windows = seq_len - _WINDOW + 1
        self.temporal_embedding = self._embedding((1, seq_len, 1, width), magnitude) if temporal_emb else None
        self.spatial_embedding = self._embedding((1, 1, num_nodes, width), magnitude) if spatial_emb else None
        # Official crop: rows N:2N (block 1 of the 4-step window).
        self.module = SynchronousGraphModule(
            num_nodes, _WINDOW, width, filters, activation="GLU",
            num_modules=windows, crop=1, init_magnitude=magnitude,
        )
        self.gated_conv = GatedDilatedConv(width, magnitude) if gated_conv else None

    @staticmethod
    def _embedding(shape: tuple[int, ...], magnitude: float) -> nn.Parameter:
        receptive = int(np.prod(shape[2:]))
        return nn.Parameter(
            mxnet_xavier_uniform_(torch.empty(shape), shape[1] * receptive, shape[0] * receptive, magnitude)
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if self.temporal_embedding is not None:
            x = x + self.temporal_embedding
        if self.spatial_embedding is not None:
            x = x + self.spatial_embedding
        batch, steps, nodes, features = x.shape
        windows = x.unfold(1, _WINDOW, 1).permute(0, 1, 4, 2, 3)
        windows = windows.reshape(batch, steps - _WINDOW + 1, _WINDOW * nodes, features)
        output = self.module(windows, adjacency)
        if self.gated_conv is not None:
            output = output + self.gated_conv(x)
        return output


class Model(nn.Module):
    """STFGNN; the temporal graph is the identity until ``fit_temporal_graph`` runs."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        num_layers: int = 3,
        filters: tuple[int, ...] = (64, 64, 64),
        first_layer_embedding: int = 64,
        use_mask: bool = True,
        temporal_emb: bool = True,
        spatial_emb: bool = True,
        gated_conv: bool = True,
        output_hidden: int = 128,
        init_magnitude: float = 0.0003,
        huber_delta: float = 1.0,
        steps_per_day: int = 288,
        dtw_band: int = 12,
        dtw_order: int = 1,
        dtw_sparsity: float = 0.01,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, num_layers, output_hidden, steps_per_day) < 1:
            raise ValueError("lengths, nodes, layers, widths and steps_per_day must be positive")
        if seq_len - (_WINDOW - 1) * num_layers < 1:
            raise ValueError(f"seq_len ({seq_len}) must exceed 3 * num_layers ({3 * num_layers})")
        if gated_conv and filters[-1] != first_layer_embedding:
            raise ValueError("the gated-conv residual needs filters[-1] == first_layer_embedding")
        if dtw_order not in (1, 2) or dtw_band < 0 or not 0.0 <= dtw_sparsity <= 1.0:
            raise ValueError("dtw_order must be 1 or 2, dtw_band >= 0 and dtw_sparsity in [0, 1]")
        if adj_mx is None:
            raise ValueError(
                "STFGNN needs the road-network adjacency adj_mx [enc_in, enc_in] for its fusion graph; "
                "use a graph dataset that ships adj_mx.npy (for example a PEMS preset)"
            )
        spatial = np.asarray(adj_mx, dtype=np.float64)
        if spatial.shape != (enc_in, enc_in):
            raise ValueError(f"adj_mx must have shape {(enc_in, enc_in)}")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, enc_in
        self.huber_delta, self.steps_per_day = huber_delta, steps_per_day
        self.dtw_band, self.dtw_order, self.dtw_sparsity = dtw_band, dtw_order, dtw_sparsity

        self.register_buffer("spatial_graph", torch.from_numpy(_binary(spatial)))
        fused = fusion_adjacency(spatial, np.eye(enc_in))
        self.register_buffer("adjacency", fused)
        self.mask = nn.Parameter((fused != 0).float()) if use_mask else None

        self.input_layer = nn.Linear(1, first_layer_embedding)
        layers = []
        steps, width = seq_len, first_layer_embedding
        for _ in range(num_layers):
            layers.append(
                STFGNL(steps, enc_in, width, list(filters), temporal_emb, spatial_emb, gated_conv, init_magnitude)
            )
            steps, width = steps - (_WINDOW - 1), filters[-1]
        self.layers = nn.ModuleList(layers)
        flat = steps * width
        self.head_hidden = nn.Parameter(torch.empty(pred_len, flat, output_hidden))
        self.head_hidden_bias = nn.Parameter(torch.zeros(pred_len, output_hidden))
        self.head_output = nn.Parameter(torch.empty(pred_len, output_hidden))
        self.head_output_bias = nn.Parameter(torch.zeros(pred_len))
        mxnet_xavier_uniform_(self.input_layer.weight, 1, first_layer_embedding, init_magnitude)
        nn.init.zeros_(self.input_layer.bias)
        for horizon in range(pred_len):
            mxnet_xavier_uniform_(self.head_hidden[horizon], flat, output_hidden, init_magnitude)
            mxnet_xavier_uniform_(self.head_output[horizon], output_hidden, 1, init_magnitude)

    @torch.no_grad()
    def fit_temporal_graph(self, series: torch.Tensor) -> np.ndarray:
        """Build ``A_TG`` from a ``[T, N]`` training series and refresh the fusion graph and mask."""
        if series.ndim != 2 or series.shape[1] != self.num_nodes or series.shape[0] < 2:
            raise ValueError(f"series must be [T >= 2, {self.num_nodes}]")
        device = self.adjacency.device
        profiles = day_segments(series.to(device=device, dtype=torch.float32), self.steps_per_day)
        distances = banded_dtw(profiles, min(self.dtw_band, profiles.shape[1] - 1), self.dtw_order)
        neighbors = max(1, int(self.num_nodes * self.dtw_sparsity)) if self.dtw_sparsity > 0 else 0
        graph = temporal_graph(distances, neighbors)
        fused = fusion_adjacency(self.spatial_graph.cpu().numpy(), graph).to(device)
        self.adjacency.copy_(fused)
        if self.mask is not None:
            self.mask.copy_((fused != 0).float())
        return graph

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.num_nodes}]")
        adjacency = self.adjacency if self.mask is None else self.mask * self.adjacency
        hidden = torch.relu(self.input_layer(x_enc.unsqueeze(-1)))
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
        flat = hidden.permute(0, 2, 1, 3).flatten(2)
        hidden = torch.relu(torch.einsum("bnf,qfh->bqnh", flat, self.head_hidden) + self.head_hidden_bias[:, None])
        return torch.einsum("bqnh,qh->bqn", hidden, self.head_output) + self.head_output_bias[:, None]


def training_series(dataset, num_nodes: int) -> torch.Tensor | None:
    """The ``[T, N]`` value series of the training split behind ``dataset``, or ``None``.

    Index-windowed datasets keep the full series and the window ends in ``idx``;
    only the history span of the training windows is returned, so no validation
    or test step enters the temporal graph.
    """
    data = getattr(dataset, "data", None)
    if data is None:
        return None
    values = torch.as_tensor(np.asarray(data), dtype=torch.float32)
    if values.ndim == 3:
        values = values[..., 0]
    idx = getattr(dataset, "idx", None)
    if idx is not None and len(idx):
        ends = np.asarray(idx).reshape(-1)
        start = max(int(ends.min()) - int(getattr(dataset, "seq_len", 1)) + 1, 0)
        values = values[start : int(ends.max()) + 1]
    if values.ndim != 2 or values.shape[1] != num_nodes or values.shape[0] < 2:
        return None
    return values


def fit_from_loader(model: Model, train_loader) -> bool:
    """Fit ``A_TG`` from the loader's training split; warn and keep the identity graph otherwise."""
    series = training_series(getattr(train_loader, "dataset", None), model.num_nodes)
    if series is None:
        warnings.warn("STFGNN: no ordered training series available; the temporal graph stays the identity",
                      stacklevel=2)
        return False
    model.fit_temporal_graph(series)
    return True


__all__ = ["Model", "STFGNL", "GatedDilatedConv", "banded_dtw", "fusion_adjacency", "fit_from_loader"]
