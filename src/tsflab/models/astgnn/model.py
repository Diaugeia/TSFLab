"""ASTGNN: trend-aware attention encoder-decoder with dynamic graph convolution.

Independent implementation of Guo et al. (IEEE TKDE 2022). The official
repository (guoshnBJTU/ASTGNN, no license) was inspected only to resolve tensor
order, residual placement, decoding, and initialization; no code was copied.

Recent-segment form (the paper's ASTGNN; the periodic ASTGNN(p) variant needs
separately sampled weekly/daily windows that the runtime batch does not carry):

- Input projection, sinusoidal temporal position, learned node (spatial
  heterogeneity) embedding, optionally smoothed by graph convolutions.
- Encoder layer: trend-aware multi-head self-attention (queries and keys are
  local 1-D convolutions over time, values linear), then a dynamic graph
  convolution ``relu(((A_hat * S) Z) W)`` with ``S = softmax(Z Z^T / sqrt(d)) / sqrt(d)``
  per time step; pre-LayerNorm residual sublayers.
- Decoder layer: causal trend-aware self-attention, trend-aware encoder-decoder
  attention (causal conv on queries, centered conv on memory keys), dynamic
  graph convolution. Prediction is autoregressive from the last observation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.adj_norm import transition_matrix
from tsflab.models._components.embed import PositionalEmbedding


class GraphConvolution(nn.Module):
    """``relu((A_hat X) Theta)`` with a bias-free ``Theta`` (used for embedding smoothing)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.theta = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return F.relu(self.theta(torch.matmul(adjacency, x)))


class DynamicGraphConvolution(nn.Module):
    """Graph convolution whose static adjacency is reweighted by spatial self-attention.

    For every sample and time step: ``S = softmax(Z Z^T / sqrt(d)) / sqrt(d)`` and
    ``Z' = relu(((A_hat elementwise S) Z) Theta)``.
    """

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.theta = nn.Linear(width, width, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, d] -> per-step node sets [B, T, N, d]
        width = x.shape[-1]
        nodes_first = x.transpose(1, 2)
        scores = torch.matmul(nodes_first, nodes_first.transpose(-1, -2)) / math.sqrt(width)
        attention = self.attention_dropout(torch.softmax(scores, dim=-1)) / math.sqrt(width)
        propagated = torch.matmul(adjacency * attention, nodes_first)
        out = F.relu(self.theta(propagated)).transpose(1, 2)
        return self.output_dropout(out)


class TrendAwareAttention(nn.Module):
    """Multi-head attention with convolutional query/key projections over time.

    ``query_causal`` / ``key_causal`` choose left-padded (causal) or centered
    convolutions of width ``kernel_size``; values use a linear projection.
    """

    def __init__(
        self, width: int, heads: int, kernel_size: int, dropout: float,
        query_causal: bool, key_causal: bool,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.heads = heads
        self.head_width = width // heads
        self.kernel_size = kernel_size
        self.query_causal = query_causal
        self.key_causal = key_causal
        self.query_conv = nn.Conv2d(width, width, (1, kernel_size))
        self.key_conv = nn.Conv2d(width, width, (1, kernel_size))
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def _convolve(self, conv: nn.Conv2d, x: torch.Tensor, causal: bool) -> torch.Tensor:
        # x: [B, N, T, d] -> conv over T keeping length T
        channels = x.permute(0, 3, 1, 2)
        total = self.kernel_size - 1
        left = total if causal else total // 2
        channels = F.pad(channels, (left, total - left))
        return conv(channels).permute(0, 2, 3, 1)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, steps, _ = x.shape
        return x.view(batch, nodes, steps, self.heads, self.head_width).transpose(2, 3)

    def forward(self, query, key, value, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._split(self._convolve(self.query_conv, query, self.query_causal))
        k = self._split(self._convolve(self.key_conv, key, self.key_causal))
        v = self._split(self.value(value))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_width)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        mixed = torch.matmul(weights, v).transpose(2, 3)
        batch, nodes, steps = mixed.shape[:3]
        return self.output(mixed.reshape(batch, nodes, steps, -1))


class PreNormResidual(nn.Module):
    """``x + dropout(sublayer(LayerNorm(x)))``."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, width, heads, kernel_size, dropout) -> None:
        super().__init__()
        self.attention = TrendAwareAttention(width, heads, kernel_size, dropout, False, False)
        self.graph = DynamicGraphConvolution(width, dropout)
        self.sublayers = nn.ModuleList([PreNormResidual(width, dropout) for _ in range(2)])

    def forward(self, x, adjacency):
        x = self.sublayers[0](x, lambda h: self.attention(h, h, h))
        return self.sublayers[1](x, lambda h: self.graph(h, adjacency))


class DecoderLayer(nn.Module):
    def __init__(self, width, heads, kernel_size, dropout) -> None:
        super().__init__()
        self.self_attention = TrendAwareAttention(width, heads, kernel_size, dropout, True, True)
        self.cross_attention = TrendAwareAttention(width, heads, kernel_size, dropout, True, False)
        self.graph = DynamicGraphConvolution(width, dropout)
        self.sublayers = nn.ModuleList([PreNormResidual(width, dropout) for _ in range(3)])

    def forward(self, x, memory, adjacency, mask):
        x = self.sublayers[0](x, lambda h: self.self_attention(h, h, h, mask))
        x = self.sublayers[1](x, lambda h: self.cross_attention(h, memory, memory))
        return self.sublayers[2](x, lambda h: self.graph(h, adjacency))


class SpatialEmbedding(nn.Module):
    """Learned per-node embedding (spatial heterogeneity), optionally graph-smoothed."""

    def __init__(self, nodes: int, width: int, smooth_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(nodes, width)
        self.smoothing = nn.ModuleList([GraphConvolution(width) for _ in range(smooth_layers)])

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding.weight
        for layer in self.smoothing:
            embedding = layer(embedding, adjacency)
        return embedding  # [N, d]


class Model(nn.Module):
    """ASTGNN on the recent history window with autoregressive decoding."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        d_model: int = 64,
        n_heads: int = 8,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
        smooth_layer_num: int = 0,
        teacher_forcing: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, d_model, n_heads, num_layers, kernel_size) < 1:
            raise ValueError("lengths, nodes, widths, heads, layers and kernel must be positive")
        if smooth_layer_num < 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("smooth_layer_num must be >= 0 and dropout in [0, 1)")
        if adj_mx is None:
            raise ValueError(
                "ASTGNN needs the road-network adjacency adj_mx [enc_in, enc_in]; "
                "use a graph dataset that ships adj_mx.npy (for example a PEMS preset)"
            )
        adjacency = np.asarray(adj_mx, dtype=np.float64)
        if adjacency.shape != (enc_in, enc_in):
            raise ValueError(f"adj_mx must have shape {(enc_in, enc_in)}")
        # A_hat = D^-1 (A + I): self-loops set to one, then row normalization.
        adjacency = adjacency.copy()
        np.fill_diagonal(adjacency, 1.0)
        self.register_buffer(
            "adjacency", torch.as_tensor(transition_matrix(adjacency), dtype=torch.float32)
        )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = enc_in
        self.teacher_forcing = teacher_forcing

        self.source_projection = nn.Linear(1, d_model)
        self.target_projection = nn.Linear(1, d_model)
        if d_model % 2:
            raise ValueError("d_model must be even for the sinusoidal position table")
        # Transformer sinusoid sin/cos(pos / 10000^(2k/d)), fixed (not trained).
        self.source_position = PositionalEmbedding(d_model, max_len=seq_len)
        self.target_position = PositionalEmbedding(d_model, max_len=pred_len)
        self.source_spatial = SpatialEmbedding(enc_in, d_model, smooth_layer_num)
        self.target_spatial = SpatialEmbedding(enc_in, d_model, smooth_layer_num)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, 1)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _check(self, x_enc: torch.Tensor) -> None:
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.num_nodes}]")

    def encode(self, x_enc: torch.Tensor) -> torch.Tensor:
        """``[B, L, N]`` history -> memory ``[B, N, L, d]``."""
        h = self.source_projection(x_enc.transpose(1, 2).unsqueeze(-1))
        h = h + self.source_position(x_enc)
        h = h + self.source_spatial(self.adjacency).unsqueeze(1)
        h = self.embedding_dropout(h)
        for layer in self.encoder_layers:
            h = layer(h, self.adjacency)
        return self.encoder_norm(h)

    def decode(self, decoder_input: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Decoder values ``[B, N, T']`` -> one-step-shifted predictions ``[B, N, T']``."""
        steps = decoder_input.shape[-1]
        h = self.target_projection(decoder_input.unsqueeze(-1))
        h = h + self.target_position(decoder_input.transpose(1, 2))
        h = h + self.target_spatial(self.adjacency).unsqueeze(1)
        h = self.embedding_dropout(h)
        mask = torch.ones(steps, steps, dtype=torch.bool, device=h.device).tril()
        for layer in self.decoder_layers:
            h = layer(h, memory, self.adjacency, mask)
        return self.generator(self.decoder_norm(h)).squeeze(-1)

    def teacher_forced(self, x_enc: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Training pass with decoder input ``[x_L, y_1, ..., y_{P-1}]`` (``target`` ``[B, P, N]``)."""
        self._check(x_enc)
        start = x_enc[:, -1:, :]
        decoder_input = torch.cat((start, target[:, :-1, :]), dim=1).transpose(1, 2)
        return self.decode(decoder_input, self.encode(x_enc)).transpose(1, 2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        self._check(x_enc)
        memory = self.encode(x_enc)
        start = x_enc[:, -1, :].unsqueeze(-1)  # [B, N, 1]
        decoder_input = start
        prediction = start
        for _ in range(self.pred_len):
            prediction = self.decode(decoder_input, memory)
            decoder_input = torch.cat((start, prediction), dim=-1)
        return prediction.transpose(1, 2)
