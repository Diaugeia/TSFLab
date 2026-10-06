"""GMAN: graph multi-attention encoder-decoder for traffic forecasting.

Independent PyTorch implementation of Zheng et al. (AAAI 2020) checked against the
official TensorFlow code at revision 45ed232f. Structure, in execution order:

* spatio-temporal embedding (STE): node2vec spatial embedding of the road graph
  through a two-layer FC, plus one-hot ``[day-of-week, time-of-day]`` through a
  two-layer FC, summed to ``[B, P + Q, N, D]``;
* encoder: ``L`` ST-attention blocks ``H + gated_fusion(spatial(H, STE), temporal(H, STE))``;
* transform attention: each future step attends to every history step with
  queries from ``STE_Q``, keys from ``STE_P`` and values from the encoder output;
* decoder: ``L`` ST-attention blocks on the future representation;
* output: two-layer FC to one value per node and step.

Every "FC" is a position-wise linear map; a nonlinear one is followed by batch
normalization over all positions and ReLU, as in the official ``FC``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsflab.models._components.gated_fusion import GatedFusion
from tsflab.models._components.marks import TIME_FEATURES
from tsflab.models._components.node2vec_embedding import node2vec_embedding

# Official masked-score value (``-2 ** 15 + 1``) for the causal temporal mask.
_MASKED_SCORE = -(2.0**15) + 1.0


def _glorot(linear: nn.Linear) -> nn.Linear:
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


class FCLayer(nn.Module):
    """Position-wise linear; when ``activate``: batch norm over all positions, then ReLU."""

    def __init__(self, in_dim: int, out_dim: int, activate: bool, bias: bool, momentum: float) -> None:
        super().__init__()
        self.linear = _glorot(nn.Linear(in_dim, out_dim, bias=bias))
        self.norm = nn.BatchNorm1d(out_dim, eps=1e-3, momentum=momentum) if activate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.norm is None:
            return x
        shape = x.shape
        return torch.relu(self.norm(x.reshape(-1, shape[-1])).reshape(shape))


class FC(nn.Sequential):
    """Official two-layer FC ``[D, D]`` (ReLU then linear) or a single layer."""

    def __init__(
        self,
        dims: list[int],
        activations: list[bool],
        momentum: float,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        layers: list[nn.Module] = []
        for in_dim, out_dim, activate in zip(dims[:-1], dims[1:], activations):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(FCLayer(in_dim, out_dim, activate, bias, momentum))
        super().__init__(*layers)


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """``[..., heads * d] -> [heads, ..., d]`` (the official split-and-concat on batch)."""
    return torch.stack(x.chunk(heads, dim=-1), dim=0)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(x.unbind(0), dim=-1)


class SpatialAttention(nn.Module):
    """Eqs. 3-5: multi-head attention over all nodes at each time step."""

    def __init__(self, heads: int, head_dim: int, momentum: float) -> None:
        super().__init__()
        width = heads * head_dim
        self.heads, self.scale = heads, head_dim**0.5
        self.query = FC([2 * width, width], [True], momentum)
        self.key = FC([2 * width, width], [True], momentum)
        self.value = FC([2 * width, width], [True], momentum)
        self.output = FC([width, width, width], [True, False], momentum)

    def forward(self, x: torch.Tensor, ste: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat((x, ste), dim=-1)
        query = _split_heads(self.query(hidden), self.heads)  # [K, B, T, N, d]
        key = _split_heads(self.key(hidden), self.heads)
        value = _split_heads(self.value(hidden), self.heads)
        scores = torch.softmax(query @ key.transpose(-1, -2) / self.scale, dim=-1)
        return self.output(_merge_heads(scores @ value))


class TemporalAttention(nn.Module):
    """Eqs. 8-10: multi-head attention over time steps of each node, optionally causal."""

    def __init__(self, heads: int, head_dim: int, momentum: float, causal: bool) -> None:
        super().__init__()
        width = heads * head_dim
        self.heads, self.scale, self.causal = heads, head_dim**0.5, causal
        self.query = FC([2 * width, width], [True], momentum)
        self.key = FC([2 * width, width], [True], momentum)
        self.value = FC([2 * width, width], [True], momentum)
        self.output = FC([width, width, width], [True, False], momentum)

    def forward(self, x: torch.Tensor, ste: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat((x, ste), dim=-1)
        # [K, B, N, T, d]
        query = _split_heads(self.query(hidden), self.heads).transpose(2, 3)
        key = _split_heads(self.key(hidden), self.heads).transpose(2, 3)
        value = _split_heads(self.value(hidden), self.heads).transpose(2, 3)
        scores = query @ key.transpose(-1, -2) / self.scale
        if self.causal:
            steps = scores.shape[-1]
            allowed = torch.ones(steps, steps, dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~allowed, _MASKED_SCORE)
        attended = torch.softmax(scores, dim=-1) @ value
        return self.output(_merge_heads(attended.transpose(2, 3)))


class STAttBlock(nn.Module):
    """Spatial and temporal attention fused by Eqs. 11-12, then a residual connection."""

    def __init__(self, heads: int, head_dim: int, momentum: float, causal: bool) -> None:
        super().__init__()
        width = heads * head_dim
        self.spatial = SpatialAttention(heads, head_dim, momentum)
        self.temporal = TemporalAttention(heads, head_dim, momentum, causal)
        # z = sigmoid(H_S W_z1 + H_T W_z2 + b_z); H = z * H_S + (1 - z) * H_T
        self.fusion = GatedFusion(width)
        _glorot(self.fusion.gate_a)
        _glorot(self.fusion.gate_b)
        self.fusion_output = FC([width, width, width], [True, False], momentum)

    def forward(self, x: torch.Tensor, ste: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(self.spatial(x, ste), self.temporal(x, ste))
        return x + self.fusion_output(fused)


class TransformAttention(nn.Module):
    """Eqs. 13-15: future steps attend to history steps through the STE."""

    def __init__(self, heads: int, head_dim: int, momentum: float) -> None:
        super().__init__()
        width = heads * head_dim
        self.heads, self.scale = heads, head_dim**0.5
        self.query = FC([width, width], [True], momentum)
        self.key = FC([width, width], [True], momentum)
        self.value = FC([width, width], [True], momentum)
        self.output = FC([width, width, width], [True, False], momentum)

    def forward(self, x: torch.Tensor, ste_history: torch.Tensor, ste_future: torch.Tensor) -> torch.Tensor:
        query = _split_heads(self.query(ste_future), self.heads).transpose(2, 3)  # [K, B, N, Q, d]
        key = _split_heads(self.key(ste_history), self.heads).transpose(2, 3)  # [K, B, N, P, d]
        value = _split_heads(self.value(x), self.heads).transpose(2, 3)
        scores = torch.softmax(query @ key.transpose(-1, -2) / self.scale, dim=-1)
        return self.output(_merge_heads((scores @ value).transpose(2, 3)))


class Model(nn.Module):
    """GMAN with a node2vec spatial embedding computed from the injected road graph."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        num_blocks: int = 3,
        num_heads: int = 8,
        head_dim: int = 8,
        steps_per_day: int = 288,
        causal_temporal: bool = True,
        output_dropout: float = 0.1,
        bn_momentum: float = 0.01,
        se_dim: int = 64,
        node2vec_p: float = 2.0,
        node2vec_q: float = 1.0,
        node2vec_walks: int = 100,
        node2vec_walk_length: int = 80,
        node2vec_window: int = 10,
        node2vec_steps: int = 1000,
        node2vec_seed: int = 0,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, num_blocks, num_heads, head_dim, steps_per_day, se_dim) < 1:
            raise ValueError("lengths, nodes, blocks, heads, widths and steps_per_day must be positive")
        if not 0.0 <= output_dropout < 1.0:
            raise ValueError("output_dropout must be in [0, 1)")
        if adj_mx is None:
            raise ValueError(
                "GMAN needs the road-network adjacency adj_mx [enc_in, enc_in] for its node2vec "
                "spatial embedding; use a graph dataset that ships adj_mx.npy (for example a PEMS preset)"
            )
        adjacency = np.asarray(adj_mx, dtype=np.float64)
        if adjacency.shape != (enc_in, enc_in):
            raise ValueError(f"adj_mx must have shape {(enc_in, enc_in)}")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, enc_in
        self.steps_per_day = steps_per_day
        width = num_heads * head_dim
        momentum = bn_momentum

        # Pre-learned node2vec vectors are fixed; only the FC on top is trained.
        spatial = node2vec_embedding(
            adjacency, se_dim, p=node2vec_p, q=node2vec_q, num_walks=node2vec_walks,
            walk_length=node2vec_walk_length, window=node2vec_window,
            steps=node2vec_steps, seed=node2vec_seed,
        )
        self.register_buffer("spatial_embedding", spatial)
        self.spatial_fc = FC([se_dim, width, width], [True, False], momentum)
        self.temporal_fc = FC([7 + steps_per_day, width, width], [True, False], momentum)
        self.input_fc = FC([1, width, width], [True, False], momentum)
        self.encoder = nn.ModuleList(
            STAttBlock(num_heads, head_dim, momentum, causal_temporal) for _ in range(num_blocks)
        )
        self.transform = TransformAttention(num_heads, head_dim, momentum)
        self.decoder = nn.ModuleList(
            STAttBlock(num_heads, head_dim, momentum, causal_temporal) for _ in range(num_blocks)
        )
        self.output_fc = FC([width, width, 1], [True, False], momentum, dropout=output_dropout)

    def _calendar(self, marks: torch.Tensor | None, steps: int) -> torch.Tensor | None:
        """``[B, steps, 2]`` integer ``(day-of-week, time-of-day)`` from the newest ``steps`` marks."""
        if marks is None or marks.shape[1] < steps:
            return None
        marks = marks[:, -steps:]
        if marks.dim() == 4:
            if marks.shape[-1] != TIME_FEATURES:
                raise ValueError(
                    "GMAN reads node-structured marks only as [time_in_day, day_in_week] "
                    f"({TIME_FEATURES} features), got {marks.shape[-1]}"
                )
            first = marks[:, :, 0].float()
            tod = torch.floor(first[..., 0] * self.steps_per_day + 1e-4).long()
            dow = torch.floor(first[..., 1] * 7 + 1e-4).long()
        elif marks.dim() == 3 and marks.shape[-1] >= 6:
            raw = marks.round().long()
            tod = (raw[..., 4] * 60 + raw[..., 5]) * self.steps_per_day // 1440
            dow = raw[..., 3]
        else:
            raise ValueError("marks must be raw [B, T, 6] stamps or node-structured [B, T, N, 2]")
        return torch.stack((dow.remainder(7), tod.clamp(0, self.steps_per_day - 1)), dim=-1)

    def _future_calendar(self, history: torch.Tensor) -> torch.Tensor:
        """Extend the last history stamp by one step per horizon (regular sampling)."""
        offsets = torch.arange(1, self.pred_len + 1, device=history.device)
        ticks = history[:, -1:, 1] + offsets
        dow = (history[:, -1:, 0] + torch.div(ticks, self.steps_per_day, rounding_mode="floor")).remainder(7)
        return torch.stack((dow, ticks.remainder(self.steps_per_day)), dim=-1)

    def _temporal_embedding(self, calendar: torch.Tensor) -> torch.Tensor:
        one_hot = torch.cat(
            (
                nn.functional.one_hot(calendar[..., 0], 7),
                nn.functional.one_hot(calendar[..., 1], self.steps_per_day),
            ),
            dim=-1,
        ).float()
        return self.temporal_fc(one_hot).unsqueeze(2)  # [B, T, 1, D]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.num_nodes}]")
        batch = x_enc.shape[0]
        history = self._calendar(x_mark_enc, self.seq_len)
        if history is None:
            history = torch.zeros(batch, self.seq_len, 2, dtype=torch.long, device=x_enc.device)
        future = self._calendar(x_mark_dec, self.pred_len)
        if future is None:
            future = self._future_calendar(history)
        temporal = self._temporal_embedding(torch.cat((history, future), dim=1))
        spatial = self.spatial_fc(self.spatial_embedding).view(1, 1, self.num_nodes, -1)
        ste = spatial + temporal  # [B, P + Q, N, D]
        ste_history, ste_future = ste[:, : self.seq_len], ste[:, self.seq_len :]

        hidden = self.input_fc(x_enc.unsqueeze(-1))
        for block in self.encoder:
            hidden = block(hidden, ste_history)
        hidden = self.transform(hidden, ste_history, ste_future)
        for block in self.decoder:
            hidden = block(hidden, ste_future)
        return self.output_fc(hidden).squeeze(-1)


__all__ = ["Model"]
