"""Independent MoFo implementation from the NeurIPS 2025 paper equations.

Each channel's look-back window is padded to whole periods (Eq. 1), sampled
into ``P`` period-structured patches whose rows hold period-aligned steps
(Eq. 2), and embedded (Eq. 3). A pre-norm Transformer attends across the ``P``
patches with a regulated-relaxation modulator over the circular period
distance (Eqs. 4, 9, 12), and a flatten-linear head forecasts the horizon
(Eq. 14). The official repository fills omissions (normalization, FFN, the
phase-identity embedding); the card records every difference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.marks import elapsed_minutes
from tsflab.models._components.revin import RevIN

#: Dropout of the SwiGLU feed-forward in the reference implementation.
FFN_DROPOUT = 0.3


def pad_to_periods(values: torch.Tensor, period: int) -> torch.Tensor:
    """Eq. (1): prepend ``X[(T mod P):P]`` so the length becomes ``P * ceil(T / P)``.

    ``values`` is ``[..., T]``. Periods are delineated backward from the last
    step, so the last observation always ends the final period row.
    """
    remainder = values.shape[-1] % period
    if not remainder:
        return values
    return torch.cat((values[..., remainder:period], values), dim=-1)


def period_structured_patches(values: torch.Tensor, period: int) -> torch.Tensor:
    """Eq. (2): ``[..., T] -> [..., P, ceil(T / P)]``; row ``i`` is ``[x_i, x_{i+P}, ...]``."""
    padded = pad_to_periods(values, period)
    cycles = padded.shape[-1] // period
    return padded.unflatten(-1, (cycles, period)).transpose(-1, -2)


def circular_period_distance(period: int) -> torch.Tensor:
    """Eq. (4): ``gamma_ij = min((i - j) mod P, (j - i) mod P)`` as a ``[P, P]`` float tensor."""
    index = torch.arange(period)
    difference = index[:, None] - index[None, :]
    return torch.minimum(difference.remainder(period), (-difference).remainder(period)).float()


def last_step_phase(marks: torch.Tensor, period: int) -> torch.Tensor:
    """Phase of the last input step within the period, from raw calendar marks.

    The phase is the absolute step count since 1970-01-01 00:00 modulo ``period``;
    the step length is the minute gap between the last two marks (at least one
    minute). For hourly data and ``period = 24`` this is the hour of day.
    """
    minutes = elapsed_minutes(marks[:, -2:])
    step = (minutes[:, 1] - minutes[:, 0]).clamp_min(1)
    return torch.div(minutes[:, 1], step, rounding_mode="floor").remainder(period)


class RMSNorm(nn.Module):
    """Eq. (34): root-mean-square normalization with a learnable scale and offset."""

    def __init__(self, width: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(width))
        self.offset = nn.Parameter(torch.zeros(width))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        rms = values.norm(2, dim=-1, keepdim=True) * values.shape[-1] ** -0.5
        return self.scale * values / (rms + self.eps) + self.offset


class SwiGLUFeedForward(nn.Module):
    """Eq. (33): ``(SiLU(Z W1 + b1) * (Z W2 + b2)) W3 + b3`` with a ``4d`` hidden width."""

    def __init__(self, width: int, expand: int = 4) -> None:
        super().__init__()
        self.gate = nn.Linear(width, expand * width)
        self.value = nn.Linear(width, expand * width)
        self.output = nn.Linear(expand * width, width)
        self.dropout = nn.Dropout(FFN_DROPOUT)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(F.silu(self.gate(values)) * self.value(values)))


class RegulatedRelaxation(nn.Module):
    """Eq. (9): ``S = 1 / (1 + exp(alpha (gamma - beta))) + exp(-gamma) / (1 + exp(alpha beta))``.

    ``alpha = sigmoid(a)`` in ``(0, 1)`` and ``beta = P * sigmoid(b)`` in ``(0, P)``
    are learnable; ``a = b = 0`` starts at ``alpha = 0.5``, ``beta = P / 2``.
    """

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        self.alpha_logit = nn.Parameter(torch.zeros(()))
        self.beta_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer("distance", circular_period_distance(period), persistent=False)

    def forward(self) -> torch.Tensor:
        alpha = self.alpha_logit.sigmoid()
        beta = self.beta_logit.sigmoid() * self.period
        gamma = self.distance
        return torch.sigmoid(-alpha * (gamma - beta)) + torch.exp(-gamma) * torch.sigmoid(-alpha * beta)


class PeriodModulatedAttention(nn.Module):
    """Eq. (12): multi-head attention over the ``P`` patches plus ``log S(Gamma; alpha, beta)``."""

    def __init__(self, width: int, period: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by head")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.modulator = RegulatedRelaxation(period)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, width = tokens.shape
        qkv = self.qkv(tokens).view(batch, count, self.heads, 3 * self.head_width)
        query, key, value = (part.transpose(1, 2) for part in qkv.chunk(3, dim=-1))
        scores = query @ key.transpose(-1, -2) * self.head_width ** -0.5
        weights = (scores + self.modulator().log()).softmax(-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, count, width)
        return self.output(attended)


class MoFoLayer(nn.Module):
    """Pre-norm layer: attention block, then SwiGLU block, each with a residual."""

    def __init__(self, width: int, period: int, heads: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(width)
        self.attention = PeriodModulatedAttention(width, period, heads)
        self.feed_forward_norm = RMSNorm(width)
        self.feed_forward = SwiGLUFeedForward(width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.feed_forward(self.feed_forward_norm(tokens))


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int,
                 d_model: int = 64, periodic: int = 24, head: int = 4,
                 d_layers: int = 1, bias: int = 1, cias: int = 1) -> None:
        super().__init__()
        if periodic < 2:
            raise ValueError("periodic must be >= 2")
        if seq_len < periodic:
            raise ValueError(
                f"MoFo needs seq_len >= periodic ({seq_len} < {periodic}): Eq. (1) pads "
                "the first incomplete period from the first complete one"
            )
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.period = periodic
        self.cycles = -(-seq_len // periodic)
        self.normalization = RevIN(enc_in, affine=True)
        self.embedding = nn.Linear(self.cycles, d_model)
        self.channel_bias = None
        if bias:
            self.channel_bias = nn.Parameter(torch.empty(1, enc_in, 1, d_model))
            nn.init.xavier_normal_(self.channel_bias)
        self.phase_embedding = None
        if cias:
            self.phase_embedding = nn.Parameter(torch.empty(periodic, d_model))
            nn.init.xavier_normal_(self.phase_embedding)
        self.layers = nn.ModuleList([MoFoLayer(d_model, periodic, head) for _ in range(d_layers)])
        self.head = nn.Linear(periodic * d_model, pred_len)

    def _patch_offsets(self, x_mark_enc: torch.Tensor | None, batch: int) -> torch.Tensor:
        """Phase-identity embedding per patch: index ``(phase_last - i) mod P`` for patch ``i``."""
        if x_mark_enc is None or x_mark_enc.ndim != 3 or x_mark_enc.shape[-1] != 6:
            raise ValueError("MoFo with cias = 1 needs raw marks [batch, seq_len, 6]")
        phase = last_step_phase(x_mark_enc, self.period)
        patches = torch.arange(self.period, device=phase.device)
        index = (phase[:, None] - patches[None, :]).remainder(self.period)
        return self.phase_embedding[index].view(batch, 1, self.period, -1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}")
        batch = x_enc.shape[0]
        normalized = self.normalization(x_enc, "norm").transpose(1, 2)
        tokens = self.embedding(period_structured_patches(normalized, self.period))
        if self.phase_embedding is not None:
            tokens = tokens + self._patch_offsets(x_mark_enc, batch)
        if self.channel_bias is not None:
            tokens = tokens + self.channel_bias
        tokens = tokens.reshape(batch * self.enc_in, self.period, -1)
        for layer in self.layers:
            tokens = layer(tokens)
        forecast = self.head(tokens.flatten(1)).view(batch, self.enc_in, self.pred_len)
        return self.normalization(forecast.transpose(1, 2), "denorm")
