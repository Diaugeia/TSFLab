"""Local BasisFormer: learnable timestamp basis, bidirectional cross-attention
coefficients, and a coefficient-weighted basis forecast (Ni et al., NeurIPS 2023).

Paper map: Coef module (Sec. 3.1, Eqs. 1-4), Forecast module (Sec. 3.2, Eq. 5),
Basis module with InfoNCE alignment and curvature smoothness (Sec. 3.3, Eqs. 6-9).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from tsflab.models._components.marks import days_from_civil

EPSILON = 1e-5


def _wn_linear(in_features: int, out_features: int) -> nn.Module:
    """Weight-normalized linear map, used for every projection of the method.

    With one input feature (the timestamp MLP's first and skip layers) each
    weight row is a scalar, so ``w = g * v / |v| = g * sign(v)``: the forecast
    loss gives ``v`` no gradient, and any weight decay shrinks ``v`` towards
    zero until ``|v|`` underflows and ``w`` becomes NaN. That layer is the same
    function as a plain linear map with weight ``g * sign(v)``, so it is kept
    plain (same initialization as ``weight_norm``, where ``g = |v|``).
    """
    if in_features == 1:
        return nn.Linear(in_features, out_features)
    return weight_norm(nn.Linear(in_features, out_features))


def normalized_timestamp(
    marks: torch.Tensor | None,
    batch: int,
    device: torch.device,
    origin_year: int,
    span_years: float,
) -> torch.Tensor:
    """Scalar ``tau`` of the first history step, shape ``[batch, 1]``.

    The paper feeds ``tau = t / T`` of the window's first time point to the basis
    MLP. Raw six-column marks ``[year, month, day, weekday, hour, minute]`` carry
    the absolute time, so ``tau`` is the elapsed time since January 1 of
    ``origin_year`` in units of ``span_years``. Missing marks give ``tau = 0``.
    """
    if marks is None or marks.ndim != 3 or marks.shape[-1] < 6:
        return torch.zeros(batch, 1, device=device)
    stamp = marks[:, 0].round().long()
    days = days_from_civil(stamp[:, 0], stamp[:, 1], stamp[:, 2])
    one = torch.ones((), dtype=torch.long, device=marks.device)
    origin = days_from_civil(one * origin_year, one, one)
    minutes = (days - origin) * 1440 + stamp[:, 4] * 60 + stamp[:, 5]
    span_minutes = span_years * 365.25 * 1440.0
    return (minutes.double() / span_minutes).float().unsqueeze(-1).to(device)


class BottleneckMLP(nn.Module):
    """Four-layer bottleneck MLP with a skip from the input to the second layer."""

    def __init__(self, input_dim: int, output_dim: int, bottleneck: int) -> None:
        super().__init__()
        self.inner = nn.Sequential(
            _wn_linear(input_dim, bottleneck), nn.ReLU(), _wn_linear(bottleneck, bottleneck)
        )
        self.skip = _wn_linear(input_dim, bottleneck)
        self.outer = nn.Sequential(
            _wn_linear(bottleneck, bottleneck), nn.ReLU(), _wn_linear(bottleneck, output_dim)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.outer(F.relu(self.inner(values) + self.skip(values)))


class CrossAttention(nn.Module):
    """Multi-head attention ``MA_H(a, b, b)``: queries from ``a``, keys/values from ``b``."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        head_dim = d_model // heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.query = _wn_linear(d_model, head_dim * heads)
        self.key = _wn_linear(d_model, head_dim * heads)
        self.value = _wn_linear(d_model, head_dim * heads)
        self.output = _wn_linear(head_dim * heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        return tokens.view(batch, length, self.heads, -1).transpose(1, 2)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self._split(self.query(queries))
        k = self._split(self.key(context))
        v = self._split(self.value(context))
        weights = self.dropout(torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1))
        mixed = (weights @ v).transpose(1, 2).reshape(queries.shape[0], queries.shape[1], -1)
        return self.output(mixed)


class CrossAttentionBlock(nn.Module):
    """Eqs. (1)-(2): ``CAB_H(a, b) = LN(FFN(a') + a')`` with ``a' = LN(MA_H(a, b, b) + a)``."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = CrossAttention(d_model, heads, dropout)
        self.ffn_in = _wn_linear(d_model, 4 * d_model)
        self.ffn_out = _wn_linear(4 * d_model, d_model)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        tokens = self.attention_norm(tokens + self.dropout(self.attention(tokens, context)))
        update = self.dropout(self.ffn_out(self.dropout(F.relu(self.ffn_in(tokens)))))
        return self.ffn_norm(tokens + update)


class BidirectionalCrossAttentionBlock(nn.Module):
    """Eqs. (3)-(4): both sets update from the *previous* representations, with
    separate parameters for the basis and the series directions."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.basis_block = CrossAttentionBlock(d_model, heads, dropout)
        self.series_block = CrossAttentionBlock(d_model, heads, dropout)

    def forward(
        self, basis: torch.Tensor, series: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.basis_block(basis, series), self.series_block(series, basis)


class CoefModule(nn.Module):
    """Sec. 3.1: ``M`` BCAB layers, then per-head scaled inner products ``c [B, H, C, N]``."""

    def __init__(self, d_model: int, heads: int, blocks: int, dropout: float) -> None:
        super().__init__()
        head_dim = d_model // heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.blocks = nn.ModuleList(
            BidirectionalCrossAttentionBlock(d_model, heads, dropout) for _ in range(blocks)
        )
        self.series_head = _wn_linear(d_model, head_dim * heads)
        self.basis_head = _wn_linear(d_model, head_dim * heads)

    def forward(self, basis: torch.Tensor, series: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            basis, series = block(basis, series)
        batch = series.shape[0]
        s = self.series_head(series).view(batch, series.shape[1], self.heads, -1).transpose(1, 2)
        b = self.basis_head(basis).view(batch, basis.shape[1], self.heads, -1).transpose(1, 2)
        return s @ b.transpose(-1, -2) * self.scale


class Model(nn.Module):
    """BasisFormer point forecaster exposing the paper's alignment and smoothness terms."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 100,
        heads: int = 16,
        basis_nums: int = 10,
        block_nums: int = 2,
        bottleneck: int = 2,
        map_bottleneck: int = 20,
        tau: float = 0.07,
        dropout: float = 0.1,
        timestamp_origin_year: int = 2010,
        timestamp_span_years: float = 10.0,
        loss_weight_infonce: float = 1.0,
        loss_weight_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, d_model, heads, basis_nums, bottleneck, map_bottleneck) < 1:
            raise ValueError("lengths, channels, widths, heads, and basis count must be positive")
        if block_nums < 0:
            raise ValueError("block_nums must be non-negative")
        if d_model < heads:
            raise ValueError("d_model must be at least heads")
        if pred_len < heads:
            raise ValueError("pred_len must be at least heads (the future basis is split into heads)")
        if pred_len < bottleneck:
            raise ValueError("pred_len must be at least bottleneck")
        if tau <= 0 or timestamp_span_years <= 0:
            raise ValueError("tau and timestamp_span_years must be positive")
        if loss_weight_infonce < 0 or loss_weight_smooth < 0:
            raise ValueError("loss weights must be non-negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.heads, self.basis_nums, self.tau = heads, basis_nums, tau
        self.timestamp_origin_year = timestamp_origin_year
        self.timestamp_span_years = timestamp_span_years
        # Eq. (9) weights, read by the spec's training objective.
        self.loss_weight_infonce = loss_weight_infonce
        self.loss_weight_smooth = loss_weight_smooth
        total = seq_len + pred_len
        head_len = pred_len // heads

        # Basis module (Sec. 3.3): tau -> N x (I + O) basis through a bottleneck MLP.
        self.basis_mlp = BottleneckMLP(1, basis_nums * total, map_bottleneck)
        # Coef-module inputs: series and basis embedded from their time axes.
        self.history_series_proj = _wn_linear(seq_len, d_model)
        self.history_basis_proj = _wn_linear(seq_len, d_model)
        self.future_series_proj = _wn_linear(pred_len, d_model)
        self.future_basis_proj = _wn_linear(pred_len, d_model)
        self.coef = CoefModule(d_model, heads, block_nums, dropout)
        # Forecast module (Sec. 3.2): head projection of z_y and head-fusion MLP.
        self.future_basis_mlp = BottleneckMLP(pred_len, heads * head_len, pred_len // bottleneck)
        self.fusion_mlp = BottleneckMLP(heads * head_len, pred_len, pred_len // bottleneck)

        # Eq. (8): second-difference operator over the I + O basis time axis.
        smooth = torch.zeros(total - 2, total)
        rows = torch.arange(total - 2)
        smooth[rows, rows] = -1.0
        smooth[rows, rows + 1] = 2.0
        smooth[rows, rows + 2] = -1.0
        self.register_buffer("smoothness_operator", smooth, persistent=False)

    @staticmethod
    def _standardize(values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True)
        return (values - mean) / (std + EPSILON)

    def basis(self, x_mark_enc: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        """Unit-norm basis ``z = [z_x, z_y]`` of shape ``[B, I + O, N]``."""
        tau = normalized_timestamp(
            x_mark_enc, batch, device, self.timestamp_origin_year, self.timestamp_span_years
        )
        z = self.basis_mlp(tau).reshape(batch, self.seq_len + self.pred_len, self.basis_nums)
        return z / torch.sqrt(z.pow(2).sum(dim=1, keepdim=True) + EPSILON)

    def history_coefficients(self, x_enc: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``c_x [B, H, C, N]`` from the standardized history and ``z_x``."""
        series = self.history_series_proj(self._standardize(x_enc).transpose(1, 2))
        basis = self.history_basis_proj(z[:, : self.seq_len].transpose(1, 2))
        return self.coef(basis, series)

    def future_coefficients(self, target: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``c_y [B, H, C, N]`` from the standardized future and ``z_y`` (training view)."""
        series = self.future_series_proj(self._standardize(target).transpose(1, 2))
        basis = self.future_basis_proj(z[:, self.seq_len :].transpose(1, 2))
        return self.coef(basis, series)

    def forecast_from(
        self, x_enc: torch.Tensor, z: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        """Eq. (5) and the head-fusion MLP, then inverse standardization."""
        batch, _, channels = x_enc.shape
        future_basis = self.future_basis_mlp(z[:, self.seq_len :].transpose(1, 2))
        future_basis = future_basis.reshape(batch, self.basis_nums, self.heads, -1).transpose(1, 2)
        per_head = coefficients @ future_basis  # [B, H, C, O // H]
        fused = self.fusion_mlp(per_head.transpose(1, 2).reshape(batch, channels, -1))
        mean = x_enc.mean(dim=1, keepdim=True)
        std = x_enc.std(dim=1, keepdim=True)
        return fused.transpose(1, 2) * (std + EPSILON) + mean

    def alignment_loss(self, history_coef: torch.Tensor, future_coef: torch.Tensor) -> torch.Tensor:
        """Eq. (6): InfoNCE over basis indices with the H head values as features."""
        anchors = history_coef.permute(0, 2, 3, 1).reshape(-1, self.basis_nums, self.heads)
        keys = future_coef.permute(0, 2, 3, 1).reshape(-1, self.basis_nums, self.heads)
        logits = (anchors @ keys.transpose(1, 2)).reshape(-1, self.basis_nums) / self.tau
        labels = torch.arange(self.basis_nums, device=logits.device).repeat(anchors.shape[0])
        return F.cross_entropy(logits, labels)

    def smoothness_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Curvature penalty of Eqs. (7)-(8), reduced as a mean absolute value."""
        return torch.einsum("st,btn->sbn", self.smoothness_operator, z).abs().mean()

    def training_terms(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(forecast, L_align, L_smooth)`` from one shared basis and history pass."""
        self._check(x_enc)
        if target.shape != (x_enc.shape[0], self.pred_len, self.enc_in):
            raise ValueError("target must be [batch, pred_len, enc_in]")
        z = self.basis(x_mark_enc, x_enc.shape[0], x_enc.device)
        history_coef = self.history_coefficients(x_enc, z)
        forecast = self.forecast_from(x_enc, z, history_coef)
        future_coef = self.future_coefficients(target, z)
        return forecast, self.alignment_loss(history_coef, future_coef), self.smoothness_loss(z)

    def _check(self, x_enc: torch.Tensor) -> None:
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must be [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        self._check(x_enc)
        z = self.basis(x_mark_enc, x_enc.shape[0], x_enc.device)
        return self.forecast_from(x_enc, z, self.history_coefficients(x_enc, z))
