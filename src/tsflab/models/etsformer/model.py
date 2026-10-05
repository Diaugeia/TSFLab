"""Independent ETSformer implementation derived from the paper equations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FrequencyAttention(nn.Module):
    """Select top-amplitude Fourier bases and extrapolate them in time."""

    def __init__(self, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k

    def forward(self, values: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        length = values.shape[1]
        spectrum = torch.fft.rfft(values, dim=1)
        candidates = spectrum[:, 1:, :]
        count = min(self.top_k, candidates.shape[1])
        if count == 0:
            zeros = values.new_zeros(values.shape)
            return zeros, values.new_zeros(values.shape[0], horizon, values.shape[2])
        indices = candidates.abs().topk(count, dim=1).indices + 1
        coefficients = torch.gather(spectrum, 1, indices)

        def synthesize(positions: torch.Tensor) -> torch.Tensor:
            phase = 2 * math.pi * positions.view(1, -1, 1, 1) * indices.unsqueeze(1) / length
            waves = coefficients.unsqueeze(1) * torch.exp(1j * phase)
            return (2.0 / length) * waves.real.sum(dim=2)

        history = synthesize(torch.arange(length, device=values.device, dtype=values.dtype))
        future = synthesize(torch.arange(length, length + horizon, device=values.device,
                                          dtype=values.dtype))
        return history, future


class ExponentialSmoothing(nn.Module):
    """Multi-head AES: AES(V)_t = alpha_h V_t + (1-alpha_h) AES(V)_{t-1}, one alpha per head.

    The ``width`` features are split into ``heads`` contiguous groups of
    ``width // heads``; each group shares one learnable smoothing weight.
    """

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_width = width // heads
        self.alpha_logit = nn.Parameter(torch.zeros(heads))
        self.initial = nn.Parameter(torch.zeros(width))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_logit.sigmoid().repeat_interleave(self.head_width).view(1, -1)
        state = self.initial.view(1, -1).expand(values.shape[0], -1)
        outputs = []
        for step in values.unbind(dim=1):
            state = alpha * step + (1.0 - alpha) * state
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class ETSLayer(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int, top_k: int, dropout: float,
                 activation: str) -> None:
        super().__init__()
        self.head_width = width // heads
        self.frequency = FrequencyAttention(top_k)
        self.growth_input = nn.Linear(width, width)
        self.smoothing = ExponentialSmoothing(width, heads)
        self.growth_output = nn.Linear(width, width)
        nonlinearity: nn.Module = nn.Sigmoid() if activation == "sigmoid" else nn.GELU()
        self.feed_forward = nn.Sequential(
            nn.Linear(width, hidden), nonlinearity, nn.Dropout(dropout),
            nn.Linear(hidden, width), nn.Dropout(dropout),
        )
        self.norm_growth = nn.LayerNorm(width)
        self.norm_feed_forward = nn.LayerNorm(width)
        self.damping_logit = nn.Parameter(torch.zeros(heads))

    def forward(self, residual: torch.Tensor, horizon: int):
        season, future_season = self.frequency(residual, horizon)
        deseasonalized = residual - season
        projected = self.growth_input(deseasonalized)
        difference = torch.diff(projected, dim=1, prepend=projected[:, :1])
        growth = self.growth_output(self.smoothing(difference))
        residual = self.norm_growth(deseasonalized - growth)
        residual = self.norm_feed_forward(residual + self.feed_forward(residual))
        gamma = self.damping_logit.sigmoid().repeat_interleave(self.head_width)
        powers = torch.arange(1, horizon + 1, device=residual.device,
                              dtype=residual.dtype).view(-1, 1)
        damping = gamma.view(1, -1).pow(powers).cumsum(dim=0)
        future_growth = growth[:, -1:, :] * damping.unsqueeze(0)
        return residual, season, growth, future_season, future_growth


class Model(nn.Module):
    """Level, growth, and seasonality decomposition forecaster."""

    def __init__(
        self, seq_len: int, pred_len: int, enc_in: int, d_model: int = 128,
        n_heads: int = 8, e_layers: int = 2, d_ff: int = 256, top_k: int = 3,
        dropout: float = 0.1, activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        del seq_len
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.pred_len = pred_len
        self.embedding = nn.Conv1d(enc_in, d_model, kernel_size=3,
                                   padding=1, padding_mode="circular")
        self.layers = nn.ModuleList([
            ETSLayer(d_model, n_heads, d_ff, top_k, dropout, activation)
            for _ in range(e_layers)
        ])
        self.season_projection = nn.ModuleList([nn.Linear(d_model, enc_in) for _ in range(e_layers)])
        self.growth_projection = nn.ModuleList([nn.Linear(d_model, enc_in) for _ in range(e_layers)])
        self.component_projection = nn.Linear(d_model, enc_in)
        self.level_alpha_logit = nn.Parameter(torch.zeros(enc_in))

    def _level(self, observations, seasons, growths):
        alpha = self.level_alpha_logit.sigmoid().view(1, -1)
        level = observations[:, 0]
        outputs = []
        for index in range(observations.shape[1]):
            season = sum(projection(values[:, index])
                         for projection, values in zip(self.season_projection, seasons))
            previous_growth = sum(projection(values[:, max(index - 1, 0)])
                                  for projection, values in zip(self.growth_projection, growths))
            level = alpha * (observations[:, index] - season) + (1 - alpha) * (level + previous_growth)
            outputs.append(level)
        return torch.stack(outputs, dim=1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        residual = self.embedding(x_enc.transpose(1, 2)).transpose(1, 2)
        seasons, growths, future_components = [], [], []
        for layer in self.layers:
            residual, season, growth, future_season, future_growth = layer(residual, self.pred_len)
            seasons.append(season)
            growths.append(growth)
            future_components.append(future_season + future_growth)
        level = self._level(x_enc, seasons, growths)[:, -1:, :]
        latent_forecast = torch.stack(future_components).sum(dim=0)
        return level.expand(-1, self.pred_len, -1) + self.component_projection(latent_forecast)
