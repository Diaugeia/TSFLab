"""TimeGrad: autoregressive RNN-conditioned denoising diffusion (Rasul et al., ICML 2021).

Independent implementation from the paper (Secs. 3-4, Algorithms 1-2, Fig. 2),
with omissions resolved by reading zalandoresearch/pytorch-ts at the revision
recorded in the card. The multivariate vector ``x_t in R^D`` (all channels) is
modelled as ``prod_t p_theta(x_t | h_{t-1})``: an RNN over lagged values, channel
embeddings and calendar features gives ``h_{t-1}``, and a conditional DDPM over
``R^D`` with a WaveNet-style denoiser ``eps_theta(x_t^n, h_{t-1}, n)`` draws ``x_t``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.ddpm_epsilon import GaussianDDPM
from tsflab.models._components.empirical_quantiles import empirical_quantiles

# Raw marks are [year, month, day, weekday, hour, minute]; fixed periods per feature.
_TIME_FEATURES = {"month": (1, 12.0, 1.0), "dayofweek": (3, 7.0, 0.0), "hour": (4, 24.0, 0.0), "minute": (5, 60.0, 0.0)}
_LEAK = 0.4  # LeakyReLU slope used throughout the denoiser (official code)


def _wrap(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Circular padding of the last axis by ``pad`` on both sides, also when ``pad`` exceeds its length."""
    length = x.shape[-1]
    index = torch.arange(-pad, length + pad, device=x.device) % length
    return x[..., index]


class _StepEmbedding(nn.Module):
    """Sinusoidal diffusion-step embedding (frequencies 10^(4j/dim)) and a two-layer SiLU MLP."""

    def __init__(self, dim: int, hidden: int, max_steps: int) -> None:
        super().__init__()
        steps = torch.arange(max_steps, dtype=torch.float32).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim, dtype=torch.float32) * 4.0 / dim)
        angles = steps * frequencies.unsqueeze(0)
        self.register_buffer("table", torch.cat((angles.sin(), angles.cos()), dim=1), persistent=False)
        self.first = nn.Linear(2 * dim, hidden)
        self.second = nn.Linear(hidden, hidden)

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        return F.silu(self.second(F.silu(self.first(self.table[n]))))


class _ResidualBlock(nn.Module):
    """Gated residual block over the variate axis (length ``D + 4``) with circular dilated conv."""

    def __init__(self, hidden: int, channels: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.dilated = nn.Conv1d(channels, 2 * channels, 3, dilation=dilation)
        self.step_projection = nn.Linear(hidden, channels)
        self.condition_projection = nn.Conv1d(1, 2 * channels, 1)
        self.output = nn.Conv1d(channels, 2 * channels, 1)
        nn.init.kaiming_normal_(self.condition_projection.weight)
        nn.init.kaiming_normal_(self.output.weight)

    def forward(self, x, condition, step):
        y = x + self.step_projection(step).unsqueeze(-1)
        y = self.dilated(_wrap(y, self.dilation))
        y = y + self.condition_projection(condition)
        gate, value = y.chunk(2, dim=1)
        y = F.leaky_relu(self.output(torch.sigmoid(gate) * torch.tanh(value)), _LEAK)
        residual, skip = y.chunk(2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class _EpsilonNetwork(nn.Module):
    """``eps_theta(x^n [S, 1, D], n [S], cond [S, 1, cond_len]) -> [S, 1, D]`` (Fig. 2).

    The variate axis is a 1-D signal: inputs are circularly padded by 2 on both
    sides, and two unpadded kernel-3 convolutions bring the length back to ``D``.
    """

    def __init__(self, target_dim, cond_length, step_dim, layers, channels, cycle, hidden, max_steps) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, channels, 1)
        self.step_embedding = _StepEmbedding(step_dim, hidden, max_steps)
        middle = max(1, target_dim // 2)
        self.upsample = nn.Sequential(
            nn.Linear(cond_length, middle), nn.LeakyReLU(_LEAK), nn.Linear(middle, target_dim), nn.LeakyReLU(_LEAK)
        )
        self.blocks = nn.ModuleList(_ResidualBlock(hidden, channels, 2 ** (i % cycle)) for i in range(layers))
        self.skip = nn.Conv1d(channels, channels, 3)
        self.output = nn.Conv1d(channels, 1, 3)
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.skip.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, x, n, cond):
        x = F.leaky_relu(self.input(_wrap(x, 2)), _LEAK)
        condition = _wrap(self.upsample(cond), 2)
        step = self.step_embedding(n)
        skips = []
        for block in self.blocks:
            x, skip = block(x, condition, step)
            skips.append(skip)
        x = torch.stack(skips).sum(0) / math.sqrt(len(self.blocks))
        return self.output(F.leaky_relu(self.skip(x), _LEAK))


class Model(nn.Module):
    """Multivariate probabilistic forecaster returning empirical quantiles of sampled paths."""

    output_type = "quantile"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        quantile_levels: list[float],
        features: str = "M",
        cell_type: str = "LSTM",
        num_layers: int = 2,
        num_cells: int = 40,
        dropout: float = 0.1,
        lags: list[int] = (1, 24),
        context_length: int = 0,
        embed_dim: int = 1,
        time_features: list[str] = ("hour", "dayofweek"),
        conditioning_length: int = 100,
        diff_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        residual_hidden: int = 64,
        step_emb_dim: int = 16,
        scaling: bool = True,
        context_loss: bool = True,
        num_samples: int = 100,
        sample_batch_size: int = 100,
    ) -> None:
        super().__init__()
        lags = sorted(set(int(lag) for lag in lags))
        if not lags or lags[0] < 1:
            raise ValueError("lags must be positive integers")
        context = context_length or seq_len - lags[-1]
        if context < 1 or context + lags[-1] > seq_len:
            raise ValueError("need 1 <= context_length and context_length + max(lags) <= seq_len")
        if features not in {"M", "S", "MS"}:
            raise ValueError("features must be M, S, or MS")
        unknown = set(time_features) - set(_TIME_FEATURES)
        if unknown:
            raise ValueError(f"unsupported time features {sorted(unknown)}")
        self.seq_len, self.pred_len, self.enc_in, self.features = seq_len, pred_len, enc_in, features
        self.lags, self.context = lags, context
        self.time_features = list(time_features)
        self.scaling, self.context_loss = scaling, context_loss
        self.num_samples, self.sample_batch_size = num_samples, sample_batch_size
        self.cell_type = cell_type
        self.register_buffer("quantile_levels", torch.tensor(list(quantile_levels)), persistent=False)

        self.embedding = nn.Embedding(enc_in, embed_dim)
        input_size = enc_in * len(lags) + enc_in * embed_dim + 2 * len(self.time_features)
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn(input_size, num_cells, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
                       batch_first=True)
        self.condition = nn.Linear(num_cells, conditioning_length)
        self.denoiser = _EpsilonNetwork(enc_in, conditioning_length, step_emb_dim, residual_layers,
                                        residual_channels, dilation_cycle_length, residual_hidden, max(500, diff_steps))
        self.diffusion = GaussianDDPM(diff_steps, beta_start, beta_end, beta_schedule)

    # ----------------------------------------------------------------- inputs
    def _calendar(self, marks: torch.Tensor | None, batch: int, steps: int, ref) -> torch.Tensor:
        """``[B, steps, 2F]`` cos/sin Fourier features of the selected calendar fields."""
        width = 2 * len(self.time_features)
        if marks is None or width == 0:
            return ref.new_zeros(batch, steps, width)
        if marks.ndim != 3 or marks.shape[:2] != (batch, steps) or marks.shape[-1] < 6:
            raise ValueError(f"marks must have shape [batch, {steps}, 6]")
        columns = []
        for name in self.time_features:
            index, period, offset = _TIME_FEATURES[name]
            angle = 2.0 * math.pi * (marks[..., index].to(ref.dtype) - offset) / period
            columns += [angle.cos(), angle.sin()]
        return torch.stack(columns, dim=-1)

    def _scale(self, x_enc: torch.Tensor) -> torch.Tensor:
        """``[B, 1, D]`` mean absolute value over the context; batch mean where a series is all zero."""
        if not self.scaling:
            return x_enc.new_ones(x_enc.shape[0], 1, self.enc_in)
        context = x_enc[:, -self.context:].abs()
        scale = context.mean(dim=1, keepdim=True)
        fallback = context.mean(dim=(0, 1), keepdim=True).expand_as(scale)
        scale = torch.where(context.sum(dim=1, keepdim=True) > 0, scale, fallback)
        return scale.clamp(min=1e-10).detach()

    def _rnn_inputs(self, sequence, positions, calendar, scale):
        """Inputs for target positions ``positions`` (lags read from ``sequence [B, T, D]``)."""
        batch = sequence.shape[0]
        lagged = torch.stack([sequence[:, positions - lag] for lag in self.lags], dim=-1)  # [B, P, D, I]
        lagged = (lagged / scale.unsqueeze(-1)).reshape(batch, len(positions), -1)
        ids = self.embedding.weight.reshape(1, 1, -1).expand(batch, len(positions), -1)
        return torch.cat((lagged, ids, calendar), dim=-1)

    def _denoise(self, condition):
        return lambda x, n: self.denoiser(x, n, condition)

    def _check(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]")

    def _marks(self, x_mark_enc, x_mark_dec, batch, ref):
        future = None if x_mark_dec is None else x_mark_dec[:, -self.pred_len:]
        if x_mark_enc is None or future is None:
            return self._calendar(None, batch, self.seq_len + self.pred_len, ref)
        return self._calendar(torch.cat((x_mark_enc, future), dim=1), batch, self.seq_len + self.pred_len, ref)

    # --------------------------------------------------------------- training
    def training_loss(self, x_enc, x_mark_enc, future, x_mark_dec) -> torch.Tensor:
        """Teacher-forced epsilon loss (Algorithm 1) over the context and/or the horizon."""
        self._check(x_enc)
        batch = x_enc.shape[0]
        sequence = torch.cat((x_enc, future), dim=1)
        scale = self._scale(x_enc)
        start = self.seq_len - self.context
        positions = torch.arange(start, self.seq_len + self.pred_len, device=x_enc.device)
        calendar = self._marks(x_mark_enc, x_mark_dec, batch, x_enc)[:, start:]
        outputs, _ = self.rnn(self._rnn_inputs(sequence, positions, calendar, scale))
        target = sequence[:, start:] / scale
        if not self.context_loss:
            outputs, target = outputs[:, self.context:], target[:, self.context:]
        condition = self.condition(outputs).reshape(-1, 1, self.condition.out_features)
        x0 = target.reshape(-1, 1, self.enc_in)
        return self.diffusion.loss(self._denoise(condition), x0)

    # -------------------------------------------------------------- inference
    def _repeat_state(self, state, count):
        if isinstance(state, tuple):
            return tuple(s.repeat_interleave(count, dim=1) for s in state)
        return state.repeat_interleave(count, dim=1)

    @torch.no_grad()
    def sample(self, x_enc, x_mark_enc=None, x_mark_dec=None, num_samples: int | None = None) -> torch.Tensor:
        """``[num_samples, B, pred_len, D]`` sample paths (Algorithm 2 run autoregressively)."""
        self._check(x_enc)
        batch = x_enc.shape[0]
        scale = self._scale(x_enc)
        start = self.seq_len - self.context
        calendar = self._marks(x_mark_enc, x_mark_dec, batch, x_enc)
        positions = torch.arange(start, self.seq_len, device=x_enc.device)
        _, state = self.rnn(self._rnn_inputs(x_enc, positions, calendar[:, start:self.seq_len], scale))
        total = num_samples or self.num_samples
        draws = []
        for offset in range(0, total, self.sample_batch_size):
            count = min(self.sample_batch_size, total - offset)
            sequence = x_enc.repeat_interleave(count, dim=0)
            scale_r = scale.repeat_interleave(count, dim=0)
            calendar_r = calendar.repeat_interleave(count, dim=0)
            state_r = self._repeat_state(state, count)
            for k in range(self.pred_len):
                position = torch.tensor([self.seq_len + k], device=x_enc.device)
                inputs = self._rnn_inputs(sequence, position, calendar_r[:, self.seq_len + k : self.seq_len + k + 1], scale_r)
                output, state_r = self.rnn(inputs, state_r)
                condition = self.condition(output).reshape(-1, 1, self.condition.out_features)
                x = self.diffusion.sample(self._denoise(condition), (condition.shape[0], 1, self.enc_in),
                                          device=x_enc.device, dtype=x_enc.dtype)
                sequence = torch.cat((sequence, x.reshape(-1, 1, self.enc_in) * scale_r), dim=1)
            draws.append(sequence[:, self.seq_len:].reshape(batch, count, self.pred_len, self.enc_in))
        return torch.cat(draws, dim=1).permute(1, 0, 2, 3)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec
        samples = self.sample(x_enc, x_mark_enc, x_mark_dec)
        quantiles = empirical_quantiles(samples, self.quantile_levels.to(samples.dtype))
        if self.features == "MS":
            quantiles = quantiles[:, :, -1:]
        return quantiles
