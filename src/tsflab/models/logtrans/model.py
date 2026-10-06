"""LogTrans: decoder-only Transformer with convolutional and LogSparse self-attention.

Independent implementation from Li et al., NeurIPS 2019 (Secs. 3-4, App. A.2);
no official code was released. Each channel is one univariate series modelled by
one shared network (paper: a collection of related series with a series-ID
embedding). The one-step-ahead model is ``z_t ~ N(mu_t, sigma_t^2)`` with
``(mu_t, sigma_t) = f(Y_t)`` and input rows ``y_t = [z_{t-1}, x_t]`` (Sec. 3).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.gaussian_parameter_head import GaussianParameterHead
from tsflab.models._components.logsparse_conv_attention import (
    ConvSelfAttention,
    logsparse_mask,
)
from tsflab.models._components.marks import normalized_time_features

_CALENDAR_FEATURES = 2  # [time_in_day, day_in_week] from the marks component


class _DecoderBlock(nn.Module):
    """Post-norm Transformer-decoder block (GPT style, ReLU feed-forward)."""

    def __init__(self, d_model, n_heads, d_ff, kernel_size, dropout) -> None:
        super().__init__()
        self.attention = ConvSelfAttention(d_model, n_heads, kernel_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def _finish(self, x, attended):
        h = self.norm1(x + self.dropout(attended))
        return self.norm2(h + self.dropout(self.feed_forward(h)))

    def forward(self, x, blocked):
        return self._finish(x, self.attention(x, blocked))

    def forward_last(self, x, blocked_row, cache):
        return self._finish(x[:, -1:], self.attention.forward_last(x, blocked_row, cache))


class Model(nn.Module):
    """Channel-independent autoregressive LogSparse Transformer with a Gaussian likelihood."""

    output_type = "distribution"
    distribution_family = "gaussian"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        d_model: int = 64,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        kernel_size: int = 6,
        sparse: bool = True,
        local_size: int = 0,
        restart_len: int = 0,
        embed_dim: int = 20,
        use_marks: bool = True,
        scaling: bool = True,
        history_likelihood: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, e_layers, embed_dim) < 1:
            raise ValueError("lengths, channels, layers and embed_dim must be positive")
        if features not in {"M", "S", "MS"}:
            raise ValueError("features must be M, S, or MS")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.use_marks = use_marks
        self.scaling = scaling
        self.history_likelihood = history_likelihood
        # Inputs y_t for t = 2 .. seq_len + pred_len of the window (z_1 has no lag).
        self.length = seq_len + pred_len - 1
        # App. A.2: learnable position embedding plus series-ID embedding (summed),
        # concatenated with the lagged value and the covariates.
        self.position = nn.Embedding(self.length, embed_dim)
        self.series_id = nn.Embedding(enc_in, embed_dim)
        width = 1 + (_CALENDAR_FEATURES if use_marks else 0) + embed_dim
        self.input_projection = nn.Linear(width, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            _DecoderBlock(d_model, n_heads, d_ff, kernel_size, dropout) for _ in range(e_layers)
        )
        self.likelihood = GaussianParameterHead(d_model, 1)
        self.register_buffer(
            "blocked",
            logsparse_mask(self.length, local=local_size, restart=restart_len, sparse=sparse),
            persistent=False,
        )

    # ----------------------------------------------------------------- inputs
    def _series(self, values: torch.Tensor) -> torch.Tensor:
        """``[B, T, C] -> [B*C, T]``."""
        batch, steps, channels = values.shape
        return values.transpose(1, 2).reshape(batch * channels, steps)

    def _covariates(self, marks: torch.Tensor | None, batch: int, steps: int, ref) -> torch.Tensor:
        """``[B*C, steps, 2]`` calendar covariates (zeros without marks)."""
        if not self.use_marks:
            return ref.new_zeros(batch * self.enc_in, steps, 0)
        if marks is None:
            return ref.new_zeros(batch * self.enc_in, steps, _CALENDAR_FEATURES)
        if marks.ndim != 3 or marks.shape[:2] != (batch, steps) or marks.shape[-1] < 6:
            raise ValueError(f"marks must have shape [batch, {steps}, 6]")
        features = normalized_time_features(marks.to(ref.dtype))
        return features.unsqueeze(1).expand(-1, self.enc_in, -1, -1).reshape(
            batch * self.enc_in, steps, _CALENDAR_FEATURES
        )

    def _scale(self, history: torch.Tensor) -> torch.Tensor:
        """DeepAR-style per-series scale ``nu = 1 + mean |z|`` over the history, ``[B*C, 1]``."""
        if not self.scaling:
            return history.new_ones(history.shape[0], 1)
        return 1.0 + history.abs().mean(dim=1, keepdim=True)

    def _embed(self, lagged, covariates, start: int) -> torch.Tensor:
        """Rows ``y_t`` for positions ``start .. start + steps - 1`` -> ``[B*C, steps, d_model]``."""
        series, steps = lagged.shape
        positions = torch.arange(start, start + steps, device=lagged.device)
        ids = torch.arange(self.enc_in, device=lagged.device).repeat(series // self.enc_in)
        embedding = self.position(positions).unsqueeze(0) + self.series_id(ids).unsqueeze(1)
        rows = torch.cat((lagged.unsqueeze(-1), covariates, embedding), dim=-1)
        return self.input_dropout(self.input_projection(rows))

    def _check(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]")

    def _window_covariates(self, x_enc, x_mark_enc, x_mark_dec):
        """Covariates x_t for t = 2 .. seq_len + pred_len, ``[B*C, length, F]``."""
        batch = x_enc.shape[0]
        if self.use_marks and x_mark_enc is not None and x_mark_dec is not None:
            future = x_mark_dec[:, -self.pred_len:]
            if future.shape[1] != self.pred_len:
                raise ValueError("x_mark_dec does not cover the forecast horizon")
            marks = torch.cat((x_mark_enc, future), dim=1)[:, 1:]
        else:
            marks = None
        return self._covariates(marks, batch, self.length, x_enc)

    def _distribution(self, hidden, scale):
        """Rescaled ``(mu, sigma)``: ``hidden [S, ..., d]`` and ``scale [S, 1]`` -> ``[S, ...]``."""
        loc, sigma = self.likelihood(hidden)
        scale = scale.view(-1, *([1] * (hidden.ndim - 2)))
        return loc.squeeze(-1) * scale, sigma.squeeze(-1) * scale

    # --------------------------------------------------------------- training
    def training_loss(self, x_enc, x_mark_enc, future, x_mark_dec) -> torch.Tensor:
        """Teacher-forced Gaussian NLL over the window (App. A.2).

        ``future`` is ``[B, pred_len, C]``. With ``history_likelihood`` the NLL also
        counts the conditioning range ``t = 2 .. seq_len``, as on the paper's
        real-world datasets; otherwise only the horizon.
        """
        self._check(x_enc)
        batch = x_enc.shape[0]
        window = self._series(torch.cat((x_enc, future), dim=1))  # [B*C, T]
        scale = self._scale(window[:, : self.seq_len])
        covariates = self._window_covariates(x_enc, x_mark_enc, x_mark_dec)
        hidden = self._embed(window[:, :-1] / scale, covariates, 0)
        for block in self.blocks:
            hidden = block(hidden, self.blocked)
        loc, sigma = self._distribution(hidden, scale)
        target = window[:, 1:]
        nll = torch.log(sigma) + 0.5 * ((target - loc) / sigma) ** 2 + 0.9189385332046727
        nll = nll.view(batch, self.enc_in, self.length)
        if self.features == "MS":
            nll = nll[:, -1:]
        if not self.history_likelihood:
            nll = nll[..., -self.pred_len:]
        return nll.mean()

    # -------------------------------------------------------------- inference
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """Autoregressive decoding: each step feeds the predicted mean back as ``z_{t-1}``."""
        del x_dec
        self._check(x_enc)
        batch = x_enc.shape[0]
        history = self._series(x_enc)  # [B*C, seq_len]
        scale = self._scale(history)
        covariates = self._window_covariates(x_enc, x_mark_enc, x_mark_dec)
        hidden = self._embed(history / scale, covariates[:, : self.seq_len], 0)
        prefix = self.blocked[: self.seq_len, : self.seq_len]
        inputs, caches = [], []
        for block in self.blocks:
            inputs.append(hidden)
            cache: dict[str, torch.Tensor] = {}
            _, cache["k"], cache["v"] = block.attention.project(hidden)
            caches.append(cache)
            hidden = block(hidden, prefix)
        loc, sigma = self._distribution(hidden[:, -1], scale)
        locs, sigmas = [loc], [sigma]
        for step in range(1, self.pred_len):
            position = self.seq_len + step - 1
            hidden = self._embed(
                loc.unsqueeze(-1) / scale, covariates[:, position : position + 1], position
            )
            row = self.blocked[position, : position + 1]
            for index, block in enumerate(self.blocks):
                inputs[index] = torch.cat((inputs[index], hidden), dim=1)
                hidden = block.forward_last(inputs[index], row, caches[index])
            loc, sigma = self._distribution(hidden[:, -1], scale)
            locs.append(loc)
            sigmas.append(sigma)
        loc = torch.stack(locs, dim=1).view(batch, self.enc_in, self.pred_len).transpose(1, 2)
        sigma = torch.stack(sigmas, dim=1).view(batch, self.enc_in, self.pred_len).transpose(1, 2)
        if self.features == "MS":
            loc, sigma = loc[..., -1:], sigma[..., -1:]
        return torch.stack((loc, sigma), dim=-1)
