"""Local PRformer: Pyramidal RNN Embeddings (PRE) feeding a variate-token Transformer.

Paper map (Yu et al., 2024, arXiv 2408.10483): per-variate PRE embedding and
encoder-projection pipeline (Eq. 1); bottom-up pyramid convolution with kernel and
stride ``K_l = Window_l / Window_{l-1}`` (Eq. 2); top-down upsampling with lateral
addition (Eq. 3); multi-scale GRU embeddings weighted by a temperature softmax and
concatenated (Eq. 4); multi-head attention across variate tokens (Eq. 5); RevIN.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import Encoder, EncoderLayer


def pyramid_chains(windows: tuple[int, ...]) -> list[list[int]]:
    """Group the configured period lengths into multiplicative pyramid chains.

    Greedy rule resolved from the official configuration method: repeatedly scan
    the (ascending) windows from the largest down, starting a chain at the first
    unassigned window and appending every smaller window that divides the last
    appended one. The smallest window may be shared by several chains; every other
    window belongs to exactly one. Each chain is returned in ascending order.
    """
    assigned = [False] * len(windows)
    chains: list[list[int]] = []
    while not all(assigned):
        chain: list[int] = []
        for index in range(len(windows) - 1, -1, -1):
            if assigned[index] and index != 0:
                continue
            window = windows[index]
            if not chain or chain[-1] % window == 0:
                chain.append(window)
                assigned[index] = True
        chains.append(sorted(chain))
    return chains


def scale_hidden_sizes(windows: tuple[int, ...], d_model: int) -> dict[int, int]:
    """GRU width per window: ``d_model / #windows`` split across the chains sharing it."""
    uses = Counter(window for chain in pyramid_chains(windows) for window in chain)
    share = d_model // len(windows)
    return {window: share // uses[window] for window in windows}


def match_length(upsampled: torch.Tensor, length: int) -> torch.Tensor:
    """Right-pad by repeating the last step (or trim) so the time axis equals ``length``."""
    missing = length - upsampled.shape[-1]
    if missing > 0:
        tail = upsampled[..., -1:].expand(*upsampled.shape[:-1], missing)
        upsampled = torch.cat((upsampled, tail), dim=-1)
    return upsampled[..., :length]


class PyramidChain(nn.Module):
    """One pyramid of Eqs. (2)-(3) with a GRU per level (Multi-Scale RNN block)."""

    def __init__(self, windows: list[int], conv_channels: int, hidden: dict[int, int]) -> None:
        super().__init__()
        self.windows = list(windows)
        # The first level compresses raw steps into the first period (kernel = window).
        self.kernels = [windows[0]] + [windows[i] // windows[i - 1] for i in range(1, len(windows))]
        self.convs = nn.ModuleList(
            nn.Conv1d(1 if level == 0 else conv_channels, conv_channels, kernel, stride=kernel)
            for level, kernel in enumerate(self.kernels)
        )
        self.grus = nn.ModuleList(
            nn.GRU(conv_channels, hidden[window], batch_first=True) for window in windows
        )

    def bottom_up(self, series: torch.Tensor) -> list[torch.Tensor]:
        """Eq. (2): ``x_l = Conv1d(x_{l-1}, kernel=K_l, stride=K_l)``; input ``[N, 1, L]``."""
        levels = []
        for conv in self.convs:
            series = conv(series)
            levels.append(series)
        return levels

    def top_down(self, levels: list[torch.Tensor]) -> list[torch.Tensor]:
        """Eq. (3), returned top level first: ``x'_{l-1} = up(x'_l)``, ``x_out = x'_{l-1} + x_{l-1}``.

        The top level enters unchanged; the upsampled stream keeps upsampling the
        previous upsampled map (nearest neighbour by the integer window ratio).
        """
        stream = levels[-1]
        fused = [stream]
        for level in range(len(levels) - 2, -1, -1):
            stream = stream.repeat_interleave(self.kernels[level + 1], dim=-1)
            stream = match_length(stream, levels[level].shape[-1])
            fused.append(stream + levels[level])
        return fused

    def forward(self, series: torch.Tensor) -> list[torch.Tensor]:
        """Last GRU hidden state per level, top level first, each ``[N, hidden]``."""
        fused = self.top_down(self.bottom_up(series))
        states = []
        for offset, features in enumerate(fused):
            gru = self.grus[len(fused) - 1 - offset]
            _, last = gru(features.transpose(1, 2))
            states.append(last[-1])
        return states


class PyramidalRNNEmbedding(nn.Module):
    """PRE: one ``d_model`` token per univariate series (Eq. 4 plus output linear)."""

    def __init__(
        self, seq_len: int, windows: tuple[int, ...], d_model: int, conv_channels: int, temperature: float
    ) -> None:
        super().__init__()
        if not windows or list(windows) != sorted(set(windows)) or windows[0] < 1:
            raise ValueError("conv_windows must be strictly increasing positive period lengths")
        if windows[-1] > seq_len:
            raise ValueError("every conv window must fit in seq_len")
        self.chain_windows = pyramid_chains(windows)
        self.hidden = scale_hidden_sizes(windows, d_model)
        if min(self.hidden.values()) < 1:
            raise ValueError("d_model is too small for the number of pyramid scales")
        self.temperature = float(temperature)
        self.chains = nn.ModuleList(
            PyramidChain(chain, conv_channels, self.hidden) for chain in self.chain_windows
        )
        self.scale_count = sum(len(chain) for chain in self.chain_windows)
        # alpha_i initialised to 1 / l (Multi-Scale RNN block).
        self.alpha = nn.Parameter(torch.full((self.scale_count,), 1.0 / self.scale_count))
        concat_width = sum(self.hidden[window] for chain in self.chain_windows for window in chain)
        self.out_linear = nn.Linear(concat_width, d_model)

    def scale_weights(self) -> torch.Tensor:
        """Eq. (4): ``beta_i = exp(alpha_i / T) / sum_j exp(alpha_j / T)``."""
        return torch.softmax(self.alpha / self.temperature, dim=0)

    def scale_states(self, values: torch.Tensor) -> list[torch.Tensor]:
        """Per-scale GRU embeddings ``h^(i)`` of every variate, each ``[B * C, hidden_i]``."""
        batch, length, channels = values.shape
        series = values.transpose(1, 2).reshape(batch * channels, 1, length)
        return [state for chain in self.chains for state in chain(series)]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """``[B, L, C]`` -> ``[B, C, d_model]``; every variate is embedded independently."""
        batch, _, channels = values.shape
        beta = self.scale_weights()
        # h = concat(beta_1 h^(1), ..., beta_l h^(l))
        embedding = torch.cat(
            [beta[i] * state for i, state in enumerate(self.scale_states(values))], dim=-1
        )
        return self.out_linear(embedding).view(batch, channels, -1)


class Model(nn.Module):
    """PRformer: RevIN, PRE variate tokens, post-norm Transformer encoder, linear head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        conv_windows: tuple[int, ...] | list[int] = (24, 48, 72, 96, 144),
        conv_channels: int = 128,
        rnn_mix_temperature: float = 0.002,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        causal_variate_mask: bool = False,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, conv_channels, d_model, n_heads, e_layers, d_ff) < 1:
            raise ValueError("lengths, channels, widths, heads, and layers must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if rnn_mix_temperature <= 0:
            raise ValueError("rnn_mix_temperature must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        # A period longer than the lookback has no complete window to convolve;
        # the official scripts never meet this case (seq_len 660-720). Such
        # periods are dropped and the remaining ones form the pyramid.
        windows = tuple(window for window in conv_windows if window <= seq_len)
        if not windows:
            raise ValueError(f"no conv window fits in seq_len {seq_len}: {list(conv_windows)}")
        self.conv_windows = windows
        self.revin = RevIN(enc_in, affine=True)
        self.embedding = PyramidalRNNEmbedding(
            seq_len, windows, d_model, conv_channels, rnn_mix_temperature
        )
        # Eq. (5): softmax(Q K^T / sqrt(d_k)) V across the C variate tokens.
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(causal_variate_mask, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must be [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        normalized = self.revin(x_enc, "norm")
        tokens = self.embedding(normalized)
        encoded, _ = self.encoder(tokens, attn_mask=None)
        forecast = self.projector(encoded).transpose(1, 2)
        return self.revin(forecast, "denorm")
