"""Clean-room DUET: distributional temporal experts plus channel attention."""
from __future__ import annotations
import math
import torch
from torch import nn
from tsflab.models._components.revin import RevIN
from tsflab.models._components.topk_expert_router import GatingMLP, topk_dense_mix


def moving_average(x, kernel):
    """Centered edge-padded moving average over time.

    Kept model-local rather than routed through
    ``tsflab.models._components.series_decomposition.EdgePaddedMovingAverage``: that
    component validates ``kernel_size`` as strictly odd, and at least one
    other consumer (BiST) relies on that odd-only validation to reject
    misconfigured even kernel sizes. This helper additionally supports even
    kernels via asymmetric padding (``left=(kernel-1)//2``,
    ``right=kernel//2``), so extending the shared component's contract to
    match it would silently change that other consumer's error behavior.
    """
    if kernel <= 1:
        return x
    left, right = (kernel - 1) // 2, kernel // 2
    padded = torch.cat((x[:, :1].expand(-1, left, -1), x, x[:, -1:].expand(-1, right, -1)), 1)
    return torch.nn.functional.avg_pool1d(padded.transpose(1, 2), kernel, stride=1).transpose(1, 2)


class TemporalExpert(nn.Module):
    def __init__(self, seq_len, width, moving_avg, dropout):
        super().__init__()
        self.moving_avg = moving_avg
        self.trend = nn.Linear(seq_len, width)
        self.seasonal = nn.Linear(seq_len, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        trend = moving_average(x, self.moving_avg)
        return self.dropout(self.trend(trend.transpose(1, 2)) + self.seasonal((x - trend).transpose(1, 2))).transpose(1, 2)


class FrequencyChannelMask(nn.Module):
    """Paper Eqs. (15)-(18) and Bernoulli resampling: learned frequency-domain channel mask.

    Channels are compared by the amplitude of their rFFT (Eq. 16) under a learnable
    Mahalanobis metric ``Q = A^T A`` (Eq. 15). Inverse distances off the diagonal are
    normalized by their row maximum (Eq. 17-18), scaled by ``gamma`` and resampled into
    a binary mask with a hard Gumbel-softmax, as in the official ``Mahalanobis_mask``.
    """

    def __init__(self, seq_len, gamma=0.99, epsilon=1e-10):
        super().__init__()
        frequencies = seq_len // 2 + 1
        self.metric = nn.Parameter(torch.randn(frequencies, frequencies))
        self.gamma, self.epsilon = gamma, epsilon

    def probabilities(self, series):
        """``series`` is ``[batch, channels, time]``; returns ``P`` as ``[batch, channels, channels]``."""
        amplitude = torch.fft.rfft(series, dim=-1).abs()
        projected = amplitude @ self.metric.T  # A (x_i - x_j) is linear, so project once
        squared = projected.square().sum(-1)
        distance = (squared[:, :, None] + squared[:, None, :]
                    - 2 * projected @ projected.transpose(1, 2)).clamp_min(0)
        eye = torch.eye(series.shape[1], device=series.device, dtype=series.dtype)
        relation = (1 / (distance + self.epsilon)) * (1 - eye)
        relation = relation / relation.amax(-1, keepdim=True).detach()
        return (relation + eye) * self.gamma

    def forward(self, series):
        p = self.probabilities(series).clamp_min(1e-12)  # guard log(0) for extreme distance ratios
        logits = torch.stack((torch.log(p / (1 - p)), torch.log((1 - p) / p)), -1)
        return torch.nn.functional.gumbel_softmax(logits, hard=True)[..., 0]


#: Finite fill for masked channel pairs, before the softmax scale (official FullAttention).
MASKED_SCORE = -math.log(1e10)


class ChannelAttention(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.heads, self.scale = heads, (width // heads) ** -0.5
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))

    def forward(self, tokens, mask):
        batch, channels, width = tokens.shape
        q, k, v = self.qkv(tokens).reshape(batch, channels, 3, self.heads, width // self.heads).unbind(2)
        scores = torch.einsum("bchd,bkhd->bhck", q, k)
        mask = mask[:, None]
        scores = scores * mask + (mask == 0).to(scores.dtype) * MASKED_SCORE  # Eq. (20)
        mixed = torch.einsum("bhck,bkhd->bchd", (scores * self.scale).softmax(-1), v).flatten(2)
        tokens = self.norm1(tokens + self.out(mixed))
        return self.norm2(tokens + self.ffn(tokens))


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, features="M", d_model=512, n_heads=8,
                 e_layers=2, d_ff=2048, dropout=0.1, fc_dropout=0.1,
                 moving_avg=25, num_experts=4, k=2, hidden_size=256, noisy_gating=True):
        super().__init__()
        if d_model % n_heads or not 1 <= k <= num_experts:
            raise ValueError("invalid attention width or expert count")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.revin = RevIN(enc_in)
        self.k = k
        # Channel-independent routing (paper Eqs. 5-8, official CI=True): the gate reads
        # each raw channel series of length ``seq_len``, before RevIN.
        self.router = GatingMLP(seq_len, num_experts, hidden_size, noisy_gating)
        kernels = [max(2, moving_avg - 2 * i) for i in range(num_experts)]
        self.experts = nn.ModuleList([TemporalExpert(seq_len, d_model, kernel, fc_dropout) for kernel in kernels])
        self.channel_mask = FrequencyChannelMask(seq_len)
        self.channel_layers = nn.ModuleList([ChannelAttention(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.head = nn.Linear(d_model, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}]")
        raw = x_enc.transpose(1, 2)  # [batch, channels, time], before RevIN as in the official code
        weights = topk_dense_mix(self.router(raw), self.k, 1e-3)  # [batch, channels, experts]
        x = self.revin(x_enc, "norm")
        expert_values = torch.stack([expert(x) for expert in self.experts], 1)
        tokens = torch.einsum("bce,bedc->bcd", weights, expert_values)
        if self.enc_in > 1:  # a single channel has no channel relations (official skips the encoder)
            mask = self.channel_mask(raw)
            for layer in self.channel_layers:
                tokens = layer(tokens, mask)
        return self.revin(self.head(tokens).transpose(1, 2), "denorm")
