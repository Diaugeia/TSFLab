"""Clean-room MICN forecast implementation from the ICLR paper.

The input is split by a multi-kernel moving-average decomposition. The trend
part is forecast by a linear regression over time (MICN-regre) or by the
window mean (MICN-mean). The seasonal part is padded with ``pred_len`` zeros,
embedded, and refined by MIC layers. In each MIC layer, every scale ``s``
downsamples with a stride-``s`` convolution (local features), applies an
isometric convolution whose kernel spans the whole downsampled sequence
(global correlations), and restores the length with a transposed convolution.
"""
from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.embed import PositionalEmbedding, TokenEmbedding
from tsflab.models._components.series_decomposition import SeriesDecomposition

#: Fixed dropout rate inside each MIC layer, as in the reference implementation.
MIC_DROPOUT = 0.05


def odd_kernel(scale: int) -> int:
    """Moving-average window for one scale: the scale itself, made odd."""
    return scale if scale % 2 else scale + 1


def downsampled_length(length: int, scale: int) -> int:
    """Length after a stride-``scale`` convolution with ``scale // 2`` padding."""
    return (length + 2 * (scale // 2) - scale) // scale + 1


class MultiScaleDecomposition(nn.Module):
    """Average several moving-average decompositions (multi-scale hybrid decomposition)."""

    def __init__(self, kernels):
        super().__init__()
        self.decompositions = nn.ModuleList([SeriesDecomposition(k) for k in kernels])

    def forward(self, values):
        pairs = [decomposition(values) for decomposition in self.decompositions]
        seasonal = torch.stack([pair[0] for pair in pairs]).mean(0)
        trend = torch.stack([pair[1] for pair in pairs]).mean(0)
        return seasonal, trend


class LocalGlobalBranch(nn.Module):
    """One scale: decompose -> downsample -> isometric conv -> upsample.

    The isometric convolution left-pads the downsampled sequence of length
    ``S`` with ``S - 1`` zeros and convolves it with a kernel of size ``S``,
    so each output step sees every earlier step of the sequence.
    """

    def __init__(self, width: int, scale: int, length: int):
        super().__init__()
        self.length = length
        self.decomposition = SeriesDecomposition(odd_kernel(scale))
        self.downsample = nn.Conv1d(width, width, scale, stride=scale, padding=scale // 2)
        self.isometric = nn.Conv1d(width, width, downsampled_length(length, scale))
        self.upsample = nn.ConvTranspose1d(width, width, scale, stride=scale)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(MIC_DROPOUT)

    def forward(self, hidden: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        seasonal, _ = self.decomposition(hidden)
        local = self.dropout(self.activation(self.downsample(seasonal.transpose(1, 2))))
        padded = torch.cat((local.new_zeros(*local.shape[:2], local.shape[2] - 1), local), dim=-1)
        global_view = self.dropout(self.activation(self.isometric(padded)))
        merged = norm((global_view + local).transpose(1, 2)).transpose(1, 2)
        restored = self.dropout(self.activation(self.upsample(merged)))[..., : self.length]
        return norm(restored.transpose(1, 2) + seasonal)


class FeedForward(nn.Module):
    """Position-wise ``Linear -> ReLU -> Dropout -> Linear`` with Xavier weights and zero biases."""

    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.layer1 = nn.Linear(width, hidden)
        self.layer2 = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.layer1, self.layer2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden):
        return self.layer2(self.dropout(torch.relu(self.layer1(hidden))))


class MICLayer(nn.Module):
    """Multi-scale local-global branches merged by a ``(scales, 1)`` Conv2d, then a feed-forward."""

    def __init__(self, width, scales, length):
        super().__init__()
        self.branches = nn.ModuleList([LocalGlobalBranch(width, scale, length) for scale in scales])
        self.norm = nn.LayerNorm(width)  # one norm shared by every branch, as in the reference
        self.merge = nn.Conv2d(width, width, kernel_size=(len(scales), 1))
        self.feed_forward = FeedForward(width, 4 * width, MIC_DROPOUT)
        self.feed_forward_norm = nn.LayerNorm(width)

    def forward(self, hidden):
        stacked = torch.stack([branch(hidden, self.norm) for branch in self.branches], dim=1)
        merged = self.merge(stacked.permute(0, 3, 1, 2)).squeeze(-2).transpose(1, 2)
        return self.feed_forward_norm(merged + self.feed_forward(merged))


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, c_out=None, d_model=64, d_layers=1,
                 dropout=0.05, conv_kernel=(12, 16), mode="regre"):
        super().__init__()
        c_out = enc_in if c_out is None else c_out
        if c_out != enc_in:
            raise ValueError("clean-room MICN requires c_out == enc_in")
        scales = tuple(int(kernel) for kernel in conv_kernel)
        if not scales or min(scales) < 2:
            raise ValueError("conv_kernel must contain scales >= 2")
        if mode not in ("regre", "mean"):
            raise ValueError("mode must be 'regre' or 'mean'")
        self.seq_len, self.pred_len, self.enc_in, self.mode = seq_len, pred_len, enc_in, mode
        length = seq_len + pred_len
        self.decomposition = MultiScaleDecomposition(tuple(odd_kernel(s) for s in scales))
        self.value_embedding = TokenEmbedding(enc_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model, max_len=max(5000, length))
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([MICLayer(d_model, scales, length) for _ in range(d_layers)])
        self.projection = nn.Linear(d_model, c_out)
        if mode == "regre":
            self.regression = nn.Linear(seq_len, pred_len)
            with torch.no_grad():
                self.regression.weight.fill_(1.0 / pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in})")
        seasonal, trend = self.decomposition(x_enc)
        if self.mode == "regre":
            trend_forecast = self.regression(trend.transpose(1, 2)).transpose(1, 2)
        else:
            trend_forecast = x_enc.mean(1, keepdim=True).expand(-1, self.pred_len, -1)
        decoder_input = torch.cat((seasonal, seasonal.new_zeros(seasonal.shape[0], self.pred_len,
                                                                seasonal.shape[2])), dim=1)
        hidden = self.value_embedding(decoder_input) + self.position_embedding(decoder_input)
        hidden = self.embedding_dropout(hidden)
        for layer in self.layers:
            hidden = layer(hidden)
        seasonal_forecast = self.projection(hidden)[:, -self.pred_len:]
        return seasonal_forecast + trend_forecast
