"""SBU-LSTM: stacked bidirectional and unidirectional LSTMs with an imputation unit.

Cui, Ke, Pu and Wang, Transportation Research Part C 2020 (arXiv 2005.11627).
The network reads the whole sensor vector ``x_t`` (width ``D``) at every step,
so all sensors share one recurrent state. Missing inputs are NaNs in ``x_enc``.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import nn

LayerKind = Literal["bidirectional", "unidirectional"]
ValueRange = Literal["unit", "unbounded"]


class ImputationLSTMDirection(nn.Module):
    """One direction of an LSTM-I layer (Eqs. 10-18).

    Imputation unit:  x~_t = act(W_I C_{t-1} + U_I h_{t-1} + b_I)            (10)
    Input update:     x_t <- m_t * x_t + (1 - m_t) * x~_t                     (12)
    Gates:            g_t = act_g(W_g x_t + U_g h_{t-1} + V_g m_t + b_g)       (13-16)
    State:            C_t = f_t * C_{t-1} + i_t * C~_t, h_t = o_t * tanh(C_t)   (17-18)
    """

    def __init__(self, input_dim: int, hidden_dim: int, bounded_imputation: bool) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # One affine map over [x_t, h_{t-1}, m_t] holds W, U, V and b of the four gates.
        self.gates = nn.Linear(2 * input_dim + hidden_dim, 4 * hidden_dim)
        # W_I, U_I and b_I of Eq. 10 over [C_{t-1}, h_{t-1}].
        self.imputation = nn.Linear(2 * hidden_dim, input_dim)
        self.bounded_imputation = bounded_imputation

    def forward(
        self, values: torch.Tensor, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hidden states ``[B, T, H]`` and imputations ``[B, T, D]`` in input order."""
        batch, steps, _ = values.shape
        hidden = values.new_zeros(batch, self.hidden_dim)
        cell = values.new_zeros(batch, self.hidden_dim)
        outputs, imputations = [], []
        for step in range(steps):
            estimate = self.imputation(torch.cat((cell, hidden), dim=-1))
            if self.bounded_imputation:
                estimate = torch.sigmoid(estimate)
            mask = observed[:, step]
            current = mask * values[:, step] + (1.0 - mask) * estimate
            forget, inp, out, candidate = self.gates(
                torch.cat((current, hidden, mask), dim=-1)
            ).chunk(4, dim=-1)
            cell = torch.sigmoid(forget) * cell + torch.sigmoid(inp) * torch.tanh(candidate)
            hidden = torch.sigmoid(out) * torch.tanh(cell)
            outputs.append(hidden)
            imputations.append(estimate)
        return torch.stack(outputs, dim=1), torch.stack(imputations, dim=1)


class ImputationLSTMLayer(nn.Module):
    """LSTM-I or BDLSTM-I layer; bidirectional outputs are averaged (Eq. 20)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, bidirectional: bool, bounded_imputation: bool
    ) -> None:
        super().__init__()
        self.forward_direction = ImputationLSTMDirection(input_dim, hidden_dim, bounded_imputation)
        self.backward_direction = (
            ImputationLSTMDirection(input_dim, hidden_dim, bounded_imputation)
            if bidirectional
            else None
        )

    def forward(
        self, values: torch.Tensor, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the layer output and the imputation-error term of Eq. 21."""
        hidden, estimate = self.forward_direction(values, observed)
        count = observed.sum().clamp_min(1.0)
        error = (observed * (values - estimate).abs()).sum() / count
        if self.backward_direction is None:
            return hidden, error
        reverse_hidden, reverse_estimate = self.backward_direction(
            values.flip(1), observed.flip(1)
        )
        reverse_error = (observed.flip(1) * (values.flip(1) - reverse_estimate).abs()).sum() / count
        return 0.5 * (hidden + reverse_hidden.flip(1)), 0.5 * (error + reverse_error)


class AveragedLSTMLayer(nn.Module):
    """Plain LSTM or BDLSTM layer (Eqs. 3-8); bidirectional outputs are averaged (Eq. 20)."""

    def __init__(self, input_dim: int, hidden_dim: int, bidirectional: bool) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.recurrent = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(values)
        if not self.bidirectional:
            return output
        forward, backward = output.chunk(2, dim=-1)
        return 0.5 * (forward + backward)


class Model(nn.Module):
    """Stacked LSTM-I/BDLSTM-I first layer and LSTM/BDLSTM layers over the sensor vector."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        layers: Sequence[LayerKind] = ("bidirectional", "unidirectional"),
        hidden_dim: int | None = None,
        imputation: bool = True,
        imputation_weight: float = 1.0,
        value_range: ValueRange = "unit",
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in) < 1:
            raise ValueError("seq_len, pred_len and enc_in must be positive")
        if not layers:
            raise ValueError("layers must name at least one layer")
        if any(kind not in ("bidirectional", "unidirectional") for kind in layers):
            raise ValueError("layers entries must be 'bidirectional' or 'unidirectional'")
        if value_range not in ("unit", "unbounded"):
            raise ValueError("value_range must be 'unit' or 'unbounded'")
        if imputation_weight < 0.0:
            raise ValueError("imputation_weight must be non-negative")
        width = enc_in if hidden_dim is None else hidden_dim
        if width < 1:
            raise ValueError("hidden_dim must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.imputation_weight = imputation_weight
        self.value_range = value_range
        # Every layer but the last is `width` wide; the last is `enc_in` wide because
        # its final hidden state is the next sensor vector (Section 4.5.1).
        widths = [width] * (len(layers) - 1) + [enc_in]
        first = layers[0] == "bidirectional"
        self.first_layer: nn.Module = (
            ImputationLSTMLayer(enc_in, widths[0], first, value_range == "unit")
            if imputation
            else AveragedLSTMLayer(enc_in, widths[0], first)
        )
        self.stacked_layers = nn.ModuleList(
            AveragedLSTMLayer(widths[index - 1], widths[index], kind == "bidirectional")
            for index, kind in enumerate(layers)
            if index > 0
        )
        # value_range="unit": x^_{T+1} = y_T (Eq. 9) for inputs scaled into [0, 1].
        # value_range="unbounded": local affine readout of y_T for standardized inputs.
        self.readout = nn.Linear(enc_in, enc_in) if value_range == "unbounded" else nn.Identity()
        self.aux_loss: torch.Tensor | None = None

    def _next_step(
        self, values: torch.Tensor, observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        error = None
        if isinstance(self.first_layer, ImputationLSTMLayer):
            hidden, error = self.first_layer(values, observed)
        else:
            hidden = self.first_layer(values)
        for layer in self.stacked_layers:
            hidden = layer(hidden)
        return self.readout(hidden[:, -1]), error

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]")
        observed = torch.isfinite(x_enc).to(x_enc.dtype)
        values = torch.where(observed.bool(), x_enc, torch.zeros_like(x_enc))
        predictions = []
        error = None
        for step in range(self.pred_len):
            prediction, step_error = self._next_step(values, observed)
            if step == 0:
                error = step_error
            predictions.append(prediction)
            if step + 1 < self.pred_len:
                # Recursive multi-step rollout: the forecast enters the window as observed.
                values = torch.cat((values[:, 1:], prediction.unsqueeze(1)), dim=1)
                observed = torch.cat((observed[:, 1:], torch.ones_like(prediction).unsqueeze(1)), dim=1)
        # Eq. 22 regularizer, added to the configured forecast loss by the trainer.
        self.aux_loss = (
            self.imputation_weight * error if self.training and error is not None else None
        )
        return torch.stack(predictions, dim=1)
