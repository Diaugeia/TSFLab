"""Temporal Fusion Transformer (Lim et al., IJF 2021, arXiv 1912.09363).

Independent PyTorch implementation of Sec. 4 of the paper, with omissions
resolved against the official TensorFlow code (google-research/tft at
e49bbfe3, Apache-2.0); no source was copied.

Input mapping to TSFLab's forecaster call:

* observed inputs z (Sec. 3): the target history. In ``M``/``S`` mode every
  channel is one entity (as in the paper's Electricity and Traffic setups) and
  its only observed input is its own value; in ``MS`` mode the sample is one
  entity whose observed inputs are all ``enc_in`` channels, the target last.
* known inputs x: the calendar fields of the raw marks (month, day, weekday,
  hour, minute) as categorical entity embeddings, over history and horizon.
* static covariates s: the entity identifier (channel index) as a categorical
  embedding, as the official Electricity/Traffic formatters do; ``MS`` has one
  entity, so its static input is a single learned vector.

Pipeline (paper equation numbers):
  static VSN -> four static context GRNs c_s, c_e, c_h, c_c (Sec. 4.3)
  past / future VSNs with context c_s (Eq. 6-8)
  LSTM encoder (initial state c_h, c_c) and decoder, gated skip (Eq. 17)
  static enrichment GRN with c_e (Eq. 18)
  causal interpretable multi-head self-attention, gated skip (Eq. 19-20)
  position-wise GRN (Eq. 21), gated skip over the whole block (Eq. 22)
  one linear map per quantile on the horizon positions (Eq. 23)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.gated_residual_network import (
    GateAddNorm,
    GatedResidualNetwork,
    VariableSelectionNetwork,
)
from tsflab.models._components.interpretable_attention import (
    InterpretableMultiHeadAttention,
    causal_mask,
)

_DEFAULT_LEVELS = [0.1, 0.5, 0.9]
# Raw mark layout: [year, month, day, weekday, hour, minute]. Year is not a
# bounded category and is left out; the other five are known categorical inputs.
_MARK_WIDTH = 6
_KNOWN_COLUMNS = (1, 2, 3, 4, 5)
_KNOWN_CARDINALITY = (13, 32, 7, 24, 60)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        d_model: int = 160,
        num_heads: int = 4,
        dropout: float = 0.1,
        quantile_levels: list[float] | None = None,
    ) -> None:
        super().__init__()
        if seq_len < 1 or pred_len < 1 or enc_in < 1:
            raise ValueError("seq_len, pred_len, and enc_in must be positive")
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("d_model must be a positive multiple of num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        levels = list(quantile_levels) if quantile_levels else list(_DEFAULT_LEVELS)
        if any(not 0.0 < level < 1.0 for level in levels):
            raise ValueError("quantile_levels must lie in (0, 1)")
        if any(left >= right for left, right in zip(levels, levels[1:])):
            raise ValueError("quantile_levels must be strictly ascending")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.output_type = "quantile"
        self.quantile_levels = levels
        self.channel_independent = features != "MS"
        num_observed = 1 if self.channel_independent else enc_in
        num_entities = enc_in if self.channel_independent else 1
        num_known = len(_KNOWN_COLUMNS)

        # Input transformations (Sec. 4.2): linear maps for real inputs, entity
        # embeddings for categorical ones, all to d_model.
        self.observed_weight = nn.Parameter(torch.empty(num_observed, d_model))
        self.observed_bias = nn.Parameter(torch.zeros(num_observed, d_model))
        self.known_embeddings = nn.ModuleList(
            nn.Embedding(size, d_model) for size in _KNOWN_CARDINALITY
        )
        self.static_embedding = nn.Embedding(num_entities, d_model)

        # Static covariate encoders (Sec. 4.3).
        self.static_selection = VariableSelectionNetwork(1, d_model, dropout=dropout)
        self.static_contexts = nn.ModuleDict(
            {
                name: GatedResidualNetwork(d_model, d_model, dropout=dropout)
                for name in ("selection", "enrichment", "state_h", "state_c")
            }
        )

        # Temporal variable selection (Eq. 6-8), separate for past and future.
        self.history_selection = VariableSelectionNetwork(
            num_known + num_observed, d_model, context_dim=d_model, dropout=dropout
        )
        self.future_selection = VariableSelectionNetwork(
            num_known, d_model, context_dim=d_model, dropout=dropout
        )

        # Temporal fusion decoder (Sec. 4.5).
        self.encoder_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.decoder_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.lstm_gate = GateAddNorm(d_model, d_model, dropout)
        self.enrichment = GatedResidualNetwork(
            d_model, d_model, context_dim=d_model, dropout=dropout
        )
        self.attention = InterpretableMultiHeadAttention(d_model, num_heads, dropout)
        self.attention_gate = GateAddNorm(d_model, d_model, dropout)
        self.position_wise = GatedResidualNetwork(d_model, d_model, dropout=dropout)
        # The official code applies no dropout in this last gate.
        self.output_gate = GateAddNorm(d_model, d_model, dropout=0.0)
        self.quantile_proj = nn.Linear(d_model, len(levels))
        self.register_buffer(
            "_mask", causal_mask(seq_len + pred_len), persistent=False
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Keras defaults of the official code: Glorot-uniform kernels, zero
        biases, orthogonal recurrent kernels, LSTM forget-gate bias 1, and
        U(-0.05, 0.05) embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.05, 0.05)
            elif isinstance(module, nn.LSTM):
                hidden = module.hidden_size
                for gate in range(4):
                    rows = slice(gate * hidden, (gate + 1) * hidden)
                    nn.init.xavier_uniform_(module.weight_ih_l0.data[rows])
                    nn.init.orthogonal_(module.weight_hh_l0.data[rows])
                nn.init.zeros_(module.bias_ih_l0)
                nn.init.zeros_(module.bias_hh_l0)
                module.bias_ih_l0.data[hidden : 2 * hidden] = 1.0
        # Per-variable Linear(1, d_model): fan_in 1, fan_out d_model.
        bound = (6.0 / (1 + self.observed_weight.shape[1])) ** 0.5
        nn.init.uniform_(self.observed_weight, -bound, bound)
        nn.init.zeros_(self.observed_bias)

    def _known_inputs(
        self, marks: torch.Tensor | None, length: int, batch: int, reference: torch.Tensor
    ) -> torch.Tensor:
        """Embed the known calendar inputs of the last ``length`` steps: [B, T, K, d]."""
        if marks is None:
            codes = torch.zeros(
                batch, length, len(_KNOWN_COLUMNS), dtype=torch.long, device=reference.device
            )
        else:
            if marks.ndim != 3 or marks.shape[0] != batch or marks.shape[-1] != _MARK_WIDTH:
                raise ValueError(
                    f"TFT expects raw marks [B, T, {_MARK_WIDTH}] "
                    "(year, month, day, weekday, hour, minute)"
                )
            if marks.shape[1] < length:
                raise ValueError("time marks do not cover the required window")
            codes = marks[:, -length:, list(_KNOWN_COLUMNS)].round().long()
        embedded = [
            table(codes[..., j].clamp(0, table.num_embeddings - 1))
            for j, table in enumerate(self.known_embeddings)
        ]
        return torch.stack(embedded, dim=-2)

    def fuse(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the network; return quantiles [B, pred_len, C_out, Q] and the
        interpretability weights (static, past, and future variable selection,
        decoder self-attention)."""
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"TFT expects input shaped (batch, {self.seq_len}, {self.enc_in})"
            )
        batch = x_enc.shape[0]
        known_past = self._known_inputs(x_mark_enc, self.seq_len, batch, x_enc)
        known_future = self._known_inputs(x_mark_dec, self.pred_len, batch, x_enc)
        if self.channel_independent:
            observed = x_enc.permute(0, 2, 1).reshape(batch * self.enc_in, self.seq_len, 1)
            entity = torch.arange(self.enc_in, device=x_enc.device).repeat(batch)
            known_past = known_past.repeat_interleave(self.enc_in, dim=0)
            known_future = known_future.repeat_interleave(self.enc_in, dim=0)
        else:
            observed = x_enc
            entity = torch.zeros(batch, dtype=torch.long, device=x_enc.device)
        observed = observed.unsqueeze(-1) * self.observed_weight + self.observed_bias

        static_vector, static_weights = self.static_selection(
            self.static_embedding(entity).unsqueeze(-2)
        )
        context = {name: grn(static_vector) for name, grn in self.static_contexts.items()}

        past, past_weights = self.history_selection(
            torch.cat([known_past, observed], dim=-2), context["selection"].unsqueeze(1)
        )
        future, future_weights = self.future_selection(
            known_future, context["selection"].unsqueeze(1)
        )
        initial = (context["state_h"].unsqueeze(0), context["state_c"].unsqueeze(0))
        encoded, state = self.encoder_lstm(past, initial)
        decoded, _ = self.decoder_lstm(future, state)
        temporal = self.lstm_gate(
            torch.cat([encoded, decoded], dim=1), torch.cat([past, future], dim=1)
        )
        enriched = self.enrichment(temporal, context["enrichment"].unsqueeze(1))
        attended, attention = self.attention(enriched, enriched, enriched, self._mask)
        fused = self.attention_gate(attended, enriched)
        fused = self.output_gate(self.position_wise(fused), temporal)
        quantiles = self.quantile_proj(fused[:, -self.pred_len :])

        if self.channel_independent:
            quantiles = quantiles.reshape(batch, self.enc_in, self.pred_len, -1)
            quantiles = quantiles.permute(0, 2, 1, 3)
        else:
            quantiles = quantiles.unsqueeze(2)
        weights = {
            "static": static_weights,
            "past": past_weights,
            "future": future_weights,
            "attention": attention,
        }
        return quantiles, weights

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec
        quantiles, _ = self.fuse(x_enc, x_mark_enc, x_mark_dec)
        return quantiles
