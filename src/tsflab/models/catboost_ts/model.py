"""Independent ordered-context oblivious-tree forecasting baseline."""

from __future__ import annotations
import torch
import torch.nn as nn
from tsflab.models._components.revin import RevIN
from tsflab.models._components.soft_tree import SoftObliviousTree

#: Cap on the dense ``(seq_len*enc_in) x (pred_len*enc_in)`` weights (8 GB of float32 on
#: the host at construction, about 32 GB on the device with gradients and Adam moments).
#: A larger cell is refused with ``ValueError`` before any allocation.
MAX_DENSE_PARAMETERS = 2_000_000_000


def dense_parameter_count(seq_len: int, pred_len: int, enc_in: int, num_estimators: int) -> int:
    """Weights of the base map plus one context map per stage."""
    return (1 + num_estimators) * (seq_len * enc_in) * (pred_len * enc_in)


class Model(nn.Module):
    """Combine symmetric trees whose stages receive prior forecast context."""
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, num_estimators: int = 16,
                 tree_depth: int = 3, learning_rate: float = 0.1,
                 temperature: float = 1.0, use_revin: bool = True,
                 max_dense_parameters: int = MAX_DENSE_PARAMETERS) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, num_estimators, tree_depth) < 1 or min(learning_rate, temperature) <= 0:
            raise ValueError("dimensions, estimators, and learning_rate must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        dense = dense_parameter_count(seq_len, pred_len, enc_in, num_estimators)
        if dense > max_dense_parameters:
            raise ValueError(f"CatBoostTS needs {dense:,} dense weights at seq_len={seq_len}, "
                             f"pred_len={pred_len}, enc_in={enc_in} (max_dense_parameters="
                             f"{max_dense_parameters:,}); they scale as (seq_len*enc_in)*(pred_len*enc_in)")
        self.learning_rate = learning_rate
        input_dim, output_dim = seq_len * enc_in, pred_len * enc_in
        self.base = nn.Linear(input_dim, output_dim)
        self.trees = nn.ModuleList([SoftObliviousTree(input_dim, output_dim, tree_depth,
                                                      temperature) for _ in range(num_estimators)])
        self.context = nn.ModuleList([nn.Linear(output_dim, input_dim, bias=False)
                                      for _ in range(num_estimators)])
        self.revin = RevIN(enc_in, affine=use_revin, enabled=use_revin)
        self.aux_loss: torch.Tensor | None = None
    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}")
        flat = self.revin(x_enc, "norm").flatten(1)
        forecast = self.base(flat)
        for stage, (tree, context) in enumerate(zip(self.trees, self.context, strict=True), start=1):
            ordered_state = flat - torch.tanh(context(forecast / float(stage)))
            forecast = forecast + self.learning_rate * tree(ordered_state)
        self.aux_loss = forecast.new_zeros(())
        return self.revin(forecast.view(-1, self.pred_len, self.enc_in), "denorm")
