"""Spatial-temporal synchronous graph convolution over a localized window graph.

STSGCN (Song et al., AAAI 2020) joins the spatial graphs of ``steps`` adjacent
time steps into one ``steps * N`` node graph and runs stacked graph convolutions
on it, so one propagation mixes neighbors in space and time at once
(STFGNN reuses the same module on its fusion graph). One module (an STSGCM):

* each layer ``l``: ``h_l = act(A' h_{l-1} W_l + b_l)`` on the ``[steps * N, C]``
  window, with ``act`` either GLU (``(A'hW1 + b1) * sigmoid(A'hW2 + b2)``) or ReLU;
* aggregation: element-wise max over the outputs of all layers;
* cropping: keep only the ``N`` rows of one time block (``crop``).

Inputs carry a window axis ``M``. With ``num_modules == M`` every window has its
own weights (the papers' "individual" modules); with ``num_modules == 1`` one
module is shared by all windows. Weights use MXNet's ``Xavier(magnitude)``
uniform rule, the initializer of both official codes, with zero biases.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def mxnet_xavier_uniform_(tensor: torch.Tensor, fan_in: int, fan_out: int, magnitude: float) -> torch.Tensor:
    """Fill ``tensor`` from ``U(-s, s)``, ``s = sqrt(magnitude / ((fan_in + fan_out) / 2))``."""
    scale = math.sqrt(magnitude / ((fan_in + fan_out) / 2.0))
    with torch.no_grad():
        return tensor.uniform_(-scale, scale)


def localized_adjacency(adj_mx, steps: int) -> torch.Tensor:
    """STSGCN localized graph ``[steps * N, steps * N]`` from a spatial adjacency.

    The spatial graph is binarized and symmetrized (``A_ij = 1`` if either
    direction has a positive weight) and placed on every diagonal block; the same
    node is linked in consecutive steps; every node gets a self-loop.
    """
    spatial = np.asarray(adj_mx, dtype=np.float64)
    if spatial.ndim != 2 or spatial.shape[0] != spatial.shape[1]:
        raise ValueError("adj_mx must be a square [N, N] matrix")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    nodes = spatial.shape[0]
    binary = ((spatial > 0) | (spatial.T > 0)).astype(np.float32)
    local = np.zeros((steps * nodes, steps * nodes), dtype=np.float32)
    for step in range(steps):
        local[step * nodes : (step + 1) * nodes, step * nodes : (step + 1) * nodes] = binary
    index = np.arange(nodes)
    for step in range(steps - 1):
        local[step * nodes + index, (step + 1) * nodes + index] = 1.0
        local[(step + 1) * nodes + index, step * nodes + index] = 1.0
    np.fill_diagonal(local, 1.0)
    return torch.from_numpy(local)


class SynchronousGraphModule(nn.Module):
    """Stacked window-graph convolutions, max aggregation, and cropping to one time block."""

    def __init__(
        self,
        num_nodes: int,
        steps: int,
        in_dim: int,
        filters: tuple[int, ...] | list[int],
        *,
        activation: str = "GLU",
        num_modules: int = 1,
        crop: int = 1,
        init_magnitude: float = 0.0003,
    ) -> None:
        super().__init__()
        filters = tuple(int(width) for width in filters)
        if not filters or min(filters) < 1 or min(num_nodes, steps, in_dim, num_modules) < 1:
            raise ValueError("num_nodes, steps, in_dim, num_modules and every filter must be positive")
        if len(set(filters)) != 1:
            raise ValueError("max aggregation needs equal filter widths")
        if activation not in {"GLU", "relu"}:
            raise ValueError("activation must be 'GLU' or 'relu'")
        if not 0 <= crop < steps:
            raise ValueError("crop must index a time block in [0, steps)")
        self.num_nodes, self.steps, self.crop = num_nodes, steps, crop
        self.activation, self.num_modules = activation, num_modules
        self.out_dim = filters[-1]
        factor = 2 if activation == "GLU" else 1
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        width = in_dim
        for out_dim in filters:
            weight = torch.empty(num_modules, width, factor * out_dim)
            for module in weight:
                mxnet_xavier_uniform_(module, width, factor * out_dim, init_magnitude)
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(torch.zeros(num_modules, 1, factor * out_dim)))
            width = out_dim

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """``x [B, M, steps * N, C]``, ``adjacency [steps * N, steps * N]`` -> ``[B, M, N, C']``."""
        if x.ndim != 4 or x.shape[2] != self.steps * self.num_nodes:
            raise ValueError(f"x must be [batch, windows, {self.steps * self.num_nodes}, features]")
        windows = x.shape[1]
        if self.num_modules not in (1, windows):
            raise ValueError(f"{self.num_modules} modules cannot serve {windows} windows")
        outputs = []
        hidden = x
        for weight, bias in zip(self.weights, self.biases):
            propagated = torch.einsum("uv,bmvc->bmuc", adjacency, hidden)
            projected = torch.einsum("bmuc,mco->bmuo", propagated, weight.expand(windows, -1, -1)) + bias
            if self.activation == "GLU":
                value, gate = projected.chunk(2, dim=-1)
                hidden = value * torch.sigmoid(gate)
            else:
                hidden = torch.relu(projected)
            outputs.append(hidden)
        block = slice(self.crop * self.num_nodes, (self.crop + 1) * self.num_nodes)
        return torch.stack([output[:, :, block] for output in outputs], dim=0).amax(dim=0)


__all__ = ["SynchronousGraphModule", "localized_adjacency", "mxnet_xavier_uniform_"]
