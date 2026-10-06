"""Deterministic node2vec vertex embeddings of a fixed weighted graph.

node2vec (Grover and Leskovec, KDD 2016) samples second-order biased random
walks and fits skip-gram with negative sampling (SGNS) to the walk corpus. The
walk transition from ``t -> v`` to ``x`` is proportional to ``w(v, x) * a(t, x)``
with ``a = 1/p`` when ``x == t``, ``1`` when the edge ``x -> t`` exists, and
``1/q`` otherwise. The first step from a start vertex is proportional to its
out-edge weights; a vertex without out-edges ends the walk.

The walk corpus is reduced to weighted (center, context) pair counts: a
context ``d`` steps away gets weight ``(window - d + 1) / window``, the
expected inclusion rate of word2vec's randomly shrunk window. SGNS is then
fitted on minibatches of pairs drawn in proportion to those counts, with
``negative`` noise vertices per pair from the unigram distribution raised to
``0.75``. Everything runs from a seeded ``torch.Generator`` on CPU, so the same
graph and settings give the same embedding.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _neighbor_table(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded out-neighbor ids ``[N, max_deg]`` and their weights (0 for padding)."""
    nodes = weights.shape[0]
    degree = (weights > 0).sum(1)
    width = max(int(degree.max().item()), 1) if nodes else 1
    order = torch.argsort((weights > 0).to(torch.int8), dim=1, descending=True, stable=True)
    neighbors = order[:, :width]
    neighbor_weights = torch.gather(weights, 1, neighbors)
    neighbor_weights = torch.where(neighbor_weights > 0, neighbor_weights, torch.zeros_like(neighbor_weights))
    return neighbors, neighbor_weights


def _simulate_walks(
    weights: torch.Tensor,
    p: float,
    q: float,
    num_walks: int,
    walk_length: int,
    generator: torch.Generator,
    chunk: int = 65536,
) -> torch.Tensor:
    """Biased walks ``[num_walks * N, walk_length]``; ``-1`` pads walks that stopped early."""
    nodes = weights.shape[0]
    edge = weights > 0
    neighbors, neighbor_weights = _neighbor_table(weights)
    has_out = neighbor_weights.sum(1) > 0
    starts = torch.arange(nodes).repeat(num_walks)
    walks = torch.full((starts.numel(), walk_length), -1, dtype=torch.long)
    for begin in range(0, starts.numel(), chunk):
        current = starts[begin : begin + chunk].clone()
        rows = torch.arange(current.numel())
        block = walks[begin : begin + chunk]
        block[:, 0] = current
        alive = torch.ones_like(current, dtype=torch.bool)
        previous = torch.full_like(current, -1)
        for step in range(1, walk_length):
            alive &= has_out[current]
            if not alive.any():
                break
            candidates = neighbors[current]
            probabilities = neighbor_weights[current].clone()
            if step > 1:
                returns = candidates == previous.unsqueeze(1)
                back_edge = edge[candidates, previous.clamp_min(0).unsqueeze(1)]
                bias = torch.where(
                    returns,
                    torch.full_like(probabilities, 1.0 / p),
                    torch.where(back_edge, torch.ones_like(probabilities), torch.full_like(probabilities, 1.0 / q)),
                )
                probabilities = probabilities * bias
            probabilities[~alive] = 1.0
            choice = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            following = candidates[rows, choice]
            previous = torch.where(alive, current, previous)
            current = torch.where(alive, following, current)
            block[:, step] = torch.where(alive, current, torch.full_like(current, -1))
    return walks


def _pair_counts(walks: torch.Tensor, nodes: int, window: int) -> torch.Tensor:
    """Symmetric window-weighted (center, context) counts ``[N, N]`` from a walk corpus."""
    counts = torch.zeros(nodes * nodes, dtype=torch.float64)
    for distance in range(1, min(window, walks.shape[1] - 1) + 1):
        left, right = walks[:, :-distance], walks[:, distance:]
        valid = (left >= 0) & (right >= 0)
        left, right = left[valid], right[valid]
        weight = (window - distance + 1) / window
        counts += weight * torch.bincount(left * nodes + right, minlength=nodes * nodes).double()
        counts += weight * torch.bincount(right * nodes + left, minlength=nodes * nodes).double()
    return counts.view(nodes, nodes)


def node2vec_embedding(
    adj_mx,
    dim: int = 128,
    *,
    p: float = 1.0,
    q: float = 1.0,
    num_walks: int = 10,
    walk_length: int = 80,
    window: int = 10,
    negative: int = 5,
    steps: int = 2000,
    batch_size: int = 4096,
    lr: float = 0.01,
    seed: int = 0,
) -> torch.Tensor:
    """Return a float32 ``[N, dim]`` node2vec embedding of the directed weighted graph ``adj_mx``.

    ``adj_mx[i, j] > 0`` is the weight of edge ``i -> j``; negative and non-finite
    entries are rejected. The returned vectors are the SGNS input ("word") vectors.
    """
    weights = torch.as_tensor(np.asarray(adj_mx, dtype=np.float64))
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("adj_mx must be a square [N, N] matrix")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("adj_mx must hold finite non-negative edge weights")
    if dim < 1 or num_walks < 1 or walk_length < 2 or window < 1 or negative < 1:
        raise ValueError("dim, num_walks, window and negative must be >= 1 and walk_length >= 2")
    if p <= 0 or q <= 0 or steps < 0 or batch_size < 1 or lr <= 0:
        raise ValueError("p, q and lr must be positive; steps >= 0; batch_size >= 1")
    nodes = weights.shape[0]
    generator = torch.Generator().manual_seed(seed)
    walks = _simulate_walks(weights, p, q, num_walks, walk_length, generator)
    counts = _pair_counts(walks, nodes, window)

    # word2vec initialization: uniform input vectors, zero output vectors.
    inputs = ((torch.rand(nodes, dim, generator=generator) - 0.5) / dim).requires_grad_()
    outputs = torch.zeros(nodes, dim, requires_grad=True)
    flat = counts.flatten()
    positive = flat.nonzero().squeeze(1)
    if positive.numel() == 0 or steps == 0:
        return inputs.detach().float()
    pair_weights = flat[positive]
    frequency = torch.bincount(walks[walks >= 0], minlength=nodes).double()
    noise = frequency.pow(0.75)
    optimizer = torch.optim.Adam([inputs, outputs], lr=lr)
    for _ in range(steps):
        drawn = positive[torch.multinomial(pair_weights, batch_size, replacement=True, generator=generator)]
        centers, contexts = drawn // nodes, drawn % nodes
        noise_ids = torch.multinomial(noise, batch_size * negative, replacement=True, generator=generator)
        center_vectors = inputs[centers]
        positive_score = (center_vectors * outputs[contexts]).sum(-1)
        negative_score = torch.einsum(
            "bd,bkd->bk", center_vectors, outputs[noise_ids].view(batch_size, negative, dim)
        )
        loss = -(F.logsigmoid(positive_score) + F.logsigmoid(-negative_score).sum(-1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return inputs.detach().float()


__all__ = ["node2vec_embedding"]
