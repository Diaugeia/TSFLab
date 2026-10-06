---
name: "node2vec_embedding"
description: "Seeded node2vec embedding of a fixed weighted graph: p/q-biased second-order walks and skip-gram with negative sampling. Use to precompute a structural spatial embedding of a road or sensor graph; not for learned or dynamic graphs, node features, or graphs too large for a dense [N, N] count matrix."
---

# node2vec_embedding

## What it does

Samples `num_walks` walks of `walk_length` vertices from every vertex. The first step
follows out-edge weights; afterwards the move `t -> v -> x` has weight
`w(v, x) * a(t, x)` with `a = 1/p` when `x == t`, `1` when the edge `x -> t` exists,
and `1/q` otherwise. A vertex without out-edges ends the walk. Pairs `d <= window`
steps apart count `(window - d + 1) / window` in both directions (the expected
rate of word2vec's shrunk window). Skip-gram with negative sampling is fitted with
Adam on `steps` minibatches of `batch_size` pairs drawn by count, with `negative`
noise vertices per pair from the unigram distribution to the power `0.75`. Input
vectors start uniform in `[-0.5/dim, 0.5/dim)`, output vectors at zero.

## When to use

Use when a model needs fixed vertex vectors that encode graph proximity (GMAN's
spatial embedding, graph positional encodings). Do not use for graphs that change
per sample or are learned, when node features should drive the embedding, or for very
large graphs: the pair counts are a dense `[N, N]` float64 matrix.

## Interface

`node2vec_embedding(adj_mx, dim=128, *, p=1.0, q=1.0, num_walks=10, walk_length=80, window=10, negative=5, steps=2000, batch_size=4096, lr=0.01, seed=0) -> torch.Tensor`

- `adj_mx`: array-like `[N, N]`; `adj_mx[i, j] > 0` is the weight of edge `i -> j`.
  Non-square, non-finite, or negative input raises `ValueError`.
- Defaults are the node2vec paper's walk settings (`r = 10`, `l = 80`, `k = 10`, `d = 128`).
- Returns float32 `[N, dim]` on CPU, no gradient. The same inputs and `seed` give the
  same result (CPU `torch.Generator`); `steps = 0` returns the initial vectors.
- No parameters or buffers: call it once at construction and store the result.
