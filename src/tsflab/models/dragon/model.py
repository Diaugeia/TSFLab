"""Local DRAGON: multivariate de Bruijn graph encoders beside a TimesNet backbone.

Independent implementation from the paper (Section 2, Figs. 2-3, Appendices A-C)
and the pinned official code (``KurbanIntelligenceLab/MultdBG-Time-Series-Library``):
``dBG/MultiDeBruijnGraph.py`` (graph construction), ``dBG/dBGSampler.py`` (node
masks), ``dBG/GraphEncoder.py`` (``GraphEncoder_Attn_new``), ``data_provider/
data_loader.py`` (``dBG_Dataset``) and ``models/TimesNet.py`` (the downstream model).

One encoder per alphabet size builds, from the scaled training series, a
multivariate de Bruijn graph (MdBG) over uniformly discretized ``(k-1)``-tuples,
optionally diffuses it with personalized PageRank and keeps the top-k sources per
node, and encodes it with graph attention. For a history window the encoder
marks the nodes of the window's tuples (exact match, else the nearest node in L1),
masks the graph-attention outputs with that node set and pools them into
``seq_len`` steps with learned time queries. The pooled graph features of all
encoders, mixed by softmax weights, are concatenated with the TimesNet embedding.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tsflab.models._components.dominant_periods import dominant_periods
from tsflab.models._components.embed import DataEmbedding
from tsflab.models._components.inception_block import InceptionBlock2d
from tsflab.models._components.marks import adapt_tslib_marks
from tsflab.models._components.revin import RevIN

# ----------------------------------------------------------------------------
# Discretization and MdBG construction (Section 2.1, Algorithm 1)
# ----------------------------------------------------------------------------


def uniform_bin_edges(series: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin edges ``[D, n_bins + 1]`` between each column's min and max.

    A constant column gets ``[-inf, inf, ..., inf]`` so all of its values fall in
    bin 0 (scikit-learn's single-bin treatment of constant features).
    """
    series = np.asarray(series, dtype=np.float64)
    low, high = series.min(axis=0), series.max(axis=0)
    rows = []
    for lo, hi in zip(low, high):
        if hi > lo:
            rows.append(np.linspace(lo, hi, n_bins + 1))
        else:
            rows.append(np.concatenate([[-np.inf], np.full(n_bins, np.inf)]))
    return np.stack(rows)


def discretize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index in ``[0, n_bins)`` of every value of ``[..., D]``, searched right over the inner edges."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty(values.shape, dtype=np.int64)
    for dim in range(values.shape[-1]):
        out[..., dim] = np.searchsorted(edges[dim, 1:-1], values[..., dim], side="right")
    return out


def build_mdbg(series: np.ndarray, k: int, n_bins: int) -> dict[str, np.ndarray]:
    """Algorithm 1 on a ``[T, D]`` series.

    Nodes ``(d, (c_t, ..., c_{t+k-2}))`` are numbered in first-appearance order
    (time step, then dimension, prefix before suffix). Every k-tuple adds the
    directed edge prefix -> suffix; at ``t = 0`` all prefixes, and at every step
    all suffixes, of the ``D`` dimensions are joined by bidirectional hyper-tuple
    edges. Each node keeps the raw ``(k-1)``-tuples mapped to it, in insertion
    order (a tuple is recorded once as a suffix and once as the next prefix).
    """
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2 or series.shape[0] < k:
        raise ValueError("the MdBG needs a [T, D] series with T >= k")
    edges = uniform_bin_edges(series, n_bins)
    codes = discretize(series, edges)
    steps, dims = series.shape
    index: dict[tuple[int, tuple[int, ...]], int] = {}
    node_dim: list[int] = []
    node_keys: list[tuple[int, ...]] = []
    features: list[list[np.ndarray]] = []

    def visit(key, raw) -> int:
        node = index.get(key)
        if node is None:
            node = len(node_dim)
            index[key] = node
            node_dim.append(key[0])
            node_keys.append(key[1])
            features.append([])
        features[node].append(raw)
        return node

    tuple_edges: set[tuple[int, int]] = set()
    suffix_ids = np.empty((steps - k + 1, dims), dtype=np.int64)
    prefix_ids = np.empty(dims, dtype=np.int64)
    for t in range(steps - k + 1):
        for d in range(dims):
            code = codes[t : t + k, d].tolist()
            raw = series[t : t + k, d]
            prefix = visit((d, tuple(code[:-1])), raw[:-1])
            suffix = visit((d, tuple(code[1:])), raw[1:])
            tuple_edges.add((prefix, suffix))
            if t == 0:
                prefix_ids[d] = prefix
            suffix_ids[t, d] = suffix
    num_nodes = len(node_dim)
    pairs = [np.array(sorted(tuple_edges), dtype=np.int64).reshape(-1, 2)]
    if dims > 1:
        left, right = np.where(~np.eye(dims, dtype=bool))
        groups = np.concatenate([prefix_ids[None], suffix_ids])
        for start in range(0, len(groups), 4096):
            chunk = groups[start : start + 4096]
            hyper = np.stack([chunk[:, left].ravel(), chunk[:, right].ravel()], axis=1)
            pairs.append(np.unique(hyper, axis=0))
    edge_list = np.unique(np.concatenate(pairs), axis=0)
    counts = np.array([len(rows) for rows in features], dtype=np.int64)
    return {
        "bin_edges": edges,
        "node_dim": np.asarray(node_dim, dtype=np.int64),
        "node_keys": np.asarray(node_keys, dtype=np.int64).reshape(num_nodes, k - 1),
        "feature_values": np.concatenate([np.stack(rows) for rows in features]).astype(np.float32),
        "feature_counts": counts,
        "feature_offsets": np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64),
        "edge_index": edge_list.T.copy(),
    }


def ppr_topk_edges(edge_index: np.ndarray, num_nodes: int, alpha: float, topk: int) -> np.ndarray:
    """Exact personalized-PageRank graph diffusion with per-column top-k sparsification.

    With ``A`` the edge indicator plus unit self-loops (summed where a loop
    exists), ``d_j = sum_i A_ij``, ``T = D^-1/2 A D^-1/2`` and
    ``S = alpha (I - (1 - alpha) T)^-1``, every node ``j`` keeps the ``topk``
    largest ``S_ij`` (ties by lower ``i``) as edges ``i -> j``.
    """
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    source, target = torch.as_tensor(edge_index, dtype=torch.long)
    adjacency.index_put_((source, target), torch.ones(source.numel(), dtype=torch.float64), accumulate=True)
    adjacency += torch.eye(num_nodes, dtype=torch.float64)
    degree = adjacency.sum(dim=0)
    inv_sqrt = degree.pow(-0.5)
    inv_sqrt[torch.isinf(inv_sqrt)] = 0.0
    transition = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    diffusion = alpha * torch.linalg.inv(torch.eye(num_nodes, dtype=torch.float64) - (1.0 - alpha) * transition)
    keep = min(topk, num_nodes)
    rows = torch.argsort(diffusion, dim=0, descending=True, stable=True)[:keep]
    columns = torch.arange(num_nodes).repeat(keep)
    return torch.stack([rows.reshape(-1), columns]).numpy()


# ----------------------------------------------------------------------------
# Graph attention
# ----------------------------------------------------------------------------


class GraphAttentionConv(nn.Module):
    """Multi-head graph attention over ``[N, in]`` node features.

    ``h = W x`` split into ``heads`` of ``out_channels``; for an edge ``j -> i``
    (self-loops on every node, existing ones replaced)
    ``e_ij = LeakyReLU_0.2(a_src . h_j + a_dst . h_i)``, softmax over the incoming
    edges of ``i``, ``out_i = sum_j alpha_ij h_j``; heads are concatenated or
    averaged, then a bias is added.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int, concat: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.weight = nn.Parameter(torch.empty(heads * out_channels, in_channels))
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels if concat else out_channels))
        nn.init.xavier_uniform_(self.weight)
        bound = math.sqrt(6.0 / (heads + out_channels))
        nn.init.uniform_(self.att_src, -bound, bound)
        nn.init.uniform_(self.att_dst, -bound, bound)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        nodes = x.shape[0]
        h = F.linear(x, self.weight).view(nodes, self.heads, self.out_channels)
        score_src = (h * self.att_src).sum(-1)
        score_dst = (h * self.att_dst).sum(-1)
        source, target = edge_index
        keep = source != target
        loops = torch.arange(nodes, device=x.device)
        source = torch.cat([source[keep], loops])
        target = torch.cat([target[keep], loops])
        logits = F.leaky_relu(score_src[source] + score_dst[target], 0.2)
        peak = torch.full((nodes, self.heads), float("-inf"), dtype=logits.dtype, device=x.device)
        peak = peak.scatter_reduce(0, target[:, None].expand_as(logits), logits.detach(), reduce="amax")
        weights = (logits - peak[target]).exp()
        total = torch.zeros(nodes, self.heads, dtype=logits.dtype, device=x.device).index_add(0, target, weights)
        weights = weights / (total[target] + 1e-16)
        out = torch.zeros_like(h).index_add(0, target, weights.unsqueeze(-1) * h[source])
        out = out.reshape(nodes, -1) if self.concat else out.mean(dim=1)
        return out + self.bias


# ----------------------------------------------------------------------------
# DRAGON encoder (Fig. 2)
# ----------------------------------------------------------------------------

_GRAPH_BUFFERS = (
    "bin_edges", "node_dim", "node_keys", "feature_values", "feature_counts", "feature_offsets", "edge_index",
)


class DragonEncoder(nn.Module):
    """One MdBG encoder: graph state from the training split, GAT, mask, time-query pooling."""

    def __init__(
        self,
        channels: int,
        seq_len: int,
        k: int,
        n_bins: int,
        d_graph: int,
        num_layers: int,
        node_feat_size: int,
        heads: int,
        dropout: float,
        use_gdc: bool,
        gdc_topk: int,
        ppr_alpha: float,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.k = k
        self.n_bins = n_bins
        self.d_graph = d_graph
        self.node_feat_size = node_feat_size
        self.dropout = dropout
        self.use_gdc = use_gdc
        self.gdc_topk = gdc_topk
        self.ppr_alpha = ppr_alpha
        self.value_linear = nn.Linear((k - 1) * node_feat_size, channels)
        layers = [GraphAttentionConv(channels, channels, heads)]
        layers += [GraphAttentionConv(channels * heads, channels, heads) for _ in range(num_layers - 2)]
        layers.append(GraphAttentionConv(channels * heads, d_graph, heads, concat=False))
        self.convs = nn.ModuleList(layers)
        self.time_queries = nn.Parameter(torch.randn(seq_len, d_graph))
        self.attn_score = nn.Linear(d_graph, d_graph)
        self._reset_graph()

    # -- graph state --------------------------------------------------------
    def _reset_graph(self) -> None:
        empty = {
            "bin_edges": torch.zeros(self.channels, self.n_bins + 1, dtype=torch.float64),
            "node_dim": torch.zeros(0, dtype=torch.long),
            "node_keys": torch.zeros(0, self.k - 1, dtype=torch.long),
            "feature_values": torch.zeros(0, self.k - 1),
            "feature_counts": torch.zeros(0, dtype=torch.long),
            "feature_offsets": torch.zeros(0, dtype=torch.long),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
        }
        for name, value in empty.items():
            self.register_buffer(name, value)
        self._lookup_cache = None

    @property
    def num_nodes(self) -> int:
        return int(self.node_dim.numel())

    @property
    def fitted(self) -> bool:
        return self.num_nodes > 0

    def fit(self, series: np.ndarray) -> None:
        """Build the MdBG of the ``[T, channels]`` training series (and its diffusion)."""
        series = np.asarray(series, dtype=np.float64)
        if series.ndim != 2 or series.shape[1] != self.channels:
            raise ValueError(f"DRAGON expects a [T, {self.channels}] training series")
        graph = build_mdbg(series, self.k, self.n_bins)
        edges = graph["edge_index"]
        if self.use_gdc:
            edges = ppr_topk_edges(edges, len(graph["node_dim"]), self.ppr_alpha, self.gdc_topk)
        graph["edge_index"] = edges
        device = self.time_queries.device
        for name in _GRAPH_BUFFERS:
            current = getattr(self, name)
            setattr(self, name, torch.as_tensor(graph[name], dtype=current.dtype).to(device))
        self._lookup_cache = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in _GRAPH_BUFFERS:
            incoming = state_dict.get(prefix + name)
            current = getattr(self, name)
            if incoming is not None and incoming.shape != current.shape:
                setattr(self, name, torch.empty(incoming.shape, dtype=current.dtype, device=current.device))
        self._lookup_cache = None
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # -- node masks (dBGMasker.generate_mask) -------------------------------
    def _lookup(self):
        if self._lookup_cache is None:
            dims = self.node_dim.cpu().numpy()
            keys = self.node_keys.cpu().numpy()
            exact = {(int(d), tuple(row.tolist())): i for i, (d, row) in enumerate(zip(dims, keys))}
            per_dim = {int(d): np.flatnonzero(dims == d) for d in np.unique(dims)}
            self._lookup_cache = (exact, per_dim, keys, {})
        return self._lookup_cache

    def nearest_node(self, dim: int, key: tuple[int, ...]) -> int:
        """Exact node, else the first node of ``dim`` within L1 distance 1, else the first L1 minimizer."""
        exact, per_dim, keys, nearest = self._lookup()
        query = (dim, key)
        if query in exact:
            return exact[query]
        if query not in nearest:
            candidates = per_dim[dim]
            distance = np.abs(keys[candidates] - np.asarray(key)).sum(axis=1)
            close = np.flatnonzero(distance < 2)
            pick = close[0] if close.size else int(np.argmin(distance))
            nearest[query] = int(candidates[pick])
        return nearest[query]

    def node_mask(self, values: torch.Tensor) -> torch.Tensor:
        """``[B, N]`` indicator of the nodes of every window ``(k-1)``-tuple of ``[B, T, channels]``."""
        codes = discretize(values.detach().cpu().double().numpy(), self.bin_edges.cpu().numpy())
        batch, steps, dims = codes.shape
        width = self.k - 1
        mask = np.zeros((batch, self.num_nodes), dtype=np.float32)
        for b in range(batch):
            for d in range(dims):
                series = codes[b, :, d].tolist()
                for start in range(steps - width + 1):
                    mask[b, self.nearest_node(d, tuple(series[start : start + width]))] = 1.0
        return torch.from_numpy(mask).to(values.device)

    # -- encoding ------------------------------------------------------------
    def sample_node_inputs(self) -> torch.Tensor:
        """``[N, node_feat_size * (k-1)]`` raw tuples drawn per node (slot ``j % count`` of ``U[0, max count)``)."""
        draws = torch.randint(
            0, int(self.feature_counts.max()), (self.num_nodes, self.node_feat_size), device=self.node_dim.device
        )
        rows = self.feature_offsets[:, None] + draws % self.feature_counts[:, None]
        return self.feature_values[rows].reshape(self.num_nodes, -1)

    def encode_nodes(self, inputs: torch.Tensor) -> torch.Tensor:
        """GAT stack: dropout before every layer, ELU after all but the last."""
        x = self.value_linear(inputs)
        for conv in self.convs[:-1]:
            x = F.elu(conv(F.dropout(x, self.dropout, self.training), self.edge_index))
        return self.convs[-1](F.dropout(x, self.dropout, self.training), self.edge_index)

    def pool(self, nodes: torch.Tensor) -> torch.Tensor:
        """Time-query attention over all ``N`` (masked) nodes: ``[N, G] -> [seq_len, G]``."""
        scores = self.time_queries @ self.attn_score(nodes).T / self.d_graph**0.5
        return scores.softmax(dim=-1) @ nodes

    def encode_window(self, row: torch.Tensor) -> torch.Tensor:
        """One window: sampled node inputs, GAT over the whole graph, mask, pooling."""
        nodes = self.encode_nodes(self.sample_node_inputs()) * row.unsqueeze(-1)
        return self.pool(nodes)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        # As in the official encoder, every window samples its own node inputs
        # and runs the GAT over the whole graph, so stored activations grow as
        # batch x edges x heads x width (about 1.2 GB per ETTh1 window for the
        # three preset encoders). With gradients enabled each window is
        # checkpointed: only its pooled output is kept and the window is
        # recomputed in backward with the same random state, so the result and
        # gradients are unchanged.
        recompute = torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        pooled = []
        for row in mask:
            if recompute:
                pooled.append(checkpoint(self.encode_window, row, use_reentrant=False, preserve_rng_state=True))
            else:
                pooled.append(self.encode_window(row))
        return torch.stack(pooled)


# ----------------------------------------------------------------------------
# TimesNet backbone (downstream model of the official experiments)
# ----------------------------------------------------------------------------


class TimesBlock(nn.Module):
    """Fold the sequence by its top-k FFT periods, 2D-convolve, re-weight by amplitude, add the input."""

    def __init__(self, length: int, width: int, d_ff: int, top_k: int, num_kernels: int) -> None:
        super().__init__()
        self.length = length
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlock2d(width, d_ff, num_kernels), nn.GELU(), InceptionBlock2d(d_ff, width, num_kernels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        periods, amplitudes = dominant_periods(x, self.top_k)
        branches = []
        for period in periods.tolist():
            padded = -(-length // period) * period
            folded = F.pad(x, (0, 0, 0, padded - length)) if padded != length else x
            image = folded.reshape(batch, padded // period, period, width).permute(0, 3, 1, 2)
            out = self.conv(image.contiguous()).permute(0, 2, 3, 1).reshape(batch, padded, width)
            branches.append(out[:, :length])
        weights = amplitudes.softmax(dim=1)[:, None, None, :]
        return (torch.stack(branches, dim=-1) * weights).sum(-1) + x


class Model(nn.Module):
    """DRAGON encoders (one per alphabet size) concatenated with a TimesNet embedding."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 16,
        d_ff: int = 32,
        e_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout: float = 0.1,
        k: int = 4,
        alphabet_sizes: tuple[int, ...] = (20, 25, 30),
        d_graph: int = 16,
        graph_layers: int = 2,
        node_feat_size: int = 4,
        graph_heads: int = 16,
        graph_dropout: float = 0.4,
        use_gdc: bool = True,
        gdc_topk: int = 32,
        ppr_alpha: float = 0.05,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, d_model, d_ff, e_layers, top_k, num_kernels, d_graph, node_feat_size) < 1:
            raise ValueError("DRAGON sizes must be positive")
        if k < 2 or seq_len < k - 1:
            raise ValueError("k must be at least 2 and seq_len at least k - 1")
        if graph_layers < 2:
            raise ValueError("graph_layers must be at least 2 (input and output attention layers)")
        if not alphabet_sizes or min(alphabet_sizes) < 1:
            raise ValueError("alphabet_sizes needs at least one positive size")
        if top_k > (seq_len + pred_len) // 2:
            raise ValueError("top_k exceeds the non-DC frequencies of seq_len + pred_len")
        if d_model % 2:
            raise ValueError("d_model must be even (sinusoidal positions)")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_graph = d_graph
        self.norm = RevIN(enc_in, eps=1e-5, affine=False)
        self.embedding = DataEmbedding(enc_in, d_model, embed_type="timeF", freq="h", dropout=dropout, time_feature_dim=4)
        self.encoders = nn.ModuleList(
            DragonEncoder(enc_in, seq_len, k, size, d_graph, graph_layers, node_feat_size, graph_heads,
                          graph_dropout, use_gdc, gdc_topk, ppr_alpha)
            for size in alphabet_sizes
        )
        self.encoder_weights = nn.Parameter(torch.randn(len(alphabet_sizes)))
        width = d_model + d_graph
        self.predict_linear = nn.Linear(seq_len, seq_len + pred_len)
        self.blocks = nn.ModuleList(
            TimesBlock(seq_len + pred_len, width, d_ff, top_k, num_kernels) for _ in range(e_layers)
        )
        self.layer_norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, enc_in)

    @property
    def fitted(self) -> bool:
        return all(encoder.fitted for encoder in self.encoders)

    def fit_graphs(self, series: np.ndarray) -> None:
        """Build every encoder's graph from the scaled ``[T, enc_in]`` training series."""
        for encoder in self.encoders:
            encoder.fit(series)

    def graph_features(self, x_enc: torch.Tensor) -> torch.Tensor:
        """``[B, seq_len, d_graph]``: softmax-weighted sum of the encoders (zeros before fitting)."""
        if not self.fitted:
            return x_enc.new_zeros(x_enc.shape[0], self.seq_len, self.d_graph)
        stacked = torch.stack([encoder(encoder.node_mask(x_enc)) for encoder in self.encoders])
        return (self.encoder_weights.softmax(0).view(-1, 1, 1, 1) * stacked).sum(0)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"DRAGON expects [B, {self.seq_len}, {self.enc_in}] inputs")
        graph = self.graph_features(x_enc)
        marks = adapt_tslib_marks(x_mark_enc, embed_type="timeF", freq="h")
        tokens = torch.cat([self.embedding(self.norm(x_enc, "norm"), marks), graph], dim=-1)
        tokens = self.predict_linear(tokens.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            tokens = self.layer_norm(block(tokens))
        forecast = self.norm(self.projection(tokens), "denorm")
        return forecast[:, -self.pred_len :]
