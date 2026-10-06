"""PDFormer: propagation delay-aware dynamic long-range Transformer (Jiang et al., AAAI 2023).

Independent implementation from the paper (arXiv 2301.07945v3), checked against
the official LibCity-based code (BUAABIGSCity/PDFormer, MIT) at a pinned revision.

- Data embedding (Eq. 2): value projection + Laplacian-eigenvector node embedding
  + minute-of-day and day-of-week embeddings + sinusoidal position.
- Encoder layer (Eq. 3-15): one multi-head block whose heads are split into
  temporal heads (attention over time per node), geographic spatial heads
  (attention over nodes within ``far_mask_delta`` hops; keys enriched by the
  delay-aware feature transformation) and semantic spatial heads (attention
  restricted to the ``dtw_delta`` most DTW-similar nodes), then an MLP.
- Output (Eq. 16): summed 1x1 skip convolutions, a time-to-horizon 1x1 conv and
  a feature-to-output 1x1 conv.

The semantic mask and the short-term traffic pattern set are fitted on the
training split only (``fit_from_series``); the official pipeline computes the
DTW profile over the whole series, including validation and test periods.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.adj_norm import symmetric_normalized_laplacian
from tsflab.models._components.embed import PositionalEmbedding
from tsflab.models._components.marks import to_calendar_spatiotemporal


# ------------------------------------------------------------ graph statistics
def laplacian_eigenvectors(adjacency: np.ndarray, count: int) -> np.ndarray:
    """``count`` smallest non-trivial eigenvectors of ``I - D^-1/2 A D^-1/2`` (zero-padded).

    Directed graphs are symmetrized first; one trivial eigenvector is skipped per
    isolated node plus the first one, as in the official code.
    """
    symmetric = np.maximum(adjacency, adjacency.T)
    isolated = int((symmetric.sum(axis=1) == 0).sum())
    _, vectors = np.linalg.eigh(symmetric_normalized_laplacian(symmetric))
    chosen = vectors[:, isolated + 1: isolated + 1 + count]
    out = np.zeros((adjacency.shape[0], count), dtype=np.float32)
    out[:, : chosen.shape[1]] = chosen
    return out


def hop_distances(adjacency: np.ndarray) -> np.ndarray:
    """Unweighted shortest-path hop counts on ``adjacency > 0`` (``inf`` when unreachable)."""
    return shortest_path(adjacency > 0, method="D", directed=True, unweighted=True)


def daily_profiles(series: torch.Tensor, steps_per_day: int) -> torch.Tensor:
    """``[T, N]`` -> ``[N, steps_per_day]`` mean over complete days (whole series if < 1 day)."""
    days = series.shape[0] // steps_per_day
    if days == 0:
        return series.transpose(0, 1)
    whole = series[: days * steps_per_day].reshape(days, steps_per_day, series.shape[1])
    return whole.mean(dim=0).transpose(0, 1)


def dtw_distance_matrix(profiles: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    """Exact DTW (absolute-difference cost) between all row pairs of ``[N, S]``.

    Dynamic program over anti-diagonals, vectorized over node pairs.
    """
    profiles = profiles.to(torch.float64)
    nodes, length = profiles.shape
    rows, cols = torch.triu_indices(nodes, nodes, offset=1)
    out = profiles.new_zeros(nodes, nodes)
    positions = torch.arange(length + 1)
    for begin in range(0, rows.numel(), chunk):
        left = profiles[rows[begin:begin + chunk]]
        right = profiles[cols[begin:begin + chunk]]
        pairs = left.shape[0]
        infinite = torch.full((pairs, length + 1), math.inf, dtype=profiles.dtype)
        two_back, one_back = infinite.clone(), infinite.clone()
        two_back[:, 0] = 0.0  # D[0, 0]
        for diagonal in range(2, 2 * length + 1):
            column = diagonal - positions
            valid = (positions >= 1) & (column >= 1) & (column <= length)
            cost = (left[:, (positions - 1).clamp(0, length - 1)]
                    - right[:, (column - 1).clamp(0, length - 1)]).abs()
            from_above = torch.cat((infinite[:, :1], one_back[:, :-1]), dim=1)
            from_corner = torch.cat((infinite[:, :1], two_back[:, :-1]), dim=1)
            best = torch.minimum(torch.minimum(from_above, one_back), from_corner)
            two_back, one_back = one_back, torch.where(valid, cost + best, infinite)
        out[rows[begin:begin + chunk], cols[begin:begin + chunk]] = one_back[:, length]
    return out + out.transpose(0, 1)


def _znorm(x: torch.Tensor) -> torch.Tensor:
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return torch.where(std > 0, (x - x.mean(dim=-1, keepdim=True)) / std.clamp_min(1e-12), 0.0)


def _cross_correlations(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """``[M, S]`` x ``[K, S]`` -> ``[M, K, 2S-1]`` cross-correlation over all shifts."""
    length = x.shape[-1]
    padded = F.pad(x, (length - 1, length - 1))
    windows = padded.unfold(-1, length, 1)  # [M, 2S-1, S]
    return torch.einsum("mws,ks->mkw", windows, centroids)


def _shape_based_distance(x, centroids):
    """SBD = 1 - max_w NCC_c(x, c); also returns the best shift index per pair."""
    norms = x.norm(dim=-1)[:, None] * centroids.norm(dim=-1)[None, :]
    ncc = _cross_correlations(x, centroids) / norms.clamp_min(1e-12).unsqueeze(-1)
    best, shift = ncc.max(dim=-1)
    return 1.0 - best, shift


def _align(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Shift each row of ``x`` by ``shift - (S-1)`` steps with zero fill (k-Shape alignment)."""
    length = x.shape[-1]
    offset = shift - (length - 1)
    index = torch.arange(length)[None, :] + offset[:, None]
    inside = (index >= 0) & (index < length)
    return torch.where(inside, x.gather(1, index.clamp(0, length - 1)), 0.0)


class _EmptyCluster(Exception):
    pass


def _kshape_run(data, clusters, iterations, generator, chunk):
    count, length = data.shape
    centering = torch.eye(length, dtype=data.dtype) - 1.0 / length

    def assign(centroids):
        distances = torch.cat([_shape_based_distance(data[i:i + chunk], centroids)[0]
                               for i in range(0, count, chunk)])
        best, labels = distances.min(dim=-1)
        if torch.bincount(labels, minlength=clusters).min() == 0:
            raise _EmptyCluster
        return labels, float((best ** 2).mean())

    centroids = data[torch.randint(0, count, (clusters,), generator=generator)].clone()
    labels, _ = assign(centroids)
    inertia = math.inf
    for _ in range(iterations):
        previous = centroids.clone()
        updated = torch.empty_like(centroids)
        for k in range(clusters):
            members = data[labels == k]
            _, shift = _shape_based_distance(members, previous[k:k + 1])
            members = _align(members, shift[:, 0])
            scatter = centering.T @ (members.T @ members) @ centering
            candidate = torch.linalg.eigh(scatter)[1][:, -1]
            plus = (members - candidate).norm(dim=-1).sum()
            minus = (members + candidate).norm(dim=-1).sum()
            updated[k] = -candidate if minus < plus else candidate
        centroids = _znorm(updated)
        new_labels, new_inertia = assign(centroids)
        if abs(inertia - new_inertia) < 1e-6 or new_inertia > inertia:
            return previous
        labels, inertia = new_labels, new_inertia
    return centroids


def kshape_centroids(series: torch.Tensor, clusters: int, iterations: int, seed: int = 0,
                     chunk: int = 131072, attempts: int = 10) -> torch.Tensor:
    """k-Shape clustering (Paparrizos & Gravano) of ``[M, S]`` series as configured upstream.

    Like the pinned tslearn 0.5.2 ``KShape``: inputs are used as given (not z-normalized),
    initial centroids are random series, each iteration aligns members to the previous
    centroid by shape-based distance, takes the leading eigenvector of the centered
    scatter (sign closer to the members), z-normalizes it, and stops when the inertia
    stalls or rises; an empty cluster restarts with a new initialization.
    """
    data = series.to(torch.float64)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(attempts):
        try:
            return _kshape_run(data, clusters, iterations, generator, chunk).float()
        except _EmptyCluster:
            continue
    raise RuntimeError("k-Shape produced an empty cluster in every attempt; lower n_cluster")


# ------------------------------------------------------------------- layers
class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch per sample while training."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class SpatialTemporalAttention(nn.Module):
    """Heterogeneous heads: temporal (Eq. 12-14), GeoSSA with DFT keys (Eq. 6-11), SemSSA (Eq. 6)."""

    def __init__(self, width, geo_heads, sem_heads, t_heads, qkv_bias, attn_drop, proj_drop):
        super().__init__()
        heads = geo_heads + sem_heads + t_heads
        if width % heads:
            raise ValueError("embed_dim must be divisible by the total number of heads")
        self.head_width = width // heads
        self.heads = {"t": t_heads, "geo": geo_heads, "sem": sem_heads}
        self.scale = self.head_width ** -0.5
        widths = {name: count * self.head_width for name, count in self.heads.items()}
        self.query = nn.ModuleDict({n: nn.Linear(width, w, bias=qkv_bias) for n, w in widths.items()})
        self.key = nn.ModuleDict({n: nn.Linear(width, w, bias=qkv_bias) for n, w in widths.items()})
        self.value = nn.ModuleDict({n: nn.Linear(width, w, bias=qkv_bias) for n, w in widths.items()})
        # Delay-aware feature transformation (Eq. 7-11): u = pattern-embedded recent window,
        # m_i / W_c from embedded pattern keys; their mix is added to the geographic keys.
        self.pattern_query = nn.Linear(width, widths["geo"])
        self.pattern_key = nn.Linear(width, widths["geo"])
        self.pattern_value = nn.Linear(width, widths["geo"])
        self.attention_dropout = nn.Dropout(attn_drop)
        self.projection = nn.Linear(width, width)
        self.projection_dropout = nn.Dropout(proj_drop)

    def _split(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], heads, self.head_width).transpose(-2, -3)

    def _attend(self, q, k, v, heads, blocked=None):
        q, k, v = (self._split(t, heads) for t in (q, k, v))
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if blocked is not None:
            scores = scores.masked_fill(blocked, float("-inf"))
        weights = self.attention_dropout(scores.softmax(dim=-1))
        out = torch.matmul(weights, v).transpose(-2, -3)
        return out.reshape(*out.shape[:-2], -1)

    def forward(self, x, recent_patterns, pattern_keys, geo_blocked, sem_blocked):
        # x: [B, T, N, d]; recent_patterns: [B, T, N, d]; pattern_keys: [P, d]
        by_node = x.transpose(1, 2)  # [B, N, T, d]
        temporal = self._attend(self.query["t"](by_node), self.key["t"](by_node),
                                self.value["t"](by_node), self.heads["t"]).transpose(1, 2)

        pattern_scores = torch.matmul(self.pattern_query(recent_patterns),
                                      self.pattern_key(pattern_keys).T) * self.scale
        delay = torch.matmul(pattern_scores.softmax(dim=-1), self.pattern_value(pattern_keys))
        geo_keys = self.key["geo"](x) + delay
        geographic = self._attend(self.query["geo"](x), geo_keys, self.value["geo"](x),
                                  self.heads["geo"], geo_blocked)
        semantic = self._attend(self.query["sem"](x), self.key["sem"](x), self.value["sem"](x),
                                self.heads["sem"], sem_blocked)
        mixed = self.projection(torch.cat((temporal, geographic, semantic), dim=-1))
        return self.projection_dropout(mixed)


class EncoderBlock(nn.Module):
    def __init__(self, width, geo_heads, sem_heads, t_heads, mlp_ratio, qkv_bias, drop,
                 attn_drop, drop_path, pre_norm):
        super().__init__()
        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.attention = SpatialTemporalAttention(width, geo_heads, sem_heads, t_heads,
                                                  qkv_bias, attn_drop, drop)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, width), nn.Dropout(drop))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, patterns, keys, geo_blocked, sem_blocked):
        if self.pre_norm:
            x = x + self.drop_path(self.attention(self.norm1(x), patterns, keys, geo_blocked, sem_blocked))
            return x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm1(x + self.drop_path(self.attention(x, patterns, keys, geo_blocked, sem_blocked)))
        return self.norm2(x + self.drop_path(self.mlp(x)))


# -------------------------------------------------------------------- model
class Model(nn.Module):
    """PDFormer for single-target node series ``[B, L, N]`` with optional calendar marks."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        embed_dim: int = 64,
        skip_dim: int = 256,
        lape_dim: int = 8,
        geo_num_heads: int = 4,
        sem_num_heads: int = 2,
        t_num_heads: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.3,
        enc_depth: int = 6,
        pre_norm: bool = True,
        s_attn_size: int = 3,
        far_mask_delta: int = 7,
        dtw_delta: int = 5,
        n_cluster: int = 16,
        cluster_max_iter: int = 5,
        cand_key_days: int = 14,
        steps_per_day: int = 288,
        add_time_in_day: bool = True,
        add_day_in_week: bool = True,
        random_flip: bool = True,
        huber_delta: float = 2.0,
        curriculum_step: int = 0,
    ) -> None:
        super().__init__()
        if adj_mx is None:
            raise ValueError(
                "PDFormer needs the road-network adjacency adj_mx [enc_in, enc_in] for its "
                "Laplacian embedding and geographic mask; use a graph dataset that ships adj_mx.npy"
            )
        adjacency = np.asarray(adj_mx, dtype=np.float64)
        if adjacency.shape != (enc_in, enc_in):
            raise ValueError(f"adj_mx must have shape {(enc_in, enc_in)}")
        if embed_dim % 2:
            raise ValueError("embed_dim must be even for the sinusoidal position table")
        if min(seq_len, pred_len, enc_in, s_attn_size, n_cluster, dtw_delta, steps_per_day) < 1:
            raise ValueError("lengths, nodes, pattern size, clusters, dtw_delta, steps_per_day must be >= 1")
        self.seq_len, self.pred_len, self.num_nodes = seq_len, pred_len, enc_in
        self.s_attn_size, self.n_cluster = s_attn_size, n_cluster
        self.dtw_delta = min(dtw_delta, enc_in)
        self.cluster_max_iter, self.cand_key_days = cluster_max_iter, cand_key_days
        self.steps_per_day = steps_per_day
        self.add_time_in_day, self.add_day_in_week = add_time_in_day, add_day_in_week
        self.random_flip = random_flip
        self.huber_delta = huber_delta
        self.curriculum_step = curriculum_step

        # Geographic mask: block pairs at >= far_mask_delta hops (official uses the transposed hop matrix).
        hops = hop_distances(adjacency).T
        self.register_buffer("geo_blocked", torch.as_tensor(hops >= far_mask_delta))
        # Semantic mask: until fitted on the training split, each node attends only to itself.
        self.register_buffer("sem_blocked", ~torch.eye(enc_in, dtype=torch.bool))
        self.register_buffer("pattern_set", torch.zeros(n_cluster, s_attn_size))
        self.register_buffer("laplacian", torch.as_tensor(laplacian_eigenvectors(adjacency, lape_dim)))
        self.register_buffer("batches_seen", torch.zeros((), dtype=torch.long))
        self.register_buffer("task_level", torch.zeros((), dtype=torch.long))

        self.value_embedding = nn.Linear(1, embed_dim)
        self.position = PositionalEmbedding(embed_dim, max_len=seq_len)
        self.minute_embedding = nn.Embedding(1440, embed_dim) if add_time_in_day else None
        self.weekday_embedding = nn.Embedding(7, embed_dim) if add_day_in_week else None
        self.laplacian_embedding = nn.Linear(lape_dim, embed_dim)
        self.embedding_dropout = nn.Dropout(drop)
        self.pattern_embedding = nn.Linear(s_attn_size, embed_dim)
        rates = torch.linspace(0, drop_path, enc_depth).tolist()
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, geo_num_heads, sem_num_heads, t_num_heads, mlp_ratio, qkv_bias,
                         drop, attn_drop, rates[i], pre_norm)
            for i in range(enc_depth)
        ])
        self.skip_convs = nn.ModuleList([nn.Linear(embed_dim, skip_dim) for _ in range(enc_depth)])
        self.time_conv = nn.Conv2d(seq_len, pred_len, kernel_size=1)  # Conv1 of Eq. 16
        self.output_conv = nn.Linear(skip_dim, 1)  # Conv2 of Eq. 16

    # --------------------------------------------------------- data fitting
    @torch.no_grad()
    def fit_from_series(self, series: torch.Tensor) -> None:
        """Fit the semantic mask (DTW of daily mean profiles) and the k-Shape pattern set.

        ``series`` is the ``[T, N]`` training-split value series in the model's input scale.
        """
        if series.ndim != 2 or series.shape[1] != self.num_nodes:
            raise ValueError(f"series must be [T, {self.num_nodes}]")
        series = series.detach().cpu().to(torch.float64)
        distances = dtw_distance_matrix(daily_profiles(series, self.steps_per_day))
        nearest = distances.argsort(dim=1, stable=True)[:, : self.dtw_delta]
        blocked = torch.ones(self.num_nodes, self.num_nodes, dtype=torch.bool)
        blocked.scatter_(1, nearest, False)
        self.sem_blocked.copy_(blocked.to(self.sem_blocked.device))

        span = min(series.shape[0], self.cand_key_days * self.steps_per_day + self.s_attn_size - 1)
        windows = series[:span].unfold(0, self.s_attn_size, 1)  # [W, N, S]
        windows = windows.reshape(-1, self.s_attn_size)
        # The official code clusters standard-scaled windows (training mean/std); apply the
        # same global scaling so the result does not depend on the runner's input scale.
        windows = (windows - series.mean()) / series.std().clamp_min(1e-12)
        clusters = min(self.n_cluster, windows.shape[0])
        centroids = kshape_centroids(windows, clusters, self.cluster_max_iter)
        self.pattern_set.zero_()
        self.pattern_set[:clusters].copy_(centroids.to(self.pattern_set.device))

    # ------------------------------------------------------------ forward
    def _recent_windows(self, values: torch.Tensor) -> torch.Tensor:
        """``[B, T, N]`` -> ``[B, T, N, S]``: the S most recent values ending at each step (zero-padded)."""
        padded = F.pad(values, (0, 0, self.s_attn_size - 1, 0))
        return padded.unfold(1, self.s_attn_size, 1)

    def _embed(self, x_enc, x_mark_enc):
        data = to_calendar_spatiotemporal(x_enc, x_mark_enc)  # [B, T, N, 3]
        h = self.value_embedding(data[..., :1])
        h = h + self.position(x_enc).unsqueeze(2)
        if self.minute_embedding is not None:
            minute = (data[..., 1] * 1440).round().long().clamp(0, 1439)
            h = h + self.minute_embedding(minute)
        if self.weekday_embedding is not None:
            weekday = (data[..., 2] * 7).round().long().clamp(0, 6)
            h = h + self.weekday_embedding(weekday)
        laplacian = self.laplacian
        if self.training and self.random_flip:
            sign = torch.randint(0, 2, (laplacian.shape[1],), device=laplacian.device) * 2 - 1
            laplacian = laplacian * sign
        h = h + self.laplacian_embedding(laplacian)
        return self.embedding_dropout(h)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.num_nodes):
            raise ValueError(f"x_enc must have shape [batch, {self.seq_len}, {self.num_nodes}]")
        patterns = self.pattern_embedding(self._recent_windows(x_enc))
        keys = self.pattern_embedding(self.pattern_set)
        h = self._embed(x_enc, x_mark_enc)
        skip = 0
        for block, skip_conv in zip(self.blocks, self.skip_convs):
            h = block(h, patterns, keys, self.geo_blocked, self.sem_blocked)
            skip = skip + skip_conv(h)
        out = self.time_conv(F.relu(skip))  # [B, P, N, skip]
        return self.output_conv(F.relu(out)).squeeze(-1)

    def curriculum_horizon(self) -> int:
        """Official curriculum: every ``curriculum_step`` batches one more horizon step enters the loss."""
        if self.curriculum_step <= 0:
            return self.pred_len
        if int(self.batches_seen) % self.curriculum_step == 0 and int(self.task_level) < self.pred_len:
            self.task_level += 1
        self.batches_seen += 1
        return int(self.task_level)
