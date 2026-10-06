"""Flat metadata catalog for reusable model components.

The catalog documents shared implementation contracts without creating a
second model hierarchy.  It is metadata only: models import the concrete
component modules directly, so catalog discovery never changes runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ComponentSpec:
    """Identity and semantic boundary of one reusable component module."""

    name: str
    module: str
    contract: str
    public_symbols: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentMatch:
    """One retrieval candidate; semantic compatibility still needs review."""

    spec: ComponentSpec
    score: int
    matched_terms: tuple[str, ...]


class ComponentCatalog:
    """Flat lookup for shared component metadata."""

    def __init__(self, specs: tuple[ComponentSpec, ...]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("component names must be unique")

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> ComponentSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown component {name!r}") from exc

    def specs(self) -> tuple[ComponentSpec, ...]:
        return tuple(self._specs[name] for name in self.names())

    def match(self, query: str, limit: int = 5) -> tuple[ComponentMatch, ...]:
        """Rank lexical candidates without claiming semantic equivalence."""
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not terms or limit < 1:
            return ()
        matches: list[ComponentMatch] = []
        for spec in self.specs():
            name_terms = set(re.findall(r"[a-z0-9]+", spec.name.lower()))
            symbol_terms = set(
                re.findall(r"[a-z0-9]+", " ".join(spec.public_symbols).lower())
            )
            keyword_terms = set(
                re.findall(r"[a-z0-9]+", " ".join(spec.keywords).lower())
            )
            contract_terms = set(re.findall(r"[a-z0-9]+", spec.contract.lower()))
            matched = terms & (
                name_terms | symbol_terms | keyword_terms | contract_terms
            )
            if not matched:
                continue
            score = sum(
                5
                if term in name_terms
                else 3
                if term in keyword_terms
                else 2
                if term in symbol_terms
                else 1
                for term in matched
            )
            matches.append(ComponentMatch(spec, score, tuple(sorted(matched))))
        matches.sort(key=lambda item: (-item.score, item.spec.name))
        return tuple(matches[:limit])


COMPONENT_CATALOG = ComponentCatalog(
    (
        ComponentSpec("adj_norm", "tsflab.models._components.adj_norm", "Dense adjacency normalization.", ("symmetric_normalized_laplacian", "lambda_rescaled_laplacian", "gcn_norm", "transition_matrix", "reverse_transition_matrix"), keywords=("adjacency", "graph", "laplacian", "normalization")),
        ComponentSpec(
            "adain_style_norm",
            "tsflab.models._components.adain_style_norm",
            "Adaptive instance normalization rescaling features to externally supplied statistics.",
            ("AdaptiveInstanceNorm1d",),
            ("adain", "adaptive", "non-stationary", "normalization", "style"),
        ),
        ComponentSpec(
            "adaptive_node_embedding_adjacency",
            "tsflab.models._components.adaptive_node_embedding_adjacency",
            "Learnable node-embedding adaptive adjacency: softmax(relu(E1 @ E2^T)).",
            ("adaptive_node_embedding_adjacency",),
            ("adaptive", "adjacency", "embedding", "graph", "node", "softmax"),
        ),
        ComponentSpec(
            "graph_conv_gru",
            "tsflab.models._components.graph_conv_gru",
            "Graph-convolutional GRU gating: r,u = sigmoid(G_g([x,h])), c = tanh(G_c([x, r*h])), h' = u*h + (1-u)*c around caller-supplied graph filters.",
            ("GraphConvGRUCell", "graph_gru_step"),
            ("gru", "graph", "graph-convolution", "recurrent", "cell", "spatiotemporal"),
        ),
        ComponentSpec(
            "node_adaptive_graph_conv",
            "tsflab.models._components.node_adaptive_graph_conv",
            "Node-adaptive Chebyshev graph convolution: per-node weights and bias generated from node embeddings through shared banks, over the softmax(relu(E E^T)) graph.",
            ("NodeAdaptiveGraphConv",),
            ("adaptive", "chebyshev", "graph", "graph-convolution", "meta-parameter", "node-adaptive", "weight-bank"),
        ),
        ComponentSpec(
            "channel_alignment",
            "tsflab.models._components.channel_alignment",
            "Slice or zero-pad the trailing feature axis to a requested width.",
            ("fit_channels",),
            ("adapter", "channel", "feature", "padding", "shape"),
        ),
        ComponentSpec(
            "channel_wise_linear",
            "tsflab.models._components.channel_wise_linear",
            "Shared or per-channel affine projection over the final sequence axis.",
            ("ChannelWiseLinear",),
            ("channel-wise", "forecast", "individual", "linear", "projection"),
        ),
        ComponentSpec(
            "deviation_memory",
            "tsflab.models._components.deviation_memory",
            "Learnable prototype memory bank with attention retrieval and a deviation score between representations.",
            ("PrototypeMemory", "PrototypeRetrieval", "deviation_score"),
            ("contrastive", "deviation", "memory", "prototype", "retrieval", "self-supervised"),
        ),
        ComponentSpec(
            "energy_frequency_pooling",
            "tsflab.models._components.energy_frequency_pooling",
            "Energy-weighted stochastic pooling of a complex spectrum across a token axis.",
            ("EnergyBasedFrequencyPooling",),
            ("energy", "frequency", "key-frequency", "pooling", "softmax", "stochastic"),
        ),
        ComponentSpec(
            "freq_band_moe",
            "tsflab.models._components.freq_band_moe",
            "Learned frequency-band decomposition with input-gated mixture-of-experts recombination.",
            ("FrequencyBandMixtureOfExperts",),
            ("band", "decomposition", "experts", "frequency", "gating", "mixture", "rfft"),
        ),
        ComponentSpec(
            "global_patch_compression_attention",
            "tsflab.models._components.global_patch_compression_attention",
            "Two-stage cross-patch attention that compresses patches into summaries before broadcasting.",
            ("GlobalPatchCompressionAttention",),
            ("attention", "compression", "cross-patch", "global", "patch", "sensor", "transformer"),
        ),
        ComponentSpec(
            "graph_masked_attention",
            "tsflab.models._components.graph_masked_attention",
            "Multi-head attention blending global dense scores with local adjacency-masked scores.",
            ("GlobalLocalGraphAttention",),
            ("adjacency", "attention", "global", "graph", "local", "mask", "spatial"),
        ),
        ComponentSpec(
            "haar_dwt1d",
            "tsflab.models._components.haar_dwt1d",
            "Lossless single-level Haar discrete wavelet transform and its inverse.",
            ("HaarDWT1D", "HaarIDWT1D"),
            ("dwt", "haar", "sub-series", "wavelet"),
        ),
        ComponentSpec(
            "harmonic_energy_gate",
            "tsflab.models._components.harmonic_energy_gate",
            "Per-channel harmonic-to-total spectral energy ratio for dual-branch gating.",
            ("HarmonicEnergyGate",),
            ("energy", "fusion", "gate", "harmonic", "periodicity", "spectral", "weighting"),
        ),
        ComponentSpec(
            "inception_block",
            "tsflab.models._components.inception_block",
            "Mean of same-padded odd square Conv2d kernels (1, 3, ..., 2k-1) with optional Kaiming fan-out initialization.",
            ("InceptionBlock2d",),
            ("conv2d", "convolution", "inception", "multi-scale", "period-2d", "timesnet"),
        ),
        ComponentSpec(
            "spectral_descriptor",
            "tsflab.models._components.spectral_descriptor",
            "Per-window spectral entropy and low/mid/high band-energy ratios of the channel-averaged power spectrum.",
            ("SpectralDescriptor",),
            ("band", "descriptor", "energy", "entropy", "fft", "power", "ratio", "spectral", "spectrum"),
        ),
        ComponentSpec(
            "mixer_block",
            "tsflab.models._components.mixer_block",
            "Pre-normalized residual time mixing then residual feature mixing (TSMixer basic block).",
            ("MixerBlock",),
            ("feature", "gelu", "layernorm", "mixer", "residual", "time"),
        ),
        ComponentSpec(
            "dlinear",
            "tsflab.models._components.dlinear",
            "Moving-average decomposition and channel-wise linear forecasting backbone.",
            ("DLinearBackbone",),
            ("decomposition", "linear", "moving-average", "seasonal", "trend"),
        ),
        ComponentSpec(
            "diffusion_conv",
            "tsflab.models._components.diffusion_conv",
            "Graph-WaveNet diffusion concatenation and projection for static supports.",
            ("DiffusionConv2d",),
            ("diffusion", "graph", "graph-wavenet", "support", "spatiotemporal"),
        ),
        ComponentSpec(
            "empirical_quantiles",
            "tsflab.models._components.empirical_quantiles",
            "Linear-interpolated empirical quantiles of forecast samples along a sample axis, appended as a trailing level axis.",
            ("empirical_quantiles",),
            ("interpolation", "probabilistic", "quantile", "samples", "sorting"),
        ),
        ComponentSpec("embed", "tsflab.models._components.embed", "Value, position, calendar, patch, and inverted embeddings.", keywords=("calendar", "embedding", "patch", "position", "token")),
        ComponentSpec(
            "gated_dilated_conv",
            "tsflab.models._components.gated_dilated_conv",
            "Causal dilated padding plus the WaveNet gated activation unit.",
            ("causal_pad", "gated_dilated_conv"),
            ("causal", "dilated", "gate", "gated-activation", "wavenet"),
        ),
        ComponentSpec(
            "decomposition_encdec",
            "tsflab.models._components.decomposition_encdec",
            "Autoformer-style progressive-decomposition encoder/decoder layers with injected self/cross mixers and bias-free trend accumulation.",
            ("DecompositionEncoderLayer", "DecompositionDecoderLayer", "decomposition_feed_forward"),
            ("autoformer", "decoder", "decomposition", "encoder", "fedformer", "progressive", "seasonal", "trend"),
        ),
        ComponentSpec(
            "dominant_periods",
            "tsflab.models._components.dominant_periods",
            "Top-k FFT period selection with per-sample amplitude weights for BLC tensors.",
            ("dominant_periods",),
            ("amplitude", "fft", "frequency", "period", "spectrum"),
        ),
        ComponentSpec(
            "fft_extrapolation_conv",
            "tsflab.models._components.fft_extrapolation_conv",
            "Zero-padded rfft convolution with per-bin complex weight and bias mapping a history to a horizon, with optional channel mixing of weight sets.",
            ("FFTExtrapolationConv",),
            ("complex", "convolution", "fft", "frequency-domain", "horizon", "zero-padding"),
        ),
        ComponentSpec(
            "flatten_forecast_head",
            "tsflab.models._components.flatten_forecast_head",
            "Shared or channel-wise linear forecast head over two flattened feature axes.",
            ("FlattenForecastHead",),
            ("channel-wise", "flatten", "forecast", "head", "linear", "patch"),
        ),
        ComponentSpec(
            "forecast_embedding",
            "tsflab.models._components.forecast_embedding",
            "Value projection plus normalized six-column raw-calendar embedding.",
            ("RawCalendarEmbedding", "ForecastEmbedding"),
            ("calendar", "covariate", "embedding", "forecast", "value"),
        ),
        ComponentSpec(
            "frequency_band_sampler",
            "tsflab.models._components.frequency_band_sampler",
            "Depth-indexed contiguous frequency-band selection over an FFT axis.",
            ("HierarchicalFrequencySampler",),
            ("band", "depth", "fft", "frequency", "hierarchical", "sampling", "spectral"),
        ),
        ComponentSpec(
            "gaussian_parameter_head",
            "tsflab.models._components.gaussian_parameter_head",
            "Independent Gaussian location/positive-scale parameter projection.",
            ("GaussianParameterHead",),
            ("distribution", "gaussian", "location", "probabilistic", "scale"),
        ),
        ComponentSpec(
            "gated_fusion",
            "tsflab.models._components.gated_fusion",
            "Learnable sigmoid gate that convexly blends two equal-shaped embeddings.",
            ("GatedFusion",),
            ("fusion", "gate", "gated", "mixture", "sigmoid"),
        ),
        ComponentSpec(
            "softmax_gate",
            "tsflab.models._components.softmax_gate",
            "Softmax feature gate: x * softmax(Linear(x)) over the last axis (PatchTSMixer gated attention).",
            ("SoftmaxGate",),
            ("gate", "gated-attention", "softmax", "feature", "mixer"),
        ),
        ComponentSpec(
            "periodic_alibi_bias",
            "tsflab.models._components.periodic_alibi_bias",
            "ALiBi attention bias with optional per-head-group periodic (triangle-wave) distance.",
            ("periodic_alibi_bias",),
            ("alibi", "bias", "periodic", "relative-position", "attention"),
        ),
        ComponentSpec(
            "frwkv_linear_attention",
            "tsflab.models._components.frwkv_linear_attention",
            "FRWKV RWKV-7-style delta-rule linear attention over tokens and the residual Linear-encoder-Linear frequency branch built on it.",
            ("FRWKVLinearAttention", "FRWKVSpectralBranch", "frwkv_state_scan"),
            ("rwkv", "linear-attention", "delta-rule", "state-recursion", "frequency-branch", "attention"),
        ),
        ComponentSpec("graph_utils", "tsflab.models._components.graph_utils", "Graph supports, Laplacians, and Chebyshev bases.", ("normalize_adj_mx", "adj_to_supports", "cheb_poly"), keywords=("adjacency", "chebyshev", "graph", "laplacian", "support")),
        ComponentSpec(
            "graph_spectral",
            "tsflab.models._components.graph_spectral",
            "Robust scaled-Laplacian and exact-order Chebyshev support construction.",
            ("scaled_laplacian", "chebyshev_polynomials", "chebyshev_supports"),
            ("adjacency", "chebyshev", "degenerate", "graph", "laplacian", "spectral"),
        ),
        ComponentSpec(
            "node2vec_embedding",
            "tsflab.models._components.node2vec_embedding",
            "Seeded node2vec (biased second-order walks + skip-gram negative sampling) vertex embedding of a fixed weighted graph.",
            ("node2vec_embedding",),
            ("node2vec", "graph", "spatial-embedding", "random-walk", "skip-gram", "positional-encoding"),
        ),
        ComponentSpec(
            "synchronous_graph_conv",
            "tsflab.models._components.synchronous_graph_conv",
            "STSGCN localized window graph and synchronous graph-convolution module (stacked GLU/ReLU GCN, max aggregation, block cropping).",
            ("SynchronousGraphModule", "localized_adjacency", "mxnet_xavier_uniform_"),
            ("graph", "spatiotemporal", "localized-graph", "synchronous", "gcn", "glu", "window"),
        ),
        ComponentSpec(
            "last_value_center",
            "tsflab.models._components.last_value_center",
            "Detached last-observed-timestep centering and restoration for BLC histories.",
            ("center_on_last_value", "restore_last_value"),
            ("centering", "detach", "last-value", "level", "residual"),
        ),
        ComponentSpec("marks", "tsflab.models._components.marks", "Canonical temporal-mark and spatiotemporal input adapters.", ("TIME_FEATURES", "TSLIB_TIME_FEATURE_DIMS", "tslib_time_feature_dimension", "adapt_tslib_marks", "encoder_timef_marks", "days_from_civil", "elapsed_minutes", "normalized_time_features", "to_spatiotemporal", "to_calendar_spatiotemporal", "future_time_features", "coerce_time_length"), keywords=("calendar", "covariate", "spatiotemporal", "timestamp", "civil-date", "epoch-minutes")),
        ComponentSpec(
            "mamba",
            "tsflab.models._components.mamba",
            "Kernel-free selective state-space mixer, normalization, and residual block.",
            ("RMSNorm", "MambaBlock", "MambaResidualBlock"),
            ("mamba", "mixer", "rmsnorm", "ssm", "state-space"),
        ),
        ComponentSpec(
            "sharpness_aware",
            "tsflab.models._components.sharpness_aware",
            "First-order sharpness-aware minimization expressed as a loss evaluated at adversarially perturbed weights.",
            ("sharpness_aware_loss",),
            ("adversarial-weights", "optimizer-agnostic", "sam", "sharpness", "training-objective", "functional-call"),
        ),
        ComponentSpec("masking", "tsflab.models._components.masking", "Attention mask construction.", ("TriangularCausalMask", "ProbMask", "LocalMask"), keywords=("attention", "causal", "mask")),
        ComponentSpec(
            "hyper_state_scan",
            "tsflab.models._components.hyper_state_scan",
            "Kernel-free scalar-state selective scan and a 2-D grid state mixer.",
            ("diagonal_selective_scan", "GridStateMixer"),
            ("grid", "hyper-state", "mamba", "scan", "ssm", "state-space"),
        ),
        ComponentSpec(
            "node_visibility",
            "tsflab.models._components.node_visibility",
            "Node-level masking and subgraph grouping for scalable node-set attention.",
            ("random_mask_tokens", "shuffle_tokens", "unshuffle_tokens", "group_into_subgraphs", "ungroup_subgraphs"),
            ("graph", "grouping", "masking", "node", "sampling", "subgraph", "visibility"),
        ),
        ComponentSpec(
            "patchtst",
            "tsflab.models._components.patchtst",
            "Patch extraction, time-series Transformer encoding, and PatchTST backbone.",
            ("PatchTSTBackbone",),
            ("backbone", "channel-independent", "patch", "transformer"),
        ),
        ComponentSpec("positional_encoding", "tsflab.models._components.positional_encoding", "Patch-transformer positional encodings.", ("positional_encoding",), keywords=("encoding", "patch", "position", "transformer")),
        ComponentSpec(
            "periodic_query_bank",
            "tsflab.models._components.periodic_query_bank",
            "Learnable per-phase vector table gathered into phase-aligned windows.",
            ("PeriodicQueryBank",),
            ("cycle", "gather", "period", "phase", "query"),
        ),
        ComponentSpec(
            "quantile_head",
            "tsflab.models._components.quantile_head",
            "Input-conditioned monotone quantile head with non-crossing outputs.",
            ("QuantileHead", "validate_quantile_levels", "DEFAULT_QUANTILE_LEVELS"),
            ("monotone", "non-crossing", "probabilistic", "quantile"),
        ),
        ComponentSpec(
            "regularized_adaptive_graph_conv",
            "tsflab.models._components.regularized_adaptive_graph_conv",
            "Stochastic embedding row-swap regularization plus a linear-complexity, node-embedding adaptive graph convolution (efficient cosine operator).",
            ("StochasticSharedEmbedding", "EfficientCosineGraphConv"),
            ("adaptive", "adjacency", "cosine", "embedding", "graph", "linear-complexity", "node", "regularization", "stochastic"),
        ),
        ComponentSpec("revin", "tsflab.models._components.revin", "Reversible instance normalization.", ("RevIN",), ("denormalization", "instance", "normalization", "reversible")),
        ComponentSpec(
            "sparse_connection_router",
            "tsflab.models._components.sparse_connection_router",
            "Shared, input-independent sparse connection routing over discrete positions.",
            ("SharedSparseConnectionRouter",),
            ("adjacency", "bernoulli", "gumbel-softmax", "interaction", "shared", "sparse", "top-k"),
        ),
        ComponentSpec("self_attention_family", "tsflab.models._components.self_attention_family", "Shared full and probabilistic attention layers.", keywords=("attention", "full", "probabilistic")),
        ComponentSpec(
            "soft_tree",
            "tsflab.models._components.soft_tree",
            "Differentiable binary and level-wise-shared tree routing with leaf interpolation.",
            ("SoftDecisionTree", "SoftObliviousTree", "binary_routes"),
            ("decision", "ensemble", "leaf", "oblivious", "routing", "soft", "tree"),
        ),
        ComponentSpec(
            "series_decomposition",
            "tsflab.models._components.series_decomposition",
            "Edge-padded moving average and residual/trend decomposition for BLC data.",
            ("EdgePaddedMovingAverage", "SeriesDecomposition"),
            ("decomposition", "moving-average", "residual", "smoothing", "trend"),
        ),
        ComponentSpec(
            "topk_expert_router",
            "tsflab.models._components.topk_expert_router",
            "Two-layer gating MLP with optional trainable noise and floor-blended top-k expert sparsification.",
            ("GatingMLP", "topk_dense_mix"),
            ("expert", "gate", "gating", "mixture", "moe", "routing", "sparse", "top-k"),
        ),
        ComponentSpec("transformer_encdec", "tsflab.models._components.transformer_encdec", "Shared Transformer encoder and decoder blocks.", ("ConvLayer", "EncoderLayer", "Encoder", "DecoderLayer", "Decoder"), keywords=("attention", "decoder", "encoder", "transformer")),
        ComponentSpec("tst_transformer", "tsflab.models._components.tst_transformer", "Time-series Transformer encoder blocks.", ("TSTEncoder",), keywords=("attention", "encoder", "time-series", "transformer")),
        ComponentSpec(
            "weight_set_router",
            "tsflab.models._components.weight_set_router",
            "Low-rank weight sharing: softmax-with-temperature routing matrix over a few weight sets and the per-channel linear mix of those sets.",
            ("WeightSetRouter", "mix_weight_sets"),
            ("low-rank", "routing", "softmax", "temperature", "weight-sharing"),
        ),
        ComponentSpec(
            "wavelet",
            "tsflab.models._components.wavelet",
            "Fixed-filter decimated and a-trous (undecimated) discrete wavelet transforms for BCL tensors.",
            ("DecimatedWaveletTransform", "UndecimatedWaveletTransform", "available_wavelets"),
            ("dwt", "haar", "multi-resolution", "subband", "undecimated", "wavelet"),
        ),
        ComponentSpec(
            "orthogonal_dwt",
            "tsflab.models._components.orthogonal_dwt",
            "Multi-level orthogonal DWT and inverse in the PyWavelets convention with zero or half-sample symmetric boundary extension.",
            ("OrthogonalDWT", "filter_bank", "coefficient_length", "coefficient_lengths", "DEC_LO", "BOUNDARY_MODES"),
            ("boundary", "coiflet", "daubechies", "dwt", "symlet", "symmetric", "wavelet", "zero-padding"),
        ),
        ComponentSpec(
            "bspline_basis",
            "tsflab.models._components.bspline_basis",
            "B-spline basis values by the Cox-de Boor recursion on shared or per-feature knots, the basis of B-spline KAN layers.",
            ("bspline_basis",),
            ("b-spline", "basis", "cox-de-boor", "kan", "kolmogorov-arnold", "knots", "spline"),
        ),
        ComponentSpec(
            "natural_cubic_spline",
            "tsflab.models._components.natural_cubic_spline",
            "Natural cubic spline control path of a sampled [..., L, C] window: coefficients, X(t), and dX/dt for neural controlled differential equations.",
            ("natural_cubic_spline_coeffs", "NaturalCubicSpline", "SplineCoeffs"),
            ("control-path", "cubic-spline", "interpolation", "natural-spline", "ncde", "neural-cde", "spline"),
        ),
        ComponentSpec(
            "topk_expert_attention",
            "tsflab.models._components.topk_expert_attention",
            "Differentiable top-k local expert self-attention with an optional shared global expert.",
            ("LocalExpertRouter", "gather_experts", "TopKExpertAttention"),
            ("attention", "expert", "mixture-of-experts", "routing", "top-k"),
        ),
        ComponentSpec(
            "differential_attention",
            "tsflab.models._components.differential_attention",
            "Differential self-attention: the RMS-renormalized difference of two softmax attention maps.",
            ("DifferentialAttention",),
            ("attention", "differential", "noise-cancelling", "rmsnorm"),
        ),
        ComponentSpec(
            "logsparse_conv_attention",
            "tsflab.models._components.logsparse_conv_attention",
            "Causal multi-head self-attention with causal-convolution queries/keys and a LogSparse (exponential-distance, optional local and restart) mask, plus an exact cached one-step path.",
            ("logsparse_mask", "ConvSelfAttention"),
            ("attention", "causal", "convolution", "logsparse", "sparse", "local", "restart", "decoder-only", "mask"),
        ),
        ComponentSpec(
            "ddpm_epsilon",
            "tsflab.models._components.ddpm_epsilon",
            "Fixed-schedule Gaussian DDPM: closed-form forward noising, epsilon-prediction MSE loss with uniform steps, and ancestral sampling with the posterior variance beta_tilde.",
            ("GaussianDDPM", "beta_schedule"),
            ("diffusion", "ddpm", "denoising", "epsilon", "noise", "schedule", "sampling", "generative", "probabilistic"),
        ),
        ComponentSpec(
            "dilated_conv_encoder",
            "tsflab.models._components.dilated_conv_encoder",
            "TS2Vec-style length-preserving dilated convolution encoder: pre-GELU residual blocks of two same-padded dilated Conv1d layers with dilation 2^i.",
            ("SamePadConv", "DilatedConvBlock", "DilatedConvEncoder"),
            ("convolution", "dilated", "encoder", "residual", "gelu", "ts2vec", "representation", "same-padding", "backbone"),
        ),
        ComponentSpec(
            "gpt2_backbone",
            "tsflab.models._components.gpt2_backbone",
            "Decoder-only GPT-2 trunk over input embeddings (learned positions, causal pre-norm blocks, final LayerNorm) with an offline loader for released safetensors weights.",
            ("GPT2Config", "GPT2Backbone", "GPT2Block", "GPT2Attention", "GPT2MLP", "LoRAAdapter", "load_gpt2_weights", "read_safetensors"),
            ("gpt2", "llm", "pretrained", "language-model", "causal", "decoder-only", "backbone", "transformer", "lora"),
        ),
        ComponentSpec(
            "segment_mlp",
            "tsflab.models._components.segment_mlp",
            "Segment-to-token or token-to-segment map: one Linear, or n >= 2 Linear layers with activation and dropout between them.",
            ("SegmentMLP", "ACTIVATIONS"),
            ("segment", "token", "mlp", "projection", "embedding", "llm", "linear"),
        ),
        ComponentSpec(
            "gated_residual_network",
            "tsflab.models._components.gated_residual_network",
            "Gated residual network LayerNorm(skip(a) + GLU(W1 ELU(W2 a + W3 c))), its GLU gate and gated add-norm, and a softmax variable-selection network over per-variable GRNs (TFT).",
            ("GatedLinearUnit", "GateAddNorm", "GatedResidualNetwork", "VariableSelectionNetwork"),
            ("grn", "glu", "gated", "residual", "variable-selection", "context", "static-covariates", "tft", "gate"),
        ),
        ComponentSpec(
            "interpretable_attention",
            "tsflab.models._components.interpretable_attention",
            "Multi-head attention whose heads share one value projection and are averaged before the output map (TFT interpretable multi-head attention), plus a causal mask helper.",
            ("InterpretableMultiHeadAttention", "causal_mask"),
            ("attention", "interpretable", "shared-value", "multi-head", "causal", "tft", "head-average"),
        ),
    )
)
