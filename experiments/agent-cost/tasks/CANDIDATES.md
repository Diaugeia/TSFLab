# Candidate papers for the from-scratch reproduction experiment

Date: 2026-10-05. Scope: arXiv forecasting papers first submitted 2026-06-01 or later.

## Search method

1. arXiv API (`export.arxiv.org/api/query`, `sortBy=submittedDate`, 300 results per query) with five queries:
   `ti:forecasting AND cat:cs.LG`, `abs:"long-term forecasting"`, `abs:ETTh1`,
   `ti:"time series" AND ti:forecasting`, `abs:"time series forecasting" AND cat:cs.LG`.
   Result: 923 unique papers. 170 of them are dated 2026-06-01 or later and match long-term/multivariate keywords.
2. Manual triage of titles and abstracts to remove surveys, benchmarks, LLM, foundation-model, traffic/weather-only, and probabilistic-only papers. 26 papers remained.
3. For each of the 26: download `arxiv.org/html/<id>`, extract every link (github.com, anonymous.4open.science, gitlab, huggingface, zenodo) and every "code ... available / released" sentence, and extract the tables that contain ETTh1.
4. For the remaining no-code papers: `gh search repos` on the method name and on a descriptive phrase; arXiv metadata `comment` field; arXiv abstract page; `huggingface.co/papers/<id>` (all returned 404, so no HF paper page and no linked repo).
5. TSFLab check: grep of all `src/tsflab/models/*/card.toml` titles/URLs and `catalog/declined.toml` for the arXiv id and the method name. TSFLab HEAD at check time: `2bfd2d54`. No candidate below appears.

Papers removed in step 3 because code is linked in the paper: MARO 2610.03494 (anonymous.4open.science), MWMixer 2609.24229, Chameleon 2609.36453 (placeholder `github.com/anonymous/Chameleon-TSF`), CARNet 2607.21681, M2Patch 2607.19404, NBDM 2608.04471, CvLoss 2608.05742, Self-Gating Attention 2607.02344. SCPaT 2608.19966 was removed because `gh search repos "semantic structured partitioning"` returns `ASEpochs/SCPaT` (created 2026-09-28).

---

## Ranked candidates

### 1. RhyMix (model) — recommended model paper

- **Title:** RhyMix: A Lightweight Adaptive Multi-Rhythm Network for Long-Term Time Series Forecasting
- **arXiv:** 2607.08234 (v1 only). **Submitted:** 2026-07-09. Comment: "38 Pages".
- **Method:** Two parallel paths with adaptive gating: a channel-independent Cyclic Path with learnable cyclic tables for periods {12, 24, 48, 168} (sigmoid-gated per period), and an MSTCN-CA path (two multi-scale blocks of depthwise dilated convolutions d = {1, 2, 4, 8} with channel attention). Each path feeds four forecasting heads (direct linear, trend-seasonal with MA kernel 25, local conv, periodic fusion); a softmax gate mixes the heads, a second softmax gate mixes the two paths; RevIN wraps the model.
- **Main table (Table 4):** L = 96. ETTh1, ETTh2, ETTm1, ETTm2, ECL, Traffic, Weather, Exchange at H = {96, 192, 336, 720}; PEMS03/04/07/08 at {12, 24, 48, 96}. Mean (std) over 5 runs.
- **ETTh1 (MSE / MAE):**

  | H | MSE | MAE |
  |---|---|---|
  | 96 | 0.360 (0.01) | 0.382 (0.01) |
  | 192 | 0.415 (0.01) | 0.411 (0.01) |
  | 336 | 0.449 (0.02) | 0.429 (0.01) |
  | 720 | 0.459 (0.02) | 0.445 (0.02) |
  | Avg | 0.421 | 0.417 |

- **No-code check:** paper text says "The source code for RhyMix will be released publicly upon acceptance of this manuscript." No link in HTML. arXiv comment field has no link. `gh search repos "RhyMix"` and `"RhyMix forecasting"` return only the unrelated PHP CMS `rhymix/rhymix`. HF paper page: 404.
- **Implementability:** Good. Given: L = 96, periods, dilations, MA kernel 25, channel-mixer bottleneck r = min(64, max(8, floor(C/8))), dropout 0.1, AdamW lr 7e-4 with cosine to 1e-6, weight decay 1e-4, grad clip 1.0, max 50 epochs, patience 6, Huber loss (delta = 1.0), init N(0, 0.02) for cyclic tables, gate hidden 16. Underspecified: batch size; hidden sizes of the per-period and head gating MLPs; conv padding/bias; the exact 5-dim "gate feature" statistics; phase offset phi "initialized to zero and fixed".
- **Compute:** about 40K parameters, 17 MB training memory. ETTh1 with 4 horizons, 50 epochs max: estimate under 15 min total on one 16 GB GPU.
- **Why good:** post-cutoff, small, full per-horizon table with std, most training settings given, many datasets for a secondary check.
- **Risk:** code may appear on GitHub if the paper is accepted during the experiment (re-check before each run). The Huber loss gives low MAE relative to MSE; an agent that uses MSE loss will miss the MAE numbers. Batch size must be guessed (TSLib default 32).

### 2. AdaRDiff (training/inference technique) — recommended technique paper

- **Title:** Learning to Difference: Adaptive Reversible Differencing (AdaRDiff) for Time Series Forecasting
- **arXiv:** 2608.28134 (v1). **Submitted:** 2026-08-28.
- **Method:** A plug-in reversible transform: per-channel learnable weights delta_1..delta_P compute residuals z_t = x_t − Σ delta_j x_{t−j}; a backbone forecasts the residuals; the forecast is restored by the inverse linear recurrence, computed in parallel as a Toeplitz convolution with the impulse response h. Training is two-phase: 30 epochs on the residual loss (no reconstruction in the gradient), then 10 epochs on the full forecast loss.
- **Main table (Table 1):** L = 720. ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, Solar at H = {96, 192, 336, 720}. Two variants: AdaRDiff-Linear and AdaRDiff-MLP. Table 3 also gives "+AdaRDiff" gains on 8 backbones (Linear, DLinear, MLP, MixLinear, SparseTSF, CycleNet, PatchTST, iTransformer) at L = 336 (averaged MSE only).
- **ETTh1 (AdaRDiff-Linear / AdaRDiff-MLP, MSE / MAE):**

  | H | Linear MSE | Linear MAE | MLP MSE | MLP MAE |
  |---|---|---|---|---|
  | 96 | 0.360 | 0.390 | 0.389 | 0.412 |
  | 192 | 0.402 | 0.419 | 0.409 | 0.434 |
  | 336 | 0.426 | 0.436 | 0.449 | 0.446 |
  | 720 | 0.423 | 0.450 | 0.487 | 0.485 |

- **No-code check:** no repository link and no code statement in the HTML (the only "official code" phrase refers to baselines). arXiv comment: none. `gh search repos "AdaRDiff"` and `"adaptive reversible differencing"`: 0 results. `gh search code "AdaRDiff"`: only paper-digest repos. HF paper page: 404.
- **Implementability:** Good. Equations for the transform, padding (repeat x_1), optional l1 reparameterization delta_eff = alpha * delta / ||delta||_1, the convolution form of the inverse, and both training phases are given. Appendix C gives the grid: P in {4, 24, 96} on ETT, init in {zero, uniform 1/P, first-order}, RevIN on/off, l1-reparam on/off, lr in {1e-3, 2e-3, 5e-3, 1e-2}, batch size in {32, 64, 256}, MLP hidden in {128, 512, 1024}; Adam, MSE. Underspecified: the selected value per dataset/horizon (only the grid is given), MLP activation, early-stopping patience, initialization of alpha, whether delta is per channel only or also shared.
- **Compute:** Linear backbone on ETTh1 is about 0.5M parameters (H × 719) and trains in seconds per epoch; 40 epochs × 4 horizons is under 10 min for one config. The full grid (about 430 configs per horizon) is not feasible; a reproduction must pick one config or a small subset.
- **Why good:** clean, self-contained, testable as a technique (plug into Linear and into an existing backbone). Strong, specific numbers.
- **Risk:** results depend on per-dataset grid choices that are not reported. The two-phase schedule is easy to get wrong. L = 720 differs from the L = 96 default in most environments.

### 3. AOSNet (model)

- **Title:** Adaptive Oscillatory-State Alignment for Time Series Forecasting
- **arXiv:** 2606.06010 (v2). **Submitted:** v1 2026-06-04, v2 2026-06-11.
- **Method:** Hilbert analytic-signal descriptors (log-amplitude, phase cos/sin, instantaneous frequency) are computed for the input and for a learnable channel-wise global oscillatory prior P (C × L); a small conv gate produces G, and the input is aligned as X̃ = (1−G)⊙X + G⊙P. A per-channel linear path (H × L) and a cross-variate attention path (embedding L→d, MHSA over channel tokens) are fused with a learnable scalar gate lambda = sigmoid(eta).
- **Main table (Table II):** L = 96. ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Solar, Traffic, Weather at H = {96, 192, 336, 720}; mean ± std over 5 runs. Also two cloud workload traces.
- **ETTh1 (MSE / MAE):**

  | H | MSE | MAE |
  |---|---|---|
  | 96 | 0.368 ± 0.002 | 0.392 ± 0.001 |
  | 192 | 0.421 ± 0.003 | 0.421 ± 0.002 |
  | 336 | 0.459 ± 0.002 | 0.440 ± 0.003 |
  | 720 | 0.462 ± 0.003 | 0.461 ± 0.002 |
  | Avg | 0.427 | 0.429 |

- **No-code check:** no link and no code statement in the HTML ("implemented in PyTorch" only). arXiv comment: none. `gh search repos "AOSNet"` returns one unrelated scheduling repo; `"oscillatory state alignment"` and `"AOSNet forecasting"`: 0 results. HF paper page: 404.
- **Implementability:** Medium. Given: descriptor equations, alignment equation, fusion, L = 96, d_model = 512, gate kernel 5. Underspecified: optimizer, lr, epochs, batch size, dropout value, number of heads, conv gate widths, epsilon in log-amplitude, Hilbert transform implementation (FFT-based is the obvious choice). Agents must fill these with TSLib defaults.
- **Compute:** about 1.45M parameters on Electricity, fewer on ETTh1 (7 channels). Estimate under 15 min for 4 horizons.
- **Why good:** earliest post-cutoff date in the set, tight std, standard L = 96 protocol, 8 standard datasets.
- **Risk:** training hyperparameters are missing, so the gap to the paper may come from defaults, not from the method. Uses Hilbert transform on a short window, where edge effects are sensitive to implementation details.

### 4. StateFlow (model)

- **Title:** StateFlow: Dual-State Recurrent Modeling for Long-Horizon Time Series Forecasting
- **arXiv:** 2607.00197 (v1). **Submitted:** 2026-06-30.
- **Method:** A recurrent encoder (VARNN) keeps a hidden state h_t and a residual-memory state e_t driven by its own one-step prediction errors: h_t = ReLU(W_h[x_t; h_{t−1}; e_{t−1}] + b_h), x̂_{t+1} = w_o^T h_t + b_o, r_t = x_{t+1} − x̂_{t+1}, e_t = tanh(W_e[r_t; e_{t−1}] + b_e). A chunk decoder summarizes the h and e trajectories with sliding windows and a linear layer maps them to the horizon; instance normalization wraps the model.
- **Main table (Table 1):** ECL, ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic at H = {96, 192, 336, 720}; baselines from the iTransformer table (L = 96).
- **ETTh1 (MSE / MAE):**

  | H | MSE | MAE |
  |---|---|---|
  | 96 | 0.364 | 0.394 |
  | 192 | 0.415 | 0.421 |
  | 336 | 0.452 | 0.440 |
  | 720 | 0.474 | 0.463 |
  | Avg | 0.426 | 0.429 |

- **No-code check:** no link and no code statement in the HTML. arXiv comment: none. `gh search repos "StateFlow forecasting"`, `"StateFlow time series"`: 0 results; `"StateFlow"` returns only Kotlin/Android repos; `"VARNN"` returns student repos unrelated to this paper (2024–2026, no StateFlow). HF paper page: 404.
- **Implementability:** Medium. Given: cell equations, L = 96, K = 32, J = 16, chunk summarizer dims T = 32 and I = 16, window w = 5, stride s = 2, MSE loss, seed 2026, two-stage training. Underspecified: optimizer, lr, epochs, batch size, stopping rule for each stage, exact chunk-decoder layout; part of the cell is defined in the earlier VARNN paper (Gharwi & Shu, 2025).
- **Compute:** about 220K (H = 96) to 1.6M (H = 720) parameters on ETTh1 (Table 6). Sequential RNN over 96 steps, channel-independent: estimate 20–40 min total for 4 horizons.
- **Why good:** a simple, unusual architecture (RNN with error feedback) that is not a recombination of common blocks, so prior knowledge helps less.
- **Risk:** depends on a second paper for the base cell; training stages and optimizer are not specified.

### 5. TA-SparseMG (model)

- **Title:** TA-SparseMG: Trend-Aware Sparse Forecasting via Multi-Scale Gating for Long-Term Time Series
- **arXiv:** 2606.27908 (v1). **Submitted:** 2026-06-26.
- **Method:** Extends SparseTSF (cross-period sparse forecasting) with a trend-aware RevIN (input statistics plus a thresholded mean/std drift correction at denormalization), a scale-adaptive gated denoiser (multi-kernel smoothing, residual gated by a depthwise conv), and a multiscale gated-attention MLP as the period-wise predictor.
- **Main table (Table 1):** L = 720. ETTh1, ETTh2, Electricity, Solar, Traffic, Weather at H = {96, 192, 336, 720}. No ETTm1/ETTm2.
- **ETTh1 (MSE / MAE):**

  | H | MSE | MAE |
  |---|---|---|
  | 96 | 0.354 | 0.384 |
  | 192 | 0.398 | 0.411 |
  | 336 | 0.433 | 0.428 |
  | 720 | 0.453 | 0.468 |

- **No-code check:** no link and no code statement in the HTML. arXiv comment: none. `gh search repos "TA-SparseMG"`, `"SparseMG"`, `"trend-aware sparse"`: 0 results. HF paper page: 404.
- **Implementability:** Weak to medium. Given: module equations, L = 720, hidden 128, batch size 4, MSE loss, 18.95K parameters. Underspecified: optimizer, lr, epochs, patience, seeds, smoothing kernel sizes, drift threshold tau_u, initial values of the learnable scalars, period selection.
- **Compute:** 18.95K parameters; batch size 4 at L = 720 gives about 2,000 steps per epoch on ETTh1. Estimate 30–60 min total for 4 horizons.
- **Why good:** builds on SparseTSF, which already exists in TSFLab (`sparsetsf`) and in other libraries, so it tests whether an environment's existing code helps.
- **Risk:** many missing settings; batch size 4 is slow; drift-threshold indicator is non-differentiable and its threshold is not given.

---

## Other no-code papers checked (not ranked)

| arXiv | Name | Date | Reason not ranked |
|---|---|---|---|
| 2609.08286 | HypLTSF (hyperbolic multi-scale) | 2026-09-08 | Reports best over L in {96, 336, 512}; lr, d_model, curvature, loss weights missing. ETTh1: 0.361/0.394, 0.401/0.418, 0.423/0.436, 0.433/0.453. |
| 2608.16098 | AsyTO (asymmetric temporal operator) | 2026-08-17 | Tiny (about 10K params), L = 720, 3 seeds; ranks, M, gamma, lr, epochs, batch size, RevIN all missing. ETTh1: 0.353/0.393, 0.389/0.415, 0.418/0.434, 0.449/0.469. |
| 2609.19670 | CoRe (loss: frequency + low-rank relational) | 2026-09-17 | Technique alternative; close to FreDF (already in TSFLab); alpha, PCA rank, number of pairs, lr not given. ETTh1: 0.376/0.391, 0.424/0.425, 0.463/0.446, 0.473/0.471. |
| 2609.33984 | HTF (Hankel–Toeplitz linear forecaster) | 2026-09-27 | Very clean and tiny, but the paper reports only horizon-averaged MSE (ETTh1 avg 0.4075), no MAE and no per-horizon values. |
| 2608.20761 | Fuzzy-MoE | 2026-08-21 | SGD lr 2e-5 given, other details thin; ETTh2 336 < 192 MSE suggests a table problem. |
| 2607.16882 | HyBDM | 2026-07-18 | Mamba-based (needs `mamba_ssm`), ETTh1 336 MSE < 192 MSE; batch size "per dataset". |
| 2609.08554 | PV-Surgery | 2026-09-08 | Technique; ETTh1 baseline numbers (iTransformer 0.454 at H = 96) do not match the standard protocol, so not comparable. |

---

## Recommended pair

- **Model paper:** RhyMix (2607.08234). Most complete hyperparameters of the no-code set, 40K parameters, L = 96, ETTh1 avg 0.421 / 0.417 with std over 5 runs.
- **Technique paper:** AdaRDiff (2608.28134). A plug-in reversible differencing transform with a two-phase training schedule; testable on a Linear backbone in minutes; ETTh1 (Linear) avg 0.403 / 0.424 at L = 720.

**NOTE:** Before each experiment run, repeat the GitHub search for both names. RhyMix says its code will be released on acceptance.
