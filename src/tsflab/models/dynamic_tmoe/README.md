---
name: "DynamicTMoE"
description: "Patch mixture of five heterogeneous experts routed by recurrent memory and RBF-MMD drift detection, plus cyclic channel relations. Use for non-stationary multivariate forecasting under concept drift; not for stable stationary series where a single simple model suffices."
---

# DynamicTMoE

## Idea

- Five fixed experts (identity, trend, seasonality via FFT gate, gated-conv fluctuation, drift MLP) process patch tokens; `topk_dense_mix` concentrates routing on `top_k` of them with a small floor.
- Routing logits come from a GRU over pooled patches blended with an anomaly-memory repository (`anomaly_repository`, `memory_gate`).
- `rbf_mmd` measures drift between the earlier and later patch windows and boosts the drift expert's logit via `drift_bias` and a learnable threshold.
- `channel_relation` combines the window's Pearson correlation with a learned cycle prototype (`cycle_relation`, selected by the window's phase from `window_phase`) to refine mixed tokens across channels; a flatten head forecasts and `revin` wraps.

## When to use

- Designed for non-stationary series with distribution shift and concept drift: MMD drift detection steers routing towards a drift expert.
- Heterogeneous experts (trend, seasonality, fluctuation) suit series whose dominant behaviour changes across windows.
- The channel-relation step mixes channels, so it assumes informative inter-channel correlation; the `[relation_period, C, C]` prototype grows quadratically with channels.

## Configure

- `enc_in`: number of channels.
- `relation_period`: cycle length of the channel-relation prototypes, in steps (preset 24, matching an hourly daily cycle). The prototype index uses the raw calendar marks `[batch, seq_len, 6]`.
- `patch_len` and `stride` are clamped to `seq_len`, so no divisibility constraint.

Other hyperparameters: preset defaults in `configs/models/DynamicTMoE.toml`; tune generically.

## Differences

Independent rewrite mapping paper Eqs. (1)-(10); `models/Dynamic_TMoE/model.py`, `memory_router.py` and `cyclic_relation.py` of `andone-07/Dynamic-TMoE` at `3e412353` (no license file, recorded `NOASSERTION`) were inspected as reference only, nothing copied.

- The paper's training orchestrator creates, aligns and prunes experts and mutates the anomaly gallery; here the pool is a fixed five experts, the repository is learnable, and `forward` performs no stateful action.
- A small routing floor keeps every expert trainable instead of exact sparse dispatch.
- The cycle prototype `R_cycle[t]` (Eq. 10) is indexed per sample by the phase of the first input step, from the raw marks (absolute step count modulo `relation_period`); without marks the phase is 0, as in the reference. The reference's own index is degenerate under timeF marks (see `[[issues]]`).
- One relation per window: `R_cur` is the Pearson correlation of the normalized window, and one prototype is used for all patches; the reference computes `R_cur` per patch from the cosine similarity of patch embeddings and reads prototype `t + p` for patch `p` (`cyclic_relation.py` L17-20, L57-70).
- `topk_dense_mix` is shared with DUET via `topk_expert_router`; the GRU/anomaly-memory pipeline stays local.

Citation: Zhu, J., Liu, S., Weng, D., Wu, Y. "Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting." ICML 2026. arXiv:2605.20678.
