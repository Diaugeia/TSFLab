# MoFo — reference

## Paper

- **Title**: MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling
- **Venue**: NeurIPS 2025
- **Abstract (shortened)**: Stable periodic patterns underpin long-term forecasting, but continuous, chaotic input partitioning and weak inductive biases hide them. MoFo treats periodicity as the correlation of period-aligned steps and the trend of period-offset steps. Period-structured patches (2D tensors from discrete sampling) put period-aligned steps in rows and offset steps in columns. A period-aware modulator adds an adaptive inductive bias through an end-to-end trainable regulated relaxation function. MoFo is competitive on standard benchmarks with high memory efficiency and fast training.

## Implementation mapping

- Padding (Eq. 1): for `r = seq_len mod periodic > 0`, `X[r:P]` is prepended (the first
  `P - r` steps of the first complete period), so periods are delineated backward from the
  last step; `seq_len >= periodic` is required for that slice to exist.
- Sampling (Eq. 2): the padded `[channels, P * N]` window becomes `[channels, P, N]`,
  `N = ceil(seq_len / P)`, row `i` = `x_i, x_{i+P}, ...`. Embedding (Eq. 3): `Linear(N, d)`.
- Offsets (official `_ias`, L196-204): per-channel bias `[1, C, 1, d]` (`bias`) and a phase
  table `[P, d]` read at `(phase_last - i) mod P` for patch `i` (`cias`), both Xavier-normal.
- Modulator (Eqs. 4, 9, 12): `Gamma_ij = min((i - j) mod P, (j - i) mod P)`,
  `S = sigmoid(-alpha (Gamma - beta)) + exp(-Gamma) sigmoid(-alpha beta)`,
  `alpha = sigmoid(a)`, `beta = P sigmoid(b)`, one pair per layer; `log S` is added to the
  scaled scores of every head.
- Layer (official L18-21, Appendix C Eqs. 33-34): `z + Attn(RMSNorm(z))`, then
  `z + SwiGLU(RMSNorm(z))`; RMSNorm has scale and offset (eps 1e-8); SwiGLU hidden `4d`,
  dropout 0.3.
- Head (Eq. 14): `Linear(P * d, pred_len)` on the flattened tokens; RevIN with affine
  parameters (official L141) normalizes and restores each channel.

## Citation

```bibtex
@inproceedings{ma2025mofo,
  author    = {Jiaming Ma and Binwu Wang and Qihe Huang and Guanjun Wang and Pengkun Wang and Zhengyang Zhou and Yang Wang},
  title     = {{MoFo}: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://github.com/PoorOtterBob/MoFo}
}
```
