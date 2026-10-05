# GraphLassoMTSF — reference

## Differences in detail

- Paper source: arXiv 2306.17090 v1 (Sections III-A, IV-A, IV-B; Eqs. 1, 4-8).
- Implementation: independent rewrite after reading the pinned official code (`HySonLab/GraphLASSO`, revision `84eb0fadae2fd2dfec7f16190e8a2fc130f23a8e`; no license file, recorded as `NOASSERTION`): `scripts/generate_adj_lasso.py` (Phase 1), `model/pytorch/model.py` (graph sampling and encoder-decoder), `model/pytorch/cell.py` (graph GRU cell), `model/pytorch/supervisor.py` (training loop), and `data/config/para_CA1_Food1.yaml` (defaults). Nothing was copied or imported.
- Phase 1 from the code: standardize the training split, add `1e-3` to the diagonal of the sample covariance, and solve the lasso with correlation scaling (`gglasso`, `do_scaling = True`, `lambda1 = 0.02`); the off-diagonal penalty and covariance rescaling are reproduced. The solver mirrors the official `gglasso` path (`glasso_problem.solve` -> `block_SGL` -> `ADMM_SGL`): connected-component screening of `|S_ij| > lambda`, then scaled ADMM per block from the identity with `rho = 1` and residual balancing, the Boyd et al. stopping rule with `tol = rtol = 1e-10` (as the script passes), at most 1000 iterations, and the sparse `Theta` as the estimate. Every `Omega` iterate is the `-log det` proximal map of one eigendecomposition, so the solver always returns and never fails on an ill-conditioned covariance. Cost: at most 1000 symmetric eigendecompositions of the largest block (about 24 s for traffic, 862 nodes, on a 16-core CPU).
- Sampling from the code: a hard Gumbel-softmax over each row of the raw precision matrix (temperature 0.5, which does not change the hard sample), once per forward pass and shared by the batch. The paper's Bernoulli sampling `A_ij ~ Bernoulli(Theta_ij)` needs a rescaling of `Theta` to `[0, 1]` that is not specified. The official code also samples a graph at evaluation, which makes metrics stochastic.
- GRU cell from the code: diffusion convolution over the transposed random-walk matrix of `A + I` for both gates (bias 1) and the candidate (bias 0) with Xavier-normal weights and Chebyshev-style recursion beyond one hop.
- Decoder from the code: starts from zeros and feeds back its own projection, replaced by the label with the curriculum probability during training.
- Defaults follow the shipped configuration (`rnn_units = 32`, one layer, one diffusion step, `cl_decay_steps = 2000`, curriculum on); only CA1_Food1 settings are released.
- The time-varying variant indexes interval graphs by the training-step counter, so it has no defined rule for windows at inference.
- The official dual-random-walk filter option is unused in the official forward pass.
- Teacher forcing is skipped for `MS` targets, which lack labels for every node. The paper trains with MAE (Eq. 8) and the code with Adam and multi-step decay; this entry uses the configured loss and catalog optimizer.

## Verification

Checked properties: the precision estimate against a sparse Gaussian graphical model and scikit-learn on the scaled covariance (before the solver moved to ADMM); the zero pattern and symmetry; the random-walk support; the diffusion recursion; the graph GRU update equations against a direct reference; hard one-edge-per-row sampling in training and the `softmax(Theta)` expectation in evaluation; the curriculum probability and teacher forcing; `training_setup` on a loader; the decoder shape; gradients. Reported benchmark numbers are not reproduction claims.

## Citation

```bibtex
@article{do2023sparsity,
  title   = {Sparsity exploitation via discovering graphical models in multi-variate time-series forecasting},
  author  = {Do, Ngoc-Dung and Hy, Truong Son and Nguyen, Duy Khuong},
  journal = {arXiv preprint arXiv:2306.17090},
  year    = {2023}
}
```
