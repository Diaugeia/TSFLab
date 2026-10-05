# MICN — reference

## Paper

Wang et al., ICLR 2023.

Transformer forecasters have high-complexity global attention and no targeted local-feature modelling. MICN combines
local features and global correlations with a multi-scale branch structure: each pattern is extracted with
down-sampled convolution (local) and isometric convolution (global), with linear complexity in sequence length for
suitable kernels. On six benchmarks it improves on the state of the art by 17.2% (multivariate) and 21.6% (univariate).

## Implementation mapping (reference at `370c69b8`)

- Isometric kernel: `run.py` L97-105 sets it to `(seq_len + pred_len + s) // s` for even `s` and
  `(seq_len + pred_len + s - 1) // s` for odd `s`, which equals the length after the stride-`s`,
  padding-`s // 2` downsampling convolution; `downsampled_length` computes the same value.
- Branch order (`models/local_global.py` L127-146): downsample, tanh, dropout; left-pad `S - 1`
  zeros; isometric conv, tanh, dropout; shared LayerNorm of `isometric + local`; transposed conv,
  tanh, dropout; truncate; shared LayerNorm of `upsampled + branch input`.
- Each branch first re-decomposes its input (`MIC.decomp`, L117, L153); the merge is a
  `Conv2d` with kernel `(scales, 1)` (L118, L157-161); the feed-forward is
  `Linear(d, 4d) -> ReLU -> Dropout -> Linear(4d, d)` with Xavier weights (L65-87).
- Seasonal input is the decomposed seasonal part concatenated with `pred_len` zeros
  (`models/model.py` L227-229); trend regression weights start at `1 / pred_len` (L211-212).

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/Wang0HWCX23,
  author    = {Huiqiang Wang and Jian Peng and Feihu Huang and Jince Wang and Junhui Chen and Yifei Xiao},
  title     = {{MICN:} Multi-scale Local and Global Context Modeling for Long-term Series Forecasting},
  booktitle = {The Eleventh International Conference on Learning Representations, {ICLR} 2023},
  publisher = {OpenReview.net},
  year      = {2023},
  url       = {https://openreview.net/forum?id=zt53IDUR1U}
}
```
