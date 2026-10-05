# AirPhyNet — reference

## Paper

- **Title**: AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction
- **Venue**: ICLR 2024 (arXiv 2402.03784, 2024-02)
- **arXiv**: https://arxiv.org/abs/2402.03784

## Abstract

Data-driven air-quality models lose long-term accuracy with sparse or incomplete data and lack physical grounding.
AirPhyNet represents two physics principles of particle movement, diffusion and advection, as differential-equation networks, integrates them through a graph, and uses latent representations for spatio-temporal relations.
On two benchmarks it reduces errors by up to 10% across 24/48/72 h lead times, sparse data, and sudden changes, and a case study shows it captures the underlying physical processes.

## Runtime contract

Inputs are `x_enc [B, seq_len, N]` and historical raw or node meteorology; the distance graph is a required construction input; a missing directed flow graph is replaced by a ring placeholder with a warning. Output is `[B, pred_len, N]`.

## Citation

```bibtex
@inproceedings{hettige2024airphynet,
  author    = {Kethmi Hirushini Hettige and Jiahao Ji and Shili Xiang and Cheng Long and Gao Cong and Jingyuan Wang},
  title     = {AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction},
  booktitle = {The Twelfth International Conference on Learning Representations (ICLR 2024)},
  year      = {2024},
  url       = {https://openreview.net/forum?id=JW3jTjaaAB}
}
```
