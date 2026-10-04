# exchange — reference

## Provenance and license

- Packaging: Lai et al. 2018 (LSTNet), https://github.com/laiguokun/multivariate-time-series-data. Checked: the GitHub API reports no license, and the README only says "the collection of the daily exchange rates of eight foreign countries ... ranging from 1990 to 2016" without naming the rate provider or terms. With no original provider and no license, TSFLab never re-hosts the file; `redistribution` is `upstream`, and `tsf data download exchange` fetches it from https://huggingface.co/datasets/thuml/Time-Series-Library (pinned sha256 in `configs/hub/datasets.json`).
- Cite: Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks (Lai et al., SIGIR 2018), https://arxiv.org/abs/1703.07015.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Rows (standard file) | 7,588 daily steps | source-reported (TFB data file) |
| Channels | 8 currencies; last column `OT` is the target convention | source-reported |
| Span | LSTNet: 1990 to 2016; file dates 1990-01-01 to 2010-10-10 (7,588 consecutive days) | source-reported (conflicting; dates nominal) |
| Missing values | none in the standard file | source-reported |

The repository neither ships nor pins this file (`dataset/` is local and the Hub has no published copy), so the numbers above are those of the standard public distribution and a different copy may differ. Run `tsf data inspect --config configs/datasets/exchange.toml` on your copy before relying on them.

## Related datasets

- [`ili`](../ili/README.md): other small non-stationary LTSF set
- [`etth1`](../etth1/README.md): standard seasonal LTSF set
