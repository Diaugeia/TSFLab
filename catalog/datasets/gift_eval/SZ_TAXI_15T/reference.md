# gift_eval/SZ_TAXI_15T — reference

## Provenance and license

- Original source: LibCity / SZ-Taxi (Zhao et al., T-GCN); https://github.com/LibCity/Bigscity-LibCity.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: not stated. The SZ-Taxi data (Zhao et al., T-GCN) live in https://github.com/lehaifeng/T-GCN, which has no LICENSE file (GitHub API: none) and no data terms in its README; LibCity (Apache-2.0 code) redistributes it without adding terms. `license` and `redistribution` stay `unknown`.
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 156 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 2,976 | source-reported |
| Total observations | 464,256 | source-reported |
| Frequency | 15-minute (15min) | source-reported |
| Short-term test windows | 7 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval/SZ_TAXI_H`](../SZ_TAXI_H/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
