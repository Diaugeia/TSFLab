# gift_eval/solar_H — reference

## Provenance and license

- Original source: NREL Solar Power Data for Integrated Variable Generation via LSTNet (Lai et al.); https://github.com/laiguokun/multivariate-time-series-data.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: not stated. The NREL/NLR page (https://www.nlr.gov/grid/solar-power-data) has no license or terms, the NLR disclaimer (https://www.nlr.gov/disclaimer.html) grants none, and the LSTNet repository (https://github.com/laiguokun/multivariate-time-series-data) has no license file; `license` and `redistribution` stay `unknown`.
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 137 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 8,760 | source-reported |
| Total observations | 1,200,120 | source-reported |
| Frequency | hourly (1h) | source-reported |
| Short-term test windows | 19 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`solar`](../../solar/README.md)
- [`gift_eval/solar_10T`](../solar_10T/README.md)
- [`gift_eval/solar_D`](../solar_D/README.md)
- [`gift_eval/solar_W`](../solar_W/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
