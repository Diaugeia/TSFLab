# gift_eval/restaurant — reference

## Provenance and license

- Original source: Recruit Restaurant Visitor Forecasting (Kaggle, Howard et al., 2017); https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: not verified. The Kaggle competition (https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting) distributes Recruit Holdings data (Hot Pepper Gourmet, AirREGI) under its own competition rules, but those pages render only with JavaScript and could not be read here, and no separate dataset license was found; `license` and `redistribution` stay `unknown`.
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 807 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 358 | source-reported |
| Total observations | 289,303 | source-reported |
| Frequency | daily (1d) | source-reported |
| Short-term test windows | 1 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval`](../README.md): the GIFT-Eval family card.
