# gift_eval/hierarchical_sales_D — reference

## Provenance and license

- Original source: Hierarchical Sales (Mancuso, Piccialli, Sudoso, 2020); https://data.mendeley.com/datasets/njdkntcpc9/1.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: the Mendeley Data records of Mancuso, Piccialli and Sudoso (https://data.mendeley.com/datasets/njdkntcpc9/1, DOI 10.17632/njdkntcpc9.1, also mirrored at UCI dataset 611, and the related https://data.mendeley.com/datasets/s8dgbs3rng/1) are licensed CC BY NC 3.0: "You are free to adapt, copy or redistribute the material, providing you attribute appropriately and do not use the material for commercial purposes." Redistribution is allowed only non-commercially with attribution.
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 118 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 1,825 | source-reported |
| Total observations | 215,350 | source-reported |
| Frequency | daily (1d) | source-reported |
| Short-term test windows | 7 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval/hierarchical_sales_W`](../hierarchical_sales_W/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
