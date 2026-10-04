# gift_eval/bitbrains_fast_storage_5T — reference

## Provenance and license

- Original source: Bitbrains traces, Grid Workloads Archive (Shen et al., 2015); http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: the GWA-T-12 page (https://atlarge-research.com/gwa-t-12/, formerly gwa.ewi.tudelft.nl) says: "This trace was graciously provided by Bitbrains IT Services Inc. To use this traces, you must include an acknowledgement to the source of the data in any published material that refers to the data. Please refer to the CCGrid 2015 paper, and please also consider referring to the Grid Workloads Archive in the acknowledgements." No SPDX license is given and redistribution is not explicitly addressed, so it is conditional on that acknowledgement.
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 1,250 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 2 | source-reported |
| Mean length per series | 8,640 | source-reported |
| Total observations | 10,800,000 | source-reported |
| Frequency | 5-minute (5min) | source-reported |
| Short-term test windows | 18 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval/bitbrains_fast_storage_H`](../bitbrains_fast_storage_H/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
