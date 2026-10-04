# gift_eval/ett1_W — reference

## Provenance and license

- Original source: ETT dataset (Zhou et al., Informer); https://github.com/zhouhaoyi/ETDataset.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CC-BY-ND-4.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- The no-derivatives clause matters for resampled or re-published copies; keep the original files unmodified when redistributing.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 1 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 7 | source-reported |
| Mean length per series | 103 | source-reported |
| Total observations | 103 | source-reported |
| Frequency | weekly (1w) | source-reported |
| Short-term test windows | 2 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`etth1`](../../etth1/README.md)
- [`ettm1`](../../ettm1/README.md)
- [`gift_eval/ett1_15T`](../ett1_15T/README.md)
- [`gift_eval/ett1_D`](../ett1_D/README.md)
- [`gift_eval/ett1_H`](../ett1_H/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
