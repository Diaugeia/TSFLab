# gift_eval/bizitobs_application — reference

## Provenance and license

- Original source: BizITObs, processed per AutoMixer (Palaskar et al., 2024); https://github.com/BizITObs/BizITObservabilityData.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CDLA-Sharing-1.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- CDLA-Sharing-1.0 asks that shared data stay under the same terms; check it before republishing derived files.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 1 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 2 | source-reported |
| Mean length per series | 8,834 | source-reported |
| Total observations | 8,834 | source-reported |
| Frequency | 10-second (10s) | source-reported |
| Short-term test windows | 15 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval`](../README.md): the GIFT-Eval family card.
