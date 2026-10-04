# gift_eval/m4_yearly — reference

## Provenance and license

- Original source: M4 Competition (Makridakis et al., 2018), via Monash; https://forecastingdata.org/.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CC-BY-4.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 22,974 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 37 | source-reported |
| Total observations | 845,109 | source-reported |
| Frequency | yearly (1y) | source-reported |
| Short-term test windows | 1 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval/m4_daily`](../m4_daily/README.md)
- [`gift_eval/m4_hourly`](../m4_hourly/README.md)
- [`gift_eval/m4_monthly`](../m4_monthly/README.md)
- [`gift_eval/m4_quarterly`](../m4_quarterly/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
