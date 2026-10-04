# gift_eval/kdd_cup_2018_with_missing_H — reference

## Provenance and license

- Original source: KDD Cup 2018 air quality, via the Monash repository; https://zenodo.org/records/4656719.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CC-BY-4.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 270 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 10,898 | source-reported |
| Total observations | 2,942,364 | source-reported |
| Frequency | hourly (1h) | source-reported |
| Short-term test windows | 20 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval/kdd_cup_2018_with_missing_D`](../kdd_cup_2018_with_missing_D/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
