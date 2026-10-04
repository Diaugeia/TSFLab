# gift_eval/electricity_D — reference

## Provenance and license

- Original source: UCI ElectricityLoadDiagrams20112014 (Trindade, 2015); https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- License of the underlying data: `CC-BY-4.0` (explicit license found for the source).
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 370 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 1,461 | source-reported |
| Total observations | 540,570 | source-reported |
| Frequency | daily (1d) | source-reported |
| Short-term test windows | 5 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`electricity`](../../electricity/README.md)
- [`gift_eval/electricity_15T`](../electricity_15T/README.md)
- [`gift_eval/electricity_H`](../electricity_H/README.md)
- [`gift_eval/electricity_W`](../electricity_W/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
