# gift_eval/jena_weather_10T — reference

## Provenance and license

- Original source: MPI for Biogeochemistry, Jena weather station; https://www.bgc-jena.mpg.de/wetter/.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: the MPI-BGC Data Download page (https://www.bgc-jena.mpg.de/wetter/weather_data.html) states "Terms of Use (as per Creative Commons CC-BY-4.0)"; the landing page itself states none. Redistribution is allowed with attribution to the Max Planck Institute for Biogeochemistry.
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 1 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 21 | source-reported |
| Mean length per series | not reported | source-reported |
| Total observations | not reported | source-reported |
| Frequency | 10-minute (10min) | source-reported (Hugging Face layout `jena_weather/10T`) |
| Short-term test windows | not reported | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`weather`](../../weather/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
