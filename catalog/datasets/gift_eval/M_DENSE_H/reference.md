# gift_eval/M_DENSE_H — reference

## Provenance and license

- Original source: LibCity / M-DENSE Madrid; https://github.com/LibCity/Bigscity-LibCity.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: LibCity's M-DENSE comes from the Madrid City Council traffic-measurement history (de Medrano and Aznarte, 2020, https://arxiv.org/abs/2003.13977, which names the municipality's open data portal as the source). The portal record "Trafico. Historico de datos del trafico desde 2013" (https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico) lists the licence "Creative Commons Attribution 4.0 International (CC BY 4.0)". The link from LibCity's 30-sensor 2018-2019 extract to that portal record is by the paper's description, not a per-file statement; redistribute with attribution to Ayuntamiento de Madrid.
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 30 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 17,520 | source-reported |
| Total observations | 525,600 | source-reported |
| Frequency | hourly (1h) | source-reported |
| Short-term test windows | 20 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval/M_DENSE_D`](../M_DENSE_D/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
