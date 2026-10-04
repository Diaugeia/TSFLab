# gift_eval/LOOP_SEATTLE_D — reference

## Provenance and license

- Original source: LibCity / LOOP Seattle (Cui et al.); https://github.com/LibCity/Bigscity-LibCity.
- Packaging: GIFT-Eval (https://arxiv.org/abs/2410.10393; data on Hugging Face `Salesforce/GiftEval`; code https://github.com/SalesforceAIResearch/gift-eval). The benchmark itself is `apache-2.0` (Hugging Face dataset card and repository LICENSE).
- GIFT-Eval wrapper: the Hugging Face card for `Salesforce/GiftEval` (https://huggingface.co/datasets/Salesforce/GiftEval) declares `license: apache-2.0` for the whole benchmark and lists no per-subset licenses; its Ethical Considerations say the release is "for research purposes only in support of an academic paper". The Apache-2.0 tag does not relicense the underlying data, so the terms below come from each original source.
- Underlying data terms: the Seattle Inductive Loop Detector Dataset README (https://github.com/zhiyongc/Seattle-Loop-Data) says "This dataset should only be used for research" and asks users to cite Cui, Ke and Wang (2018) or Cui, Henrickson, Ke and Wang (2019). The repository has no LICENSE file (GitHub API: none); LibCity (Apache-2.0 code) redistributes it without adding terms. Redistribution is conditional on research-only use and citation.
- Bytes are not bundled; download them with `tsf data prepare --from gift-eval`.
- Redistribution: `upstream`. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use). The license above still binds the data.

## Structure and statistics

| Item | Value | Basis |
| --- | --- | --- |
| Series | 323 | source-reported (GIFT-Eval paper, Table 13) |
| Variates per series | 1 | source-reported |
| Mean length per series | 365 | source-reported |
| Total observations | 117,895 | source-reported |
| Frequency | daily (1d) | source-reported |
| Short-term test windows | 2 | source-reported |

All values are source-reported; nothing here was measured from local files because GIFT-Eval data are not bundled. The loader reports the series count through its own windowing, not through this table.

## Related datasets

- [`gift_eval/LOOP_SEATTLE_5T`](../LOOP_SEATTLE_5T/README.md)
- [`gift_eval/LOOP_SEATTLE_H`](../LOOP_SEATTLE_H/README.md)
- [`gift_eval`](../README.md): the GIFT-Eval family card.
