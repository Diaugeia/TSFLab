# gift_eval — reference

## Provenance and license

- Authors: Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, Doyen Sahoo (Salesforce AI Research).
- Paper: https://arxiv.org/abs/2410.10393. Data: https://huggingface.co/datasets/Salesforce/GiftEval. Code: https://github.com/SalesforceAIResearch/gift-eval. Leaderboard: https://huggingface.co/spaces/Salesforce/GIFT-Eval.
- License of the benchmark packaging: `apache-2.0` (Hugging Face dataset card; the repository LICENSE is Apache License 2.0, Copyright 2024 Salesforce, Inc.).
- Each underlying dataset keeps its own license. The member cards record it: CC-BY-4.0 for the Monash, M4, KDD Cup 2018, UCI Electricity, M-DENSE, and Jena weather series; CC-BY-ND-4.0 for ETT; CDLA-Sharing-1.0 for BizITObs; CC-BY-NC-3.0 (non-commercial) for Hierarchical Sales; custom terms for Bitbrains (free use with acknowledgement) and LOOP Seattle (research use with citation, no formal license); and `unknown` for the Solar, SZ-Taxi, and Restaurant sources. `redistribution: "unknown"` on a member means no explicit terms were found, not that republishing is allowed.
- TSFLab does not bundle these bytes. Download with `tsf data prepare --from gift-eval`; it links `./dataset/gift_eval` to the download directory.
- Redistribution: `upstream` for the family and every member. TSFLab never re-hosts GIFT-Eval files; `uv run tsf data prepare --from gift-eval` fetches them from https://huggingface.co/datasets/Salesforce/GiftEval (Apache-2.0 packaging, research use; sub-datasets keep their own licenses).

## Structure and statistics

All numbers are source-reported (GIFT-Eval paper, Table 13, and the repository's dataset properties); nothing is measured locally because the data are not bundled.

- Benchmark: 23 datasets, 144,000 series, 177 million observations, 7 domains, 10 frequencies, 97 configurations; 8 multivariate datasets (Jena Weather, ETT1/ETT2, BizITObs Application/Service/L2C, Bitbrains Fast Storage/Rnd) and 15 univariate ones.
- This repository: 53 presets, each one dataset-frequency pair at the short-term horizon. Series counts range from 1 (ETT, Jena Weather, BizITObs Application and L2C, saugeenday, us_births) to 48,000 (M4 monthly); the per-series cards list counts, lengths, and horizons, and the generated table below lists the members.
- Short-term horizons follow frequency: 60 for 10s, 48 for 5min/10min/15min/hourly, 30 for daily, 8 for weekly, 12 for monthly, 8 for quarterly, 6 for yearly; M4 keeps its competition horizons (14 daily, 13 weekly, 18 monthly, 48 hourly, 8 quarterly, 6 yearly). Medium and long terms extend the short horizon by 10x and 15x where enough windows exist.

## Related datasets

- LTSF counterparts: [`etth1`](../etth1/README.md), [`ettm1`](../ettm1/README.md), [`electricity`](../electricity/README.md), [`solar`](../solar/README.md), [`weather`](../weather/README.md).
- Traffic with spatial structure: [`pems_bay`](../pems_bay/README.md), [`metr_la`](../metr_la/README.md).
