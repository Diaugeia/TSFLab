# Submitting data to the TSFLab Leaderboard

The leaderboard at **[the TSFLab Space](https://huggingface.co/spaces/Diaugeia/TSFLab)** is an open board
you can check: every row is rebuilt from the submission evidence in `results/` of the
Hugging Face model repository
[`Diaugeia/TSFLab-Checkpoints`](https://huggingface.co/Diaugeia/TSFLab-Checkpoints) —
the single source of truth for results. The generated board is `board/leaderboard.json`
in the same repository; the site downloads it at build time
(`pipeline/fetch_board.py`). GitHub keeps code only.

## TL;DR

1. Build one bundle per run with `tsf result submit` (or write one `submission.json`).
2. Open a pull request that adds it under
   `submissions/<track>/<dataset>/<model>/<run_id>/` (a staging folder). CI validates
   it with `pipeline/validate.py` (`tsf repo check`, step `web-submissions`).
3. After review, a maintainer uploads it with `tsf result hub results push`, which
   regenerates `board/`, and removes it from the staging folder. The next site build
   shows it.

Maintainers with a Hub token skip the pull request:
`uv run tsf result hub results push work_dirs/_submissions`.

## Directory layout

```
results/                                                    # in Diaugeia/TSFLab-Checkpoints
  time_series/<dataset>/<model>/<run_id>/submission.json    # ETTh1, ETTh2, electricity, solar, traffic, weather, …
  realtime/<track>/<model>/<run_id>/submission.json         # realtime/stock_hs300 → shown as the "Stock" track
```

The staging folder `submissions/` in this repository uses the same layout. The
TSEval-era CSI-300 bundles are archived under `legacy/` of that repository and are
not ranked.

`run_id` is any unique folder name; the convention is
`<model>_<dataset>_sl<seqlen>_pl<predlen>_seed<seed>_<timestamp>`.

## `submission.json` format

Two shapes are accepted (same record fields). **Prefer the flat shape** for new
submissions:

```jsonc
{
  "schema_version": "1.0.0",
  "model": "PatchTST",          // must match a TSFLab model name (see `tsf catalog search`)
  "dataset_id": "ETTh1",        // ETTh1 … weather, or stock_hs300
  "track": "time_series",       // "time_series" or "realtime"
  "seed": 2021,                 // one seed per file (see "Multiple runs" below)
  "results": [
    {
      "horizon": 192,           // pred_len: 192 for time_series, 5 for stock
      "metrics": {              // nulls allowed; mse drives ranking
        "mse": 0.4521, "mae": 0.4310, "rmse": 0.6724,
        "corr": 0.62, "wape": 0.51, "rse": 0.60
      },
      "timing": { "fit_time_sec": 812.4, "inference_time_sec": 7.1 }
    }
  ]
}
```

The bundle shape `{ "manifest": {...}, "datasets": [...], "records": [ <flat record>, … ] }`
is also accepted — the aggregator reads `records[]`. Both are validated against
`pipeline/contract.schema.json`.

`dataset_id → display name`: `stock_hs300 → Stock-HS300`; every other id is shown
verbatim. `track=realtime` + `dataset_id=stock_hs300` lands in the **Stock** track.

## Multiple runs / averaging across seeds

To report a mean over several seeds, **submit one file per run** (same `model` +
`dataset_id` + `horizon`, different `seed` and `run_id`). The aggregator groups them
and the leaderboard row shows:

- each metric = **mean across runs**,
- `<metric>_std` (e.g. `mse_std`) = sample standard deviation (omitted for a single run),
- `n_runs` = number of runs (the **Runs** column),
- `submission_ids` = every run folded in.

No config needed — drop 5 seed files in and the row reports `n_runs: 5` with averaged
metrics. (Example today: `weather` MoFo/Kronos are already `n_runs: 2`.)

## What is and isn't submission-driven

| Block | Source |
|---|---|
| `time_series/*` (all 8 datasets) | aggregated from `results/time_series/` |
| Stock **regression** (mse/mae/corr) | aggregated from `results/realtime/stock_hs300/` (the TSEval-era rows are in `legacy/`, not ranked) |
| Stock **quant** (returns/Sharpe/…) | archived in `legacy/board/` — no raw quant submissions yet |
| **Air quality** (`Air-CHNCities`) | **curated** — `board/curated.json`, raw inputs not uploaded |

Curated blocks are applied on every rebuild (see `overlay_curated` in
`pipeline/build_leaderboard.py`). To make them submission-driven, add the
corresponding `submission.json` files and they'll replace the curated rows.

## Real-time rounds

Real-time tracks (`stock_hs300`, `traffic_pems_sb`, `air_airnow_us`, and the other 8 listed in
`docs/en/realtime.md`) are
evaluated in weekly **rounds**. Each Monday the `weekly` workflow
releases new data, scores rounds whose target window is now observed, and opens
a new round:

```
submissions/realtime/<track>/rounds/<round_id>/
    round.json              # the round: cutoff, target timestamps, channels, deadline
    forecasts/<model>.json  # one ForecastSubmission per method
    scores.json             # written by the workflow once the truth is released
```

To take part, open a pull request that adds `forecasts/<YourModel>.json` for an
open round **before its `deadline`**. The file is a `ForecastSubmission`
(`src/tsflab/core/schema/forecast_submission.schema.json`):
`predictions` has shape `(len(target_timestamps), len(channels))` in the
round's raw units. CI checks the shape and rejects the file if the pull request
was last updated after the deadline, so forecasts always precede their truth.
Only the workflow may write `round.json` and `scores.json`.

Catalog models can produce a forecast directly (GPU recommended):

```bash
uv run tsf realtime forecast --track traffic_pems_sb --model DLinear
```

Scores use MSE/MAE on channels z-scored with statistics frozen at the round's
cutoff. The site shows the mean rank over scored rounds and rank stability
(Kendall's tau between consecutive rounds).

## Build & validate locally

```bash
python pipeline/validate.py                                     # contract check of submissions/
python pipeline/build_leaderboard.py --no-write                 # preview the staged bundles
python pipeline/build_leaderboard.py --source DIR --out FILE    # a board from any bundle folder
bun run build                                                   # fetch board/ from the Hub, then next build
python pipeline/fetch_board.py --from DIR                       # offline: use local board files instead
```

Ranking is per `(track, dataset, horizon)` by **MSE** (lower is better). Weights are
**not** part of a submission and are never required to get on the board — a row earns
its place with its result, trajectory, and report. Maintainers archive the
checkpoints of top-ranked runs in `checkpoints/` of
[`Diaugeia/TSFLab-Checkpoints`](https://huggingface.co/Diaugeia/TSFLab-Checkpoints)
(`tsf result hub push-top`) for bit-level reproducibility.
