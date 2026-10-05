# Rolling real-time tracks

← [Documentation index](README.md)

Static benchmarks are fixed snapshots, so they cannot show how a model
generalizes to data that arrive after it was designed. Real-time tracks evaluate
every method in weekly **rounds** whose target data do not exist yet when the
forecasts are made.

## Tracks

There are 11 tracks (`tsf realtime list`), each declared in `configs/realtime/<id>.toml`. Hourly tracks use a
168-step input and a 24-step horizon (a week of context, one forecast day, the
same as the PeMS tracks); daily stock tracks use 20 and 5 trading
days. All hourly stamps are naive in the track's `tz` (UTC unless noted).

| Track | Source | Auth | License / terms | Bootstrap history | Frequency · horizon |
| --- | --- | --- | --- | --- | --- |
| `stock_hs300` | AKShare, forward-adjusted closes → daily log returns | none | vendor terms forbid redistribution (script: `tsf realtime update --bootstrap`) | AKShare, from 2019-01-02 | trading days · 5 |
| `stock_nasdaq100` | Nasdaq API, then Yahoo chart API, Sina last resort → daily log returns | none | vendor terms forbid redistribution (script: `tsf realtime update --bootstrap`) | Nasdaq API, from 2019-01-02 | trading days · 5 |
| `stock_sp500` | constituents from `datasets/s-and-p-500-companies` (Wikipedia fallback); closes as above | none | list ODC-PDDL; prices script-only (`tsf realtime update --bootstrap`) | Nasdaq API, from 2019-01-02 | trading days · 5 |
| `traffic_pems_{ba,la,sac,sb}` | Caltrans PeMS clearinghouse `station_5min` (Districts 4, 7, 3, 8), summed to hourly flow | optional free PeMS account | public domain (Caltrans PeMS conditions of use) | UltraTraffic store (2019–2023) | hourly · 24 |
| `air_airnow_us` | EPA AirNow `HourlyData` file products, US PM2.5 (preliminary) | none | AirNow data use guidelines: attribution, preliminary and unvalidated notice | AirNow archive files, 56 days | hourly · 24 |
| `weather_openmeteo_temp` | Open-Meteo Historical Forecast API, 2 m temperature at 82 US and EU cities | none | CC BY 4.0 data; the free API is non-commercial (collection only) | same API, 365 days | hourly · 24 |
| `solar_openmeteo_ghi` | Open-Meteo, global horizontal irradiance (`shortwave_radiation`) at 55 PV-relevant sites | none | as above | same API, 365 days | hourly · 24 |
| `grid_ercot` | ERCOT hourly load, 8 weather zones: `Native_Load` archives + MIS report NP6-345-CD | none | ERCOT terms: unmodified, credit ERCOT | `Native_Load` archives, from 2019 | hourly · 24 (America/Chicago) |

Credentials are read from the environment and never written to disk:

| Variable | Used by | Where to get it |
| --- | --- | --- |
| `PEMS_USER`, `PEMS_PASSWORD` | `traffic_pems_*` (optional) | free account at https://pems.dot.ca.gov |

The PeMS credentials are optional: they add the live weekly increments. Without
them, the traffic tracks use their UltraTraffic history only. The stock, AirNow,
Open-Meteo, and ERCOT tracks need no key. Notes per source:

- **Stocks.** Nasdaq and Yahoo both return split-adjusted, not dividend-adjusted,
  closes, so either serves a day identically; the Sina fallback is
  dividend-adjusted and can differ by the dividend yield on ex-dividend days.
  The Nasdaq historical endpoint answers only windows that end near the present,
  which is how updates and bootstraps call it. Each symbol is cached under
  `dataset/realtime/_cache/`, so an interrupted bootstrap resumes. Constituents
  are frozen at bootstrap.
- **Open-Meteo.** The Historical Forecast API stitches the analysis hours of the
  operational models, which is the same product the live forecast endpoint shows
  for recent hours, so history and weekly updates share one definition (the ERA5
  archive endpoint lags by days and differs slightly). The newest three hours
  are forecasts and are not stored. Temperature is its own track so all channels
  share one scale and unit; irradiance is a second track because it is the
  weather driver of PV output and needs no key. Free-tier limits (600 calls per
  minute, 5,000 per hour, 10,000 per day, counted per location and 14 days)
  allow the 365-day bootstrap in batches of 10 sites.
- **AirNow.** The files are preliminary and unvalidated (AQS holds the validated
  values months later); negative PM2.5 readings are raw instrument noise and
  are kept. Each file is about 0.75 MB, so the 8-week bootstrap is about 1 GB
  fetched once and then pulled from the Hub.
- **ERCOT.** Hour-ending values in local clock time are stored at the start of the
  hour they cover; the repeated hour of the autumn clock change is averaged. The
  archive is refreshed monthly and the daily report files stay listed for about a
  month, so a weekly update never leaves a gap. The ERCOT Public API
  (`api.ercot.com`, registration at https://apiexplorer.ercot.com plus a
  subscription key) offers deeper history and the solar and wind reports, whose
  public files keep only about a week; it is not used by default.
- **PeMS (UltraTraffic history).** The four traffic tracks bootstrap from a
  local UltraTraffic parquet store, not from the live API. The `UltraTraffic_CL`
  archive (inside `TrafficCL.zip`) holds hourly total flow per Caltrans PeMS
  station for Districts 4 (`PEMS_BA`), 7 (`PEMS_LA`), 3 (`PEMS_SAC`), and 8
  (`PEMS_SB`), one file per year from 2003 to 2023. Its publisher is not
  identified; the raw PeMS data are public domain under the PeMS conditions of
  use. Build the store once with
  `uv run tsf data prepare --from ultratraffic --archive TrafficCL.zip`
  (written to `dataset/ultratraffic`, override with `ULTRATRAFFIC_ROOT`). The
  tracks read the `static` panels from 2019 and keep the 2023 station set:
  2,472 stations (District 4), 1,926 (7), 801 (3), and 1,105 (8). Stations per
  year range from 1,177 to 2,472 (District 4), 1,295 to 1,927 (7), 84 to 801 (3),
  and 108 to 1,252 (8). The archive carries no station coordinates, so the
  tracks have no adjacency. The weekly increments come from the clearinghouse
  `station_5min` files summed to hourly flow, which match the archive's
  `station_hour` flow values.

A new track is bootstrapped once by a maintainer (`tsf realtime update --bootstrap --track T
--push`); the weekly workflow then pulls the store from the Hub and appends. The
stock tracks are never uploaded: `--push` prints a skip line for them, and each
machine that runs them builds its own store with
`tsf realtime update --bootstrap --track <track>`.

## Data releases

Each track has an append-only panel store (`dataset/realtime/<track>/`,
parquet by year plus a manifest). An update only adds new timestamps or fills
cells that were missing, never rewrites observed history, and records a release
with a content hash. Releases of the hosted tracks are mirrored to the folder
`realtime/<track>/` of the Hugging Face dataset `Diaugeia/TSFLab-Datasets` (owner
overridable with `TSFLAB_HUB_OWNER`), one commit per release, so every round can be
reproduced from a pinned revision. Only tracks whose card says
`[source] redistribution = "hosted"` are uploaded; `--push` skips the others.

## Tracks as static datasets

Every track also has a static dataset preset, `rt_<track>` (`configs/datasets/rt/`),
so catalog models can be trained and compared on a frozen snapshot with the usual
`tsf run`. The `realtime_panel_ts` / `realtime_panel_st` loaders read the track's
panel store at a pinned release (`version`) or Hub `revision`, fill gaps causally,
split chronologically 7:1:2 with one z-score from the training rows, and add
calendar marks; tracks whose mode is `spatiotemporal` use the node-covariate
layout of `forecast.export_bundle`. Provide the store under
`dataset/realtime/<track>` or set `revision` to pull one from the Hub. Static results are not live results: only the
rolling rounds evaluate data that did not exist at submission time.

## Rounds

`tsf realtime open` creates a round on the latest release:

- `cutoff` — the last observed timestamp;
- `target_timestamps` — `horizon` steps that start after a submission window
  (`submission_hours`), so forecasts always precede their truth;
- `deadline` — one second before the first target;
- `norm_mean` / `norm_std` — per-channel statistics frozen at the cutoff, used to
  z-score errors so that channels with different scales are comparable.

Rounds, forecasts, and scores live in
`apps/web/submissions/realtime/<track>/rounds/<round_id>/` as reviewable
evidence. Every stored instant carries its timezone offset.

## Forecasts

A forecast is a `ForecastSubmission` (`predictions` of shape
`(horizon, channels)` in raw units). Three sources fill a round:

- **Baselines** (`Naive`, `SeasonalNaive`, `WindowMean`) are submitted by the
  weekly workflow on CPU, so every round has anchors.
- **Catalog models**: `tsf realtime forecast --track T --model M` exports the
  round's history in the index-windowed `cauair` layout, trains and early-stops
  through the standard runner, and forecasts the window at the cutoff.
  Time-series models see stations as channels; spatiotemporal models also get
  calendar covariates. Needs the full install; a GPU is recommended.
- **Anyone** can open a pull request adding `forecasts/<Model>.json`. CI rejects
  it if the pull request was last updated after the deadline.

## Scoring and stability

Once a later release covers at least `min_coverage` of the target cells,
`tsf realtime score` computes MSE and MAE over the observed cells and ranks the
forecasts. `apps/web/data/realtime/<track>.json` summarizes mean rank and mean
MSE per method over scored rounds, plus rank stability: Kendall's tau between
the rankings of consecutive rounds.

## Weekly automation

The `weekly` workflow runs `python -m tsflab.realtime weekly` on a
CPU runner every Monday (update → score → open → baselines) and proposes the
results as a pull request that merges after `tsf repo check` passes. The real-time package imports no torch outside
`forecast`, so the job installs only `pydantic`, `pandas`, `pyarrow`, `requests`,
`huggingface_hub`, and `akshare` (only the Sina fallback and CSI-300 need it).

## Replaying history

`tsf realtime replay --track T --end DATE --weeks N [--models ...]` opens rounds
at past weekly cutoffs on a store that already contains the following weeks,
forecasts them (catalog models train only on data up to each cutoff), and scores
them immediately. Replayed rounds are written under `work_dirs/_realtime_replay/`
and never mix with live rounds.
