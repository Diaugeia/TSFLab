<div align="center">

# 🚀 TSFLab

**A fully automated, continuously updated platform for time series forecasting**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Models: 311](https://img.shields.io/badge/models-311-orange.svg)](docs/en/models.md)
[![Datasets: 93](https://img.shields.io/badge/datasets-93-teal.svg)](docs/en/workflows.md#data)
[![Real-time tracks: 11](https://img.shields.io/badge/real--time%20tracks-11-purple.svg)](docs/en/realtime.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Every forecasting method, one interface, one protocol, evaluated on the data
we already have **and** on data that did not exist when the method was written.

</div>

> 🧪 **Latest features land on the [`dev`](https://github.com/Diaugeia/TSFLab/tree/dev) branch first.** `main` is the stable, versioned release line.

---

## 🧭 Why TSFLab

Forecasting papers multiply every year, yet each one can compare against only a
few baselines, re-run under its own code and data conventions. Keeping hundreds
of methods in one benchmark by hand, verifying their code, and re-evaluating them
on new data no longer scales, and fixed benchmark snapshots cannot show how a
model holds up as the world moves on.

TSFLab automates that loop. Coding agents read new papers, implement them
behind one verified interface, and evaluate them under one protocol on static
datasets and on **rolling real-time tracks** that refresh every week. TSFLab
ships no agent of its own: it is the infrastructure (catalog, contracts, data,
protocols, run records) that any coding agent operates through declarative skills.

---

## ✨ What is inside

| Module | What it does |
| --- | --- |
| 📚 **Paper reading** | Scans arXiv and Hugging Face Papers, deduplicates against the catalog, and records each paper's structure, equations, and pinned official code |
| 🧩 **Code & interface** | 313 methods as peers in one flat catalog, composed from 65 shared components, one forecasting signature, and an executable admission contract recorded in every card; each card states its fidelity to the paper and pinned official code, a six-slot composition, the data characteristics it fits, and its data-dependent parameters |
| 🗃️ **Data** | 93 dataset presets in seven domains (energy 25, transport 18, environment 16, finance 14, cloud-web 9, healthcare 7, sales 4; 79 conventional, including the GIFT-Eval family, and 14 spatiotemporal or covariate; 11 of them are frozen releases of the real-time tracks), each with a card, its benchmark suites, a verified redistribution class, and one TSFLab protocol; plus 11 rolling real-time tracks (stocks, traffic, air quality, weather, grid, solar) |
| ⚙️ **Experiments** | Declarative TOML sweeps, pre-run validation, seeds, budgets, GPU leases, queues, and recovery; `tsf data analyze` profiles a dataset, `tsf catalog match` maps its characteristics to models and components, and `tsf model compose` dry-runs a recombination for AutoResearch |
| 🏆 **Release & compare** | Run records → submissions → a leaderboard recomputed from evidence; weights as pinned `hf://` bundles |

---

## 🏁 Quick start

**Work in the repository with an agent:**

```bash
git clone https://github.com/Diaugeia/TSFLab.git
cd TSFLab
codex          # or any other coding agent
```

```text
> Set up the environment for my GPU.
> Benchmark DLinear, PatchTST and iTransformer on ETTh1 and give me a leaderboard.
> Implement the paper at <arXiv URL> as a catalog model and admit it.
> Forecast this week's traffic round with PatchTST and submit it.
```

**Or install the framework and scaffold your own project:**

```bash
uv tool install "git+https://github.com/Diaugeia/TSFLab"   # provides `tsf`
tsf init my-forecasting-project --modules data,models,experiments,release,autoresearch
cd my-forecasting-project        # default: all modules; also writes the agent guide and skills
tsf data download etth1          # pinned, checksum-verified from the Hub
tsf run configs/runs/example.toml --dry-run
tsf run configs/runs/example.toml
```

Scaffolded run configs inherit the installed catalog through `tsflab://`
paths, so upgrading TSFLab upgrades their defaults. Install by module with
extras: `tsflab[data]`, `[models]`, `[experiments]`, `[hub]`, `[realtime]`,
`[autoresearch]`, or `[all]`.

**Read the catalog progressively:**

```bash
uv run tsf catalog                                      # counts, then the next commands
uv run tsf catalog search "reversible normalization"    # L0: one line per match
uv run tsf catalog show PatchTST                        # L1: card facts and short README
uv run tsf catalog show revin --depth 2                 # L2: adds the reference (--depth 3: paths)
uv run tsf data analyze etth1                           # profile a dataset (train-only statistics)
uv run tsf catalog match etth1                          # models and components that fit its characteristics
uv run tsf realtime list
```

Search accepts `--kind model|component|dataset` and, for models, `--capability`.
See [docs/en/workflows.md](docs/en/workflows.md#reading-the-catalog).

---

## 📈 Rolling real-time evaluation

Each week the `weekly` workflow releases new observations (versioned on
the Hugging Face Hub), scores the rounds whose target window is now observed, and
opens a new round. Forecasts must be submitted **before** their targets exist, so
no model, including ours, can have seen its evaluation data.

| Track | Data | Setting | Horizon |
| --- | --- | --- | --- |
| `stock_hs300` | CSI-300 constituents, daily log returns (AKShare) | time series | 5 trading days |
| `stock_nasdaq100` | NASDAQ-100 constituents, daily log returns (Nasdaq API, Yahoo fallback) | time series | 5 trading days |
| `stock_sp500` | S&P 500 constituents, daily log returns (Nasdaq API, Yahoo fallback) | time series | 5 trading days |
| `traffic_pems_{ba,la,sac,sb}` | Caltrans PeMS Districts 4, 7, 3, 8: hourly flow at 2,472 / 1,926 / 801 / 1,105 stations | spatiotemporal | 24 h |
| `air_airnow_us` | EPA AirNow hourly PM2.5, US monitors (no key) | spatiotemporal | 24 h |
| `weather_openmeteo_temp`, `solar_openmeteo_ghi` | Open-Meteo hourly temperature at 82 US/EU cities, irradiance at 55 PV sites | spatiotemporal | 24 h |
| `grid_ercot` | ERCOT hourly load in 8 weather zones | spatiotemporal | 24 h |

```bash
uv run tsf realtime forecast --track traffic_pems_sb --model DLinear   # produce a forecast
uv run tsf realtime replay --track traffic_pems_sb --end 2023-12-25 --weeks 12   # backtest the protocol
```

Each track is also a frozen static dataset (`rt_<track>`, `tsf catalog list --kind dataset`). See [docs/en/realtime.md](docs/en/realtime.md).

---

## 🤝 Contributing

There are three ways to take part:

1. **Propose a method or report a problem** — open an *Add a paper* or *Report a
   problem or ask a question* issue. An agent triages it; accepted papers and
   reproducible fixes are checked with `tsf repo check`, reviewed by a second agent
   pass, and merged by the `agent` workflow.
2. **Submit results** — add a `submission.json` under `apps/web/submissions/`
   (see [SUBMITTING.md](apps/web/SUBMITTING.md)); CI validates it against the
   contract, and a maintainer moves it to the results repository on the Hugging
   Face Hub (`Diaugeia/TSFLab-Checkpoints`) that the leaderboard is built from.
3. **Forecast a real-time round** — add `forecasts/<YourModel>.json` to an open
   round before its deadline.

The literature is also scanned weekly by the `agent` workflow. See
[CONTRIBUTING.md](CONTRIBUTING.md) for code contributions.

---

## 📖 Documentation

- [Workflow documentation](docs/en/README.md): catalog, models, data, AutoResearch, admission, experiments
- [Projects and the Hub](docs/en/hub.md): `tsf init`, `hf://` assets, results and checkpoints (`TSFLab-Checkpoints`)
- [Real-time tracks](docs/en/realtime.md): rounds, forecasts, scoring, weekly automation

Exact command options stay in `tsf <command> --help`.

---

## 🗂️ Repository layout

| Path | Contents |
| --- | --- |
| `src/tsflab/` | One package per module, mirrored by the CLI: `catalog` (cards, registries, admission), `data`, `models` (flat catalog, `_components`, `_slots`), `experiments` (config, runner, evaluation, execution), `release` (Hub, submissions), `realtime`, `research` (rounds, recombination), `agent` (assets, tasks, `tsf init`), `core` (contracts), `cli` |
| `configs/`, `catalog/` | Run, model, and dataset presets, real-time track configs, `configs/fixtures/` (smoke and synthetic test inputs, not datasets), dataset cards, declined papers |
| `dataset/` | Local dataset bytes fetched with `tsf data download` (not packaged) |
| `apps/web/` | TSFLab Leaderboard: static site, submission pipeline, `submissions/`, real-time rounds |
| `experiments/` | Local research workspace; only `*/scripts/` is tracked |

---

## 📜 License

TSFLab is released under the [MIT License](LICENSE). Copyright © 2026 **Diaugeia.AI**.

Ordinary paper architectures are maintained locally under the project license.
Released pretrained foundation models use optional official packages and unchanged
checkpoints through the offline runtime boundary; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Real-time data remain subject to
their providers' terms (Caltrans PeMS, AirNow, Open-Meteo, ERCOT, and the stock
vendors behind AKShare, Nasdaq, and Yahoo). Each dataset card records its license
and redistribution class; `upstream` and `script` data are never re-hosted.

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Diaugeia/TSFLab&type=Date)](https://star-history.com/#Diaugeia/TSFLab&Date)
