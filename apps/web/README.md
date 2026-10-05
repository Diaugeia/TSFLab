<div align="center">

# 📊 TSFLab Leaderboard

**Open, reproducible time-series forecasting leaderboard**

[![Live](https://img.shields.io/badge/live-Hugging%20Face%20Space-8c6f24.svg)](https://huggingface.co/spaces/Diaugeia/TSFLab)
[![🤗 Space](https://img.shields.io/badge/🤗%20Space-Diaugeia/TSFLab-yellow.svg)](https://huggingface.co/spaces/Diaugeia/TSFLab)
[![🤗 Datasets](https://img.shields.io/badge/🤗%20Datasets-TSFLab--Datasets-orange.svg)](https://huggingface.co/datasets/Diaugeia/TSFLab-Datasets)
[![Next.js](https://img.shields.io/badge/Next.js-static%20export-black.svg?logo=next.js)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Every entry is a community submission — one agent trajectory and one verified result —
ranked transparently across tracks, datasets, and horizons.

[**English**](README.md) | [**中文**](README_zh.md)

</div>

---

## 🧭 What is TSFLab Leaderboard

TSFLab Leaderboard is the public scoreboard for [TSFLab](https://github.com/Diaugeia/TSFLab):
**TSFLab is where experiments run; TSFLab Leaderboard is where they're shown, in the open.**
Most forecasting numbers are impossible to check — a paper reports them, a leaderboard
reprints them, nobody re-runs them. TSFLab Leaderboard works the other way around: every row is a
committed **submission you can open** — the result, the agent's trajectory, and a
readable report — so the board stays comparable, auditable, and reproducible. It's a
function of the evidence, not a table someone pasted in.

This folder holds the website and the pipeline that turns submissions into the ranked
board. The evidence itself — every `submission.json` — lives on the Hugging Face Hub in
`results/` of [`Diaugeia/TSFLab-Checkpoints`](https://huggingface.co/Diaugeia/TSFLab-Checkpoints),
next to the generated `board/leaderboard.json` that the site downloads at build time.

The board ranks TSFLab 1.0 results only. All TSEval-era results (the former static
board, the CSI-300 stock bundles, the curated air-quality and quant blocks) are archived
under `legacy/` of that repository and are not shown. Until the 1.0 benchmark results
are published the site shows a "results coming" message.

---

## ✨ Highlights

- 🏆 **Submission-driven** — the board is rebuilt from `results/` on the Hub on every upload; nothing is hand-edited.
- 🔬 **Reproducible & auditable** — each submission carries metrics + trajectory + run metadata; multi-seed runs are averaged with `n_runs` and std.
- 📈 **Method Evolution chart** — publication year vs MSE for 100+ methods, with a best-so-far (SOTA) frontier (ECharts; pan/hover/log).
- 💹 **More than regression** — a Stock track with both forecasting metrics *and* a quant backtest view (P&L, Sharpe, drawdown), plus an Air-Quality track.
- 🌏 **Bilingual & themed** — full EN / 中文, light/dark, on a self-contained static site (no backend, no cold start).
- ⚡ **Build once, ship twice** — one CI artifact deploys to Cloudflare Pages (primary) and a Hugging Face Space (mirror).

---

## 🔗 Live & data

- 🌐 **Site:** [Hugging Face Space](https://huggingface.co/spaces/Diaugeia/TSFLab) (deployed manually: `bun run build`, then upload `out/`)
- 📦 **Datasets** (on Hugging Face): [`Diaugeia/TSFLab-Datasets`](https://huggingface.co/datasets/Diaugeia/TSFLab-Datasets) — `static/` benchmark sets (ETT, electricity, solar, traffic, weather, …) and `realtime/` track panels
- 🧠 **Results + checkpoints:** [`Diaugeia/TSFLab-Checkpoints`](https://huggingface.co/Diaugeia/TSFLab-Checkpoints) — `results/` (every submission), `board/` (the generated leaderboard), `checkpoints/` (weights of top-ranked runs only), `legacy/` (TSEval archive). A submission carries no weights and never needs a `.pth` to rank.

---

## 📊 Tracks & datasets

| Category | Track | Datasets | Source |
|---|---|---|---|
| Common / static | `time_series` | ETTh1, ETTm1, ETTh2, ETTm2, electricity, solar, traffic, weather | submission-driven |
| Real-time | `stock` | Stock-HS300 (CSI-300) — regression | from submissions |
| Real-time | rolling rounds | 11 weekly tracks (`configs/realtime/`) | weekly workflow, kept in the repository |

Each block is ranked per `(track, dataset, horizon)` by **MSE** (lower is better).

---

## 📤 Submit & upload

> Full format + multi-seed averaging rules: **[SUBMITTING.md](SUBMITTING.md)**.

Open a pull request with one bundle per run under `submissions/` (a staging folder; CI
validates it). A maintainer uploads accepted bundles to the Hub and removes them here:

```bash
python3 pipeline/build_leaderboard.py --no-write              # preview the staged bundles
uv run tsf result hub results push apps/web/submissions       # maintainers: upload + regenerate board/
```

```jsonc
{
  "model": "PatchTST",        // must match a TSFLab model name
  "dataset_id": "ETTh1",      // ETTh1 … weather, or stock_hs300
  "track": "time_series",     // "time_series" | "realtime"
  "seed": 2021,
  "results": [{ "horizon": 192, "metrics": { "mse": 0.45, "mae": 0.43, "corr": 0.62 } }]
}
```

**Averaging:** submit several files with the same `model`/`dataset`/`horizon` and
different `seed` — the row reports the **mean**, `n_runs`, and `<metric>_std`.

---

## ⚙️ How the board is built

```
tsf result hub results push <bundles>          upload to results/ of Diaugeia/TSFLab-Checkpoints
  └─ tsf result hub results board               results/ → board/leaderboard.json + board/model-meta.json
bun run build
  ├ python3 pipeline/fetch_board.py              board/*.json → data/ (+ real-time block from Git)
  └ next build                                   Next static export → out/
deploy out/ to the Hugging Face Space (static) → Diaugeia/TSFLab
  (formerly TSEval: the legacy Space Diaugeia/TSEval and tseval.diaugeia.ai redirect to it)
```

- `pipeline/validate.py` — TSF-Core contract schema + TSFLab-binding check (of the staging folder).
- `pipeline/build_leaderboard.py` — aggregates submissions (mean / std / `n_runs`), ranks by MSE; optional curated overlay (`--curated`); none on the 1.0 board.
- `pipeline/fetch_board.py` — downloads the board before `next build`; `--from DIR` uses local files.
- `pipeline/build_model_meta.py` — builds `model-meta.json` (publication years) from a TSFLab checkout.

---

## 🛠️ Develop

```bash
bun install
bun run dev      # http://localhost:3000
bun run build    # static export → out/
```

---

## 🗂️ Repository layout

```
app/, src/, lib/, components/   self-contained Next app (UI + EN/中文 copy + design tokens)
  src/leaderboard.tsx           orchestrator (category/track/view + URL state)
  src/dataset-card.tsx          per-dataset card (filters + chart slot + table)
  src/results-table.tsx         the ranked table
  src/evolution-chart.tsx       Method Evolution chart (ECharts)
  src/quant-visualization.tsx   stock P&L + prediction-accuracy charts
  src/lib/, src/ui/             metrics, model types, dataset order, shared UI
data/                           fetched at build time (not in Git); data/realtime/ from the weekly workflow
submissions/                    PR staging folder + real-time rounds (see submissions/README.md)
pipeline/                       validate + build_leaderboard + fetch_board + build_model_meta
```

---

## 🔗 Related

- [TSFLab](https://github.com/Diaugeia/TSFLab) — the forecasting library that produces submissions and supplies model metadata.
- [Diaugeia.AI](https://diaugeia.ai) — open infrastructure for AI research.

---

## 📜 License

Released under the [MIT License](LICENSE). Copyright © 2026 **Diaugeia.AI**.

<div align="center">

διαύγεια · open, reproducible time-series forecasting.

</div>
