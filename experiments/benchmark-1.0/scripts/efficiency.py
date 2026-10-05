"""Accuracy versus cost of the benchmark runs: table, Pareto fronts, figure.

    python experiments/benchmark-1.0/scripts/efficiency.py --phases main main-core main-rest \
        --out-dir experiments/benchmark-1.0/efficiency [--horizon 192] [--datasets etth1 ...]

Joins each succeeded run (``report.py`` rows of the given phases) with its
``profile.csv`` row (parameters, MACs, inference peak VRAM, latency) and writes:

- ``efficiency.csv``: one row per (model, dataset, horizon), the first succeeded run;
- ``pareto.csv``: for each (dataset, horizon), the models on the front of
  (fewer parameters, lower MSE): no other model has both;
- ``efficiency_<horizon>.pdf`` / ``.png``: one panel per dataset, MSE gap to the best
  run against parameters (log), every model a gray dot, the front a blue line with
  its models labeled.

Parameters and MACs depend only on the architecture and the data shape. Latency and
training time in ``profile.csv`` are measured on shared GPUs during the sweep, so they
are reported but not used for the front; use an exclusive latency probe for those.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report  # noqa: E402  (same folder)

CORE = ["etth1", "etth2", "ettm1", "ettm2", "electricity", "traffic", "weather", "solar"]
INK, MUTED, GRID, SURFACE, FRONT = "#0b0b0b", "#8a8984", "#e6e5e0", "#fcfcfb", "#2a78d6"


def number(text: str | None) -> float | None:
    """'155166.1793 M' -> 155166.1793 (unit dropped; columns keep one unit each)."""
    if text in (None, ""):
        return None
    match = re.match(r"\s*([-+0-9.eE]+)", str(text))
    return float(match.group(1)) if match else None


def collect(phases: list[str]) -> list[dict]:
    rows, seen = [], set()
    for phase in phases:
        ledger = HERE.parent / "runs" / phase / "queued.json"
        if not ledger.exists():
            continue
        for row in report.rows_of(phase):
            if row["status"] != "succeeded" or row["mse"] is None:
                continue
            key = (row["model"], row["dataset"], int(row["pred_len"]))
            if key in seen:  # one record per cell (re-runs may duplicate a cell)
                continue
            seen.add(key)
            rows.append(row)
    return rows


def pareto(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """(name, params, mse) -> front sorted by params: each point has lower mse than every smaller one."""
    front, best = [], math.inf
    for name, params, mse in sorted(points, key=lambda p: (p[1], p[2])):
        if mse < best:
            front.append((name, params, mse))
            best = mse
    return front


def figure(rows: list[dict], horizon: int, datasets: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [d for d in datasets if any(r["dataset"] == d and r["pred_len"] == horizon for r in rows)]
    if not shown:
        return
    cols = 4
    nrows = math.ceil(len(shown) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(7.0, 1.75 * nrows + 0.35), squeeze=False)
    plt.rcParams.update({"font.size": 6.5})
    for ax in axes.flat[len(shown):]:
        ax.axis("off")
    for ax, d in zip(axes.flat, shown):
        pts = [(r["model"], r["total_params"], r["mse"]) for r in rows
               if r["dataset"] == d and r["pred_len"] == horizon and r["total_params"]]
        best = min(p[2] for p in pts)
        gap = lambda m: 100.0 * (m - best) / best  # noqa: E731
        ax.set_facecolor(SURFACE)
        ax.scatter([p[1] for p in pts], [gap(p[2]) for p in pts], s=7, color=MUTED, alpha=0.55,
                   linewidths=0, zorder=2)
        front = pareto(pts)
        ax.plot([p[1] for p in front], [gap(p[2]) for p in front], color=FRONT, lw=1.2, zorder=3)
        ax.scatter([p[1] for p in front], [gap(p[2]) for p in front], s=14, color=FRONT,
                   edgecolors=SURFACE, linewidths=0.8, zorder=4)
        for name, x, m in front[-4:]:  # label the low-error end of the front
            ax.annotate(name, (x, gap(m)), xytext=(3, 3), textcoords="offset points",
                        fontsize=5.2, color=INK)
        ax.set_xscale("log")
        top = sorted(gap(p[2]) for p in pts)
        ax.set_ylim(-1, max(5.0, top[int(0.8 * (len(top) - 1))] * 1.15))  # clip the long tail
        ax.set_title(f"{d}  (n={len(pts)})", fontsize=7, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=5.5, length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    fig.supxlabel("parameters (log scale)", fontsize=6.5, color=INK)
    fig.supylabel(f"MSE above the best run, % (horizon {horizon})", fontsize=6.5, color=INK)
    fig.tight_layout(pad=0.4)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phases", nargs="+", default=["main", "main-core", "main-rest"])
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "efficiency")
    ap.add_argument("--horizon", type=int, default=192)
    ap.add_argument("--datasets", nargs="+", default=CORE)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(args.phases)
    for r in rows:
        r["pred_len"] = int(r["pred_len"])
        for key in ("total_params", "peak_vram_mb", "latency_avg_ms"):
            r[key] = number(r.get(key))
        r["total_macs_m"] = number(r.get("total_macs_m"))
    fields = ["model", "dataset", "seq_len", "pred_len", "mse", "mae", "total_params", "total_macs_m",
              "peak_vram_mb", "latency_avg_ms", "train_time_sec", "run_id"]
    with (args.out_dir / "efficiency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["dataset"], r["pred_len"], r["mse"])))
    fronts = []
    for d in sorted({r["dataset"] for r in rows}):
        for h in sorted({r["pred_len"] for r in rows if r["dataset"] == d}):
            pts = [(r["model"], r["total_params"], r["mse"]) for r in rows
                   if r["dataset"] == d and r["pred_len"] == h and r["total_params"]]
            for name, params, mse in pareto(pts):
                fronts.append({"dataset": d, "pred_len": h, "model": name, "total_params": int(params),
                               "mse": mse, "n": len(pts)})
    with (args.out_dir / "pareto.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "pred_len", "model", "total_params", "mse", "n"])
        writer.writeheader()
        writer.writerows(fronts)
    figure(rows, args.horizon, args.datasets, args.out_dir / f"efficiency_{args.horizon}")
    print(f"{len(rows)} runs, {len(fronts)} front points -> {args.out_dir}")


if __name__ == "__main__":
    main()
