"""Paper main figure: how far methods are from the best, and which architecture families lead.

    python experiments/benchmark-1.0/scripts/efficiency.py ...            # writes efficiency.csv
    python experiments/benchmark-1.0/scripts/main_figure.py efficiency.csv --out fig_main [--horizon 192]

(a) one row per dataset: every point-forecast method's MSE above the best run (%),
    gray dots, the median, and a 5% guide line;
(b) one row per architecture category (primary card tag, as in ``make_runs.py``):
    the rank percentile of its methods over the same datasets (0 = best), a box per
    category in one hue, with the method count.

Writes ``<out>.pdf``, ``<out>.png`` and ``<out>.csv`` (the plotted values).
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_runs  # noqa: E402  (same folder: categories from the cards)

CORE = ["etth1", "etth2", "ettm1", "ettm2", "electricity", "traffic", "weather", "solar"]
INK, MUTED, GRID, SURFACE, ACCENT = "#0b0b0b", "#8a8984", "#e6e5e0", "#fcfcfb", "#2a78d6"
CLIP = 60.0  # % above the best; larger gaps are drawn at the edge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("efficiency", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=192)
    ap.add_argument("--datasets", nargs="+", default=CORE)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.efficiency.open())
            if int(r["pred_len"]) == args.horizon and r["dataset"] in args.datasets]
    models = {m["name"]: m for m in make_runs.models()}
    make_runs.categorize(list(models.values()))
    category = {name: m["category"] for name, m in models.items()}

    gaps: dict[str, list[tuple[str, float]]] = {}
    ranks: dict[str, list[float]] = {}
    out_rows = []
    datasets = [d for d in args.datasets if any(r["dataset"] == d for r in rows)]
    for d in datasets:
        mse = {r["model"]: float(r["mse"]) for r in rows if r["dataset"] == d}
        best = min(mse.values())
        order = sorted(mse, key=mse.get)
        for i, name in enumerate(order):
            gap = 100.0 * (mse[name] - best) / best
            pct = i / (len(order) - 1) if len(order) > 1 else 0.0
            gaps.setdefault(d, []).append((name, gap))
            ranks.setdefault(category.get(name, "other"), []).append(pct)
            out_rows.append({"dataset": d, "model": name, "category": category.get(name, "other"),
                             "mse": mse[name], "gap_pct": round(gap, 3), "rank_pct": round(pct, 4)})
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 6.5})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (ax_a, ax_b):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=6, length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    # (a) gap to the best run per dataset
    for y, d in enumerate(reversed(datasets)):
        values = [min(g, CLIP) for _, g in gaps[d]]
        jitter = [((i * 0.6180339) % 1.0 - 0.5) * 0.45 for i in range(len(values))]
        ax_a.scatter(values, [y + j for j in jitter], s=5, color=MUTED, alpha=0.6, linewidths=0, zorder=2)
        ax_a.plot([st.median(values)] * 2, [y - 0.32, y + 0.32], color=ACCENT, lw=1.6, zorder=3)
        within = sum(g <= 5.0 for _, g in gaps[d])
        ax_a.text(CLIP * 1.02, y, f"{within}/{len(values)}", va="center", fontsize=5.5, color=INK)
    ax_a.axvline(5.0, color=INK, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax_a.set_yticks(range(len(datasets)))
    ax_a.set_yticklabels(list(reversed(datasets)), color=INK)
    ax_a.set_xlim(-1, CLIP * 1.12)
    ax_a.set_xlabel(f"MSE above the best run, % (horizon {args.horizon}; >{CLIP:.0f}% drawn at the edge)",
                    color=INK, fontsize=6)
    ax_a.set_title("(a) distance to the best method", loc="left", fontsize=7, color=INK)

    # (b) rank percentile per architecture category
    cats = sorted(ranks, key=lambda c: st.median(ranks[c]))
    box = ax_b.boxplot([ranks[c] for c in cats], vert=False, widths=0.55, patch_artist=True,
                       showfliers=False, medianprops={"color": SURFACE, "lw": 1.4},
                       whiskerprops={"color": MUTED, "lw": 0.8}, capprops={"color": MUTED, "lw": 0.8})
    for patch in box["boxes"]:
        patch.set(facecolor=ACCENT, edgecolor=ACCENT, alpha=0.85)
    ax_b.set_yticks(range(1, len(cats) + 1))
    ax_b.set_yticklabels([f"{c} ({len({r['model'] for r in out_rows if r['category'] == c})})" for c in cats],
                         color=INK)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1)
    ax_b.set_xlabel("rank percentile over the datasets (0 = best)", color=INK, fontsize=6)
    ax_b.set_title("(b) architecture families", loc="left", fontsize=7, color=INK)

    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=220)
    print(f"{len(datasets)} datasets, {len(category)} cards, {len(out_rows)} points -> {args.out}")


if __name__ == "__main__":
    main()
