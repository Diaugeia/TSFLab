"""Write the re-run cells of a phase: failed cells of chosen categories, as a CSV for make_runs --cells.

    python experiments/benchmark-1.0/scripts/report.py --phase main --csv main.csv
    python experiments/benchmark-1.0/scripts/rerun_failed.py main.csv --category oom --out oom.csv
    python experiments/benchmark-1.0/scripts/make_runs.py --phase main --cells oom.csv \
        --out experiments/benchmark-1.0/runs/main-oom1            # then enqueue with an exclusive policy
    python experiments/benchmark-1.0/scripts/make_runs.py --phase main --cells oom2.csv --batch-scale 0.5 \
        --out experiments/benchmark-1.0/runs/main-oom2            # still OOM alone on a GPU: halve the batch

A cell already succeeded in any of ``--also`` report CSVs is never re-run, so a
(model, dataset, horizon, seed) has at most one successful record.
"""

import argparse
import csv
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("report", type=Path, help="report.py --csv output of the phase")
ap.add_argument("--also", type=Path, nargs="*", default=[], help="report CSVs of earlier re-runs")
ap.add_argument("--category", nargs="+", default=["oom"], help="categories to re-run (oom, timed_out, error)")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

done, wanted = set(), {}
for path in [args.report, *args.also]:
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = (row["model"], row["dataset"], row["pred_len"])
            if row["status"] == "succeeded":
                done.add(key)
            elif row.get("category") in args.category:
                wanted[key] = row
cells = sorted(k for k in wanted if k not in done)
with args.out.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["model", "dataset", "pred_len"])
    writer.writerows(cells)
print(f"{len(cells)} cells ({', '.join(args.category)}) -> {args.out}")
