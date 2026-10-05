"""Summarize the runs of benchmark sweeps: status, diagnosis, error, time, profile.

    uv run python experiments/benchmark-1.0/scripts/report.py --phase pilot [--csv out.csv]

Reads ``queued.json`` of the phase (the sweeps ``enqueue.py`` added), then each
run directory under ``work_dirs/_runs``. Prints counts and every failed cell with
the last error line of its latest attempt log.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ERROR = re.compile(r"(Error|Exception|error:|Killed|Traceback)")


def last_error(log: Path) -> str:
    if not log.exists():
        return ""
    lines = log.read_text(errors="replace").splitlines()
    hits = [line.strip() for line in lines if ERROR.search(line) and not line.lstrip().startswith("File ")]
    return (hits[-1] if hits else (lines[-1].strip() if lines else ""))[:240]


def cell(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest.get("config", manifest)
    attempts = sorted(run_dir.glob("attempts/*.json"), key=lambda p: int(p.stem))
    last = json.loads(attempts[-1].read_text()) if attempts else {}
    row = {
        "run_id": run_dir.name,
        "model": cfg.get("model", {}).get("name"),
        "dataset": cfg.get("dataset", {}).get("alias") or cfg.get("dataset", {}).get("name"),
        "seq_len": cfg.get("task", {}).get("seq_len"),
        "pred_len": cfg.get("task", {}).get("pred_len"),
        "batch_size": cfg.get("training", {}).get("batch_size"),
        "status": last.get("status", "pending"),
        "attempts": len(attempts),
        "diagnosis": last.get("diagnosis", ""),
        "category": "",
        "error": "",
        "train_time_sec": None,
        "epochs_run": None,
        "mse": None,
        "mae": None,
    }
    result = run_dir / "result.json"
    if result.exists():
        res = json.loads(result.read_text())
        row["train_time_sec"] = round(res.get("train_time_sec") or 0, 1)
        row["mse"] = res.get("metrics", {}).get("mse")
        row["mae"] = res.get("metrics", {}).get("mae")
    events = run_dir / "events.jsonl"
    if events.exists():
        steps = [json.loads(line).get("step") for line in events.read_text().splitlines() if line.strip()]
        row["epochs_run"] = max((s for s in steps if isinstance(s, int)), default=None)
    row["category"] = "ok" if row["status"] == "succeeded" else row["status"]
    if row["status"] != "succeeded":
        logs = sorted(run_dir.glob("attempt-*.log"), key=lambda p: int(p.stem.split("-")[1]))
        row["error"] = last_error(logs[-1]) if logs else ""
        text = logs[-1].read_text(errors="replace") if logs else ""
        if row["status"] == "failed":
            if "OutOfMemoryError" in row["error"] or "out of memory" in row["error"]:
                row["category"] = "oom"
            elif row["error"].startswith("ValueError") and "Epoch 1/" not in text:
                # the model or its spec refused the cell at construction: not applicable
                row["category"] = "n/a"
            else:
                row["category"] = "error"
    return row


def profiles(work_dirs: Path) -> dict[str, dict]:
    out = {}
    for path in work_dirs.glob("*/*/profile.csv"):
        with path.open() as stream:
            for rec in csv.DictReader(stream):
                out[rec["run_id"]] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True)
    ap.add_argument("--csv", type=Path, help="write one row per run")
    args = ap.parse_args()
    phase_dir = HERE.parent / "runs" / args.phase
    ledger = json.loads((phase_dir / "queued.json").read_text())
    rows = []
    for entry in ledger.values():
        sweep = json.loads((Path(entry["sweep"]) / "sweep.json").read_text())
        for run in sweep.get("runs", []):
            run_dir = Path(run if isinstance(run, str) else run.get("directory") or run.get("path"))
            if (run_dir / "manifest.json").exists():
                rows.append(cell(run_dir))
    work_dirs = Path(next(iter(ledger.values()))["sweep"]).parents[1]
    prof = profiles(work_dirs)
    for row in rows:
        p = prof.get(row["run_id"], {})
        for key in ("total_params", "peak_vram_mb", "latency_avg_ms"):
            row[key] = p.get(key)
    counts = collections.Counter(r["status"] for r in rows)
    print(f"{len(rows)} runs: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    cats = collections.Counter(r["category"] for r in rows)
    print("by category: " + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))
    order = {"error": 0, "oom": 1, "timed_out": 2, "n/a": 3}
    for r in sorted((r for r in rows if r["category"] not in ("ok", "pending", "running")),
                    key=lambda r: (order.get(r["category"], 9), r["model"], r["dataset"])):
        print(f"  {r['category']:10s} {r['model']:22s} {r['dataset']:12s} {r['error']}")
    done = [r for r in rows if r["train_time_sec"]]
    if done:
        times = sorted(r["train_time_sec"] for r in done)
        print(f"train time (s): median {times[len(times) // 2]}, p90 {times[int(len(times) * 0.9)]}, max {times[-1]}")
    if args.csv:
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
