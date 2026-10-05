"""Prepare every run file of a phase and add it to one TSFLab queue by priority.

    uv run python experiments/benchmark-1.0/scripts/enqueue.py --phase pilot --queue <dir>
    uv run tsf run --backend queue work <dir> --slots <n>     # in tmux

Each run file becomes one prepared sweep (``tsf run --prepare-only``); the queue
priority comes from ``plan.json``. Sweeps share the GPUs through the policy's
cooperative leases, so several worker slots never exceed
``max_processes_per_gpu`` per GPU. ``queued.json`` in the phase directory maps
run files to sweep directories.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def tsf(*args: str) -> dict:
    out = subprocess.run([sys.executable, "-m", "tsflab.cli.main", *args, "--json"],
                         cwd=ROOT, capture_output=True, text=True)
    text = out.stdout
    try:
        return json.loads(text[text.index("{"):])
    except ValueError:
        raise SystemExit(f"tsf {' '.join(args)} failed:\n{out.stdout}\n{out.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True)
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--jobs", default="24")
    ap.add_argument("--policy", type=Path, help="default: policy-<phase>.toml, else policy-main.toml")
    ap.add_argument("--heavy-policy", type=Path,
                    help="policy for heavy-dataset run files (default: policy-<phase>-heavy.toml if present)")
    ap.add_argument("--weight", choices=["light", "heavy", "all"], default="all",
                    help="enqueue only light or heavy run files (heavy ones need fewer runs per GPU)")
    ap.add_argument("--tier", choices=["t1", "t2", "all"], default="all", help="enqueue only one model tier")
    args = ap.parse_args()
    phase_dir = HERE.parent / "runs" / args.phase
    policy = args.policy or next(p for p in (HERE / f"policy-{args.phase}.toml", HERE / "policy-main.toml")
                                 if p.exists())
    heavy_policy = args.heavy_policy or next(
        (p for p in (HERE / f"policy-{args.phase}-heavy.toml",) if p.exists()), policy)
    plan = json.loads((phase_dir / "plan.json").read_text())
    ledger_path = phase_dir / "queued.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    for item in plan["files"]:
        if (item["file"] in ledger or args.weight not in ("all", item.get("weight"))
                or args.tier not in ("all", item.get("tier"))):
            continue
        chosen = heavy_policy if item.get("weight") == "heavy" else policy
        prepared = tsf("run", str(phase_dir / item["file"]), "--policy", str(chosen),
                       "--gpus", args.gpus, "--jobs", args.jobs, "--prepare-only")
        if not prepared.get("ok"):
            raise SystemExit(f"{item['file']}: {prepared}")
        added = tsf("run", "--backend", "queue", "add", str(args.queue),
                    "--run", prepared["directory"], "--priority", str(item["priority"]))
        ledger[item["file"]] = {"sweep": prepared["directory"], "priority": item["priority"], "policy": chosen.name,
                                "cells": item["cells"], "queue": added}
        ledger_path.write_text(json.dumps(ledger, indent=1))
        print(f"p{item['priority']}  {item['cells']:6d}  {item['file']}")
    print(f"{len(ledger)} sweeps in {args.queue}")


if __name__ == "__main__":
    main()
