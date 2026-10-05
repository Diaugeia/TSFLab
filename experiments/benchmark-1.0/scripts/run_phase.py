"""Run one benchmark phase stage by stage on a GPU server.

    setsid nohup python experiments/benchmark-1.0/scripts/run_phase.py --phase main \
        --queue-root /data/cshen20/runs/TSFLab/queues > run_phase.log 2>&1 &

Stages run in order, each to completion before the next starts:
tier 1 light, tier 1 heavy, tier 2 light, tier 2 heavy (coverage first; heavy
datasets use ``policy-<phase>-heavy.toml``). A stage enqueues its run files with
``enqueue.py``, starts the queue worker (``tsf run --backend queue work --once``
detaches one job per sweep), and polls the jobs until none is queued or running.
``--stages`` selects a subset, e.g. ``t1-light t1-heavy``. The run files must exist
(``make_runs.py --phase <phase> --validate``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGES = ["t1-light", "t1-heavy", "t2-light", "t2-heavy"]


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def job_states(queue: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in queue.glob("*/job.json"):
        try:
            status = json.loads(job.read_text()).get("status", "?")
        except (OSError, ValueError):
            status = "?"
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True)
    ap.add_argument("--queue-root", type=Path, required=True)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--jobs", default="24")
    ap.add_argument("--slots", default="64", help="worker slots = sweeps started at once")
    ap.add_argument("--poll", type=float, default=120)
    args = ap.parse_args()
    py = sys.executable
    for stage in args.stages:
        tier, weight = stage.split("-")
        queue = args.queue_root / f"{args.phase}-{stage}"
        policy = HERE / f"policy-{args.phase}.toml"
        heavy = HERE / f"policy-{args.phase}-heavy.toml"
        cmd = [py, str(HERE / "enqueue.py"), "--phase", args.phase, "--queue", str(queue),
               "--tier", tier, "--weight", weight, "--jobs", args.jobs, "--policy", str(policy)]
        if heavy.exists():
            cmd += ["--heavy-policy", str(heavy)]
        log(f"stage {stage}: enqueue")
        subprocess.run(cmd, cwd=ROOT, check=True)
        if not queue.exists():
            log(f"stage {stage}: nothing to run")
            continue
        subprocess.run([py, "-m", "tsflab.cli.main", "run", "--backend", "queue", "work", str(queue),
                        "--slots", args.slots, "--once"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        log(f"stage {stage}: started {job_states(queue)}")
        while True:
            states = job_states(queue)
            if not states.get("queued") and not states.get("running"):
                break
            time.sleep(args.poll)
            # jobs left queued (more sweeps than slots) start as slots free up
            if states.get("queued"):
                subprocess.run([py, "-m", "tsflab.cli.main", "run", "--backend", "queue", "work", str(queue),
                                "--slots", args.slots, "--once"], cwd=ROOT, stdout=subprocess.DEVNULL)
        log(f"stage {stage}: done {job_states(queue)}")
    log("phase done")


if __name__ == "__main__":
    main()
