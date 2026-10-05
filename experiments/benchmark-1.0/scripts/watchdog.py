"""Keep a shared server responsive while benchmark sweeps run.

    python experiments/benchmark-1.0/scripts/watchdog.py --min-available-gb 100 --log <file>

Every ``--interval`` seconds it reads ``MemAvailable`` from ``/proc/meminfo``.
Below ``--min-available-gb`` it sends SIGKILL to the largest run process of this
user (``tsflab.experiments.run_config``, with its data-loader children); the
cell is then recorded as failed and can be re-run with ``tsf run resume``. Only
this user's run processes are ever touched. Standard library only.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

MARKER = "tsflab.experiments.run_config"


def available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    return float("inf")


def run_processes() -> dict[int, tuple[int, str, int]]:
    """pid -> (rss_kb, payload, ppid) for this user's run processes."""
    uid, found = os.getuid(), {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            cmd = (entry / "cmdline").read_bytes().split(b"\0")
            if not any(MARKER.encode() in part for part in cmd):
                continue
            status = dict(line.split(":", 1) for line in (entry / "status").read_text().splitlines() if ":" in line)
            rss = int(status.get("VmRSS", "0 kB").split()[0])
            ppid = int(status.get("PPid", "0").strip())
            payload = next((c.decode() for c in cmd if c.startswith(b"/") and b"_runs/" in c), "")
            found[int(entry.name)] = (rss, payload, ppid)
        except (OSError, ValueError):
            continue
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-available-gb", type=float, default=100)
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--log", type=Path)
    args = ap.parse_args()

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        if args.log:
            with args.log.open("a") as stream:
                stream.write(line + "\n")

    log(f"watchdog start: kill the largest run below {args.min_available_gb} GB available")
    while True:
        free = available_gb()
        if free < args.min_available_gb:
            procs = run_processes()
            # group data-loader workers with their run (same payload): kill the run with the largest total RSS
            totals: dict[str, int] = {}
            for rss, payload, _ in procs.values():
                totals[payload] = totals.get(payload, 0) + rss
            if totals:
                victim = max(totals, key=totals.get)
                pids = [pid for pid, (_, payload, _) in procs.items() if payload == victim]
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                log(f"available {free:.0f} GB: killed {Path(victim).name or '?'} "
                    f"({totals[victim] / 1048576:.1f} GB RSS, {len(pids)} processes)")
                time.sleep(5)
                continue
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
