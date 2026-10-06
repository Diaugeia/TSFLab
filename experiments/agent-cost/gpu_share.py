"""Measure how long each session shared its GPU with another session.

    python3 gpu_share.py <queue.log | runs-root> ... > gpu_share.tsv

Reads the given queue logs, or every *.queue*.log in a given directory (lines "<date> <time> start|done <name> gpu <g>")
and prints, per session, its GPU, start and end, the seconds another session ran on the same
GPU, and that time as a fraction of the session. A session still running ends now. GPU time and
wall-clock comparisons use only sessions with shared_fraction == 0 (or report both).
"""

import re
import sys
import time
from pathlib import Path

LINE = re.compile(r"^(\S+ \S+) (start|done) (\S+) gpu (\S+)")


def main() -> None:
    logs = []
    for a in map(Path, sys.argv[1:]):
        logs += sorted(a.glob("*.queue*.log")) if a.is_dir() else [a]
    span: dict[str, list] = {}
    for log in logs:
        for line in log.read_text().splitlines():
            m = LINE.match(line)
            if not m:
                continue
            t = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            kind, name, gpu = m.group(2), m.group(3), m.group(4)
            s = span.setdefault(name, [gpu, None, None])
            if kind == "start" and (s[1] is None or t < s[1]):
                s[0], s[1] = gpu, t
            if kind == "done" and (s[2] is None or t > s[2]):
                s[2] = t
    now = time.time()
    rows = [(n, g, t0, t1 if t1 and t1 > t0 else now) for n, (g, t0, t1) in span.items() if t0 is not None]
    print("name\tgpu\tstart\tend\tshared_seconds\tshared_fraction")
    for n, g, t0, t1 in sorted(rows, key=lambda r: r[2]):
        # Union of the other sessions' intervals on this GPU, clipped to this session.
        others = sorted((max(a, t0), min(b, t1)) for m, gg, a, b in rows if m != n and gg == g and a < t1 and b > t0)
        shared, cur = 0.0, None
        for a, b in others:
            if cur and a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                if cur:
                    shared += cur[1] - cur[0]
                cur = [a, b]
        if cur:
            shared += cur[1] - cur[0]
        fmt = lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x))
        print(f"{n}\t{g}\t{fmt(t0)}\t{fmt(t1)}\t{shared:.0f}\t{shared / max(t1 - t0, 1):.3f}")


if __name__ == "__main__":
    main()
