"""Per-session flags and clean group numbers for the paper.

    python3 clean_tables.py <runs-root> <out-root>... > clean_numbers.json

Flags per session:
  cudnn   the agent's stream contains CUDNN_STATUS_NOT_INITIALIZED: the session
          ran in the broken environment (uv run re-synced cu13 cuDNN next to the
          cu126 torch) and its cost, time, and fidelity measure that, not the arm.
  shared  fraction of the session's wall-clock time during which another
          session of the campaign ran on the same GPU, from every session's
          start, duration, and assigned GPU under <runs-root> (all campaigns).
Clean groups drop cudnn sessions (the same rule for every arm). Token, cost,
validity, and fidelity use all clean sessions; wall-clock and GPU time use only
clean sessions with shared == 0 and report how many those are.
"""
import json, statistics, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_tables import session, STAGES  # noqa: E402

def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def spans(runs_root: Path):
    out = {}
    for m in runs_root.glob("*/*/metrics.json"):
        meta = json.loads((m.parent / "meta.json").read_text())
        met = json.loads(m.read_text())
        if not meta.get("started"):
            continue
        t0 = ts(meta["started"])
        out[str(m.parent)] = (str(meta.get("train_gpu")), t0, t0 + (met.get("wall_seconds") or 0))
    return out

def shared_fraction(key, sp):
    g, a, b = sp[key]
    if b <= a:
        return 0.0
    iv = sorted((max(a, x0), min(b, x1)) for k, (h, x0, x1) in sp.items() if k != key and h == g and x0 < b and x1 > a)
    tot, cur = 0.0, None
    for s, e in iv:
        if cur and s <= cur[1]:
            cur = (cur[0], max(cur[1], e))
        else:
            if cur:
                tot += cur[1] - cur[0]
            cur = (s, e)
    if cur:
        tot += cur[1] - cur[0]
    return tot / (b - a)

def cudnn(run: Path):
    return any("CUDNN_STATUS_NOT_INITIALIZED" in p.read_text(errors="ignore") for p in run.glob("stream.*.jsonl"))

def summarize(ss):
    if not ss:
        return None
    mean = lambda k, xs=ss: statistics.mean(x[k] for x in xs) if xs else None
    errs = [e for x in ss for e in x["errs"]]
    excl = [x for x in ss if x["shared"] == 0]
    return {
        "n": len(ss), "tasks": dict(sorted(defaultdict(int, {t: sum(x["task"] == t for x in ss) for t in {x["task"] for x in ss}}).items())),
        "completion": mean("completion"), "valid": mean("valid"),
        "within5": (sum(e <= 0.05 for e in errs) / len(errs)) if errs else None, "n_valid_cells": len(errs),
        "tokens_m": mean("tokens_m"), "cost": mean("cost"), "turns": mean("turns"), "interventions": mean("interventions"),
        "stage_tokens_m": {k: statistics.mean(x["stage_tokens_m"][k] for x in ss) for k in STAGES},
        "n_exclusive": len(excl), "wall_h_exclusive": mean("wall_h", excl), "gpu_h_exclusive": mean("gpu_h", excl),
        "wall_h_all": mean("wall_h"), "not_produced": sum(not x["produced"] for x in ss),
        "err_pct": sorted(round(100 * e, 2) for e in errs),
    }

def main():
    runs_root = Path(sys.argv[1]); sp = spans(runs_root)
    ss = []
    for root in sys.argv[2:]:
        for meta in sorted(Path(root).glob("*/meta.json")):
            s = session(meta.parent)
            if not s:
                continue
            s["root"] = Path(root).name
            s["cudnn"] = cudnn(meta.parent)
            s["shared"] = round(shared_fraction(str(meta.parent), sp), 3) if str(meta.parent) in sp else None
            ss.append(s)
    groups = defaultdict(lambda: {"raw": [], "clean": []})
    for s in ss:
        env = s["arm"] if s["arm"] in ("tslib", "tfb", "empty") else f"{s['arm']}@{s['version']}"
        for kind in (s["kind"], "all"):
            k = " | ".join((s["agent"], s["model"] or s["agent"], env, kind))
            groups[k]["raw"].append(s)
            if not s["cudnn"]:
                groups[k]["clean"].append(s)
    out = {k: {"raw": summarize(v["raw"]), "clean": summarize(v["clean"])} for k, v in sorted(groups.items())}
    per = [{k: s[k] for k in ("root", "name", "task", "arm", "version", "model", "cudnn", "shared", "valid", "tokens_m", "cost", "wall_h", "produced")} for s in ss]
    print(json.dumps({"sessions": per, "groups": out}, indent=1))

main()
