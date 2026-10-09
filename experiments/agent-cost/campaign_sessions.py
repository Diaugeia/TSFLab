"""Export every finished session of the given roots as one JSON record (all measures, no grouping).

    python3 campaign_sessions.py <out-root>... > sessions.json

Reads meta.json, metrics.json (analyze.py), success.json (check_success.py), review.{a,b,c}.json,
process.json (process_stats.py), and quality.json (code_quality.py). Validity follows paper_tables.verdict:
two reviewers, a third on disagreement; a session whose values were not produced by a run has completion 0.

Per task kind:
  reproduce    completion = share of required cells with finite MSE/MAE; valid = the same cells if the session
               is valid; errs = |MSE - paper| / paper of valid cells (strategies: cells with the strategy)
  benchmark    completion = share of the 8 methods with a result; valid = the same if the session is valid;
               mse = {method: MSE} of valid sessions (for rankings)
  autoresearch completion = 1 if a final method beats the target on validation over three seeds; valid = the
               same if the reviewers accept the session; gain = mean relative validation improvement over the
               target across horizons; test_reads = test rows reported
"""

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_tables import load, verdict  # noqa: E402


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def ar_gain(run: Path, task: dict, success: dict):
    rows = load(run / "workspace" / "results" / "results.json") or []
    if not isinstance(rows, list):
        return None, 0
    targets = success.get("targets") or {}
    method = success.get("method")
    test_reads = sum(1 for r in rows if isinstance(r, dict) and r.get("split") == "test")
    if not method or not targets:
        return None, test_reads
    gains = []
    for p in task.get("pred_lens", []):
        v = [r["mse"] for r in rows if isinstance(r, dict) and r.get("method") == method and r.get("split") == "val"
             and int(r.get("pred_len", -1)) == p and finite(r.get("mse"))]
        t = targets.get(str(p))
        if v and finite(t) and t > 0:
            gains.append((t - statistics.mean(v)) / t)
    return (statistics.mean(gains) if gains else None), test_reads


def record(run: Path):
    meta, m = load(run / "meta.json"), load(run / "metrics.json")
    if not meta or not m:
        return None
    task = load(HERE / "tasks" / f"{meta.get('task', '').removesuffix('.json')}.json") or {}
    s = load(run / "success.json") or m.get("success") or {}
    kind = task.get("task", "reproduce")
    sub = task.get("kind", "method")
    produced = verdict(run, "reported") is not False
    session_valid = verdict(run) is True
    rec = {"name": run.name, "root": run.parent.name, "task": task.get("id"), "kind": kind, "sub": sub,
           "arm": meta.get("arm"), "version": (meta.get("tsflab_commit") or "")[:8],
           "agent": meta.get("agent", "claude"), "model": meta.get("model"), "produced": produced,
           "session_valid": session_valid}
    if kind == "reproduce":
        cells = s.get("cells") or []
        required = max(len(task.get("cells") or []), 1)
        ok = [c for c in cells if c.get("ok")] if produced else []
        v = ok if session_valid else []
        rec.update(completion=len(ok) / required, valid=len(v) / required,
                   errs=[abs(c["rel_err_mse"]) for c in v if c.get("rel_err_mse") is not None
                         and (sub == "method" or c.get("variant") == "with")])
    elif kind == "benchmark":
        ms = s.get("methods") or []
        ok = [x for x in ms if x.get("ok")] if produced else []
        rec.update(completion=len(ok) / max(len(task.get("methods") or []), 1),
                   valid=(len(ok) / max(len(task.get("methods") or []), 1)) if session_valid else 0.0,
                   mse={x["method"]: x["mse"] for x in ok} if session_valid else {})
    else:
        succ = bool(s.get("success")) and produced
        gain, reads = ar_gain(run, task, s)
        rec.update(completion=float(succ), valid=float(succ and session_valid), gain=gain, test_reads=reads,
                   final_method=s.get("method"))
    tok = sum(sum(v.values()) for v in (m.get("tokens_by_action") or {}).values())
    disc = (m.get("discovery") or {}).get("token_share")
    rec.update(tokens_m=tok / 1e6, read_share=disc, cost=m.get("cost_usd_luna") or 0.0,
               wall_h=(m.get("wall_seconds") or 0) / 3600, gpu_h=(m.get("gpu_busy_seconds") or 0) / 3600,
               turns=m.get("turns") or 0, interventions=m.get("interventions") or 0,
               stage_tokens=m.get("stage_tokens") or {})
    rec["cudnn"] = any("CUDNN_STATUS_NOT_INITIALIZED" in p.read_text(errors="ignore") for p in run.glob("stream.*.jsonl"))
    rec.update(load(run / "process.json") or {})
    q = load(run / "quality.json") or {}
    rec.update({f"code_{k}": v for k, v in q.items()})
    return rec


def main():
    out = [r for root in sys.argv[1:] for meta in sorted(Path(root).glob("*/meta.json")) if (r := record(meta.parent))]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
