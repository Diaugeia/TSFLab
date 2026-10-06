"""Collect finished sessions into one table: aggregate.py <out-root>... > sessions.csv

One row per session directory (one that holds meta.json). Columns: task, kind,
arm, version, agent, model, completion (share of required cells), success,
validity by each reviewer and their agreement, mean |relative MSE error| against
the paper, strategy gain vs paper gain, wall hours, GPU-busy hours, tokens,
cost at GPT-6 Luna prices, discovery token share, tool calls, interventions,
and code added.
"""

import csv
import json
import sys
from pathlib import Path

COLS = ["session", "task", "kind", "arm", "version", "agent", "model", "completion", "success", "valid_a",
        "valid_b", "agree", "valid", "mean_abs_rel_err_mse", "gain", "paper_gain", "wall_h", "gpu_busy_h",
        "tokens_m", "cost_usd_luna", "discovery_share", "read_calls", "write_calls", "run_calls",
        "interventions", "code_lines_added", "discovery_errors", "gpu_overrides"]


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def row(run: Path) -> dict:
    meta = load(run / "meta.json") or {}
    m = load(run / "metrics.json") or {}
    s = (m.get("success") or load(run / "success.json")) or {}
    cells = s.get("cells") or []
    ra, rb = load(run / "review.a.json") or {}, load(run / "review.b.json") or {}
    rc = load(run / "review.c.json") or {}
    va, vb, vc = ra.get("valid"), rb.get("valid"), rc.get("valid")
    gains = [g for g in (s.get("gains") or []) if "gain" in g]
    pg = [g for g in (s.get("gains") or []) if "paper_gain" in g]
    tok = sum(sum(v.values()) for v in (m.get("tokens_by_action") or {}).values())
    calls = m.get("tool_calls") or {}
    code = (m.get("code_changes") or {}).get("code", {})
    errs = (ra.get("discovery_errors") or []) if isinstance(ra.get("discovery_errors"), list) else []
    return {
        "session": run.name, "task": meta.get("task", "").removesuffix(".json"), "kind": None,
        "arm": meta.get("arm"), "version": meta.get("tsflab_commit", "")[:8], "agent": meta.get("agent", "claude"),
        "model": meta.get("model"),
        "completion": round(sum(c.get("ok", False) for c in cells) / len(cells), 3) if cells else 0.0,
        "success": s.get("success"), "valid_a": va, "valid_b": vb,
        "agree": (va == vb) if va is not None and vb is not None else None,
        # Two reviewers decide; a third breaks a disagreement.
        "valid": (va if va == vb else vc) if va is not None and vb is not None else va,
        "mean_abs_rel_err_mse": s.get("mean_abs_rel_err_mse"),
        "gain": round(sum(g["gain"] for g in gains) / len(gains), 4) if gains else None,
        "paper_gain": round(sum(g["paper_gain"] for g in pg) / len(pg), 4) if pg else None,
        "wall_h": round((m.get("wall_seconds") or 0) / 3600, 3), "gpu_busy_h": round((m.get("gpu_busy_seconds") or 0) / 3600, 3),
        "tokens_m": round(tok / 1e6, 3), "cost_usd_luna": round(m.get("cost_usd_luna") or 0, 4),
        "discovery_share": round((m.get("discovery") or {}).get("token_share") or 0, 3),
        "read_calls": calls.get("read", 0), "write_calls": calls.get("write", 0), "run_calls": calls.get("run", 0),
        "interventions": m.get("interventions"), "code_lines_added": code.get("added", 0),
        "discovery_errors": len(errs),
        "gpu_overrides": m.get("gpu_overrides"),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    w = csv.DictWriter(sys.stdout, fieldnames=COLS)
    w.writeheader()
    for root in sys.argv[1:]:
        for meta in sorted(Path(root).glob("*/meta.json")):
            if not (meta.parent / "metrics.json").exists():
                continue  # still running
            r = row(meta.parent)
            t = load(here / "tasks" / f"{r['task']}.json") or {}
            r["kind"] = t.get("kind", "method")
            w.writerow(r)


if __name__ == "__main__":
    main()
