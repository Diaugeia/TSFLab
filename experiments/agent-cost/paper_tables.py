"""Numbers for the paper's tab:main, tab:ablation, and fig:compare from finished sessions.

    python3 paper_tables.py <out-root>... > paper_numbers.json

Per session: completion is the share of required cells with finite MSE and MAE (zero when
the reviewers find the results were not produced by a run); valid cells are the completed
cells of a session the reviewers accept (two reviewers, a third breaks a disagreement);
a valid cell is within 5% when |MSE - paper MSE| / paper MSE <= 0.05 (for strategies, the
cells with the strategy). Groups are (agent, model, arm, version, kind); completion and
validity are means over sessions, the 5% share is pooled over valid cells, and cost columns
are means over sessions. Stage tokens and per-cell errors feed fig:compare.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["collect", "read", "implement", "check", "run", "compare"]
STAGE_MAP = {"before-first-stage": "collect", "done": "compare"}


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def verdict(run: Path, key: str | None = None):
    """Final reviewer decision on the whole session (key=None) or on one check."""
    rs = [load(run / f"review.{r}.json") or {} for r in "abc"]

    def one(r):
        if not r:
            return None
        if key is None:
            return r.get("valid")
        v = r.get(key)
        return v.get("ok") if isinstance(v, dict) else None

    a, b, c = (one(r) for r in rs)
    if a is None or b is None:
        return a if a is not None else b
    return a if a == b else c


def session(run: Path) -> dict | None:
    meta, m = load(run / "meta.json"), load(run / "metrics.json")
    if not meta or not m:
        return None
    s = m.get("success") or load(run / "success.json") or {}
    task = load(HERE / "tasks" / f"{meta.get('task', '').removesuffix('.json')}.json") or {}
    kind = task.get("kind", "method")
    cells = s.get("cells") or []
    required = max(len(task.get("cells") or []), 1)  # strategy tasks list the cells with and without it
    produced = verdict(run, "reported") is not False
    ok = [c for c in cells if c.get("ok")] if produced else []
    valid = verdict(run) is True
    vcells = ok if valid else []
    errs = [abs(c["rel_err_mse"]) for c in vcells
            if c.get("rel_err_mse") is not None and (kind == "method" or c.get("variant") == "with")]
    st = defaultdict(float)
    for k, v in (m.get("stage_tokens") or {}).items():
        st[STAGE_MAP.get(k, k)] += v
    tok = sum(sum(v.values()) for v in (m.get("tokens_by_action") or {}).values())
    return {
        "name": run.name, "task": task.get("id"), "kind": kind, "arm": meta.get("arm"),
        "version": (meta.get("tsflab_commit") or "")[:8], "agent": meta.get("agent", "claude"),
        "model": meta.get("model"), "completion": len(ok) / required, "valid": len(vcells) / required,
        "errs": errs, "wall_h": (m.get("wall_seconds") or 0) / 3600, "gpu_h": (m.get("gpu_busy_seconds") or 0) / 3600,
        "tokens_m": tok / 1e6, "cost": m.get("cost_usd_luna") or 0.0, "turns": m.get("turns") or 0,
        "interventions": m.get("interventions") or 0, "stage_tokens_m": {k: st.get(k, 0.0) / 1e6 for k in STAGES},
        "produced": produced,
    }


def summarize(ss: list[dict]) -> dict:
    mean = lambda k: statistics.mean(x[k] for x in ss)
    errs = [e for x in ss for e in x["errs"]]
    return {
        "n": len(ss), "completion": mean("completion"), "valid": mean("valid"),
        "within5": (sum(e <= 0.05 for e in errs) / len(errs)) if errs else None, "n_valid_cells": len(errs),
        "wall_h": mean("wall_h"), "gpu_h": mean("gpu_h"), "tokens_m": mean("tokens_m"), "cost": mean("cost"),
        "turns": mean("turns"), "interventions": mean("interventions"),
        "stage_tokens_m": {k: statistics.mean(x["stage_tokens_m"][k] for x in ss) for k in STAGES},
        "err_pct": sorted(round(100 * e, 2) for e in errs),
        "not_produced": sum(not x["produced"] for x in ss),
    }


def main() -> None:
    ss = [s for root in sys.argv[1:] for meta in sorted(Path(root).glob("*/meta.json"))
          if (s := session(meta.parent))]
    groups = defaultdict(list)
    for s in ss:
        env = s["arm"] if s["arm"] in ("tslib", "tfb", "empty") else f"{s['arm']}@{s['version']}"
        model = s["model"] or s["agent"]
        groups[(s["agent"], model, env, s["kind"])].append(s)
        groups[(s["agent"], model, env, "all")].append(s)
        groups[(s["agent"], model, env, "tasks:" + ",".join(sorted({"rhymix", "aosnet"} & {s["task"]})) or "-")].append(s)
    out = {" | ".join(k): summarize(v) for k, v in sorted(groups.items()) if not k[3].endswith(":")}
    print(json.dumps({"sessions": len(ss), "groups": out}, indent=1))


if __name__ == "__main__":
    main()
