"""Check a session's results against its task.

    check_success.py <task.json> <workspace> [--quiet]

Reproduce: success when results/results.json has a finite MSE and MAE for every
required cell and results/report.md exists. Reports the relative error against
the paper's numbers where the task gives them.
Autoresearch: success when a final method has three validation seeds that beat
the target on every prediction length and exactly one test read per length.
Exit code 0 on success, 1 otherwise; prints a JSON summary unless --quiet.
"""

import json
import math
import sys
from pathlib import Path


def load_results(ws: Path) -> list[dict]:
    p = ws / "results" / "results.json"
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def check_reproduce(task: dict, ws: Path) -> dict:
    rows = load_results(ws)
    got = {(str(r.get("dataset", "")).lower(), int(r.get("pred_len", -1))): r for r in rows if "pred_len" in r}
    cells, missing = [], []
    for c in task["cells"]:
        r = got.get((c["dataset"].lower(), c["pred_len"]))
        ok = bool(r) and finite(r.get("mse")) and finite(r.get("mae"))
        cell = {"dataset": c["dataset"], "pred_len": c["pred_len"], "ok": ok}
        if ok:
            cell.update(mse=r["mse"], mae=r["mae"])
            for m in ("mse", "mae"):
                ref = c.get(f"paper_{m}")
                if finite(ref) and ref:
                    cell[f"rel_err_{m}"] = (r[m] - ref) / ref
        else:
            missing.append(f"{c['dataset']}@{c['pred_len']}")
        cells.append(cell)
    report = (ws / "results" / "report.md").exists()
    errs = [abs(c["rel_err_mse"]) for c in cells if "rel_err_mse" in c]
    return {
        "success": not missing and report,
        "missing": missing,
        "report": report,
        "mean_abs_rel_err_mse": sum(errs) / len(errs) if errs else None,
        "cells": cells,
    }


def check_autoresearch(task: dict, ws: Path) -> dict:
    rows = load_results(ws)
    target = task["target_val_mse"]
    targets = target if isinstance(target, dict) else {str(p): target for p in task["pred_lens"]}
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(str(r.get("method")), []).append(r)
    best = None
    for method, rs in by_method.items():
        ok = True
        for p in task["pred_lens"]:
            val = [r for r in rs if r.get("split") == "val" and int(r.get("pred_len", -1)) == p and finite(r.get("mse"))]
            test = [r for r in rs if r.get("split") == "test" and int(r.get("pred_len", -1)) == p]
            seeds = {r.get("seed") for r in val}
            mean_val = sum(r["mse"] for r in val) / len(val) if val else math.inf
            if len(seeds) < 3 or mean_val >= targets[str(p)] or len(test) != len(seeds):
                ok = False
        if ok:
            best = method
    return {"success": best is not None, "method": best, "n_candidates": len(by_method)}


def main() -> None:
    task = json.loads(Path(sys.argv[1]).read_text())
    ws = Path(sys.argv[2])
    quiet = "--quiet" in sys.argv
    out = check_reproduce(task, ws) if task["task"] == "reproduce" else check_autoresearch(task, ws)
    if not quiet:
        print(json.dumps(out, indent=2))
    sys.exit(0 if out["success"] else 1)


if __name__ == "__main__":
    main()
