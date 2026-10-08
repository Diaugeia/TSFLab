"""Check a session's results against its task.

    check_success.py <task.json> <workspace> [--quiet]

Reproduce: success when results/results.json has a finite MSE and MAE for every
required cell and results/report.md exists. Reports the relative error against
the paper's numbers where the task gives them.
Autoresearch: success when a final method has three validation seeds that beat
the target on every prediction length and exactly one test read per length.
Benchmark: success when results/results.json has a finite MSE and MAE for every
listed method and results/report.md exists; methods are matched by name, ignoring
case and non-alphanumeric characters.
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


def key(r: dict) -> tuple:
    """A cell is identified by dataset and horizon, plus backbone and variant for strategies."""
    try:
        h = int(r.get("pred_len", -1))
    except (TypeError, ValueError):
        h = -1
    return (str(r.get("dataset", "")).lower(), h,
            str(r.get("backbone", "") or "").lower(), str(r.get("variant", "") or "").lower())


def check_reproduce(task: dict, ws: Path) -> dict:
    rows = load_results(ws)
    strategy = task.get("kind") == "strategy"
    got = {}
    for r in rows:
        k = key(r)
        got[k if strategy else k[:2] + ("", "")] = r
    cells, missing = [], []
    for c in task["cells"]:
        r = got.get(key(c))
        ok = bool(r) and finite(r.get("mse")) and finite(r.get("mae"))
        cell = {"dataset": c["dataset"], "pred_len": c["pred_len"], "ok": ok}
        if strategy:
            cell.update(backbone=c.get("backbone"), variant=c.get("variant"))
        if ok:
            cell.update(mse=r["mse"], mae=r["mae"])
            for m in ("mse", "mae"):
                ref = c.get(f"paper_{m}")
                if finite(ref) and ref:
                    cell[f"rel_err_{m}"] = (r[m] - ref) / ref
            ref = c.get("paper_mse")
            # More than 3x off the paper usually means another scale (e.g. unnormalized data).
            if finite(ref) and ref and not (ref / 3 <= r["mse"] <= ref * 3):
                cell["scale_suspect"] = True
        else:
            missing.append(f"{c['dataset']}@{c['pred_len']}" + (f"/{c.get('backbone')}/{c.get('variant')}" if strategy else ""))
        cells.append(cell)
    report = (ws / "results" / "report.md").exists()
    errs = [abs(c["rel_err_mse"]) for c in cells if "rel_err_mse" in c]
    gains = None
    if strategy:
        # Relative MSE gain of the strategy per (dataset, horizon, backbone), measured and reported.
        gains = []
        for c in cells:
            if c.get("variant") != "with":
                continue
            base = next((b for b in cells if b.get("variant") == "without" and b["dataset"] == c["dataset"]
                         and b["pred_len"] == c["pred_len"] and b.get("backbone") == c.get("backbone")), None)
            ref_w = next((x for x in task["cells"] if key(x) == key(c)), {})
            ref_b = next((x for x in task["cells"] if base and key(x) == key(base)), {})
            g = {"dataset": c["dataset"], "pred_len": c["pred_len"], "backbone": c.get("backbone")}
            if c.get("ok") and base and base.get("ok") and base["mse"]:
                g["gain"] = (base["mse"] - c["mse"]) / base["mse"]
            if finite(ref_w.get("paper_mse")) and finite(ref_b.get("paper_mse")) and ref_b["paper_mse"]:
                g["paper_gain"] = (ref_b["paper_mse"] - ref_w["paper_mse"]) / ref_b["paper_mse"]
            gains.append(g)
    return {
        "gains": gains,
        "success": not missing and report,
        "missing": missing,
        "report": report,
        "mean_abs_rel_err_mse": sum(errs) / len(errs) if errs else None,
        "scale_suspect": sum(1 for c in cells if c.get("scale_suspect")),
        "cells": cells,
    }


def check_autoresearch(task: dict, ws: Path) -> dict:
    rows = load_results(ws)
    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(str(r.get("method")), []).append(r)
    target = task.get("target_val_mse")
    if target is None:
        # The agent measures the target itself: mean validation MSE of the target method.
        tr = by_method.pop(task["target_method"], [])
        targets = {}
        for p in task["pred_lens"]:
            v = [r["mse"] for r in tr if r.get("split") == "val" and int(r.get("pred_len", -1)) == p and finite(r.get("mse"))]
            targets[str(p)] = sum(v) / len(v) if v else -math.inf
    else:
        targets = target if isinstance(target, dict) else {str(p): target for p in task["pred_lens"]}
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
    return {"success": best is not None, "method": best, "n_candidates": len(by_method), "targets": targets}


def check_benchmark(task: dict, ws: Path) -> dict:
    def norm(name) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())
    got = {norm(r.get("method")): r for r in load_results(ws)}
    methods, missing = [], []
    for m in task["methods"]:
        r = got.get(norm(m))
        ok = bool(r) and finite(r.get("mse")) and finite(r.get("mae"))
        methods.append({"method": m, "ok": ok, **({"mse": r["mse"], "mae": r["mae"]} if ok else {})})
        if not ok:
            missing.append(m)
    report = (ws / "results" / "report.md").exists()
    return {"success": not missing and report, "missing": missing, "report": report, "methods": methods}


def main() -> None:
    task = json.loads(Path(sys.argv[1]).read_text())
    ws = Path(sys.argv[2])
    quiet = "--quiet" in sys.argv
    check = {"reproduce": check_reproduce, "benchmark": check_benchmark}.get(task["task"], check_autoresearch)
    out = check(task, ws)
    if not quiet:
        print(json.dumps(out, indent=2))
    sys.exit(0 if out["success"] else 1)


if __name__ == "__main__":
    main()
