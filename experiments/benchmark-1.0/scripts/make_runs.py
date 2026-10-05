"""Generate the TSFLab 1.0 static-benchmark run files.

The sweep loader takes one global product, does not read the dataset cards'
``[protocol]``, and does not fill data-dependent model parameters. This script
does both: for every dataset it writes an overlay (``alias``, channel counts,
batch size) and for every (dataset group, model class, parameter signature) one
run file whose ``[sweep.extend]`` lists the models before the datasets, so the
overlay wins the merge (``experiments/config/loader.py``).

    uv run python experiments/benchmark-1.0/scripts/make_runs.py --phase pilot
    uv run python experiments/benchmark-1.0/scripts/make_runs.py --phase main

Outputs go to ``experiments/benchmark-1.0/runs/<phase>/`` (not tracked); the
script and the execution policies next to it are the record.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SEED = 2024

# Dataset groups. Lookbacks and horizons come from each card's [protocol];
# phase "main" uses the first (shortest) lookback, phase "lookback" the rest.
GROUPS = {
    "long": ["etth1", "etth2", "ettm1", "ettm2", "electricity", "traffic", "weather", "solar",
             "wind", "aqshunyi", "aqwan", "czelan", "zafnoo", "exchange"],
    "short": ["ili", "covid19", "fred_md", "nn5", "nasdaq", "nyse", "wike2000"],
    "spatial": ["metr_la", "pems03", "pems04", "pems07", "pems08", "pems_bay"],
}
PILOT = {"long": ["etth1"], "spatial": ["pems08"]}
# Smoke: one epoch at the longest horizon on the extreme shapes (fewest channels,
# shortest lookback, most channels, most nodes) to catch shape and memory errors.
SMOKE = {"long": ["traffic"], "short": ["nasdaq", "ili", "wike2000"], "spatial": ["pems07"]}

# One batch size per dataset, shared by every model (training settings are part
# of the protocol fingerprint, so a per-model batch would split the cohort).
def batch_size(channels: int, spatial: bool) -> int:
    if spatial:
        return 64
    if channels <= 32:
        return 128
    if channels <= 400:
        return 32
    return 16

# Cells a model's own parameter schema rejects; recorded in plan.json as not run.
EXCLUDE = {
    ("CALF", "wike2000"): "CALF caps enc_in at 1024 (calf/spec.py); wike2000 has 2000 channels",
}
# DRAGON refuses graphs above 32,768 nodes in training_setup (after construction, so --validate cannot see it).
for _d in ("traffic", "electricity", "solar", "covid19", "nn5", "wike2000"):
    EXCLUDE[("DRAGON", _d)] = ("DRAGON's dense personalized-PageRank graph exceeds 32,768 nodes on this dataset "
                               "(dragon/model.py raises before epoch 1)")

# Coverage first: each model gets one primary category from its card tags (first
# match wins). Tier 1 = the well-known representatives below; a category with
# none of them gets one model (cards tagged "baseline" first, then the newest
# paper). Tier 2 = the rest.
CATEGORIES = [
    ("llm", {"llm", "gpt2", "pretrained-backbone"}),
    ("diffusion", {"diffusion", "generative"}),
    ("ssm", {"ssm", "mamba"}),
    ("gnn", {"gnn", "graph-learning", "graph-attention"}),
    ("transformer", {"transformer"}),
    ("rnn", {"rnn"}),
    ("cnn", {"cnn", "dilated-convolution"}),
    ("mlp", {"mlp", "mixer"}),
    ("linear", {"linear"}),
    ("classical", {"statistical", "tree", "boosting", "ensemble"}),
]
TIER1 = set("""
DLinear RLinear NLinear SparseTSF CycleNet FITS
PatchTST iTransformer Informer Autoformer FEDformer Crossformer TimeXer
TimesNet TimeMixer TSMixer NBeats NHiTS SOFTS TQNet
ModernTCN MICN SCINet TCNForecasterTS SegRNN LSTMForecasterTS
S_Mamba MambaTS S4 CrossGNN MSGNet Kronos
CALF AutoTimes LLM4TS TALON CDPM NsDiff
XGBoostTS LightGBMTS RidgeRegressionTS ExpSmoothingTS
QuantileDLinear QuantilePatchTST DeepAR TiRex
GWNet STID STAEformer AGCRN DCRNN MTGNN D2STGNN STGCN
""".split())

# Datasets queued after the light ones (many channels or nodes).
HEAVY = {"electricity", "traffic", "solar", "covid19", "wike2000", "pems07"}
PRIORITY = {("t1", "light"): 40, ("t1", "heavy"): 30, ("t2", "light"): 20, ("t2", "heavy"): 10}


def categorize(rows: list[dict]) -> None:
    """Set m["category"] and m["tier"] in place."""
    cards = {}
    for path in (ROOT / "src/tsflab/models").glob("*/card.toml"):
        cfg = tomllib.loads(path.read_text())
        cards[cfg.get("name")] = cfg
    for m in rows:
        cfg = cards.get(m["name"], {})
        tags = set(cfg.get("tags", []))
        m["category"] = next((c for c, keys in CATEGORIES if tags & keys), "other")
        m["_rank"] = ("baseline" not in tags, -int(cfg.get("paper", {}).get("year") or 0), m["name"])
    covered = {(model_class(m), m["category"]) for m in rows if m["name"] in TIER1}
    for m in sorted(rows, key=lambda r: r["_rank"]):
        key = (model_class(m), m["category"])
        if m["name"] in TIER1:
            m["tier"] = "t1"
        elif key not in covered:
            m["tier"] = "t1"
            covered.add(key)
        else:
            m["tier"] = "t2"

# Models whose design does not fit a whole dataset group (the model raises at
# construction); recorded in plan.json as not run, with the reason.
EXCLUDE_GROUPS = {
    ("GMRL", "spatial"): "needs several source modalities per location (enc_in = num_sources x num_locations); "
                         "PEMS / METR-LA have one",
    ("MoSSL", "spatial"): "needs several modalities per node (enc_in = num_modalities x num_nodes); PEMS / METR-LA have one",
    ("MGSFformer", "spatial"): "needs seq_len to be a multiple of 24; the spatial protocol uses 12",
    ("PCATransformer", "long"): "forecasts one target from reduced covariates (task.features = 'MS'); the protocol is 'M'",
    ("PCATransformer", "short"): "forecasts one target from reduced covariates (task.features = 'MS'); the protocol is 'M'",
}

STEPS_PER_DAY_KEYS = ("num_time_in_day", "steps_per_day", "time_in_day_size")
CHANNEL_KEYS = ("enc_in", "dec_in", "c_out")


def card(name: str) -> dict:
    return tomllib.loads((ROOT / "catalog/datasets" / name / "card.toml").read_text())


def models() -> list[dict]:
    out = subprocess.run([sys.executable, "-m", "tsflab.cli.main", "catalog", "list", "--kind", "model", "--json"],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout
    presets = {}
    for path in sorted((ROOT / "configs/models").glob("*.toml")):
        cfg = tomllib.loads(path.read_text())
        presets[cfg["model"]["name"]] = (path, cfg["model"].get("params", {}))
    rows = []
    for item in json.loads(out):
        path, params = presets[item["name"]]
        rows.append({**item, "preset": path, "params": params})
    return rows


def model_class(m: dict) -> str | None:
    """Benchmark class of a model, or None when no static dataset fits it."""
    caps, modes = set(m["capabilities"]), set(m["task_modes"])
    if "time_series" in modes:
        if "quantile-output" in caps:
            return "quantile"
        if "distribution-output" in caps:
            return "distribution"
        return "point"
    if "spatiotemporal" in modes:
        return "spatial"
    return None  # covariate-only (air-quality models): no static dataset has the covariates


CLASS = {
    "point": {"loss": "mse", "metrics": None},
    "quantile": {"loss": "quantile", "metrics": ["crps", "wql", "coverage_80", "width_80", "mae", "mse"]},
    "distribution": {"loss": "nll_gaussian", "metrics": ["crps", "wql", "coverage_80", "width_80", "mae", "mse"]},
    "spatial": {"loss": "masked_mae", "metrics": None},
}


def signature(m: dict, spatial: bool) -> tuple[str, ...]:
    keys = CHANNEL_KEYS + (("num_nodes",) + STEPS_PER_DAY_KEYS if spatial else ())
    return tuple(k for k in keys if k in m["params"])


# --- minimal TOML writer (strings, numbers, booleans, lists) ---
def value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(value(x) for x in v) + "]"
    raise TypeError(v)


def dump(doc: dict) -> str:
    lines = [f"{k} = {value(v)}" for k, v in doc.items() if not isinstance(v, dict)]
    def table(prefix, d):
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        if scalars:
            lines.append(f"\n[{prefix}]")
            lines.extend(f"{k} = {value(v)}" for k, v in scalars.items())
        for k, v in d.items():
            if isinstance(v, dict):
                table(f"{prefix}.{k}", v)
    for k, v in doc.items():
        if isinstance(v, dict):
            table(k, v)
    return "\n".join(lines) + "\n"


def write(path: Path, doc: dict, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {header}\n# Generated by experiments/benchmark-1.0/scripts/make_runs.py; do not edit.\n" + dump(doc))


def overlay(out: Path, name: str, sig: tuple[str, ...], spatial: bool, batch_scale: float = 1.0) -> Path:
    shape = card(name)["shape"]
    channels = int(shape["channels"])
    params = {k: channels for k in sig if k in CHANNEL_KEYS + ("num_nodes",)}
    params.update({k: 288 for k in sig if k in STEPS_PER_DAY_KEYS})  # 5-minute data
    batch = max(1, int(batch_size(channels, spatial) * batch_scale))
    suffix = "" if batch_scale == 1.0 else f"__b{batch}"  # a reduced batch is part of the file name (flagged)
    path = out / "overlays" / f"{name}__{'-'.join(sig) or 'none'}{suffix}.toml"
    doc = {"extends": [os.path.relpath(ROOT / "configs/datasets" / f"{name}.toml", path.parent)],
           "dataset": {"alias": name},
           "training": {"batch_size": batch}}
    if params:
        doc["model"] = {"params": params}
    write(path, doc, f"{name}: alias, channel counts, batch size")
    return path


def horizons(phase: str, preds: list[int]) -> list[int]:
    return {"pilot": preds[:1], "smoke": preds[-1:]}.get(phase, preds)


def root_doc(phase: str, tag: str, c: str, spatial: bool, seq: int) -> dict:
    doc = {
        "extends": [],
        "experiment": {"description": f"TSFLab 1.0 static benchmark ({phase}): {tag}",
                       "random_seed": SEED, "work_dir": "./work_dirs",
                       "runtime": {"device": "cuda", "use_multi_gpu": False,
                                   "device_ids": [0], "num_workers": 4}},
        "task": {"mode": "spatiotemporal" if spatial else "time_series",
                 "seq_len": seq, "label_len": seq // 2, "features": "M"},
        "training": {"loss": CLASS[c]["loss"]},
    }
    if phase == "smoke":
        doc["training"].update({"epochs": 1, "patience": 1})
    evaluation = {"enable_profile": phase != "pilot"}  # params, MACs, inference VRAM and latency
    if CLASS[c]["metrics"]:
        evaluation["metrics"] = CLASS[c]["metrics"]
    doc["evaluation"] = evaluation
    return doc


def check(out: Path, m: dict, ov: Path, doc: dict, preds) -> str | None:
    """Load one (model, dataset) cell per horizon with TSFLab's loader; return the first error."""
    from tsflab.experiments.config.loader import load_config
    path = out / "_check" / "cell.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    for pred in preds:
        cell = {**doc, "extends": [os.path.relpath(p, path.parent) for p in
                                   (ROOT / "configs/base.toml", m["preset"], ov)],
                "task": {**doc["task"], "pred_len": pred}}
        path.write_text(dump(cell))
        try:
            load_config(str(path))
        except Exception as exc:  # the loader's own message is the exclusion reason
            return f"pred_len {pred}: {type(exc).__name__}: " + " ".join(str(exc).split())[:300]
    return None


def generate(phase: str, out: Path, validate: bool = False, only: set[str] | None = None) -> list[dict]:
    groups = {"pilot": PILOT, "smoke": SMOKE}.get(phase, GROUPS)
    by_class: dict[str, list[dict]] = {}
    skipped = []
    rows = models()
    categorize(rows)
    if only:
        missing = only - {m["name"] for m in rows}
        if missing:
            raise SystemExit(f"unknown models: {sorted(missing)}")
        rows = [m for m in rows if m["name"] in only]
    for m in rows:
        c = model_class(m)
        (by_class.setdefault(c, []) if c else skipped).append(m)
    plan = []
    for group, datasets in groups.items():
        for (name, g), reason in EXCLUDE_GROUPS.items():
            if g == group:
                for d in datasets:
                    EXCLUDE.setdefault((name, d), reason)
        spatial = group == "spatial"
        classes = ["spatial"] if spatial else ["point", "quantile", "distribution"]
        proto = {d: card(d)["protocol"] for d in datasets}
        for c in classes:
            sigs: dict[tuple, list[dict]] = {}
            for m in by_class.get(c, []):
                sigs.setdefault(signature(m, spatial), []).append(m)
            if validate:
                for sig, members in sigs.items():
                    for d in datasets:
                        ov = overlay(out, d, sig, spatial)
                        seqs = proto[d]["seq_lens"]
                        for seq in (seqs[:1] if phase in ("pilot", "smoke", "main") else seqs[1:]):
                            preds = horizons(phase, proto[d]["pred_lens"])
                            doc = root_doc(phase, "check", c, spatial, seq)
                            for m in members:
                                if (m["name"], d) not in EXCLUDE:
                                    err = check(out, m, ov, doc, preds)
                                    if err:
                                        EXCLUDE[(m["name"], d)] = f"seq_len {seq}, {err}"
            # models with excluded cells get their own run files, without those datasets
            split: dict[tuple, list[dict]] = {}
            for sig, members in sigs.items():
                for m in members:
                    drop = tuple(sorted(d for d in datasets if (m["name"], d) in EXCLUDE))
                    split.setdefault((sig, drop), []).append(m)
            tiers: dict[tuple, list[dict]] = {}
            for (sig, drop), members in split.items():
                for m in members:
                    tiers.setdefault((sig, drop, m["tier"]), []).append(m)
            for (sig, drop, tier), members in sorted(tiers.items()):
              for weight in ("light", "heavy"):
                # cells with the same lookback set share one run file
                buckets: dict[tuple, list[str]] = {}
                for d in (d for d in datasets if d not in drop and (d in HEAVY) == (weight == "heavy")):
                    seqs = proto[d]["seq_lens"]
                    chosen = seqs[:1] if phase in ("pilot", "smoke", "main") else seqs[1:]
                    for s in chosen:
                        preds = horizons(phase, proto[d]["pred_lens"])
                        buckets.setdefault((s, tuple(preds)), []).append(d)
                for (seq, preds), ds in sorted(buckets.items()):
                    tag = (f"{group}__{c}__{'-'.join(sig) or 'none'}__sl{seq}__{tier}-{weight}"
                           + (f"__no-{'-'.join(drop)}" if drop else ""))
                    path = out / f"{tag}.toml"
                    ovs = [os.path.relpath(overlay(out, d, sig, spatial), out) for d in ds]
                    doc = root_doc(phase, tag, c, spatial, seq)
                    doc["extends"] = [os.path.relpath(ROOT / "configs/base.toml", out)]
                    doc["sweep"] = {"extend": {"models": [os.path.relpath(m["preset"], out) for m in members],
                                               "datasets": ovs},
                                    "task": {"pred_len": list(preds)}}
                    write(path, doc, f"{len(members)} models x {len(ds)} datasets x {len(preds)} horizons")
                    plan.append({"file": path.name, "priority": PRIORITY[(tier, weight)], "tier": tier,
                                 "weight": weight, "group": group, "class": c, "signature": list(sig),
                                 "seq_len": seq, "models": [m["name"] for m in members],
                                 "categories": sorted({m["category"] for m in members}), "datasets": ds,
                                 "pred_lens": list(preds), "cells": len(members) * len(ds) * len(preds)})
    excluded = [{"model": m, "dataset": d, "reason": r} for (m, d), r in sorted(EXCLUDE.items())
                if any(d in ds for ds in groups.values())]
    shutil.rmtree(out / "_check", ignore_errors=True)
    plan.sort(key=lambda p: (-p["priority"], p["file"]))
    tier1 = sorted((m["category"], m["name"]) for m in rows if m.get("tier") == "t1" and model_class(m))
    (out / "plan.json").write_text(json.dumps({"phase": phase, "seed": SEED, "files": plan, "tier1": tier1,
                                               "not_run": sorted(m["name"] for m in skipped),
                                               "excluded_cells": excluded}, indent=1))
    return plan


def generate_cells(phase: str, out: Path, cells: dict[tuple[str, str], list[int]],
                   batch_scale: float = 1.0) -> list[dict]:
    """One run file per (model, dataset) for exactly the listed horizons (re-runs).

    ``phase`` names the protocol phase whose lookback is used (``main`` -> the
    first card lookback, ``lookback`` -> the others).
    """
    rows = {m["name"]: m for m in models()}
    categorize(list(rows.values()))
    plan = []
    for (name, d), preds in sorted(cells.items()):
        m = rows[name]
        c = model_class(m)
        spatial = d in GROUPS["spatial"]
        sig = signature(m, spatial)
        proto = card(d)["protocol"]
        seqs = proto["seq_lens"][:1] if phase in ("pilot", "smoke", "main") else proto["seq_lens"][1:]
        weight = "heavy" if d in HEAVY else "light"
        for seq in seqs:
            tag = f"{name}__{d}__sl{seq}" + ("" if batch_scale == 1.0 else f"__bscale{batch_scale:g}")
            path = out / f"{tag}.toml"
            ov = overlay(out, d, sig, spatial, batch_scale)
            doc = root_doc(phase, tag, c, spatial, seq)
            doc["extends"] = [os.path.relpath(ROOT / "configs/base.toml", out)]
            doc["sweep"] = {"extend": {"models": [os.path.relpath(m["preset"], out)],
                                       "datasets": [os.path.relpath(ov, out)]},
                            "task": {"pred_len": sorted(preds)}}
            write(path, doc, f"re-run: {name} on {d}, horizons {sorted(preds)}, batch scale {batch_scale:g}")
            plan.append({"file": path.name, "priority": PRIORITY[(m["tier"], weight)], "tier": m["tier"],
                         "weight": weight, "group": "rerun", "class": c, "signature": list(sig),
                         "seq_len": seq, "models": [name], "categories": [m["category"]], "datasets": [d],
                         "pred_lens": sorted(preds), "cells": len(preds), "batch_scale": batch_scale})
    plan.sort(key=lambda p: (-p["priority"], p["file"]))
    (out / "plan.json").write_text(json.dumps({"phase": phase, "seed": SEED, "files": plan, "tier1": [],
                                               "not_run": [], "excluded_cells": []}, indent=1))
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["pilot", "smoke", "main", "lookback"], required=True)
    ap.add_argument("--out", type=Path, help="default: experiments/benchmark-1.0/runs/<phase>")
    ap.add_argument("--models", nargs="+", help="only these models (e.g. to re-check fixed models)")
    ap.add_argument("--cells", type=Path,
                    help="CSV with model,dataset,pred_len: write one re-run file per (model, dataset)")
    ap.add_argument("--batch-scale", type=float, default=1.0, help="with --cells: scale the batch (flagged)")
    ap.add_argument("--validate", action="store_true",
                    help="load every (model, dataset) cell with the TSFLab loader and exclude the rejected ones")
    args = ap.parse_args()
    out = (args.out or HERE.parent / "runs" / args.phase).resolve()
    shutil.rmtree(out, ignore_errors=True)
    if args.cells:
        import csv
        cells: dict[tuple[str, str], list[int]] = {}
        with args.cells.open() as stream:
            for rec in csv.DictReader(stream):
                cells.setdefault((rec["model"], rec["dataset"]), []).append(int(rec["pred_len"]))
        plan = generate_cells(args.phase, out, cells, args.batch_scale)
    else:
        plan = generate(args.phase, out, validate=args.validate, only=set(args.models or ()))
    total = sum(p["cells"] for p in plan)
    excluded = json.loads((out / "plan.json").read_text())["excluded_cells"]
    print(f"{len(plan)} run files, {total} cells, {len(excluded)} excluded (model, dataset) pairs -> {out}")
    for p in plan:
        print(f"  p{p['priority']}  {p['cells']:6d}  {p['file']}")


if __name__ == "__main__":
    main()
