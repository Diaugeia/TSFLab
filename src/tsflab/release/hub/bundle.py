"""Package a finished run into a self-describing weights bundle and load it back.

A bundle is one directory, published unchanged under
``checkpoints/<track>/<dataset>/<model>/<run_id>/`` in the checkpoints model
repository (``Diaugeia/TSFLab-Checkpoints``)::

    manifest.json       identity, shapes, provenance, per-file SHA-256
    model.safetensors   the best checkpoint's state_dict
    record.json         the run's TSF-Core RunRecord (metrics, config, env)
    README.md           a model card rendered from the manifest

Packing and loading need ``torch`` and ``safetensors`` (``tsflab[hub]``).
"""

from __future__ import annotations

import json
from pathlib import Path

from tsflab.release.hub.fetch import fetch, sha256_file
from tsflab.release.hub.uri import HubURI, parse

MANIFEST_SCHEMA_VERSION = 1
WEIGHTS = "model.safetensors"
MANIFEST = "manifest.json"
RECORD = "record.json"
CARD = "README.md"
CHECKPOINT_NAMES = ("best_checkpoint.pth", "latest.pth")


def _require(module: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:
        raise RuntimeError(
            f"{module} is required for hub bundles; install `tsflab[hub]`"
        ) from exc


def _record_for(run: str | Path, work_root: Path) -> Path:
    candidate = Path(run)
    if candidate.suffix == ".json" and candidate.is_file():
        return candidate
    matches = sorted(work_root.glob(f"work_dirs/*/*/records/{run}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one record for run {run!r} under "
            f"{work_root / 'work_dirs'}, found {len(matches)}"
        )
    return matches[0]


def checkpoint_candidates(record: Path) -> list[Path]:
    """Checkpoint locations for one record, most specific first.

    The managed runner writes ``<work_dir>/_runs/<run_id>/result.json`` (with
    ``checkpoint_path``) and ``<work_dir>/_runs/<run_id>/checkpoints/``, next to
    the ``<work_dir>/<dataset>/<model>/records/<run_id>.json`` record. The older
    layout kept ``<work_dir>/<dataset>/<model>/checkpoints/<run_id>/``.
    """
    run_id = record.stem
    # <work_dir>/<dataset>/<model>/records/<run_id>.json
    work_dir = record.parents[3] if len(record.parents) > 3 else record.parent
    run_dir = work_dir / "_runs" / run_id
    candidates: list[Path] = []
    result = run_dir / "result.json"
    if result.is_file():
        try:
            recorded = json.loads(result.read_text(encoding="utf-8")).get("checkpoint_path")
        except (OSError, json.JSONDecodeError):
            recorded = None
        if recorded:
            # A run copied back from another machine keeps its absolute path;
            # fall back to the same file name inside the local run directory.
            candidates += [Path(recorded), run_dir / "checkpoints" / Path(recorded).name]
    for base in (run_dir / "checkpoints", record.parent.parent / "checkpoints" / run_id):
        candidates += [base / name for name in CHECKPOINT_NAMES]
    return candidates


def find_run(run: str | Path, work_root: Path) -> tuple[Path, Path]:
    """Return ``(record.json, checkpoint)`` for a run id or record path."""
    record = _record_for(run, work_root)
    candidates = checkpoint_candidates(record)
    for candidate in candidates:
        if candidate.is_file():
            return record, candidate
    searched = ", ".join(dict.fromkeys(str(c.parent) for c in candidates))
    raise FileNotFoundError(f"no checkpoint for run {record.stem!r}; looked in {searched}")


def _state_dict(checkpoint: Path) -> dict:
    torch = _require("torch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError(f"{checkpoint} does not contain a tensor state_dict")
    # safetensors rejects shared storage; clone so every tensor owns its bytes.
    return {key: value.detach().contiguous().clone() for key, value in state.items()}


def _card(manifest: dict) -> str:
    metrics = manifest.get("metrics", {})
    rows = "\n".join(
        f"| {name} | {value:.6g} |"
        for name, value in sorted(metrics.items())
        if isinstance(value, (int, float))
    )
    return f"""---
library_name: tsflab
tags: [time-series, forecasting, tsflab, {manifest['model']}]
---

# {manifest['model']} · {manifest['dataset_id']} · pred_len {manifest['pred_len']}

Trained with [TSFLab](https://github.com/Diaugeia/TSFLab)
{manifest.get('framework_version') or ''} (commit `{manifest.get('git_sha') or 'unknown'}`).

| Field | Value |
| --- | --- |
| run_id | `{manifest['run_id']}` |
| seq_len / pred_len | {manifest['seq_len']} / {manifest['pred_len']} |
| seed | {manifest['seed']} |

| Metric | Value |
| --- | --- |
{rows}

```python
from tsflab.release import hub
state_dict, manifest = hub.load_state_dict("hf://<repo>@<revision>/{manifest['path']}")
```
"""


def pack(record_path: Path, checkpoint: Path, out_dir: Path) -> dict:
    """Write a bundle for one run into ``out_dir`` and return its manifest."""
    save_file = _require("safetensors.torch").save_file
    record = json.loads(record_path.read_text(encoding="utf-8"))
    config = record.get("config") or {}
    env = record.get("env") or {}
    result = (record.get("results") or [{}])[0]
    run_id = result.get("run_id") or record_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(_state_dict(checkpoint), str(out_dir / WEIGHTS))
    (out_dir / RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model": record["model"],
        "dataset_id": record["dataset_id"],
        "track": record.get("track"),
        "run_id": run_id,
        "seed": record.get("seed"),
        "seq_len": config.get("seq_len"),
        "label_len": config.get("label_len"),
        "pred_len": config.get("pred_len", result.get("horizon")),
        "features": config.get("features"),
        "metrics": {k: v for k, v in (result.get("metrics") or {}).items() if v is not None},
        "framework_version": env.get("framework_version"),
        "git_sha": env.get("git_sha"),
        "source_checkpoint": checkpoint.name,
        "path": bundle_path(record["dataset_id"], record["model"], run_id, record.get("track")),
    }
    manifest["files"] = {
        name: sha256_file(out_dir / name) for name in (WEIGHTS, RECORD)
    }
    (out_dir / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / CARD).write_text(_card(manifest), encoding="utf-8")
    return manifest


CHECKPOINTS_PREFIX = "checkpoints"


def bundle_path(dataset_id: str, model: str, run_id: str, track: str | None = None) -> str:
    """Return the canonical in-repository directory of one bundle."""
    return f"{CHECKPOINTS_PREFIX}/{track or 'time_series'}/{dataset_id}/{model}/{run_id}"


def load_manifest(uri: str | HubURI, cache_root: Path | None = None) -> dict:
    """Fetch the manifest of a bundle addressed by its directory URI."""
    base = parse(uri) if isinstance(uri, str) else uri
    path = fetch(HubURI(base.repo_id, base.revision, f"{base.path.rstrip('/')}/{MANIFEST}", base.repo_type), cache_root=cache_root)
    return json.loads(path.read_text(encoding="utf-8"))


def load_state_dict(uri: str | HubURI, cache_root: Path | None = None) -> tuple[dict, dict]:
    """Return ``(state_dict, manifest)`` for a published bundle, checksum-verified."""
    load_file = _require("safetensors.torch").load_file
    base = parse(uri) if isinstance(uri, str) else uri
    manifest = load_manifest(base, cache_root)
    weights = HubURI(base.repo_id, base.revision, f"{base.path.rstrip('/')}/{WEIGHTS}", base.repo_type)
    path = fetch(weights, sha256=manifest["files"][WEIGHTS], cache_root=cache_root)
    return load_file(str(path)), manifest
