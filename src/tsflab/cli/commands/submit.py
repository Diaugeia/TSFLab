#!/usr/bin/env python3
"""tsf result submit — package runs into self-contained Submission Reports.

Locates a run's ``record.json`` (+ ``profile.csv`` + research events) and
assembles a schema-valid ``tsflab.core.SubmissionReport`` bundle (machine results +
audit trajectory + human report) under ``work_dirs/_submissions/``. Depends only
on ``tsf_core`` + stdlib.

Results are published on the Hugging Face Hub, in ``results/`` of the model
repository ``Diaugeia/TSFLab-Checkpoints``; the leaderboard is generated from
there. Maintainers upload bundles with ``tsf result hub results push``. External
contributors open a pull request on https://github.com/Diaugeia/TSFLab that adds
the bundle under ``apps/web/submissions/`` (a staging folder); a maintainer
moves accepted bundles to the Hub.

Examples
--------
    uv run tsf result submit --dataset ETTh1 --model iTransformer --latest
    uv run tsf result submit --all                          # every record under work_dirs/
    uv run tsf result submit --all --dataset ETTh1 --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tsflab.core.paths import working_root

ROOT = working_root()
WORK = ROOT / "work_dirs"

LEADERBOARD_REPO = "https://github.com/Diaugeia/TSFLab"
PROFILE_INT_FIELDS = {"total_params", "trainable_params", "non_trainable_params"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "_", (text or "").lower()).strip("_") or "x"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_user() -> str | None:
    try:
        r = subprocess.run(
            ["git", "config", "user.name"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _find_record(dataset: str, model: str, run_id: str | None, work: Path = WORK) -> Path:
    rdir = work / dataset / model / "records"
    if not rdir.exists():
        sys.exit(f"error: no records dir at {rdir} — run an experiment first (records appear there).")
    if run_id:
        p = rdir / f"{run_id}.json"
        if not p.exists():
            sys.exit(f"error: record not found: {p}")
        return p
    files = sorted(rdir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"error: no record.json under {rdir}")
    return files[-1]


def find_records(work: Path, dataset: str | None = None, model: str | None = None) -> list[Path]:
    """Every ``<work>/<dataset>/<model>/records/*.json``; ``_``-prefixed dirs are skipped."""
    pattern = f"{dataset or '*'}/{model or '*'}/records/*.json"
    return sorted(p for p in work.glob(pattern) if not p.parents[2].name.startswith("_"))


def _read_profiles(path: Path) -> dict[str, dict]:
    """``run_id -> profile`` for one ``profile.csv``."""
    from tsflab.core import PROFILE_FIELDS

    profiles: dict[str, dict] = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                out: dict = {}
                for k in PROFILE_FIELDS:
                    v = row.get(k)
                    if v in (None, ""):
                        continue
                    if k in PROFILE_INT_FIELDS:
                        try:
                            out[k] = int(float(v))
                        except ValueError:
                            continue
                    else:
                        out[k] = v
                if row.get("run_id") and out:
                    profiles.setdefault(row["run_id"], out)
    except Exception:
        return {}
    return profiles


def _load_profile(model_dir: Path, run_id: str, cache: dict | None = None) -> dict | None:
    pf = model_dir / "profile.csv"
    if not pf.exists() or not run_id:
        return None
    if cache is None:
        return _read_profiles(pf).get(run_id)
    if pf not in cache:
        cache[pf] = _read_profiles(pf)
    return cache[pf].get(run_id)


def trajectory_index() -> dict[str, list[dict]]:
    """``run_id -> events`` over every research round, read once (bulk packaging)."""
    from tsflab.research.rounds import list_rounds, read_events

    index: dict[str, list[dict]] = {}
    for state in list_rounds():
        events = read_events(state["id"])
        run_ids = {(event.get("details") or {}).get("run_id") for event in events} - {None}
        for run_id in run_ids:
            index.setdefault(run_id, []).extend({"round": state["id"], **event} for event in events)
    return index


def _gather_trajectory(dest: Path, run_id: str, index: dict | None = None) -> dict:
    """Collect research events for ``run_id`` or synthesize a minimal record."""
    if index is None:
        from tsflab.research.rounds import events_for_run

        events = events_for_run(run_id)
    else:
        events = list(index.get(run_id, []))
    synthetic = not events
    if synthetic:
        events = [
            {
                "synthetic": True,
                "note": "No research round was associated with this experiment. "
                "Reconstructed from run artifacts.",
                "run_id": run_id,
                "time": _now(),
            }
        ]
    events.sort(key=lambda event: event.get("time", ""))
    with open(dest, "w") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
    return {"n_events": len(events), "synthetic": synthetic}


def _render_report(record, ds_spec) -> str:
    """Human-readable Markdown report for one submission (renders on the Hub)."""
    def cell(v):
        return "—" if v is None else str(v)

    rows = "\n".join(
        f"| {hr.horizon} | {cell(hr.metrics.mse)} | {cell(hr.metrics.mae)} | "
        f"{cell(hr.metrics.rmse)} | {cell(hr.metrics.corr)} | `{hr.run_id}` |"
        for hr in record.results
    )
    prof = record.results[0].profile
    prof_lines = "(none)"
    if prof is not None:
        items = {k: v for k, v in prof.model_dump().items() if v is not None}
        if items:
            prof_lines = " · ".join(f"{k} {v}" for k, v in items.items())
    env = record.env.model_dump()
    env_line = " · ".join(f"{k} {v}" for k, v in env.items() if v is not None)
    return f"""# TSFLab Leaderboard Submission — {record.model} / {record.dataset_id}

- **track** `{record.track}` · **mode** `{record.mode}` · **seed** {record.seed}
- **dataset** `{ds_spec.id}@{ds_spec.version}`
- **created** {record.created_at or ""}

## Metrics by horizon

| pred_len | MSE | MAE | RMSE | Corr | run_id |
|---:|---|---|---|---|---|
{rows}

## Profile

{prof_lines}

## Environment

{env_line}

_Generated by `tsf result submit` · Diaugeia / TSFLab Leaderboard._
"""


def build_submission(rec_path: Path, *, submitter: str, out_root: Path, track: str | None = None,
                     dataset_version: str = "1.0.0", index: dict | None = None,
                     profiles: dict | None = None, skip_existing: bool = False) -> tuple[Path, dict | None]:
    """Write one bundle for ``rec_path``; return ``(bundle dir, info)``, ``info`` None if skipped.

    The dataset name is the record's ``<dataset>/<model>/records/`` directory.
    Raises ``ValueError`` when the record is empty or fails the contract.
    """
    from tsflab import core as tsf_core

    rec_data = json.loads(rec_path.read_text())
    if not rec_data.get("results"):
        raise ValueError(f"record has no results: {rec_path}")
    run_id = rec_data["results"][0].get("run_id") or rec_path.stem
    submission_id = f"{_slug(submitter)}__{run_id}"
    sub_dir = out_root / submission_id
    if skip_existing and (sub_dir / "submission.json").is_file():
        return sub_dir, None

    model_dir = rec_path.parent.parent
    prof = _load_profile(model_dir, run_id, profiles)
    if prof:
        rec_data["results"][0]["profile"] = prof
    try:
        record = tsf_core.RunRecord(**rec_data)
    except Exception as exc:
        raise ValueError(f"record.json failed contract validation ({rec_path}): {exc}") from exc

    track = track or record.track
    ds_spec = tsf_core.DatasetSpec(
        id=_slug(model_dir.parent.name),
        version=dataset_version,
        mode=record.mode,
        track=track,
    )
    sub_dir.mkdir(parents=True, exist_ok=True)

    traj_path = sub_dir / "trajectory.jsonl"
    traj_info = _gather_trajectory(traj_path, run_id, index)
    traj_sha = _sha256_file(traj_path)

    report_path = sub_dir / "report.md"
    report_path.write_text(_render_report(record, ds_spec))
    report_sha = _sha256_file(report_path)

    files = [
        tsf_core.FileRef(path="trajectory.jsonl", sha256=traj_sha, bytes=traj_path.stat().st_size, role="trajectory"),
        tsf_core.FileRef(path="report.md", sha256=report_sha, bytes=report_path.stat().st_size, role="report"),
    ]
    files_sha = _sha256_bytes(
        json.dumps([f.model_dump() for f in files], sort_keys=True).encode()
    )
    manifest = tsf_core.SubmissionManifest(
        submission_id=submission_id,
        submitter=submitter,
        track=track,
        created_at=_now(),
        files=files,
        files_sha256=files_sha,
    )
    report = tsf_core.SubmissionReport(
        manifest=manifest,
        datasets=[ds_spec],
        records=[record],
        trajectories=[
            tsf_core.TrajectoryRef(
                path="trajectory.jsonl", sha256=traj_sha,
                synthetic=traj_info["synthetic"], n_events=traj_info["n_events"],
            )
        ],
        reports=[tsf_core.ReportArtifact(path="report.md", sha256=report_sha, format="md")],
    )
    document = report.model_dump(mode="json")
    (sub_dir / "submission.json").write_text(json.dumps(document, indent=2, ensure_ascii=False))
    from tsflab.release.hub.results import results_path

    return sub_dir, {
        "submission_id": submission_id, "track": track, "dataset": f"{ds_spec.id}@{ds_spec.version}",
        "horizons": len(record.results), "trajectory": traj_info,
        "path_in_repo": results_path(document, submission_id),
    }


def _next_steps(out_root: Path, path_in_repo: str | None = None) -> None:
    print("\nNext — publish the bundle(s):")
    print(f"  maintainers: uv run tsf result hub results push {out_root}   (add --dry-run first)")
    if path_in_repo:
        print(f"               -> {path_in_repo}/ in the TSFLab-Checkpoints repository")
    staged = path_in_repo.replace("results/", "apps/web/submissions/", 1) if path_in_repo else \
        "apps/web/submissions/<track>/<dataset>/<model>/<submission_id>"
    print(f"  contributors: open a pull request on {LEADERBOARD_REPO} that adds the bundle")
    print(f"               under {staged}/")
    print("  (exact steps + accepted formats: apps/web/SUBMITTING.md in that repo)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tsf result submit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="Dataset name (work_dirs/<dataset>/...); a filter with --all")
    ap.add_argument("--model", help="Model name (work_dirs/<dataset>/<model>/...); a filter with --all")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--run-id", help="Specific run_id to submit")
    g.add_argument("--latest", action="store_true", help="Use the newest record.json (default)")
    g.add_argument("--all", action="store_true",
                   help="Package every record under --work-dir (bulk; e.g. a whole sweep)")
    ap.add_argument("--work-dir", type=Path, default=WORK, help="Records root (default: work_dirs)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="With --all: keep bundles that already exist in the output dir")
    ap.add_argument("--track", default=None, help="Override track (default: from record)")
    ap.add_argument("--submitter", default=None, help="Submitter name (default: git user)")
    ap.add_argument("--dataset-version", default="1.0.0", help="DatasetSpec version to pin")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: work_dirs/_submissions)")
    args = ap.parse_args(argv)

    submitter = args.submitter or _git_user() or "anonymous"
    out_root = Path(args.out_dir) if args.out_dir else args.work_dir / "_submissions"

    if args.all:
        records = find_records(args.work_dir, args.dataset, args.model)
        if not records:
            sys.exit(f"error: no records under {args.work_dir}/<dataset>/<model>/records/")
        index, profiles = trajectory_index(), {}
        built = skipped = 0
        failures: list[str] = []
        for number, rec_path in enumerate(records, start=1):
            try:
                _, info = build_submission(rec_path, submitter=submitter, out_root=out_root,
                                           track=args.track, dataset_version=args.dataset_version,
                                           index=index, profiles=profiles,
                                           skip_existing=args.skip_existing)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                failures.append(str(exc).splitlines()[0])
                continue
            if info is None:
                skipped += 1
            else:
                built += 1
            if number % 500 == 0:
                print(f"  {number}/{len(records)} records ...", flush=True)
        print(f"Submission Reports under {out_root}: {built} built, {skipped} kept, {len(failures)} failed "
              f"(of {len(records)} records)")
        for message in failures[:20]:
            print(f"  failed: {message}", file=sys.stderr)
        _next_steps(out_root)
        return 1 if failures else 0

    if not args.dataset or not args.model:
        ap.error("--dataset and --model are required unless --all is given")
    rec_path = _find_record(args.dataset, args.model, args.run_id, args.work_dir)
    try:
        sub_dir, info = build_submission(rec_path, submitter=submitter, out_root=out_root,
                                         track=args.track, dataset_version=args.dataset_version)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    print(f"Submission Report built: {sub_dir}")
    print(f"  submission_id : {info['submission_id']}")
    print(f"  track/dataset : {info['track']} / {info['dataset']}")
    print(f"  records       : 1 ({info['horizons']} horizon[s])")
    print(f"  trajectory    : {info['trajectory']['n_events']} event(s)"
          + (" [SYNTHETIC — use `tsf run --round <id>` next time]" if info["trajectory"]["synthetic"] else ""))
    print("  files         : submission.json, trajectory.jsonl, report.md")
    _next_steps(out_root, info["path_in_repo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
