"""tsf realtime — rolling real-time tracks: data releases, rounds, forecasts, scores.

    tsf realtime list
    tsf realtime update    --track T [--bootstrap | --no-fetch] [--pull] [--push] [--repo OWNER/NAME]
    tsf realtime open      --track T [--now ISO]
    tsf realtime baselines --track T [--round R]
    tsf realtime forecast  --track T --model M [--round R] [--set section.key=value ...]
    tsf realtime validate  FILE... [--received-at ISO]
    tsf realtime score     --track T
    tsf realtime weekly    --track T [T ...] [--pull] [--push]
    tsf realtime replay    --track T --end DATE --weeks N [--models M ...]

Everything except ``forecast`` runs without torch, so the weekly job fits a
CPU-only CI runner. ``forecast`` trains a catalog model on the round's history
with the standard runner and needs a (GPU) machine with the full install.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from tsflab.release.hub.uri import default_repo
from tsflab.realtime import rounds as R
from tsflab.realtime.store import PanelStore
from tsflab.realtime.tracks import get_track, list_tracks
from tsflab.core.realtime import ForecastSubmission

SUMMARY_DIR = Path("apps") / "web" / "data" / "realtime"


def _latest_round(track: str, round_id: str | None):
    if round_id:
        return R.load_round(track, round_id)
    specs = R.list_rounds(track)
    if not specs:
        raise RuntimeError(f"track {track!r} has no open round; run `tsf realtime open`")
    return specs[-1]


def _update(track_id: str, pull: bool, push: bool, repo: str) -> dict | None:
    from tsflab.realtime import sources

    from tsflab.realtime.publish import push_refusal

    track = get_track(track_id)
    store = PanelStore(track_id)
    refusal = push_refusal(track_id)
    if pull and not store.exists:
        if refusal:
            print(f"{track_id}: not hosted on the Hub, nothing to pull; "
                  f"build it with `tsf realtime update --bootstrap --track {track_id}`")
        else:
            from tsflab.realtime.publish import pull_track

            pull_track(store, repo)
    if not store.exists:
        raise RuntimeError(f"no local store for {track_id!r}; run `tsf realtime update --bootstrap` first")
    last = pd.Timestamp(store.manifest()["last_timestamp"])
    start = last - pd.Timedelta(days=2)  # overlap re-reads late-arriving cells
    end = pd.Timestamp.now(tz=track.tz).tz_localize(None).floor("h")
    new = sources.fetch(track, start, end, store.manifest()["channels"])
    if new.empty or new.index.max() <= last:
        print(f"{track_id}: no new observations after {last}")
        return None
    release = store.append(new, note="weekly update")
    print(f"{track_id}: release {release['version']} -> last {release['last_timestamp']}")
    if push:
        if refusal:
            print(f"skip push: {refusal}")
        else:
            from tsflab.realtime.publish import push_release

            release["hf_revision"] = push_release(store, repo)
    return release


def _push(store: PanelStore, repo: str) -> None:
    """Upload the store with ``--push``, or print why the track is never uploaded."""
    from tsflab.realtime.publish import push_refusal, push_release

    refusal = push_refusal(store.track)
    if refusal:
        print(f"skip push: {refusal}")
        return
    print("hub revision:", push_release(store, repo, create=True))


def _score(track_id: str) -> dict:
    track = get_track(track_id)
    store = PanelStore(track_id)
    for spec in R.list_rounds(track_id):
        if (R.round_dir(track_id, spec.round_id) / "scores.json").is_file():
            continue
        scores = R.score_round(spec, store, track.min_coverage)
        status = "waiting for truth" if scores is None else f"scored {len(scores)} forecast(s)"
        print(f"{track_id} {spec.round_id}: {status}")
    summary = R.track_summary(track_id)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / f"{track_id}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_overrides(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        section, _, name = key.rpartition(".")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        out.setdefault(section, {})[name] = parsed
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsf realtime", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    for name in ("update", "open", "baselines", "forecast", "score"):
        p = sub.add_parser(name)
        p.add_argument("--track", required=True)
        if name == "update":
            p.add_argument("--push", action="store_true", help="publish the release to the Hub")
            p.add_argument("--repo", default=default_repo("TSFLab-Datasets"))
            p.add_argument("--pull", action="store_true", help="fetch the published store first")
            mode = p.add_mutually_exclusive_group()
            mode.add_argument("--bootstrap", action="store_true",
                              help="create the store from the full history instead of appending")
            mode.add_argument("--no-fetch", action="store_true",
                              help="skip the source fetch; with --push, publish the local store as is")
        if name == "open":
            p.add_argument("--now")
        if name in {"baselines", "forecast"}:
            p.add_argument("--round")
        if name == "forecast":
            p.add_argument("--model", required=True)
            p.add_argument("--submitter")
            p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
            p.add_argument("--work-dir", type=Path, default=Path("work_dirs") / "_realtime")
    validate = sub.add_parser("validate")
    validate.add_argument("files", nargs="+", type=Path)
    validate.add_argument("--received-at", help="trusted arrival time (e.g. PR creation) checked against the deadline")
    weekly = sub.add_parser("weekly")
    weekly.add_argument("--track", nargs="+", required=True)
    weekly.add_argument("--pull", action="store_true")
    weekly.add_argument("--push", action="store_true")
    weekly.add_argument("--repo", default=default_repo("TSFLab-Datasets"))
    replay = sub.add_parser("replay", help="backtest the rolling protocol on historical weeks")
    replay.add_argument("--track", required=True)
    replay.add_argument("--end", required=True)
    replay.add_argument("--weeks", type=int, default=12)
    replay.add_argument("--models", nargs="*", default=[])
    replay.add_argument("--out", type=Path, default=None)
    replay.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args(argv)

    try:
        if args.action == "list":
            for track in list_tracks():
                store = PanelStore(track.id)
                last = store.manifest()["last_timestamp"] if store.exists else "not bootstrapped"
                print(f"{track.id:18s} {track.mode:15s} freq={track.freq:3s} H={track.horizon:<4d} last={last}")
            return 0
        if args.action == "update":
            if args.bootstrap:
                from tsflab.realtime import sources

                store = PanelStore(args.track)
                release = store.append(sources.bootstrap(get_track(args.track)), note="bootstrap")
                print(f"{args.track}: bootstrapped through {release['last_timestamp']}")
                if args.push:
                    _push(store, args.repo)
            elif args.no_fetch:
                from tsflab.realtime.publish import push_release

                store = PanelStore(args.track)
                if not store.exists:
                    raise RuntimeError(f"no local store for {args.track!r}; run `tsf realtime update --bootstrap` first")
                if not args.push:
                    raise RuntimeError("--no-fetch only makes sense with --push")
                # Publishing is the only action here, so a refused track is an error.
                print("hub revision:", push_release(store, args.repo, create=True))
            else:
                _update(args.track, args.pull, args.push, args.repo)
            return 0
        if args.action == "open":
            now = pd.Timestamp(args.now).to_pydatetime() if args.now else None
            spec = R.open_round(get_track(args.track), PanelStore(args.track), now=now)
            print(f"{spec.track} {spec.round_id}: cutoff {spec.cutoff}, targets "
                  f"{spec.target_timestamps[0]} .. {spec.target_timestamps[-1]}, deadline {spec.deadline}")
            return 0
        if args.action == "baselines":
            from tsflab.realtime.baselines import run_baselines

            spec = _latest_round(args.track, args.round)
            for submission in run_baselines(spec, PanelStore(args.track), get_track(args.track)):
                print("wrote", R.write_forecast(submission, spec))
            return 0
        if args.action == "forecast":
            from tsflab.realtime.forecast import forecast_with_model

            spec = _latest_round(args.track, args.round)
            submission = forecast_with_model(spec, PanelStore(args.track), get_track(args.track),
                                             args.model, args.work_dir, _parse_overrides(args.set),
                                             args.submitter)
            print("wrote", R.write_forecast(submission, spec))
            return 0
        if args.action == "validate":
            failures = 0
            for path in args.files:
                try:
                    submission = ForecastSubmission.model_validate_json(path.read_text(encoding="utf-8"))
                    spec = R.load_round(submission.track, submission.round_id)
                    if args.received_at:
                        submission = submission.model_copy(update={"submitted_at": args.received_at})
                    submission.check_against(spec)
                    print(f"ok   {path}")
                except (OSError, ValueError) as exc:
                    failures += 1
                    print(f"FAIL {path}: {exc}", file=sys.stderr)
            return 1 if failures else 0
        if args.action == "score":
            summary = _score(args.track)
            print(f"{args.track}: {len(summary['scored_rounds'])} scored round(s)")
            return 0
        if args.action == "weekly":
            from tsflab.realtime.baselines import run_baselines

            for track_id in args.track:
                release = _update(track_id, args.pull, args.push, args.repo)
                _score(track_id)
                if release is not None:
                    spec = R.open_round(get_track(track_id), PanelStore(track_id),
                                        hf_revision=release.get("hf_revision"))
                    for submission in run_baselines(spec, PanelStore(track_id), get_track(track_id)):
                        R.write_forecast(submission, spec)
                    print(f"{track_id}: opened {spec.round_id} with baselines")
            return 0
        if args.action == "replay":
            from tsflab.realtime.replay import replay as run_replay

            summary = run_replay(get_track(args.track), PanelStore(args.track), end=pd.Timestamp(args.end),
                                 weeks=args.weeks, models=args.models,
                                 overrides=_parse_overrides(args.set))
            text = json.dumps(summary, indent=2)
            if args.out:
                args.out.write_text(text + "\n", encoding="utf-8")
            print(text)
            return 0
    except (KeyError, RuntimeError, ValueError, FileNotFoundError, PermissionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
