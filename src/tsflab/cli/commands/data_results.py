"""Dataset and result resource command routing behind the public CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tsflab.cli.runtime import ROOT, passthrough


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _dataset_record_payload(record: object, facts: dict[str, object] | None = None) -> dict[str, object]:
    from dataclasses import asdict

    payload = asdict(record)
    payload["card"] = f"catalog/datasets/{record.name}/README.md"
    if facts is not None:
        payload["facts"] = facts  # curated card front matter (level 0/1)
    return payload


def dataset_read(action: str, rest: list[str]) -> int:
    """Dataset list/show/search/audit, reached through `tsf catalog` and `tsf data audit`."""
    from tsflab.catalog.cards.datasets import dataset_facts
    from tsflab.catalog.cards.resources import audit_resource_cards, dataset_records

    records = dataset_records(ROOT)
    if action == "list":
        if rest not in ([], ["--json"]):
            print("usage: tsf catalog list --kind dataset [--json]", file=sys.stderr)
            return 2
        from tsflab.catalog.cards.schema import DOMAINS

        facts = dataset_facts(ROOT)
        payload = [_dataset_record_payload(record) for record in records]
        for record in payload:
            record["domain"] = str(facts.get(record["name"], {}).get("domain", ""))
            record["benchmarks"] = list(facts.get(record["name"], {}).get("benchmarks", []))
        if rest == ["--json"]:
            _print(payload)
        else:  # grouped by domain, in the schema's domain order
            order = {domain: index for index, domain in enumerate(DOMAINS)}
            current = None
            for record in sorted(payload, key=lambda r: (order.get(r["domain"], len(order)), r["name"])):
                if record["domain"] != current:
                    current = record["domain"]
                    count = sum(r["domain"] == current for r in payload)
                    print(f"# {current or 'unknown'} ({count})")
                modes = ",".join(record["task_modes"])
                print(f"{record['name']}\t{record['loader']}\t{modes}\t{record['alias']}")
        return 0
    if action == "show":
        from tsflab.catalog.cards.show import existing, parse_show, show_card
        from tsflab.catalog.cards.resources import dataset_card_path

        parsed = parse_show("tsf catalog show --kind dataset", "dataset preset name", rest)
        selected = next((record for record in records if record.name == parsed.name), None)
        if selected is None:
            family = [r for r in records if r.loader == parsed.name and r.dataset_id]
            if not family:
                print(f"Unknown dataset preset {parsed.name!r}", file=sys.stderr)
                return 2
            card_path = dataset_card_path(ROOT, parsed.name)
            facts = {"kind": "dataset-family", "members": len(family)}
            legacy = {"name": parsed.name, "kind": "dataset-family",
                      "card": card_path.relative_to(ROOT).as_posix(),
                      "facts": dataset_facts(ROOT, [parsed.name]).get(parsed.name, {}),
                      "members": [r.name for r in family]}
            paths = existing(ROOT, card_path.relative_to(ROOT).as_posix())
            return show_card(ROOT, card_path, parsed, facts=facts, paths=paths, legacy=legacy)
        legacy = _dataset_record_payload(
            selected, dataset_facts(ROOT, [selected.name]).get(selected.name, {})
        )
        # config, loader, and task modes are already in the card header.
        facts = {"path": selected.path or "(loader-defined)"}
        card_path = dataset_card_path(ROOT, selected.name)
        paths = existing(
            ROOT,
            card_path.relative_to(ROOT).as_posix(),
            selected.config,
            selected.path,
        )
        return show_card(ROOT, card_path, parsed, facts=facts, paths=paths, legacy=legacy)
    if action == "search":
        from tsflab.catalog.cards.search import search_command

        by_name = {record.name: record for record in records}

        def augment(match: dict[str, object]) -> dict[str, object]:
            return {**_dataset_record_payload(by_name[str(match["name"])]), **match}

        return search_command(
            ROOT, rest, prog="tsf catalog search --kind dataset", kind="dataset", augment=augment
        )
    if rest:
        print("tsf data audit takes no arguments", file=sys.stderr)
        return 2
    failures = [error for error in audit_resource_cards(ROOT) if "catalog/datasets" in error]
    for failure in failures:
        print(f"ERROR: {failure}")
    failing = sum(any(f"catalog/datasets/{record.name}" in item for item in failures) for record in records)
    print(f"Dataset cards: {len(records) - failing}/{len(records)} complete and current")
    return 1 if failures else 0


def data_command(args: list[str]) -> int:
    """Route dataset scaffolding, preparation, inspection, plotting, and publishing."""
    usage = (
        "usage: tsf data {add,prepare,inspect,analyze,plot,download,publish,audit} [args...]\n"
        "       tsf data prepare [--from traffic|ultratraffic|gift-eval|tfb|dcrnn] [args...]\n"
        "Find and read datasets with `tsf catalog search|show --kind dataset`."
    )
    if not args or args[0] in {"-h", "--help", "help"}:
        print(usage)
        return 0
    action, rest = args[0], args[1:]
    if action == "audit":
        return dataset_read("audit", rest)
    scripts = {
        "add": "new_dataset.py",
        "prepare": "pre_process.py",
        "inspect": "dataset_characteristics.py",
        "analyze": "dataset_analyze.py",
        "plot": "visual_data.py",
    }
    if action in {"download", "publish"}:
        return _hub_dataset_command(action, rest)
    if action == "prepare":
        source, rest = _extract_from(rest)
        if source == "ultratraffic":
            from tsflab.data.prepare.ultratraffic import main as convert_ultratraffic

            return convert_ultratraffic(rest)
        if source == "traffic":
            return passthrough("convert_traffic.py", rest)
        if source == "gift-eval":
            return passthrough("gift_eval_download.py", rest)
        if source == "tfb":
            from tsflab.data.prepare.tfb import main as fetch_tfb

            return fetch_tfb(rest)
        if source == "dcrnn":
            from tsflab.data.prepare.dcrnn import main as fetch_dcrnn

            return fetch_dcrnn(rest)
        if source is not None:
            print(f"unknown --from source {source!r}; choose traffic, ultratraffic, gift-eval, tfb, or dcrnn",
                  file=sys.stderr)
            return 2
    script = scripts.get(action)
    if script is None:
        print(usage, file=sys.stderr)
        return 2
    return passthrough(script, rest)


def _extract_from(args: list[str]) -> tuple[str | None, list[str]]:
    """Remove ``--from SOURCE`` from ``args``."""
    rest, source, skip = [], None, False
    for index, arg in enumerate(args):
        if skip:
            skip = False
        elif arg == "--from" and index + 1 < len(args):
            source, skip = args[index + 1], True
        elif arg.startswith("--from="):
            source = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    return source, rest


def _hub_dataset_command(action: str, rest: list[str]) -> int:
    """Download published preset files, or publish local ones (maintainers)."""
    from tsflab.release import hub

    parser = argparse.ArgumentParser(prog=f"tsf data {action}")
    parser.add_argument("presets", nargs="*")
    parser.add_argument("--root", type=Path, default=Path("dataset"),
                        help="local dataset root (default: ./dataset)")
    if action == "download":
        parser.add_argument("--all", action="store_true", help="every published preset")
        parser.add_argument("--list", action="store_true", help="list published presets")
        parser.add_argument("--check", action="store_true",
                            help="check that every pinned file still resolves (no download)")
    else:
        parser.add_argument("--repo", default=hub.DEFAULT_STATIC_REPO)
        parser.add_argument("--create", action="store_true", help="create the repo if missing")
        parser.add_argument("--private", action="store_true", help="create the repo as private")
        parser.add_argument("--path", action="append", default=[], dest="paths",
                            help="also publish a whole subtree of the root (repeatable)")
    parsed = parser.parse_args(rest)
    try:
        if action == "publish":
            if not parsed.presets and not parsed.paths:
                parser.error("name presets or --path subtrees to publish")
            revisions = hub.publish_presets(parsed.presets, parsed.root, parsed.repo,
                                            paths=tuple(parsed.paths), create=parsed.create, private=parsed.private,
                                            root=ROOT)
            for preset, revision in revisions.items():
                print(f"{preset}\t{revision}")
            print("Updated configs/hub/datasets.json; commit it with the release.")
            return 0
        if parsed.check:
            from tsflab.release.hub.datasets import check_manifest, load_manifest

            issues = check_manifest(ROOT)
            for issue in issues:
                print(f"ERROR: {issue}")
            manifest = load_manifest(ROOT)
            total = len(manifest["files"]) + len(manifest.get("upstream", {}))
            print(f"Pinned dataset files: {total - len(issues)}/{total} reachable")
            return 1 if issues else 0
        published = hub.available_presets(ROOT)
        if parsed.list:
            for preset in published:
                print(preset)
            return 0
        presets = published if parsed.all else parsed.presets
        if not presets:
            parser.error("name presets, or pass --all or --list")
        for preset in presets:
            files = hub.fetch_preset(preset, parsed.root, ROOT)
            print(f"{preset}\t{len(files)} file(s) verified under {parsed.root}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def result_command(args: list[str]) -> int:
    """Route aggregation, ranking, plotting, reporting, submission, leaderboard, and the Hub."""
    usage = (
        "usage: tsf result {aggregate,rank,plot,report,predictions,board,submit,leaderboard,hub} "
        "[args...]\n"
        "       tsf result hub {pack,push,list,pull,init} [args...]"
    )
    if not args or args[0] in {"-h", "--help", "help"}:
        print(usage)
        return 0
    action, rest = args[0], args[1:]
    if action == "board":
        from tsflab.cli.commands.result_board import board_command

        return board_command(rest)
    if action == "hub":
        from tsflab.cli.commands.hub import hub_command

        return hub_command(rest)
    scripts = {
        "aggregate": "aggregate_results.py",
        "rank": "rank_models.py",
        "plot": "plot_bubble.py",
        "report": "report.py",
        "predictions": "visualize_predictions.py",
        "submit": "submit.py",
        "leaderboard": "leaderboard_build.py",
    }
    script = scripts.get(action)
    if script is None:
        print(usage, file=sys.stderr)
        return 2
    return passthrough(script, rest)
