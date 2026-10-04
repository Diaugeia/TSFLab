"""Download GIFT-EVAL datasets from HuggingFace and symlink into the project.

Usage examples
--------------
# Download all datasets to default location (~/.cache/gift_eval):
  uv run tsf data prepare --from gift-eval

# Download to a custom location:
  uv run tsf data prepare --from gift-eval --output-dir /data/gift_eval

# Download specific datasets only:
  uv run tsf data prepare --from gift-eval --datasets electricity/15T ett1/H m4_monthly

# Link an already-downloaded directory (skip download):
  uv run tsf data prepare --from gift-eval --link-only --output-dir /data/gift_eval

The script creates a symlink at ``./dataset/gift_eval`` pointing to the
download directory so that TOML configs with ``path = "./dataset/gift_eval"``
work out of the box.
"""

from __future__ import annotations

import argparse
import os
import sys

# All 55 dataset/freq combinations in GIFT-EVAL
ALL_DATASETS = [
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/D",
    "LOOP_SEATTLE/H",
    "M_DENSE/D",
    "M_DENSE/H",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
    "bizitobs_service",
    "car_parts_with_missing",
    "covid_deaths",
    "electricity/15T",
    "electricity/D",
    "electricity/H",
    "electricity/W",
    "ett1/15T",
    "ett1/D",
    "ett1/H",
    "ett1/W",
    "ett2/15T",
    "ett2/D",
    "ett2/H",
    "ett2/W",
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "hospital",
    "jena_weather/10T",
    "jena_weather/D",
    "jena_weather/H",
    "kdd_cup_2018_with_missing/D",
    "kdd_cup_2018_with_missing/H",
    "m4_daily",
    "m4_hourly",
    "m4_monthly",
    "m4_quarterly",
    "m4_weekly",
    "m4_yearly",
    "restaurant",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
    "solar/10T",
    "solar/D",
    "solar/H",
    "solar/W",
    "temperature_rain_with_missing",
    "us_births/D",
    "us_births/M",
    "us_births/W",
]

HF_REPO = "Salesforce/GiftEval"
DEFAULT_OUTPUT = os.path.expanduser("~/.cache/gift_eval")
SYMLINK_PATH = os.path.join("dataset", "gift_eval")


def _download(datasets: list[str], output_dir: str) -> None:
    """Download datasets from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    # Group by base name to avoid redundant downloads.
    # HF dataset configs correspond to the base dataset name (before '/').
    # The frequency subdirectories are part of the on-disk layout.
    bases: dict[str, list[str]] = {}
    for name in datasets:
        parts = name.split("/")
        base = parts[0]
        bases.setdefault(base, []).append(name)

    total = len(datasets)
    done = 0

    for base, members in sorted(bases.items()):
        for name in members:
            dest = os.path.join(output_dir, name)
            if os.path.exists(os.path.join(dest, "dataset_info.json")):
                done += 1
                print(f"  [{done}/{total}] {name}  (already exists, skipping)")
                continue

            print(f"  [{done + 1}/{total}] Downloading {name} ...")
            try:
                snapshot_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    allow_patterns=f"{name}/*",
                    local_dir=output_dir,
                )
                done += 1
                print(f"  [{done}/{total}] {name}  ✓")
            except Exception as exc:
                done += 1
                print(f"  [{done}/{total}] {name}  FAILED: {exc}", file=sys.stderr)


def _make_symlink(output_dir: str) -> None:
    """Create ./dataset/gift_eval symlink pointing to output_dir."""
    target = os.path.abspath(output_dir)
    link = SYMLINK_PATH

    os.makedirs(os.path.dirname(link), exist_ok=True)

    if os.path.islink(link):
        existing = os.readlink(link)
        if os.path.abspath(existing) == target:
            print(f"Symlink already correct: {link} -> {target}")
            return
        os.remove(link)
    elif os.path.exists(link):
        print(
            f"Warning: {link} exists and is not a symlink. "
            "Skipping symlink creation.",
            file=sys.stderr,
        )
        return

    os.symlink(target, link)
    print(f"Symlink created: {link} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GIFT-EVAL datasets and symlink into the project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help=f"Download destination (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific datasets to download (e.g. electricity/15T m4_monthly). "
        "Omit to download all 55.",
    )
    parser.add_argument(
        "--link-only",
        action="store_true",
        help="Only create the symlink, skip downloading.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_datasets",
        help="Print all available dataset names and exit.",
    )
    args = parser.parse_args()

    if args.list_datasets:
        for name in ALL_DATASETS:
            print(name)
        return

    output_dir = os.path.expanduser(args.output_dir)

    if not args.link_only:
        targets = args.datasets if args.datasets else ALL_DATASETS
        # Validate names
        for name in targets:
            if name not in ALL_DATASETS:
                print(
                    f"Warning: '{name}' is not a known GIFT-EVAL dataset.",
                    file=sys.stderr,
                )
        print(f"Downloading {len(targets)} dataset(s) to {output_dir} ...")
        _download(targets, output_dir)

    _make_symlink(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
