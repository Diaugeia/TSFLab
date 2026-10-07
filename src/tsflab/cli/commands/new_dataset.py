#!/usr/bin/env python3
"""Scaffold a new TSFLab dataset in one command.

Two patterns:

* ``custom``   — a plain flat-multivariate CSV (a ``date`` column + numeric
  channels). NO code, just a config that reuses the built-in ``custom`` loader.
* ``single``   — an unusual layout / synthetic generation needing a bespoke
  ``_read_data``. Generates the dataset class, schema, ``DATASET_NAME_MAP``
  entry, and a config.

Examples
--------
    uv run tsf data add --name my_csv --pattern custom \
        --path ./dataset/my_csv/my_csv.csv --target OT

    uv run tsf data add --name my_special --pattern single \
        --path ./dataset/my_special/my_special.csv --target OT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tsflab.core.paths import repository_root, require_checkout

ROOT = repository_root()
DS_DIR = ROOT / "src" / "tsflab" / "data" / "datasets"
SCHEMA_DIR = ROOT / "src" / "tsflab" / "data" / "schemas" / "datasets"
NAME_MAP_FILE = ROOT / "src" / "tsflab" / "catalog" / "registry" / "datasets.py"
DS_CONFIG_DIR = ROOT / "configs" / "datasets"


def _config_custom(name, path, target) -> str:
    return (
        "[dataset]\n"
        'name = "custom"\n'
        f'path = "{path}"\n\n'
        "[dataset.params]\n"
        f'target = "{target}"\n'
        "scale = true\n"
        "split_ratio = [0.7, 0.1, 0.2]\n"
    )


def _config_single(name, path, target) -> str:
    return (
        "[dataset]\n"
        f'name = "{name}"\n'
        f'path = "{path}"\n\n'
        "[dataset.params]\n"
        f'target = "{target}"\n'
        "scale = true\n"
        "split_ratio = [0.7, 0.1, 0.2]\n"
    )


def _schema_single(name) -> str:
    return (
        f'"""Parameter schema for the {name} dataset."""\n\n'
        "from pydantic import Field\n\n"
        "from tsflab.data.schemas.base import DatasetParameters\n\n\n"
        "class DatasetParameterConfig(DatasetParameters):\n"
        f'    """Validated {name} dataset parameters."""\n\n'
        "    target: str\n"
        "    scale: bool = True\n"
        "    split_ratio: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.2])\n"
    )


def _dataset_single(name) -> str:
    cls = "Dataset_" + "".join(p.capitalize() for p in name.split("_"))
    return f'''"""{name} dataset implementation (SCAFFOLD).

Replace the body of ``_read_data`` with the real loading logic. It must return
``(series_data, time_stamp)`` as numpy arrays for the requested split.
"""
from __future__ import annotations

from typing import Tuple, cast

import numpy as np
import pandas as pd

from tsflab.data.schemas.datasets.{name} import DatasetParameterConfig
from tsflab.catalog.registry import DATASET_REGISTRY
from tsflab.data.datasets.base import ForecastingDataset


class {cls}(ForecastingDataset):
    """The {name} dataset."""

    def _read_data(
        self,
        flag: str,
        features: str,
        target: str,
        split_ratio: tuple[float, float, float],
        scale: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: replace with the real loader. This template reads a CSV with a
        # `date` column + numeric channels and splits it by ratio. Set the class
        # attribute `max_rows` when the protocol uses only the first rows.
        df_raw = self._limit_rows(pd.read_csv(self.file_path))
        cols = [c for c in df_raw.columns if c != "date"]
        if features == "S":
            cols = [target]
        df_data = df_raw[cols]

        num_samples = len(df_data)
        border1, border2 = self._get_borders(flag, split_ratio, num_samples)

        if scale:
            train_len = self._train_len(split_ratio, num_samples)
            data = self._apply_scaling(df_data.to_numpy(), train_len)
        else:
            data = df_data.to_numpy()

        df_stamp = df_raw[["date"]] if "date" in df_raw.columns else df_raw.iloc[:, :0].copy()
        if "date" not in df_stamp.columns:
            df_stamp["date"] = 0
        time_stamp = np.asarray(self._build_time_stamp(df_stamp))

        series_data = np.asarray(data[border1:border2])
        time_stamp = time_stamp[border1:border2]
        return cast(np.ndarray, series_data), cast(np.ndarray, time_stamp)


def register() -> None:
    """Register the {name} dataset."""
    DATASET_REGISTRY.register(
        "{name}", {cls}, DatasetParameterConfig,
        task_modes=frozenset({{"time_series"}}),
    )
'''


def _insert_name_map(name: str) -> str:
    text = NAME_MAP_FILE.read_text()
    if f'"{name}":' in text:
        return "already-present"
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("DATASET_NAME_MAP"))
    close = next(i for i in range(start, len(lines)) if lines[i].rstrip() == "}")
    entry = f'    # Scaffolded by `tsf data add`\n    "{name}": "tsflab.data.datasets.{name}",'
    lines.insert(close, entry)
    NAME_MAP_FILE.write_text("\n".join(lines) + "\n")
    return "inserted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="Dataset key / file name (snake_case)")
    ap.add_argument("--pattern", required=True, choices=["custom", "single"])
    ap.add_argument("--path", default=None, help="Dataset file (default: ./dataset/<name>/<name>.csv)")
    ap.add_argument("--target", default="OT", help="Target column name (default: OT)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()
    try:
        require_checkout("tsf data add")
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None

    name = args.name
    data_path = args.path or f"./dataset/{name}/{name}.csv"
    cfg_path = DS_CONFIG_DIR / f"{name}.toml"

    targets: dict[Path, str] = {}
    if args.pattern == "custom":
        targets[cfg_path] = _config_custom(name, data_path, args.target)
    else:  # single
        targets[DS_DIR / f"{name}.py"] = _dataset_single(name)
        targets[SCHEMA_DIR / f"{name}.py"] = _schema_single(name)
        targets[cfg_path] = _config_single(name, data_path, args.target)

    existing = [p for p in targets if p.exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing files (use --force):", file=sys.stderr)
        for p in existing:
            print(f"  {p.relative_to(ROOT)}", file=sys.stderr)
        raise SystemExit(1)

    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    status = "n/a"
    if args.pattern == "single":
        status = _insert_name_map(name)

    card_dir = ROOT / "catalog" / "datasets" / name
    if not (card_dir / "card.toml").exists() or args.force:
        from tsflab.catalog.cards.render import card_files

        todo = "TODO(card-v1)"
        files = card_files("dataset", {
            "name": name,
            "domain": todo,
            "benchmarks": [],
            "tags": [todo, "dataset", name],
            # script until the source terms are verified: never re-host unchecked data.
            "source": {"name": todo, "url": todo, "citation": todo, "citation_url": todo,
                       "license": todo, "redistribution": "script"},
            "shape": {"frequency": todo, "target": args.target or todo, "stats_basis": "measured"},
            "protocol": {"protocol": todo},
        }, f"{todo}: what the data is. Use for ...; not for ...", {
            "Overview": f"{todo}: what it is, domain, scale, and why it is used.",
            "Protocol and pitfalls": f"{todo}: split, scaling, lookbacks/horizons, and known pitfalls.",
        })
        card_dir.mkdir(parents=True, exist_ok=True)
        for filename, text in files.items():
            (card_dir / filename).write_text(text, encoding="utf-8")

    print(f"✓ Scaffolded dataset '{name}' (pattern: {args.pattern})")
    for path in targets:
        print(f"  + {path.relative_to(ROOT)}")
    if args.pattern == "single":
        print(f"  ~ DATASET_NAME_MAP: {status}")
    print(f"  + catalog/datasets/{name}/card.toml and README.md  (TODO placeholders)")
    print()
    print("Next steps:")
    if args.pattern == "single":
        print(f"  1. Implement the loader in src/tsflab/data/datasets/{name}.py (_read_data).")
    print(f"  - Fill every TODO in catalog/datasets/{name}/card.toml and README.md with verified facts")
    print("    (source, license, statistics, protocol); `tsf data audit` fails until done.")
    print(f"  - Measure characteristics: `tsf data analyze {name} --write-card`.")
    print(f"  - Put the data at {data_path}, then reference the config from a")
    print(f"    run config via `extends = [..., \"../datasets/{name}.toml\", ...]`.")


if __name__ == "__main__":
    main()
