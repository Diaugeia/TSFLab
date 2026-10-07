"""Official split and scaling protocol of every dataset preset.

This module is the single source of truth for split borders. The loaders in
``tsflab.data.datasets`` call :func:`split_borders` and :func:`train_rows`, and
:func:`split_protocol` reports the same numbers for a preset without training
anything. :func:`load_splits` returns the split arrays scaled exactly as
``tsf run`` scales them, optionally from an explicit file, so a custom loop
never re-derives the protocol.

Row indices count data rows from the first row after the CSV header. Ranges
are half-open ``[start, end)``.

The import is torch-free; the loader classes are imported only when a preset
is resolved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tomllib
from typing import Any, Sequence

import numpy as np

SPLITS = ("train", "val", "test")
#: Protocol kinds: a ratio split of the rows, fixed window-centre index files,
#: or a split that only the loader can compute (rolling or store-backed).
KINDS = ("chronological", "window-index", "loader-defined")


def split_borders(
    num_rows: int, split_ratio: Sequence[float], seq_len: int = 0
) -> dict[str, tuple[int, int]]:
    """Return the ``(start, end)`` rows each split reads.

    The ratio is normalized by its sum. Validation and test start ``seq_len``
    rows before their own first row, so their first input window ends exactly
    at the split border and every target row belongs to the split.
    """
    total = sum(split_ratio)
    cum_ratios = [sum(split_ratio[: i + 1]) / total for i in range(len(SPLITS))]
    borders: dict[str, tuple[int, int]] = {}
    for idx, flag in enumerate(SPLITS):
        start = int(cum_ratios[idx - 1] * num_rows) - seq_len if idx > 0 else 0
        borders[flag] = (start, int(cum_ratios[idx] * num_rows))
    return borders


def train_rows(num_rows: int, split_ratio: Sequence[float]) -> int:
    """Return the number of leading rows the scaler is fitted on (the train split)."""
    return split_borders(num_rows, split_ratio)["train"][1]


@dataclass(frozen=True)
class SplitRange:
    """Rows one split reads: inputs start at ``start``, own (target) rows at ``own_start``."""

    start: int
    own_start: int
    end: int


@dataclass(frozen=True)
class SplitProtocol:
    """Resolved split and scaling rule of one dataset preset."""

    dataset: str
    loader: str
    kind: str
    file: str
    seq_len: int
    file_rows: int | None = None
    rows_used: int | None = None
    row_limit: int | None = None
    rows_basis: str = ""
    split_ratio: tuple[float, ...] | None = None
    ratio_of: str = "rows used"
    splits: dict[str, SplitRange] = field(default_factory=dict)
    scaling: str = ""
    notes: tuple[str, ...] = ()
    card_protocol: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping."""
        return asdict(self)

    def ratio_text(self) -> str:
        return ":".join(f"{value:g}" for value in self.split_ratio or ())

    def border_text(self) -> str:
        """One line: own rows of each split, then the input overlap."""
        if not self.splits:
            return "loader-defined"
        own = ", ".join(f"{flag} [{r.own_start}, {r.end})" for flag, r in self.splits.items())
        if self.kind == "window-index":
            return (f"{own} (target rows of the window centres); inputs start {self.seq_len - 1} rows "
                    f"before each centre (seq_len {self.seq_len})")
        val, test = self.splits["val"], self.splits["test"]
        return (f"{own}; val/test inputs start seq_len rows earlier "
                f"(seq_len {self.seq_len}: val {val.start}, test {test.start})")

    def rows_text(self) -> str:
        if self.rows_used is None:
            return "loader-defined"
        text = f"[0, {self.rows_used})"
        if self.file_rows is not None and self.file_rows != self.rows_used:
            text += f" of {self.file_rows} rows in the file (the loader keeps the first {self.row_limit})"
        elif self.file_rows is not None:
            text += " (every row of the file)"
        return text

    def facts(self) -> dict[str, str]:
        """Short generated facts for the dataset L1 page (``tsf catalog show``)."""
        if self.kind == "loader-defined":
            return {"split": f"loader-defined ({self.loader}); see [protocol] and `tsf data splits {self.dataset}`"}
        return {
            "rows used": self.rows_text(),
            "split": f"chronological {self.ratio_text()} of the {self.ratio_of}",
            "split borders": self.border_text(),
            "scaling": self.scaling,
            "protocol facts": f"`tsf data splits {self.dataset} [--seq-len N] [--path FILE] [--json]`",
        }

    def render(self) -> str:
        """Human-readable report printed by ``tsf data splits``."""
        lines = [f"dataset:     {self.dataset} (loader {self.loader}, {self.kind} split)",
                 f"file:        {self.file}"]
        if self.card_protocol:
            lines.append(f"card:        {self.card_protocol}")
        if self.kind == "loader-defined":
            lines += [f"note:        {note}" for note in self.notes]
            return "\n".join(lines)
        lines.append(f"rows used:   {self.rows_text()}  [{self.rows_basis}]")
        lines.append(f"split ratio: {self.ratio_text()} of the {self.ratio_of}")
        lines.append(f"seq_len:     {self.seq_len}")
        lines.append("")
        lines.append(f"{'split':<7}{'rows read (inputs)':<22}{'own rows (targets)':<22}rows")
        for flag, r in self.splits.items():
            lines.append(f"{flag:<7}{f'[{r.start}, {r.end})':<22}{f'[{r.own_start}, {r.end})':<22}"
                         f"{r.end - r.own_start}")
        lines.append("")
        lines.append(f"scaling:     {self.scaling}")
        lines += [f"note:        {note}" for note in self.notes]
        lines.append(f"python:      tsflab.data.protocol.load_splits({self.dataset!r}, seq_len={self.seq_len})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preset resolution
# ---------------------------------------------------------------------------


def _root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    from tsflab.core.paths import repository_root

    return repository_root()


def _preset(dataset: str, root: Path) -> tuple[str, dict[str, Any]]:
    """Return ``(preset name, [dataset] table)`` for a preset name (case-insensitive)."""
    config_root = root / "configs" / "datasets"
    candidates = {path.relative_to(config_root).with_suffix("").as_posix(): path
                  for path in config_root.rglob("*.toml")}
    name = next((key for key in candidates if key.lower() == dataset.lower()), None)
    if name is None:
        # Accept the loader's registered name (for example ``ETTh1``) as well.
        for key, path in sorted(candidates.items()):
            table = tomllib.loads(path.read_text(encoding="utf-8")).get("dataset", {})
            if str(table.get("name", "")).lower() == dataset.lower() and not table.get("id"):
                name = key
                break
    if name is None:
        known = ", ".join(sorted(key for key in candidates if "/" not in key))
        raise ValueError(f"unknown dataset preset {dataset!r}; known presets: {known}")
    return name, tomllib.loads(candidates[name].read_text(encoding="utf-8")).get("dataset", {})


def _card_protocol(root: Path, name: str) -> dict[str, Any]:
    card = root / "catalog" / "datasets" / name / "card.toml"
    if not card.is_file():
        return {}
    payload = tomllib.loads(card.read_text(encoding="utf-8"))
    return {"protocol": payload.get("protocol", {}), "shape": payload.get("shape", {})}


def _spec(loader: str):
    from tsflab.catalog.registry.datasets import DATASET_REGISTRY, register_dataset_by_name

    register_dataset_by_name(loader)
    return DATASET_REGISTRY.get(loader)


def _params(spec, table: dict[str, Any]) -> dict[str, Any]:
    """Validated loader parameters with schema defaults, as the runner passes them."""
    params = dict(table.get("params", {}))
    if spec.params_schema is not None:
        params = spec.params_schema(**params).model_dump()
    params.pop("adj_norm", None)
    return params


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    from tsflab.core.paths import working_root

    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else working_root() / candidate


def _display(path: Path) -> str:
    from tsflab.core.paths import working_root

    try:
        return path.relative_to(working_root()).as_posix()
    except ValueError:
        return str(path)


def _default_seq_len(card: dict[str, Any], root: Path) -> int:
    """First lookback of the card's protocol, else the runner's base ``task.seq_len``."""
    seq_lens = card.get("protocol", {}).get("seq_lens") or []
    if seq_lens:
        return int(seq_lens[0])
    base = tomllib.loads((root / "configs" / "base.toml").read_text(encoding="utf-8"))
    return int(base["task"]["seq_len"])


def _default_pred_len(card: dict[str, Any]) -> int:
    pred_lens = card.get("protocol", {}).get("pred_lens") or []
    return int(pred_lens[0]) if pred_lens else 0


def _scaling_text(params: dict[str, Any], fit_rows: int) -> str:
    if not params.get("scale", True):
        return "none (scale = false); metrics use raw values"
    rule = ("per-channel mean/std (zero std set to 1)" if params.get("norm_each_channel")
            else "per-channel z-score (sklearn StandardScaler)")
    return (f"{rule} fitted on train rows [0, {fit_rows}) of the rows used and applied to every "
            "split; metrics use scaled values unless task.inverse = true")


def split_protocol(
    dataset: str,
    seq_len: int | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
    pred_len: int | None = None,
    root: Path | None = None,
) -> SplitProtocol:
    """Resolve the official split of ``dataset`` from the loader's own rules.

    Parameters
    ----------
    dataset : str
        Preset name (``etth1``, ``weather``, ``pems08``; case-insensitive).
    seq_len : int, optional
        Lookback; validation and test inputs start this many rows early.
        Default: the first lookback in the card's ``[protocol]``.
    path : path-like, optional
        Apply the preset's protocol to this file (or bundle directory) instead
        of the preset path, for example a raw copy of the CSV elsewhere.
    pred_len : int, optional
        Horizon; used only for window-index bundles (default: first card horizon).
    root : Path, optional
        Repository root holding ``configs/`` and ``catalog/``.
    """
    root = _root(root)
    name, table = _preset(dataset, root)
    card = _card_protocol(root, name)
    seq_len = _default_seq_len(card, root) if seq_len is None else int(seq_len)
    loader = str(table.get("name", ""))
    spec = _spec(loader)
    params = _params(spec, table)
    location = _resolve_path(path if path is not None else str(table.get("path", "")))
    common = dict(dataset=name, loader=loader, file=_display(location), seq_len=seq_len,
                  card_protocol=str(card.get("protocol", {}).get("protocol", "")))

    from tsflab.data.datasets.base import ForecastingDataset

    cls = spec.dataset_class
    if isinstance(cls, type) and issubclass(cls, ForecastingDataset):
        return _chronological(cls, params, location, card, common)
    if spec.storage == "directory" and all((location / f"idx_{flag}.npy").is_file() for flag in SPLITS):
        horizon = _default_pred_len(card) if pred_len is None else int(pred_len)
        return _window_index(location, params, seq_len, horizon, common)
    return SplitProtocol(kind="loader-defined", **common, notes=(
        "this loader computes its split itself (rolling windows or a frozen store); "
        "read the card [protocol] and build splits with tsflab.data.build_data_loader",))


def _chronological(cls, params, location: Path, card, common) -> SplitProtocol:
    import inspect

    ratio = params.get("split_ratio")
    if ratio is None:
        ratio = inspect.signature(cls.__init__).parameters["split_ratio"].default
    ratio = tuple(float(value) for value in ratio)
    limit = cls.max_rows
    if location.is_file():
        file_rows, basis = cls.count_rows(str(location)), "counted in the file"
    else:
        length = card.get("shape", {}).get("length")
        if not isinstance(length, int):
            raise FileNotFoundError(f"{location} not found; download it with `tsf data download "
                                    f"{common['dataset']}` or pass --path")
        file_rows, basis = None, "file absent; card [shape].length"
        rows_used = length if limit is None else min(length, limit)
    if file_rows is not None:
        rows_used = file_rows if limit is None else min(file_rows, limit)
    borders = split_borders(rows_used, ratio, common["seq_len"])
    plain = split_borders(rows_used, ratio)
    splits = {flag: SplitRange(borders[flag][0], plain[flag][0], borders[flag][1]) for flag in SPLITS}
    notes = []
    if limit is not None:
        notes.append(f"the loader keeps only the first {limit} rows; later rows are never read")
    if not cls.has_header:
        notes.append("the file has no header row; row 0 is its first line")
    sentinels = params.get("missing_sentinels") or []
    if sentinels:
        notes.append(f"values {sentinels} are missing; filled causally (forward fill, back fill only "
                     "at the series start) before scaling")
    if splits["val"].start < 0:
        notes.append(f"seq_len {common['seq_len']} is longer than the train split; inputs would start "
                     "before row 0")
    return SplitProtocol(
        kind="chronological", **common, file_rows=file_rows, rows_used=rows_used, row_limit=limit,
        rows_basis=basis, split_ratio=ratio, splits=splits,
        scaling=_scaling_text(params, train_rows(rows_used, ratio)), notes=tuple(notes),
    )


def _npz_shape(bundle: Path, key: str = "data") -> tuple[int, ...]:
    """Read an array shape from an ``.npz`` member header without loading the array."""
    import zipfile

    with zipfile.ZipFile(bundle) as archive, archive.open(f"{key}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, _, _ = reader(handle)
    return tuple(shape)


def _window_index(location: Path, params, seq_len: int, pred_len: int, common) -> SplitProtocol:
    centres = {flag: np.load(location / f"idx_{flag}.npy").reshape(-1) for flag in SPLITS}
    splits = {flag: SplitRange(int(c.min()) - seq_len + 1, int(c.min()) + 1, int(c.max()) + 1 + pred_len)
              for flag, c in centres.items()}
    bundle = location / str(params.get("npz_name", "his.npz"))
    rows = _npz_shape(bundle)[0] if bundle.is_file() else None
    built = {}
    if (location / "split.json").is_file():
        built = json.loads((location / "split.json").read_text(encoding="utf-8"))
    ratio = tuple(float(value) for value in built.get("splits", ())) or None
    notes = [f"window centres are fixed in idx_*.npy; a centre c reads rows [c - seq_len + 1, c] "
             f"and predicts [c + 1, c + pred_len] (pred_len {pred_len} here)"]
    if built:
        notes.append(f"the centres were built for seq_len {built.get('seq_len')} and pred_len "
                     f"{built.get('pred_len')}; other lengths keep the same centres")
    if splits["train"].start < 0:
        notes.append(f"seq_len {seq_len} reaches before row 0 for the first train windows")
    shared = splits["train"].end - splits["val"].own_start
    if shared > 0:
        notes.append(f"the ratio splits windows, not rows: adjacent splits share up to {shared} target rows "
                     "(the last train targets overlap the first val targets)")
    if params.get("scale", True):
        train_end = built.get("train_end")
        fit = f" fitted on rows [0, {train_end})" if train_end is not None else ""
        scaling = f"z-score with the bundle mean/std ({bundle.name}){fit}; applied to every split"
    else:
        scaling = "none (scale = false): values and calendar covariates stay raw; metrics use raw values"
    return SplitProtocol(
        kind="window-index", **common, file_rows=rows, rows_used=rows, rows_basis="bundle data shape",
        split_ratio=ratio, ratio_of="windows", splits=splits, scaling=scaling, notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Split arrays
# ---------------------------------------------------------------------------


@dataclass
class SplitArrays:
    """Scaled split arrays plus the statistics to invert them."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    mean: np.ndarray | None
    std: np.ndarray | None
    protocol: SplitProtocol

    def __getitem__(self, flag: str) -> np.ndarray:
        return getattr(self, flag)


def load_splits(
    dataset: str,
    seq_len: int | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
    features: str = "M",
    pred_len: int | None = None,
    root: Path | None = None,
) -> SplitArrays:
    """Return the train/val/test arrays exactly as ``tsf run`` builds them.

    Each array holds the rows its split reads (see :class:`SplitRange`):
    validation and test include the ``seq_len`` input rows before their border.
    Scaling, truncation, and imputation are the loader's own. ``path`` applies
    the preset's protocol to another copy of the file.
    """
    protocol = split_protocol(dataset, seq_len, path=path, pred_len=pred_len, root=root)
    if protocol.kind == "loader-defined":
        raise ValueError(f"{protocol.dataset} has a loader-defined split; build it with "
                         "tsflab.data.build_data_loader")
    root = _root(root)
    _, table = _preset(protocol.dataset, root)
    spec = _spec(protocol.loader)
    params = _params(spec, table)
    location = _resolve_path(path if path is not None else str(table.get("path", "")))
    root_path, data_file = spec.resolve_location(str(location), None)
    horizon = max(1, pred_len or 1)
    arrays: dict[str, np.ndarray] = {}
    dataset_obj = None
    for flag in SPLITS:
        dataset_obj = spec.dataset_class(root_path=root_path, data_path=data_file,
                                         size=(protocol.seq_len, 0, horizon), flag=flag,
                                         features=features, **params)
        if protocol.kind == "chronological":
            arrays[flag] = np.asarray(dataset_obj.data)
        else:
            r = protocol.splits[flag]
            arrays[flag] = np.asarray(dataset_obj.data[max(r.start, 0):r.end])
    mean = getattr(dataset_obj, "_mean", None)
    std = getattr(dataset_obj, "_std", None)
    if protocol.kind == "window-index" and getattr(dataset_obj, "value_mean", None) is not None:
        mean, std = np.asarray([dataset_obj.value_mean]), np.asarray([dataset_obj.value_std])
    return SplitArrays(arrays["train"], arrays["val"], arrays["test"], mean, std, protocol)
