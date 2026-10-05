"""Parameter schema for the real-time panel datasets."""

from pydantic import Field

from tsflab.data.schemas.base import DatasetParameters


class DatasetParameterConfig(DatasetParameters):
    """Validated parameters for ``realtime_panel_ts`` / ``realtime_panel_st``.

    Parameters
    ----------
    track : str
        Real-time track id (``configs/realtime/<track>.toml``); the store lives
        at ``<dataset.path>`` = ``dataset/realtime/<track>``.
    version : str | None
        Release ``version`` from the store manifest to freeze the data at. The
        panel is cut at that release's ``last_timestamp``; the latest release
        additionally has its content hash verified. ``None`` reads the whole
        local store, which keeps growing and is not reproducible.
    revision : str | None
        Hugging Face dataset revision (commit) to pull the track from instead of
        a local store; cached under ``<dataset.path>/../_hub/<revision>``.
    repo_id : str | None
        Hub dataset repository (default ``Diaugeia/TSFLab-Datasets``; the track
        is read from its ``realtime/<track>/`` folder).
    start, end : str | None
        Optional inclusive time bounds applied before the split.
    split_ratio : tuple[float, float, float]
        Chronological train/val/test fractions.
    scale : bool
        Z-score with one mean/std taken from the training rows only.
    calendar : bool
        Attach calendar covariates (spatiotemporal layout).
    max_windows : int | None
        Evenly subsample windows per split (smoke runs).
    """

    track: str
    version: str | None = None
    revision: str | None = None
    repo_id: str | None = None
    start: str | None = None
    end: str | None = None
    split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2)
    scale: bool = True
    calendar: bool = True
    max_windows: int | None = Field(default=None, gt=0)
