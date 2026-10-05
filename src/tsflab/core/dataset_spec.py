"""DatasetSpec — the identity + protocol contract for an evaluated dataset.

A DatasetSpec is the "ID card" that pins *which exact bytes* and *which
evaluation protocol* a result was produced against. Its three-part identity
``(id, version, sha256)`` is what every leaderboard row references — never a
bare ``id`` — so that swapping data or splits is detectable.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import SCHEMA_VERSION, TaskMode, Track


class DatasetSpec(BaseModel):
    """Standardized identity + protocol for a benchmark dataset.

    Carries publication facts that do not belong in the short runtime
    ``DatasetConfig``: content digest, frequency, horizons, track, and optional
    Hugging Face coordinates. Optional facts remain null when unavailable; a
    published submission should pin every fact it claims.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    id: str = Field(
        pattern=r"^[a-z0-9_\-]+$",
        description="Dataset identifier; should map to a TSFLab dataset name key.",
    )
    version: str = Field(
        default="1.0.0",
        description="Immutable published version (Static: v1/v2; RealTime: as-of tag).",
    )
    sha256: str | None = Field(
        default=None,
        description="Content-address of the dataset snapshot (manifest self-hash).",
    )
    mode: TaskMode = Field(description="Forecasting data setting (task.mode).")
    track: Track | None = Field(
        default=None,
        description="Leaderboard track; defaults to `mode` when omitted.",
    )
    freq: str | None = Field(default=None, description="Sampling frequency, e.g. '1h', '15min'.")
    horizons: list[int] = Field(
        default_factory=list,
        description="Official prediction lengths for this dataset (empty = TBD).",
    )
    num_nodes: int | None = Field(default=None, description="Node count (spatiotemporal/covariate).")
    num_channels: int | None = Field(default=None, description="Channel count (time_series).")
    target: str | None = Field(default=None, description="Target variable/feature name, if applicable.")
    hf_path: str | None = Field(default=None, description="Hugging Face repo path, e.g. 'Diaugeia/TSFLab-Datasets:static/ett1'.")
    hf_revision: str | None = Field(default=None, description="Pinned HF commit/tag the version resolves to.")
    source_config: str | None = Field(default=None, description="Originating TSFLab dataset TOML path.")

    @model_validator(mode="after")
    def _default_track_from_mode(self) -> "DatasetSpec":
        if self.track is None:
            self.track = self.mode
        return self
