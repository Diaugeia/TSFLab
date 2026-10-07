"""ETT dataset implementations."""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np
import pandas as pd

from tsflab.data.schemas.datasets.ett import DatasetParameterConfig
from tsflab.catalog.registry import DATASET_REGISTRY
from tsflab.data.datasets.base import ForecastingDataset


class _ETTDataset(ForecastingDataset):
    """ETT split following the original Informer setup.

    Only the first ``max_rows`` rows (12 + 4 + 4 = 20 months) are used; the
    remaining rows of the CSV are never read. The split ratio applies to those
    rows.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "S",
        target: str = "OT",
        split_ratio: tuple[float, float, float] = (12, 4, 4),
        scale: bool = True,
        target_channel: int | None = None,
        norm_each_channel: bool = False,
    ):
        super().__init__(
            root_path,
            data_path,
            size,
            flag,
            features,
            target,
            split_ratio,
            scale,
            target_channel,
            norm_each_channel,
        )

    def _read_data(
        self,
        flag: str,
        features: str,
        target: str,
        split_ratio: tuple[float, float, float],
        scale: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read the ETT data and return split series and timestamps."""
        df_raw = self._limit_rows(pd.read_csv(self.file_path))
        num_samples = len(df_raw)
        border1, border2 = self._get_borders(flag, split_ratio, num_samples)

        if features in {"M", "MS"}:
            df_data = cast(pd.DataFrame, df_raw.iloc[:, 1:].copy())
        else:
            df_data = cast(pd.DataFrame, df_raw.loc[:, [target]].copy())

        if scale:
            train_len = self._train_len(split_ratio, num_samples)
            data = self._apply_scaling(df_data.to_numpy(), train_len)
        else:
            data = df_data.to_numpy()

        data = np.asarray(data)

        time_stamp = np.asarray(self._build_time_stamp(df_raw))
        series_data = np.asarray(data[border1:border2])
        time_stamp = time_stamp[border1:border2]
        return cast(np.ndarray, series_data), cast(np.ndarray, time_stamp)


class Dataset_ETT_hour(_ETTDataset):
    """ETT hourly dataset: the first 20 months at 24 rows per day."""

    max_rows = 20 * 30 * 24  # 14,400 rows (12/4/4 months of 30 days)


class Dataset_ETT_minute(_ETTDataset):
    """ETT 15-minute dataset: the first 20 months at 96 rows per day."""

    max_rows = 20 * 30 * 24 * 4  # 57,600 rows (12/4/4 months of 30 days)


def register() -> None:
    """Register ETT datasets by name."""
    modes = frozenset({"time_series"})
    DATASET_REGISTRY.register("ETTh1", Dataset_ETT_hour, DatasetParameterConfig, task_modes=modes)
    DATASET_REGISTRY.register("ETTh2", Dataset_ETT_hour, DatasetParameterConfig, task_modes=modes)
    DATASET_REGISTRY.register("ETTm1", Dataset_ETT_minute, DatasetParameterConfig, task_modes=modes)
    DATASET_REGISTRY.register("ETTm2", Dataset_ETT_minute, DatasetParameterConfig, task_modes=modes)
