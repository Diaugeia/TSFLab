"""Base dataset class for forecasting tasks."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import ClassVar, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from tsflab.data.protocol import split_borders, train_rows


class ForecastingDataset(ABC, Dataset):
    """Base class for forecasting datasets.

    Parameters
    ----------
    root_path : str
        Root directory for the dataset.
    data_path : str
        Dataset file name.
    size : tuple[int, int, int]
        Sequence length, label length, prediction length.
    flag : str, optional
        Split flag: "train", "val", or "test".
    features : str, optional
        Feature mode ("M", "S", "MS").
    target : str, optional
        Target column name.
    split_ratio : tuple[float, float, float], optional
        Train/val/test split ratios.
    scale : bool, optional
        Whether to scale features.
    target_channel : int or None, optional
        Index of the model's target value channel within the scaled feature
        matrix. When set, ``inverse_transform`` is anchored on this channel's
        statistics so a single-column (target-only) prediction round-trips
        correctly even when the matrix carries covariate channels. ``None``
        (default) keeps the original behavior (the scaler inverts whatever
        columns it is handed).
    norm_each_channel : bool, optional
        When ``True`` normalize with per-channel mean/std computed explicitly
        on the training split. When ``False`` (default) the original
        ``StandardScaler`` path is used (also per-column, byte-identical to
        the previous behavior). The flag mainly exists so the value/covariate
        split needed by covariate (node) task mode is computed under our own
        control rather than sklearn's.

    Notes
    -----
    The official split of every subclass is defined by ``max_rows`` (rows kept
    from the file start before splitting), ``split_borders``, and
    ``train_rows`` in :mod:`tsflab.data.protocol`; ``tsf data splits`` prints it.
    """

    #: Rows kept from the start of the file before the split (``None`` keeps all).
    max_rows: ClassVar[int | None] = None
    #: Whether the first line of the file is a header row.
    has_header: ClassVar[bool] = True

    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: tuple[int, int, int],
        flag: str = "train",
        features: str = "S",
        target: str = "OT",
        split_ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
        scale: bool = True,
        target_channel: int | None = None,
        norm_each_channel: bool = False,
    ):
        super().__init__()
        self.file_path = os.path.join(root_path, data_path)
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.scale = scale
        self.target_channel = target_channel
        self.norm_each_channel = norm_each_channel
        self.scaler = None
        # Per-channel statistics captured when ``norm_each_channel`` is on (or
        # ``target_channel`` anchoring is requested). ``None`` when the plain
        # ``StandardScaler`` path is used.
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.data, self.time_stamp = self._read_data(
            flag, features, target, split_ratio, scale
        )

    def __len__(self) -> int:
        """Return number of windows available in the split."""
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> Tuple:
        """Return one input/target window and optional timestamps.

        Parameters
        ----------
        index : int
            Window start index.

        Returns
        -------
        tuple
            (input_series, output_series, input_stamp, output_stamp)
        """
        input_start = index
        input_end = input_start + self.seq_len
        output_start = input_end - self.label_len
        output_end = input_end + self.pred_len

        input_series = self.data[input_start:input_end]
        output_series = self.data[output_start:output_end]

        if self.time_stamp is not None:
            input_stamp = self.time_stamp[input_start:input_end]
            output_stamp = self.time_stamp[output_start:output_end]
        else:
            input_stamp = np.zeros((input_end - input_start, 6), dtype=np.float32)
            output_stamp = np.zeros((output_end - output_start, 6), dtype=np.float32)

        return input_series, output_series, input_stamp, output_stamp

    def _apply_scaling(
        self,
        values: np.ndarray,
        train_len: int,
    ) -> np.ndarray:
        """Z-score ``values`` (``(T, C)``) using the training split statistics.

        The default path (``norm_each_channel`` off) uses sklearn's
        ``StandardScaler`` exactly as before, so behavior is unchanged. When
        ``norm_each_channel`` is on, per-channel mean/std are computed directly
        on the first ``train_len`` rows.

        Regardless of branch, the per-channel ``mean``/``std`` are cached on
        ``self._mean`` / ``self._std`` so ``inverse_transform`` can anchor on a
        chosen ``target_channel`` for the covariate task mode.

        Parameters
        ----------
        values : np.ndarray
            Full ``(T, C)`` feature matrix (before slicing the split window).
        train_len : int
            Number of leading rows that make up the training split; statistics
            are fit on ``values[:train_len]`` only.

        Returns
        -------
        np.ndarray
            The scaled ``(T, C)`` matrix.
        """
        values = np.asarray(values, dtype=np.float64)
        train_data = values[:train_len]

        if self.norm_each_channel:
            mean = train_data.mean(axis=0)
            std = train_data.std(axis=0)
            std = np.where(std == 0, 1.0, std)
            self._mean = mean
            self._std = std
            scaled = (values - mean) / std
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)
            scaled = self.scaler.transform(values)
            # Mirror the scaler's stats so target-channel anchoring works in
            # both branches (StandardScaler is per-column too).
            self._mean = np.asarray(self.scaler.mean_, dtype=np.float64)
            self._std = np.asarray(self.scaler.scale_, dtype=np.float64)

        return scaled

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data using the fitted statistics.

        When ``target_channel`` is set, ``data`` is treated as the model's
        target output and inverted with that single channel's mean/std. This
        keeps the value channel correct even when the original feature matrix
        also held covariate channels with their own statistics. Otherwise the
        original column-wise scaler inversion is used.

        Parameters
        ----------
        data : np.ndarray
            Scaled data, shape ``(N, C)``.

        Returns
        -------
        np.ndarray
            Unscaled data.
        """
        if self.target_channel is not None and self._mean is not None:
            # Anchor on the value channel's statistics. The model emits the
            # target channel(s); invert with that channel's mean/std regardless
            # of how many columns ``data`` carries.
            mean = float(self._mean[self.target_channel])
            std = float(self._std[self.target_channel])
            return np.asarray(data) * std + mean

        data = np.asarray(data)

        if self._mean is not None and self.scaler is None:
            # ``norm_each_channel`` path without target anchoring: invert
            # column-wise with the cached per-channel statistics.
            if data.shape[-1] < len(self._std):
                # MS / reduced-channel output: anchor on the last k channels
                # (the target channel(s), placed last in TS datasets).
                k = data.shape[-1]
                return data * self._std[len(self._std) - k:] + self._mean[len(self._mean) - k:]
            return data * self._std + self._mean

        if self.scaler is None:
            return data
        # MS / reduced-channel target output: the scaler was fit on all C
        # channels but the model emits fewer (the target channel(s), which TS
        # datasets place last). Anchor on the last k channels' stats — the full
        # per-column ``scaler.inverse_transform`` broadcast-fails on a (N, k<C)
        # array. Full-width (M mode, k == C) stays byte-identical.
        n = self.scaler.scale_.shape[0]
        if data.shape[-1] < n:
            k = data.shape[-1]
            return data * self.scaler.scale_[n - k:] + self.scaler.mean_[n - k:]
        return self.scaler.inverse_transform(data)

    def _build_time_stamp(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Generate time feature matrix from a date column.

        Parameters
        ----------
        df_raw : pandas.DataFrame
            Raw dataframe containing a "date" column.

        Returns
        -------
        np.ndarray
            Time feature matrix with year/month/day/weekday/hour/minute.
        """
        df_stamp = pd.DataFrame()
        df_stamp["date"] = pd.to_datetime(df_raw["date"])
        df_stamp["year"] = df_stamp.date.dt.year
        df_stamp["month"] = df_stamp.date.dt.month
        df_stamp["day"] = df_stamp.date.dt.day
        df_stamp["weekday"] = df_stamp.date.dt.weekday
        df_stamp["hour"] = df_stamp.date.dt.hour
        df_stamp["minute"] = df_stamp.date.dt.minute
        df_stamp = df_stamp.drop(["date"], axis=1).values
        return df_stamp

    @classmethod
    def count_rows(cls, file_path: str) -> int:
        """Return the number of data rows in ``file_path`` (blank lines skipped)."""
        with open(file_path, "rb") as handle:
            rows = sum(1 for line in handle if line.strip())
        return rows - 1 if cls.has_header else rows

    def _limit_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep the first ``max_rows`` rows of ``frame`` (all rows when unset)."""
        return frame.iloc[: self.max_rows, :]

    def _train_len(self, split_ratio: tuple[float, float, float], num_samples: int) -> int:
        """Rows the scaler is fitted on: the training split of the rows used."""
        return train_rows(num_samples, split_ratio)

    def _get_borders(
        self,
        flag: str,
        split_ratio: tuple[float, float, float],
        num_samples: int,
    ) -> Tuple[int, int]:
        """Return ``(start, end)`` rows of ``flag``; see ``protocol.split_borders``."""
        return split_borders(num_samples, split_ratio, self.seq_len)[flag]

    @abstractmethod
    def _read_data(
        self,
        flag: str,
        features: str,
        target: str,
        split_ratio: tuple[float, float, float],
        scale: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read data for a split and return series and timestamps.

        Parameters
        ----------
        flag : str
            Split flag: "train", "val", or "test".
        features : str
            Feature mode ("M", "S", "MS").
        target : str
            Target column name.
        split_ratio : tuple[float, float, float]
            Train/val/test split ratios.
        scale : bool
            Whether to scale features.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Series data and timestamp features.
        """
        raise NotImplementedError
