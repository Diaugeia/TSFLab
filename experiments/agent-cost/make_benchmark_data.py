"""Export the raw files of the benchmarking task from TSFLab real-time panels.

    python make_benchmark_data.py <realtime-root> <out-dir>

For each track, the last full window that ends at END (hourly) is written as one CSV,
`date` plus one column per channel, the form every arm receives. Gaps are filled by
time interpolation (at most 6 hours) and then carried forward; the number of filled
values is printed and stored next to the file.
"""

import glob
import json
import sys
from pathlib import Path

import pandas as pd

END = "2026-09-30 23:00:00"
TRACKS = {"grid_ercot": 2 * 365 * 24, "weather_openmeteo_temp": 365 * 24}


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for track, rows in TRACKS.items():
        df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / track / "panel" / "*.parquet"))))
        df = df[~df.index.duplicated(keep="last")].sort_index()
        idx = pd.date_range(end=pd.Timestamp(END), periods=rows, freq="h")
        df = df.reindex(idx)
        missing = int(df.isna().sum().sum())
        df = df.interpolate(method="time", limit=6).ffill().bfill()
        df.index.name = "date"
        df.reset_index().to_csv(out / f"{track}.csv", index=False, float_format="%.4f")
        info = {"track": track, "start": str(idx[0]), "end": str(idx[-1]), "rows": rows,
                "channels": df.shape[1], "filled_values": missing}
        (out / f"{track}.json").write_text(json.dumps(info, indent=1) + "\n")
        print(info)


if __name__ == "__main__":
    main()
