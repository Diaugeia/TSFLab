"""Fetch METR-LA from the official DCRNN sources and build the TSFLab node bundle.

DCRNN (Li et al., ICLR 2018) ships ``metr-la.h5`` (5-minute speeds, 34272 x 207)
through the Google Drive folder linked from its README, and the sensor graph
``adj_mx.pkl`` in its GitHub repository. This command downloads both (pinned
SHA-256; the pickle from commit ``602afd9d``), reads the h5 value block with
``h5py``, and builds the bundle exactly as ``tsf data prepare --from traffic``
does (calendar covariates, 12/12 windows, 7:1:2 split). Every output is checked
against the pinned SHA-256 of the published bundle; a mismatch is an error.

    uv run tsf data prepare --from dcrnn                         # -> dataset/metr_la
    uv run tsf data prepare --from dcrnn --h5 metr-la.h5         # an already downloaded h5

If Google Drive refuses the download, open the folder URL in a browser, save
``metr-la.h5``, and pass it with ``--h5``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

from tsflab.data.prepare.fetch import cache_dir, download_gdrive, download_url, sha256_file, verify_sha256

FOLDER_URL = "https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
H5_ID = "1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC"
H5_SHA256 = "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05"
COMMIT = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
ADJ_URL = f"https://raw.githubusercontent.com/liyaguang/DCRNN/{COMMIT}/data/sensor_graph/adj_mx.pkl"
ADJ_SHA256 = "a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3"
#: The published ``dataset/metr_la`` bundle (``SOURCE.txt``, retrieved 2026-10-03).
OUTPUTS = {
    "adj_mx.npy": "9b198090f595fb5b988e6cde9c267b305bad36cd3f34d17848be357ad8678765",
    "his.npz": "1a5e8de4007d29687a0dd32c348e264ce32ddec765199b7c70f5e7d8ab08f4e8",
    "idx_train.npy": "8818412af45945db7f5337592e650d36626284598f3434f54fb496da00cee118",
    "idx_val.npy": "2463e2c7bee32344217165a8c575f9b371402e4ae607861a7b80dc11440c3fab",
    "idx_test.npy": "5ac96fff7ffcfb10e19bf9edb75ab6fb32ad739aca17c4c8263330d99a064a98",
    "split.json": "5b65e32d1645f08fbf97608241689bd408cd8d853ba20d4042472eb462e83b98",
}


def read_values(h5: Path):
    """The ``(T, N)`` float32 speed matrix of DCRNN's pandas ``fixed``-format h5."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("reading metr-la.h5 needs `h5py` (`uv sync --extra data`)") from exc
    import numpy as np

    # The 2019 py2-era pandas file is not readable by current pandas.read_hdf; read the block directly.
    with h5py.File(h5, "r") as handle:
        return handle["df"]["block0_values"][:].astype(np.float32)


def build(h5: Path, adj: Path, out: Path) -> list[Path]:
    """Build the bundle in a staging directory, verify it, then move it to ``out``."""
    from tsflab.cli.commands.convert_traffic import _load_adjacency, build_bundle

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out, prefix=".dcrnn.") as staging:
        build_bundle(read_values(h5), staging, adj=_load_adjacency(str(adj), None), add_time=True, freq_min=5)
        for name, digest in OUTPUTS.items():
            verify_sha256(Path(staging) / name, digest)
        written = []
        for name in OUTPUTS:
            (Path(staging) / name).replace(out / name)
            written.append(out / name)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsf data prepare --from dcrnn", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", type=Path, help="an already downloaded metr-la.h5")
    parser.add_argument("--adj", type=Path, help="an already downloaded adj_mx.pkl")
    parser.add_argument("--out", type=Path, default=Path("dataset") / "metr_la",
                        help="bundle directory (default: ./dataset/metr_la)")
    args = parser.parse_args(argv)
    raw = cache_dir("dcrnn")
    try:
        h5 = args.h5 or download_gdrive(
            H5_ID, raw / "metr-la.h5", H5_SHA256,
            manual=f"Download metr-la.h5 from {FOLDER_URL} in a browser and pass it with --h5 PATH.")
        adj = args.adj or download_url(ADJ_URL, raw / "adj_mx.pkl", ADJ_SHA256)
        verify_sha256(h5, H5_SHA256)
        verify_sha256(adj, ADJ_SHA256)
        if all((args.out / name).is_file() and sha256_file(args.out / name) == digest
               for name, digest in OUTPUTS.items()):
            print(f"{args.out} already verified")
            return 0
        for path in build(h5, adj, args.out):
            print(f"{path}: sha256 verified")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
