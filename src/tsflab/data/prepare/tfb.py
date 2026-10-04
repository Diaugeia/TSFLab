"""Fetch the TFB forecasting archive and convert its members to TSFLab wide CSVs.

TFB (Qiu et al., PVLDB 2024) distributes its pre-processed forecasting datasets
as one Google Drive archive. Each member ``forecasting/<Name>.csv`` uses TFB's
long layout (``date,data,cols[,name]``: one contiguous block per channel). This
command pivots a member to the wide CSV that the ``custom`` loader reads (``date``
plus one column per channel, channels in order of first appearance), exactly as
TFB ``ts_benchmark/data/utils.py::read_data`` does. Values and dates are unchanged.

The archive, every member, and every output are checked against pinned SHA-256
values; a mismatch is an error. Outputs go to ``<out>/<Dir>/<Name>.csv``, the
path of the TSFLab preset.

    uv run tsf data prepare --from tfb                                  # every covered set
    uv run tsf data prepare --from tfb --datasets fred_md NASDAQ
    uv run tsf data prepare --from tfb --archive forecasting.zip        # an already downloaded archive

If Google Drive refuses the download, open the archive URL in a browser, save
the zip, and pass it with ``--archive``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import pandas as pd

from tsflab.data.prepare.fetch import cache_dir, download_gdrive, sha256_file

ARCHIVE_ID = "1vgpOmAygokoUt235piWKUjfwao6KwLv7"
ARCHIVE_URL = f"https://drive.google.com/file/d/{ARCHIVE_ID}/view"
ARCHIVE_SHA256 = "01794d0000ccc7e033523481cfa117f647c9740d6efbf0626f56228f1bcf785c"


@dataclass(frozen=True)
class TFBSet:
    """One archive member, its TSFLab preset, conversion options, and pinned digests."""

    name: str
    preset: str
    member_sha256: str
    sha256: str
    #: Dates from the first block and every block by position (TFB's per-block date rotation).
    positional: bool = False
    #: Rename the last channel to ``OT`` (Time-Series-Library target convention).
    rename_last_to_ot: bool = False

    @property
    def relative(self) -> str:
        return f"{self.name}/{self.name}.csv"


#: Digests from each dataset's ``dataset/<Dir>/SOURCE.txt`` (retrieved 2026-10-03).
SETS = {s.name: s for s in (
    # License forbids re-hosting (card class ``script``).
    TFBSet("FRED-MD", "fred_md", "88763bed6f1207a6918b7ee7efb2283dbc68ea78102e8303fa26486a2ed97c15",
           "bfdf46330a3de901873c95df8ff520638d080b9c264be8cc5fa41c36a86d056f", rename_last_to_ot=True),
    TFBSet("NASDAQ", "nasdaq", "11ee1dfad4659957fe19b06e35d36de9e9ced3d8298766e368c2ba05432f2a25",
           "ab9efecb4bcc5a6a5b5dfd0207ce40c13072924f5497c32f82d75fcf0be9f4b4", positional=True),
    TFBSet("NYSE", "nyse", "0c7786baac1a0f244ca48254d27a57221e15a4b96eca628df25baf4df04ce502",
           "0befbd24145635b20bd77828491df1fe11073e2ddda97aca95f7ba382fcdccb6", positional=True),
    TFBSet("Wike2000", "wike2000", "b3c4c2c9f1f591e066e1b8d5e981d5052b90712e7226fff8bfe900545aa46422",
           "8da398cd540fbbd2ef8d52f779eb09b4b8b87b1851dd81ca743b4126741b46fd", rename_last_to_ot=True),
    # Hosted by TSFLab; rebuilt here so the hosted files stay reproducible.
    TFBSet("AQShunyi", "aqshunyi", "c4e3ecefc53c3ff4da545b5da15d19e3a1342e31251444a91821073000a77d7e",
           "8aae73ca69d9eea0ba7ee08387d7c9bfd15f0198fa5a1f50e6a69ffb3f943212"),
    TFBSet("AQWan", "aqwan", "e30e969666f795c3763fbabc58424bfc5751422db6660a110f504b7df8529af2",
           "16b973031c43416333782cbb12d073fe7cb4df2cbb129d6cf07afff14e1441a3"),
    TFBSet("CzeLan", "czelan", "b09c1dfa7279eff3aa53c883f0be71ebf2ef58d2980863ec01c38316a4beb15d",
           "6a59b7efdc111c615d5444c5a7a4de7e281623760162ce2c23313dc36229131b"),
    TFBSet("ZafNoo", "zafnoo", "7dffc18f7655fcf191fa424b5c720eac888c233e6e639819406f460d4fdf62e3",
           "a4df4902e4b0f5b6ed5521c06a48ee51a9d3fabd0c2e7077328fa6d508886204"),
    TFBSet("Wind", "wind", "db0b78b5d32440cbeee2fc47dcd6d81f28c33b76da36aa343b28b6dff1c023c7",
           "8d0a06a4de54f4c87b77a5bca162b7cca29392c34334ad31d3e6f52587bc1c49"),
    TFBSet("NN5", "nn5", "012736b21e2fe17420c488346ce879ee593331a5bbdc599ed43a6d6ac9ef2dd5",
           "8078ec48f58351763c0721171352008212e2878ebbd8361ab52511199785347a", rename_last_to_ot=True),
    TFBSet("Covid-19", "covid19", "60930b1bc46867659876a007d707136500898f06247b8b34e65e6f4c266ab004",
           "43f574ee1c9b07882ce696ee88a5ebc4d0db3ac6748af2fa2a44121279694685"),
)}


def resolve(names: list[str] | None) -> list[TFBSet]:
    """Map TFB names or preset ids (any case) to sets; ``None`` means every set."""
    if not names:
        return list(SETS.values())
    lookup = {key.lower(): s for s in SETS.values() for key in (s.name, s.preset)}
    unknown = [name for name in names if name.lower() not in lookup]
    if unknown:
        raise ValueError(f"unknown TFB dataset(s) {unknown}; choose from {', '.join(SETS)}")
    return [lookup[name.lower()] for name in names]


def long_to_wide(frame: pd.DataFrame, *, positional: bool = False, rename_last_to_ot: bool = False) -> pd.DataFrame:
    """Pivot a TFB long frame (``date,data,cols[,name]``) to ``date`` plus one column per channel."""
    cols = frame["cols"].unique()
    n = int(frame["cols"].value_counts().max())
    if len(frame) != n * len(cols):
        raise ValueError("ragged blocks: every channel needs the same number of rows")
    dates = frame["date"].iloc[:n].reset_index(drop=True)
    out = {"date": dates}
    for j, col in enumerate(cols):
        block = frame.iloc[j * n:(j + 1) * n]
        if not (block["cols"] == col).all():
            raise ValueError(f"block {j} ({col}) is not contiguous")
        same = block["date"].reset_index(drop=True) == dates
        if not same.all():
            if not positional:
                raise ValueError(f"block {col} dates differ from block 0; TFB reads such files by position")
            print(f"  block {col}: {int((~same).sum())} date labels differ from block 0; "
                  "kept by position as TFB read_data does")
        out[col] = block["data"].to_numpy()
    wide = pd.DataFrame(out)
    if rename_last_to_ot:
        if "OT" in wide.columns:
            raise ValueError("the frame already has an OT column")
        wide = wide.rename(columns={wide.columns[-1]: "OT"})
    return wide


def convert(archive: Path, chosen: list[TFBSet], out: Path) -> list[Path]:
    """Convert the chosen members of ``archive`` into ``out`` and verify every digest."""
    written = []
    with zipfile.ZipFile(archive) as source:
        for item in chosen:
            target = out / item.relative
            if target.is_file() and sha256_file(target) == item.sha256:
                print(f"{item.name}: {target} already verified")
                written.append(target)
                continue
            payload = source.read(f"forecasting/{item.name}.csv")
            actual = hashlib.sha256(payload).hexdigest()
            if actual != item.member_sha256:
                raise ValueError(f"checksum mismatch for archive member forecasting/{item.name}.csv: "
                                 f"expected {item.member_sha256}, got {actual}")
            wide = long_to_wide(pd.read_csv(io.BytesIO(payload)), positional=item.positional,
                                rename_last_to_ot=item.rename_last_to_ot)
            text = wide.to_csv(index=False).encode("utf-8")
            actual = hashlib.sha256(text).hexdigest()
            if actual != item.sha256:
                raise ValueError(f"checksum mismatch for converted {item.relative}: expected {item.sha256}, "
                                 f"got {actual} (pandas {pd.__version__})")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text)
            print(f"{item.name}: {target} {wide.shape[0]} rows x {wide.shape[1] - 1} channels, sha256 verified")
            written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsf data prepare --from tfb", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", metavar="NAME",
                        help="TFB names or preset ids (default: all of " + ", ".join(SETS) + ")")
    parser.add_argument("--archive", type=Path, help="an already downloaded TFB forecasting zip")
    parser.add_argument("--out", type=Path, default=Path("dataset"), help="dataset root (default: ./dataset)")
    parser.add_argument("--list", action="store_true", help="print the covered datasets and exit")
    args = parser.parse_args(argv)
    if args.list:
        for item in SETS.values():
            print(f"{item.name}\t{item.preset}\t{item.relative}")
        return 0
    try:
        chosen = resolve(args.datasets)
        archive = args.archive
        if archive is None:
            archive = download_gdrive(
                ARCHIVE_ID, cache_dir("tfb") / "forecasting.zip", ARCHIVE_SHA256,
                manual=f"Download {ARCHIVE_URL} in a browser and pass the zip with --archive PATH.")
        elif sha256_file(archive) != ARCHIVE_SHA256:
            raise ValueError(f"{archive} is not the pinned TFB archive (sha256 {ARCHIVE_SHA256})")
        convert(archive, chosen, args.out)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
