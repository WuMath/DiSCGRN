#!/usr/bin/env python3
"""Extract the prespecified BEELINE runs from Zenodo record 7682713.

Download beeline_datasets.zip separately, then pass its local path here. The
archive checksum is checked before any extraction. Existing files are checked
byte-for-byte and never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


ARCHIVE_MD5 = "857cb335da79d09865cd7db2549fbda4"
SERIES = (("dyn-LI", 100), ("dyn-LI", 200), ("dyn-LI", 500),
          ("dyn-LL", 100), ("dyn-LL", 200), ("dyn-LL", 500),
          ("dyn-CY", 200))
FILES = ("ExpressionData.csv", "PseudoTime.csv", "GroundTruth.csv", "refNetwork.csv")


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/beeline"))
    args = parser.parse_args()
    if md5_file(args.archive) != ARCHIVE_MD5:
        raise ValueError("Archive MD5 differs from Zenodo record 7682713")
    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        expected = [f"Synthetic/{network}/{network}-{cells}-{run}/{filename}"
                    for network, cells in SERIES for run in range(1, 11)
                    for filename in FILES]
        missing = sorted(set(expected) - names)
        if missing:
            raise ValueError(f"Archive is missing {len(missing)} expected files: {missing[:5]}")
        count = 0
        for name in expected:
            destination = args.output_root / name
            contents = archive.read(name)
            if destination.exists():
                if destination.read_bytes() != contents:
                    raise ValueError(f"Existing file differs from archive: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            count += 1
    print(f"Verified {len(expected)} files, extracted {count} new files to {args.output_root}")


if __name__ == "__main__":
    main()
