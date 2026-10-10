#!/usr/bin/env python3
"""Make the historical ``data/`` code paths resolve to the public ``Data/`` layout."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "Data"
    legacy = root / "data"
    data.mkdir(exist_ok=True)
    if legacy.exists():
        if legacy.samefile(data):
            print("data/ and Data/ already resolve to the same directory")
            return
        raise RuntimeError("data/ exists but is not Data/; inspect it before continuing")
    if legacy.is_symlink():
        raise RuntimeError("data/ is a broken symlink; inspect it before continuing")
    legacy.symlink_to("Data", target_is_directory=True)
    print("Created relative link data -> Data")


if __name__ == "__main__":
    main()
