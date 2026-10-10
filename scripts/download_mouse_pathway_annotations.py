#!/usr/bin/env python3
"""Cache official mouse GO, MGI identifier, and KEGG pathway snapshots."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SOURCES = {
    "MOUSE-mod.gaf.gz": "https://current.geneontology.org/annotations/gaf/MOUSE-mod.gaf.gz",
    "go-basic.obo": "https://current.geneontology.org/ontology/go-basic.obo",
    "MGI_Gene_Model_Coord.rpt": "https://www.informatics.jax.org/downloads/reports/MGI_Gene_Model_Coord.rpt",
    "kegg_mouse_pathways.tsv": "https://rest.kegg.jp/list/pathway/mmu",
    "kegg_mouse_pathway_genes.tsv": "https://rest.kegg.jp/link/mmu/pathway",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path) -> None:
    if path.stat().st_size < 100:
        raise ValueError(f"Download is unexpectedly small: {path}")
    if ".gz" in path.suffixes:
        with gzip.open(path, "rb") as handle:
            if not handle.read(100):
                raise ValueError(f"Empty gzip archive: {path}")
    else:
        with path.open("rb") as handle:
            if not handle.read(100):
                raise ValueError(f"Empty text file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/external/mouse_pathway_annotations_20261007"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "sources": {}}
    for name, url in SOURCES.items():
        destination = args.output_dir / name
        if destination.exists():
            validate(destination)
            print(f"[cached] {name}", flush=True)
        else:
            pending = args.output_dir / f"{name}.part"
            subprocess.run(["curl", "--fail", "--location", "--retry", "4",
                            "--retry-delay", "3", "--continue-at", "-",
                            "--output", str(pending), url], check=True)
            validate(pending)
            pending.rename(destination)
            print(f"[downloaded] {name}", flush=True)
        manifest["sources"][name] = {"url": url, "bytes": destination.stat().st_size,
                                     "sha256": sha256(destination)}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
