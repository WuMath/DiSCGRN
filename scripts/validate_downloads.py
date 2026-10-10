#!/usr/bin/env python3
"""Validate raw inputs and produce a machine-readable provenance manifest."""

from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from pathlib import Path

import pandas as pd

from ctcpgrn.io import read_dream_signed_network, read_gse74243_expression, sha256sum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/manifest.json"))
    args = parser.parse_args()

    raw = args.root.resolve()
    files = sorted(path for path in raw.rglob("*") if path.is_file())
    records = []
    for path in files:
        integrity = "not_applicable"
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                while handle.read(1024 * 1024):
                    pass
            integrity = "gzip_ok"
        elif path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
                if bad:
                    raise RuntimeError(f"Corrupt ZIP member: {bad}")
            integrity = "zip_ok"
        records.append(
            {
                "path": str(path.relative_to(raw.parent)),
                "bytes": path.stat().st_size,
                "sha256": sha256sum(path),
                "integrity": integrity,
            }
        )

    expression_path = raw / "gse74243" / "GSE74243_matrix_RMAlog2expressionvalues.txt.gz"
    expression, metadata = read_gse74243_expression(expression_path)
    gse_summary = {
        "n_gene_rows": int(len(expression)),
        "n_samples": int(len(metadata)),
        "strains": metadata.groupby("strain").size().astype(int).to_dict(),
        "stages": sorted(metadata["stage"].unique().tolist(), key=lambda x: float(x[1:])),
        "n_unique_times": int(metadata["time"].nunique()),
    }

    gold_files = sorted(raw.glob("dream4/**/*goldstandard_signed.tsv"))
    dream_records = []
    for path in gold_files:
        edges = read_dream_signed_network(path)
        genes = set(edges["source"]) | set(edges["target"])
        dream_records.append(
            {
                "path": str(path.relative_to(raw.parent)),
                "n_genes": len(genes),
                "n_signed_edges": int((edges["sign"] != 0).sum()),
                "n_positive": int((edges["sign"] > 0).sum()),
                "n_negative": int((edges["sign"] < 0).sum()),
            }
        )
    payload = {"files": records, "gse74243": gse_summary, "dream4_signed_networks": dream_records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), **gse_summary, "dream_networks": len(dream_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

