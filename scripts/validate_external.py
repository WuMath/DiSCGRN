#!/usr/bin/env python3
"""Validate CollecTRI and ChIP-Atlas downloads and record provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ctcpgrn.io import sha256sum


def main() -> None:
    collectri_path = Path("data/external/collectri_mouse.tsv")
    collectri = pd.read_csv(collectri_path, sep="\t", low_memory=False)
    chip_files = sorted(Path("data/external/chipatlas_mm10").glob("*.tsv"))
    chip_records = []
    for path in chip_files:
        table = pd.read_csv(path, sep="\t", low_memory=False)
        sample_columns = list(table.columns[2:])
        lung_columns = [column for column in sample_columns if "lung" in column.lower()]
        chip_records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256sum(path),
                "n_target_rows": len(table),
                "n_experiment_columns": len(sample_columns),
                "n_lung_columns": len(lung_columns),
                "lung_columns": lung_columns,
            }
        )
    manifest = {
        "collectri": {
            "path": str(collectri_path),
            "source_url": "https://omnipathdb.org/interactions?datasets=tf_target&resources=CollecTRI&organisms=10090&format=tsv&fields=sources,references&genesymbols=1",
            "bytes": collectri_path.stat().st_size,
            "sha256": sha256sum(collectri_path),
            "n_rows": len(collectri),
            "n_regulators": int(collectri["source_genesymbol"].nunique()),
            "n_targets": int(collectri["target_genesymbol"].nunique()),
        },
        "chip_atlas": {
            "genome": "mm10",
            "promoter_window_kb": 5,
            "index_url": "https://chip-atlas.org/data/target_genes_analysis.json",
            "index_sha256": sha256sum("data/external/chipatlas_target_genes_analysis.json"),
            "tables": chip_records,
        },
    }
    output = Path("data/external_manifest.json")
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

