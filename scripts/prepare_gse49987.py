#!/usr/bin/env python3
"""Prepare a prespecified hematopoietic TF panel from GEO GSE49987.

The time-point replicates are independent cell samples, not longitudinal
replicates. The shared FDCPmix time-zero samples enter each lineage only as
the common baseline. This script does not fit a model or select genes by their
observed trajectory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PANEL = (
    "Gata1", "Gata2", "Spi1", "Tal1", "Klf1", "Cebpa", "Cebpb",
    "Cebpe", "Epor", "Mpo", "Itgam", "Kit", "Myb", "Runx1",
)


def line_number(path: Path, prefix: str) -> int:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle):
            if line.startswith(prefix):
                return number
    raise ValueError(f"Missing {prefix!r} in {path}")


def sample_metadata(path: Path) -> pd.DataFrame:
    wanted = {"!Sample_geo_accession", "!Sample_title"}
    found: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            key = line.split("\t", 1)[0]
            if key in wanted:
                found[key] = [value.strip().strip('"') for value in line.rstrip("\n").split("\t")[1:]]
            if line.startswith("!series_matrix_table_begin"):
                break
    if set(found) != wanted:
        raise ValueError(f"Incomplete GEO sample metadata: {set(found)}")
    if len(found["!Sample_geo_accession"]) != len(found["!Sample_title"]):
        raise ValueError("GEO sample IDs and titles have different lengths")
    rows = []
    for sample, title in zip(found["!Sample_geo_accession"], found["!Sample_title"]):
        baseline = re.fullmatch(r"FDCP T0 rep([1-3])", title)
        series = re.fullmatch(r"([EN])diff (\d+)h rep([1-3])", title)
        if baseline:
            for lineage in ("erythroid", "myeloid"):
                rows.append({"sample_id": sample, "title": title, "lineage": lineage,
                             "time_h": 0.0, "replicate": int(baseline.group(1)),
                             "shared_baseline": True})
        elif series:
            rows.append({"sample_id": sample, "title": title,
                         "lineage": "erythroid" if series.group(1) == "E" else "myeloid",
                         "time_h": float(series.group(2)),
                         "replicate": int(series.group(3)), "shared_baseline": False})
        else:
            raise ValueError(f"Unrecognized time-course sample: {sample} {title}")
    metadata = pd.DataFrame(rows)
    if metadata.sample_id.nunique() != 177 or len(metadata) != 180:
        raise ValueError("Unexpected GSE49987 sample count")
    counts = metadata.groupby(["lineage", "time_h"]).size()
    if not counts.eq(3).all():
        raise ValueError("Each lineage/time point should have three samples")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/gse49987"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/gse49987_hematopoietic"))
    args = parser.parse_args()
    matrix_path = args.raw_dir / "GSE49987_series_matrix.txt.gz"
    annot_path = args.raw_dir / "GPL7202.annot.gz"
    metadata = sample_metadata(matrix_path)
    annotation = pd.read_csv(annot_path, sep="\t", skiprows=line_number(annot_path, "ID\t"),
                             usecols=["ID", "Gene symbol"], dtype=str)
    annotation = annotation.rename(columns={"ID": "ID_REF", "Gene symbol": "gene"})
    annotation = annotation[annotation.gene.isin(PANEL)].drop_duplicates("ID_REF")
    if annotation.empty:
        raise ValueError("No prespecified genes mapped to GPL7202")
    matrix = pd.read_csv(matrix_path, sep="\t",
                         skiprows=line_number(matrix_path, "!series_matrix_table_begin") + 1,
                         low_memory=False)
    matrix = matrix[matrix.ID_REF.isin(annotation.ID_REF)].copy()
    matrix = matrix.merge(annotation, on="ID_REF", validate="one_to_one")
    matrix = matrix.drop(columns="ID_REF").set_index("gene")
    sample_ids = metadata.sample_id.drop_duplicates().tolist()
    if set(matrix.columns) != set(sample_ids):
        raise ValueError("Expression matrix columns do not match GEO sample IDs")
    matrix = matrix[sample_ids].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Nonfinite expression value in panel")
    # Several probes may map to one gene. Their sample-wise median is fixed
    # independently of trajectory behavior, avoiding outcome-driven selection.
    collapsed = matrix.groupby(level=0).median().reindex([g for g in PANEL if g in matrix.index])
    counts = annotation.groupby("gene").size().to_dict()
    observed = metadata.merge(collapsed.T.rename_axis("sample_id").reset_index(),
                              on="sample_id", validate="many_to_one")
    means = observed.groupby(["lineage", "time_h"], as_index=False)[list(collapsed.index)].mean()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output_dir / "sample_metadata.tsv", sep="\t", index=False)
    observed.to_csv(args.output_dir / "replicate_expression.tsv", sep="\t", index=False)
    means.to_csv(args.output_dir / "mean_trajectories.tsv", sep="\t", index=False)
    details = {"accession": "GSE49987", "platform": "GPL7202",
               "panel_prespecified": list(PANEL), "mapped_genes": list(collapsed.index),
               "probe_counts": counts, "unique_samples": len(sample_ids),
               "lineage_timepoints": means.groupby("lineage").size().to_dict(),
               "note": "GEO deposited values, no second log transform; independent samples per time; shared T0"}
    (args.output_dir / "processing_manifest.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
