#!/usr/bin/env python3
"""Count measured CollecTRI regulators and variance-ranked genes in GSE74243.

The analysis mirrors the existing real-data preprocessing. It does not treat
the chosen 50-gene panel as a count of all measured or expressed genes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.external import read_collectri
from ctcpgrn.io import read_gse74243_expression


def count_one(expression: pd.DataFrame, metadata: pd.DataFrame,
              regulator_catalogue: set[str], excluded_times: set[float]) -> dict:
    training = metadata[~metadata.time.isin(excluded_times)]
    samples = training["sample"].tolist()
    work = expression[["Gene ID", "Gene Symbol", *samples]].copy()
    work = work[work["Gene Symbol"].notna()].copy()
    work["Gene Symbol"] = work["Gene Symbol"].astype(str).str.strip()
    work = work[work["Gene Symbol"].ne("")].copy()
    measured_symbols = set(work["Gene Symbol"])
    work["selection_variance"] = np.nanvar(work[samples].to_numpy(dtype=float), axis=1, ddof=1)
    work = work[np.isfinite(work.selection_variance)].copy()
    work = work.sort_values(["selection_variance", "Gene Symbol", "Gene ID"],
                            ascending=[False, True, True], kind="stable")
    work = work.drop_duplicates("Gene Symbol", keep="first")
    eligible_symbols = set(work["Gene Symbol"])
    measured_regulators = regulator_catalogue & measured_symbols
    eligible_regulators = regulator_catalogue & eligible_symbols
    return {
        "n_training_samples": len(samples),
        "n_probe_rows_with_symbol": int(expression["Gene Symbol"].notna().sum()),
        "n_unique_measured_symbols": len(measured_symbols),
        "n_unique_finite_variance_symbols": len(eligible_symbols),
        "n_collectri_regulators_measured": len(measured_regulators),
        "n_collectri_regulators_finite_variance": len(eligible_regulators),
        "n_nonregulator_genes_finite_variance": len(eligible_symbols - eligible_regulators),
        "selected_regulators": 20,
        "selected_other_genes": 30,
        "selected_total": 50,
        "selection_rule": "one fixed Nkx2-1 anchor; 19 further measured CollecTRI sources by training variance; 30 other genes by training variance",
    }


def main() -> None:
    root = Path("data/processed/real_study_20261004")
    expression, metadata = read_gse74243_expression(
        Path("data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz"))
    metadata = metadata[metadata.time > 9.5].copy()
    expression = expression[["Gene ID", "Gene Symbol", *metadata["sample"].tolist()]]
    collectri = read_collectri(Path("data/external/collectri_mouse.tsv"))
    catalogue = set(collectri.regulator.dropna().astype(str))
    output = {"collectri_unique_source_symbols": len(catalogue), "cohorts": {}}
    output["cohorts"]["full"] = count_one(expression, metadata, catalogue, set())
    for fold in sorted((root / "model/cv").glob("fold_*")):
        split = json.loads((fold / "split_info.json").read_text())
        output["cohorts"][fold.name] = count_one(
            expression, metadata, catalogue, set(split["heldout_times"]))
    destination = Path("results/real_data_20261007/candidate_pool_audit.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
