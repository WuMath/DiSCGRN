#!/usr/bin/env python3
"""Prepare gene panels and strain-level trajectories from GSE74243."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.io import (
    build_gse74243_groups,
    collapse_duplicate_symbols,
    read_gse74243_expression,
    select_variable_genes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/gse74243"))
    parser.add_argument("--n-genes", type=int, default=50)
    parser.add_argument("--n-regulators", type=int, default=20)
    parser.add_argument("--collectri", type=Path, default=Path("data/external/collectri_mouse.tsv"))
    args = parser.parse_args()

    expression, metadata = read_gse74243_expression(args.input)
    expression = collapse_duplicate_symbols(expression)
    # Only the set of CollecTRI source identities is used here, never its
    # TF-target pairs, signs, scores or references. This guarantees enough
    # candidate regulators while keeping edge evidence hidden until evaluation.
    collectri = pd.read_csv(args.collectri, sep="\t", usecols=["source_genesymbol"])
    regulator_catalogue = set(collectri["source_genesymbol"].dropna().astype(str))
    sample_columns = list(expression.columns[2:])
    variance = expression[sample_columns].var(axis=1, ddof=1)
    regulator_rows = expression[expression["Gene Symbol"].astype(str).isin(regulator_catalogue)]
    regulator_variance = regulator_rows[sample_columns].var(axis=1, ddof=1)
    regulators = regulator_rows.loc[regulator_variance.nlargest(args.n_regulators).index]
    remaining = expression[~expression["Gene Symbol"].isin(set(regulators["Gene Symbol"]))]
    targets = select_variable_genes(remaining, args.n_genes - len(regulators))
    panel = pd.concat([regulators, targets], ignore_index=True)
    panel["is_candidate_regulator"] = panel["Gene Symbol"].isin(regulator_catalogue)
    # Z-score each gene within strain using all biological samples. Stage labels
    # are not used, so this cannot create stage-alignment leakage.
    for strain, strain_meta in metadata.groupby("strain"):
        columns = strain_meta["sample"].tolist()
        values = panel[columns].to_numpy(dtype=float)
        means = np.nanmean(values, axis=1, keepdims=True)
        scales = np.nanstd(values, axis=1, ddof=1, keepdims=True)
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
        panel.loc[:, columns] = (values - means) / scales
    groups = build_gse74243_groups(panel, metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel[["Gene ID", "Gene Symbol", "is_candidate_regulator"]].to_csv(
        args.output_dir / "gene_panel.tsv", sep="\t", index=False
    )
    metadata.to_csv(args.output_dir / "sample_metadata.tsv", sep="\t", index=False)
    payload = {"genes": panel["Gene Symbol"].astype(str).tolist()}
    for group in groups:
        payload[f"{group.name}_times"] = group.times
        payload[f"{group.name}_values"] = group.values
        payload[f"{group.name}_weights"] = group.weights
        payload[f"{group.name}_n_replicates"] = group.n_replicates
    np.savez_compressed(args.output_dir / "trajectories.npz", **payload)
    summary = {
        "n_genes": len(payload["genes"]),
        "n_candidate_regulators": int(panel["is_candidate_regulator"].sum()),
        "groups": {
            group.name: {
                "n_times": len(group.times),
                "first_time": float(group.times.min()),
                "last_time": float(group.times.max()),
                "replicates_min": int(group.n_replicates.min()),
                "replicates_max": int(group.n_replicates.max()),
            }
            for group in groups
        },
        "time_convention": "E stages use embryonic day; P0=20.0 and Pn=20+n",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
