#!/usr/bin/env python3
"""Plot the untouched BEELINE holdout after development-only DiSCGRN tuning."""

import argparse
from pathlib import Path

import pandas as pd

from plot_five_method_grn_benchmark import dataset_metric_matrix, grouped_panels, macro_panels


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "results/runs/grn_accuracy_20261009/beeline_tuned_holdout_20261009"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--li-only", action="store_true",
                        help="Plot only the three dyn-LI cell-count conditions")
    args = parser.parse_args()
    root = args.root
    summary = pd.read_csv(root / "comparison_holdout_by_condition.tsv", sep="\t")
    if args.li_only:
        summary = summary[summary.dataset.isin(["dyn-LI-100", "dyn-LI-200", "dyn-LI-500"])].copy()
    summary = summary[summary.method != "DiSCGRN"].copy()
    summary.loc[summary.method == "DiSCGRN_tuned", "method"] = "DiSCGRN"
    summary["domain"] = "BEELINE"
    datasets = summary.dataset.drop_duplicates().tolist()
    labels = [dataset.replace("dyn-", "") for dataset in datasets]
    grouped_panels(summary, "BEELINE", datasets, labels,
                   root / "beeline_holdout_by_condition_bar.pdf", size=(15.0, 10.0))
    dataset_metric_matrix(summary, "BEELINE", datasets, labels,
                          root / "beeline_holdout_dataset_rows.pdf")
    macro = summary.groupby(["domain", "method"], as_index=False)[
        ["aupr", "precision", "recall", "f1", "f0_1"]].mean()
    macro_panels(macro, root / "beeline_holdout_macro_bar.pdf")
    print(root / "beeline_holdout_by_condition_bar.pdf")
    print(root / "beeline_holdout_dataset_rows.pdf")
    print(root / "beeline_holdout_macro_bar.pdf")


if __name__ == "__main__":
    main()
