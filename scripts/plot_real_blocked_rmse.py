#!/usr/bin/env python3
"""Plot DiSCGRN held-out RMSE by time block from existing real-data results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FOLDS = (
    ("fold_01", "Embryonic\nE13.5–E19.5"),
    ("fold_02", "Perinatal\nP0–P9"),
    ("fold_03", "Postnatal\nP11–P24"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="blocked_cv_by_fold_strain.tsv")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output PDF path")
    args = parser.parse_args()

    table = pd.read_csv(args.input, sep="\t")
    table = table.loc[table.method.eq("discgrn")]
    values = []
    for fold, _ in FOLDS:
        fold_values = table.loc[table.fold.eq(fold), "pooled_rmse"]
        if len(fold_values) != 3 or fold_values.nunique() != 1:
            raise ValueError(f"Expected one pooled DiSCGRN RMSE across three strains for {fold}")
        values.append(float(fold_values.iloc[0]))

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.7, 3.7), layout="constrained")
    bars = ax.bar(range(3), values, width=0.58,
                  color=["#5B8DB8", "#3C719C", "#285579"])
    ax.set_xticks(range(3), [label for _, label in FOLDS])
    ax.set_ylabel("Held-out one-step RMSE")
    ax.set_ylim(0, max(values) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
