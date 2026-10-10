#!/usr/bin/env python3
"""Plot comparable edge-ranking results from the frozen GRN benchmark summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT = Path(__file__).resolve().parents[1] / "results/runs/grn_accuracy_20261009/summary"
ORDER = ["DiSCGRN", "Euler--DP", "EDISON", "GENIE3", "GRNBoost2", "dynGENIE3", "SCODE"]
COLORS = {"DiSCGRN": "#2563a6", "Euler--DP": "#6b7280", "EDISON": "#bb6940",
          "GENIE3": "#a376c4", "GRNBoost2": "#5f9f82", "dynGENIE3": "#d2a436",
          "SCODE": "#ca657c"}


def dream_plot(summary: pd.DataFrame, output: Path) -> None:
    subset = summary[summary.domain == "DREAM_simulated"]
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    groups = ["T25", "T41"]
    width = 0.105
    for i, method in enumerate(ORDER):
        y = [subset[(subset.dataset == group) & (subset.method == method)].mean_aupr.iloc[0]
             for group in groups]
        x = np.arange(len(groups)) + (i - (len(ORDER) - 1) / 2) * width
        ax.bar(x, y, width=width, label=method, color=COLORS[method])
    ax.set_xticks(range(len(groups)), ["25 observations", "41 observations"])
    ax.set_ylabel("Mean directed-edge AP")
    ax.set_ylim(0, 0.65)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "dream_stage_network_aupr.pdf", bbox_inches="tight")
    plt.close(fig)


def beeline_plot(summary: pd.DataFrame, output: Path) -> None:
    subset = summary[summary.domain == "BEELINE"]
    datasets = ["dyn-LI-100", "dyn-LI-200", "dyn-LI-500", "dyn-LL-100",
                "dyn-LL-200", "dyn-LL-500", "dyn-CY-200"]
    methods = [method for method in ORDER if method != "Euler--DP"]
    matrix = np.array([[subset[(subset.dataset == dataset) &
                                 (subset.method == method)].mean_aupr.iloc[0]
                        for method in methods] for dataset in datasets])
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)), datasets)
    for i in range(len(datasets)):
        for j in range(len(methods)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if value >= 0.60 else "#172334", fontsize=9)
    fig.colorbar(image, ax=ax, label="Mean directed-edge AP", fraction=0.046)
    fig.tight_layout()
    fig.savefig(output / "beeline_static_network_aupr.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    summary = pd.read_csv(args.summary_root / "dataset_summary.tsv", sep="\t")
    dream_plot(summary, args.summary_root)
    beeline_plot(summary, args.summary_root)
    print(args.summary_root / "dream_stage_network_aupr.pdf")
    print(args.summary_root / "beeline_static_network_aupr.pdf")


if __name__ == "__main__":
    main()
