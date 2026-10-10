#!/usr/bin/env python3
"""Consistent-colour bar charts for the five-method GRN accuracy benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "results/runs/grn_accuracy_20261009/five_method_reconstruction"
METHODS = ["DiSCGRN", "GENIE3", "GRNBoost2", "dynGENIE3", "SCODE"]
METRICS = [("aupr", "AUPR"), ("precision", "Precision"),
           ("recall", "Recall"), ("f1", "F1"), ("f0_1", "F0.1")]
COLORS = {"DiSCGRN": "#0072B2", "GENIE3": "#D55E00",
          "GRNBoost2": "#009E73", "dynGENIE3": "#CC79A7", "SCODE": "#E69F00"}


def grouped_panels(summary: pd.DataFrame, domain: str, datasets: list[str],
                   labels: list[str], output: Path, *, size: tuple[float, float]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=size, constrained_layout=True)
    axes = axes.ravel()
    x = np.arange(len(datasets))
    width = 0.16
    for axis, (metric, title) in zip(axes, METRICS):
        for method_index, method in enumerate(METHODS):
            values = []
            for dataset in datasets:
                part = summary[(summary.domain == domain) &
                               (summary.dataset == dataset) & (summary.method == method)]
                if len(part) != 1:
                    raise ValueError(f"Missing {domain}, {dataset}, {method}")
                values.append(float(part[metric].iloc[0]))
            positions = x + (method_index - 2) * width
            axis.bar(positions, values, width=width, color=COLORS[method], label=method)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=25 if len(datasets) > 2 else 0,
                        ha="right" if len(datasets) > 2 else "center")
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.20)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, legend_labels, loc="center", frameon=False, ncol=2)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def dataset_metric_matrix(summary: pd.DataFrame, domain: str, datasets: list[str],
                          labels: list[str], output: Path) -> None:
    """One dataset per row and one metric per column, with fixed method colors."""
    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, len(METRICS),
                             figsize=(15.5, 2.05 * n_rows + 1.0),
                             sharey=True, squeeze=False)
    for row, (dataset, label) in enumerate(zip(datasets, labels)):
        for col, (metric, title) in enumerate(METRICS):
            axis = axes[row, col]
            subset = summary[(summary.domain == domain) &
                             (summary.dataset == dataset)].set_index("method")
            if set(subset.index) != set(METHODS) or len(subset) != len(METHODS):
                raise ValueError(f"Expected exactly five methods for {domain}/{dataset}")
            axis.bar(range(len(METHODS)),
                     [float(subset.loc[method, metric]) for method in METHODS],
                     width=0.72, color=[COLORS[method] for method in METHODS])
            axis.set_ylim(0, 1.0)
            axis.set_xticks([])
            axis.grid(axis="y", alpha=0.18)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="y", labelsize=8, length=2)
            if row == 0:
                axis.set_title(title, fontsize=11, pad=8)
            if col == 0:
                axis.set_ylabel(label, fontsize=10, rotation=0,
                                labelpad=40, ha="right", va="center")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[method]) for method in METHODS]
    fig.legend(handles, METHODS, loc="lower center", ncol=5,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.095, right=0.995, top=0.94,
                        bottom=0.11 if n_rows == 2 else 0.07,
                        hspace=0.28, wspace=0.18)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def macro_panels(macro: pd.DataFrame, output: Path) -> None:
    part = macro[macro.domain == "BEELINE"].set_index("method")
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.4), constrained_layout=True)
    for axis, (metric, title) in zip(axes, METRICS):
        values = [part.loc[method, metric] for method in METHODS]
        axis.bar(range(len(METHODS)), values, color=[COLORS[method] for method in METHODS])
        axis.set_title(title)
        axis.set_xticks(range(len(METHODS)), METHODS, rotation=55, ha="right")
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.20)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    summary = pd.read_csv(root / "summary_by_dataset.tsv", sep="\t")
    macro = pd.read_csv(root / "macro_summary.tsv", sep="\t")
    grouped_panels(summary, "DREAM_derived", ["T25", "T41"],
                   ["25 time points", "41 time points"],
                   root / "dream_five_metrics_bar.pdf", size=(11.5, 9.0))
    dataset_metric_matrix(summary, "DREAM_derived", ["T25", "T41"],
                          ["T=25", "T=41"],
                          root / "dream_five_metrics_dataset_rows.pdf")
    datasets = summary.loc[summary.domain == "BEELINE", "dataset"].drop_duplicates().tolist()
    labels = [dataset.replace("dyn-", "") for dataset in datasets]
    grouped_panels(summary, "BEELINE", datasets, labels,
                   root / "beeline_five_metrics_by_condition_bar.pdf", size=(11.5, 9.0))
    dataset_metric_matrix(summary, "BEELINE", datasets, labels,
                          root / "beeline_five_metrics_dataset_rows.pdf")
    macro_panels(macro, root / "beeline_five_metrics_macro_bar.pdf")
    (root / "method_colors.tsv").write_text(
        "method\tcolor_hex\n" + "".join(f"{name}\t{COLORS[name]}\n" for name in METHODS))
    for path in sorted(root.glob("*bar.pdf")):
        print(path)
    print(root / "dream_five_metrics_dataset_rows.pdf")
    print(root / "beeline_five_metrics_dataset_rows.pdf")


if __name__ == "__main__":
    main()
