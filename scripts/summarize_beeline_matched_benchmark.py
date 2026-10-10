#!/usr/bin/env python3
"""Summarize matched-input BEELINE runs without treating failures as zero."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("DiSCGRN", "EDISON", "GENIE3", "GRNBoost2")
METRICS = ("aupr_ap", "aupr_lift", "auroc", "max_f1", "precision_at_k")


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/beeline_matched_summary_20261007"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    roots_by_dataset = {}
    for root in args.roots:
        protocol = json.loads((root / "protocol.json").read_text())
        label = protocol.get("dataset_label")
        if label in roots_by_dataset:
            raise ValueError(f"Duplicate input dataset label: {label}")
        roots_by_dataset[label] = root
    frames = [pd.read_csv(root / "all_metrics.tsv", sep="\t") for root in args.roots]
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["dataset", "run_id", "method"]).any():
        raise ValueError("Duplicate dataset/run/method records")
    if not set(data.method).issubset(METHODS):
        raise ValueError("Unexpected method")
    data.to_csv(args.output_root / "all_metrics.tsv", sep="\t", index=False)
    rng = np.random.default_rng(20261007)
    rows, pairs, common = [], [], []
    for dataset, group in data.groupby("dataset", sort=True):
        expected_runs = set(group.run_id)
        if len(expected_runs) != 10 or set(group.method) != set(METHODS):
            raise ValueError(f"Incomplete method inventory for {dataset}")
        all_wide = group.pivot(index="run_id", columns="method", values="status")
        if not all_wide.index.to_series().isin(expected_runs).all():
            raise ValueError("Inconsistent runs")
        complete = all_wide.index[all_wide.eq("ok").all(axis=1)]
        for method in METHODS:
            part = group[group.method.eq(method)]
            if set(part.run_id) != expected_runs:
                raise ValueError(f"Missing {method} record in {dataset}")
            good = part[part.status.eq("ok")]
            row = {"dataset": dataset, "method": method, "n_planned": len(part),
                   "n_evaluable": len(good), "n_common_four_methods": len(complete),
                   "n_genes": int(part.n_genes.iloc[0]),
                   "k_true_edges": int(part.k_true_edges.dropna().iloc[0])}
            for metric in METRICS:
                values = good[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                if len(values):
                    row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bootstrap_mean(values, rng)
                common_values = good[good.run_id.isin(complete)][metric].to_numpy(dtype=float)
                common.append({"dataset": dataset, "method": method, "metric": metric,
                               "n": len(common_values),
                               "mean": float(common_values.mean()) if len(common_values) else np.nan})
            if method == "DiSCGRN":
                root = roots_by_dataset.get(dataset)
                if root is None:
                    raise ValueError(f"Missing input root for {dataset}")
                records = [json.loads((root / run / "DiSCGRN_metrics.json").read_text())
                           for run in expected_runs
                           if (root / run / "DiSCGRN_metrics.json").exists()]
                if len(records) != len(expected_runs):
                    raise ValueError("Missing DiSCGRN diagnostics")
                row["n_all_segments_converged"] = sum(
                    all(item["detail"]["converged_segments"]) for item in records)
                row["n_selected_change"] = sum(
                    item["detail"]["change_index"] is not None for item in records)
            rows.append(row)
        for comparator in METHODS[1:]:
            for metric in METRICS:
                values = group.pivot(index="run_id", columns="method", values=metric)
                matched = values[["DiSCGRN", comparator]].dropna()
                diff = (matched.DiSCGRN - matched[comparator]).to_numpy(dtype=float)
                lo, hi = bootstrap_mean(diff, rng)
                pairs.append({"dataset": dataset, "comparator": comparator,
                              "metric": metric, "n_pairs": len(diff),
                              "mean_difference": float(diff.mean()), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(rows).to_csv(args.output_root / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(pairs).to_csv(args.output_root / "paired.tsv", sep="\t", index=False)
    pd.DataFrame(common).to_csv(args.output_root / "common_success_summary.tsv", sep="\t", index=False)
    environment = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "arboreto"):
        try:
            environment[name] = version(name)
        except PackageNotFoundError:
            environment[name] = "not installed in summarization environment"
    environment["runner_protocols"] = {
        str(root): json.loads((root / "protocol.json").read_text())
        for root in args.roots
    }
    environment["EDISON"] = "1.1.2 (verified in each successful run log)"
    (args.output_root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(pd.DataFrame(rows)[["dataset", "method", "n_evaluable", "n_planned",
                              "aupr_ap_mean", "auroc_mean", "max_f1_mean",
                              "precision_at_k_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
