#!/usr/bin/env python3
"""Paired comparison of BEELINE edge ranking before and after expression dropout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def mean_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    samples = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).astype(float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("results/beeline_extended_valid_20261009"))
    parser.add_argument("--dropout-root", type=Path,
                        default=Path("results/beeline_dropout_20261009"))
    args = parser.parse_args()
    original = pd.read_csv(args.baseline_root / "summary/all_metrics.tsv", sep="\t")
    rng = np.random.default_rng(20261009)
    records = []
    for topology in ("dyn-LI", "dyn-LL", "dyn-CY"):
        source = original[original.dataset.eq(f"{topology}-200")]
        for label, fraction in (("drop20", 0.2), ("drop50", 0.5)):
            experiment = f"{topology}-200-{label}"
            perturbed = pd.read_csv(args.dropout_root / experiment / "all_metrics.tsv", sep="\t")
            joined = source.merge(perturbed, on=["run_id", "method"],
                                  suffixes=("_original", "_dropout"), validate="one_to_one")
            if len(joined) != 40:
                raise ValueError(f"Expected 40 paired method records for {experiment}")
            for method, group in joined.groupby("method", sort=False):
                both = group[group.status_original.eq("ok") & group.status_dropout.eq("ok")]
                for metric in ("aupr_ap", "auroc", "max_f1"):
                    if both.empty:
                        continue
                    before = both[f"{metric}_original"].to_numpy(float)
                    after = both[f"{metric}_dropout"].to_numpy(float)
                    change = after - before
                    lo, hi = mean_interval(change, rng)
                    records.append({"dataset": experiment, "topology": topology,
                                    "dropout_fraction": fraction, "method": method,
                                    "metric": metric, "n_paired": len(both),
                                    "original_mean_on_pairs": before.mean(),
                                    "dropout_mean_on_pairs": after.mean(),
                                    "mean_change_dropout_minus_original": change.mean(),
                                    "ci_low": lo, "ci_high": hi})
    output = args.dropout_root / "paired_dropout_summary.tsv"
    pd.DataFrame(records).to_csv(output, sep="\t", index=False)
    print(pd.read_csv(output, sep="\t").query("metric == 'aupr_ap'").to_string(index=False))


if __name__ == "__main__":
    main()
