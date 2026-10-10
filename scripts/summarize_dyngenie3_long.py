#!/usr/bin/env python3
"""Pair official dynGENIE3 whole-series rankings with frozen DiSCGRN systems."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("results/runs/weighted_score_20261004"))
    args = parser.parse_args()
    dyn_root = args.root / "dyngenie3_long"
    new = pd.read_csv(dyn_root / "system_aupr.tsv", sep="\t")
    if len(new) != 60 or not new.status.eq("ok").all():
        raise ValueError("Require 60 successful dynGENIE3 runs")
    old = pd.read_csv(args.root / "native_grn_benchmark" / "system_aupr.tsv", sep="\t")
    disc = old[old.method.eq("DiSCGRN")][["seed", "T", "changed", "network_aupr"]]
    joined = disc.merge(new[["seed", "T", "changed", "network_aupr", "runtime_seconds"]],
                        on=["seed", "T", "changed"], suffixes=("_discgrn", "_dyngenie3"),
                        validate="one_to_one")
    if len(joined) != 60:
        raise ValueError("Unmatched systems")
    joined.to_csv(dyn_root / "paired_systems.tsv", sep="\t", index=False)
    rng = np.random.default_rng(20261009)
    rows = []
    for length in (25, 41):
        for condition in ("stationary", "changed", "all"):
            selected = joined[joined["T"].eq(length)]
            if condition != "all":
                selected = selected[selected.changed.eq(condition == "changed")]
            diff = (selected.network_aupr_discgrn - selected.network_aupr_dyngenie3).to_numpy(float)
            lo, hi = interval(diff, rng)
            rows.append({"T": length, "condition": condition, "n_systems": len(selected),
                         "discgrn_mean": selected.network_aupr_discgrn.mean(),
                         "dyngenie3_mean": selected.network_aupr_dyngenie3.mean(),
                         "mean_difference": diff.mean(), "ci_low": lo, "ci_high": hi,
                         "dyngenie3_median_runtime_s": selected.runtime_seconds.median()})
    output = pd.DataFrame(rows)
    output.to_csv(dyn_root / "paired_summary.tsv", sep="\t", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
