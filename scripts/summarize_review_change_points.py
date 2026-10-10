#!/usr/bin/env python3
"""Summarize the existing locked-test change-point records without refitting."""

from pathlib import Path

import numpy as np
import pandas as pd
from ctcpgrn.paths import results_root


SOURCE = results_root() / "revision_comparison_h80/comparison_30systems.tsv"
OUT = results_root() / "change_points"
METHODS = ("ct", "euler", "dplsq", "two_stage")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE, sep="\t")
    rows = []
    for method in METHODS:
        group = frame[frame.method == method]
        changed, null = group[group.changed], group[~group.changed]
        rows.append({
            "method": method,
            "n_changed": len(changed),
            "within_1_5_count": int(changed.cp_f1.sum()),
            "exact_count": int(changed.cp_mae.eq(0).sum()),
            "conditional_mae": changed.cp_mae.mean(),
            "matched_for_mae": int(changed.cp_mae.notna().sum()),
            "n_null": len(null),
            "no_change_count": int(null.cp_f1.sum()),
            "balanced_accuracy": (changed.cp_f1.mean() + null.cp_f1.mean()) / 2,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "locked_change_point_summary.tsv", sep="\t", index=False)

    wide = frame.pivot(index=["changed", "seed"], columns="method", values="cp_f1")
    rng = np.random.default_rng(20261003)
    paired = []
    for method in METHODS[1:]:
        changed = (wide.loc[True, "ct"] - wide.loc[True, method]).to_numpy()
        null = (wide.loc[False, "ct"] - wide.loc[False, method]).to_numpy()
        ci_draws = (changed[rng.integers(len(changed), size=(20000, len(changed)))].mean(axis=1)
                    + null[rng.integers(len(null), size=(20000, len(null)))].mean(axis=1)) / 2
        paired.append({
            "comparison": f"ct_minus_{method}",
            "balanced_accuracy_difference": (changed.mean() + null.mean()) / 2,
            "ci_low": np.quantile(ci_draws, 0.025),
            "ci_high": np.quantile(ci_draws, 0.975),
        })
    pd.DataFrame(paired).to_csv(OUT / "locked_change_point_paired.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__":
    main()
