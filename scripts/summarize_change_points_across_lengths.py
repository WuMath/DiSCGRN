#!/usr/bin/env python3
"""Summarize frozen change-point tests using actual-time tolerance throughout."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctcpgrn.paths import results_root


TOLERANCE = 1.5


def main() -> None:
    root = results_root()
    out = root / "change_points"
    out.mkdir(parents=True, exist_ok=True)
    short = pd.read_csv(out / "locked_change_point_summary.tsv", sep="\t")
    rows = []
    names = {"ct": "DiSCGRN", "euler": "Euler--DP", "dplsq": "DPLSQ--DP",
             "two_stage": "Two-stage"}
    for row in short.itertuples(index=False):
        rows.append({"T": 13, "method": names[row.method], "n_changed": int(row.n_changed),
                     "within_1_5_actual_time": int(row.within_1_5_count),
                     "exact": int(row.exact_count), "n_stationary": int(row.n_null),
                     "correct_no_change": int(row.no_change_count),
                     "planned_changed": 15, "planned_stationary": 15})

    manifest = pd.read_csv(root / "revision_e6" / "input" / "manifest.tsv", sep="\t")
    long_manifest = manifest[manifest.split.eq("e4test") & manifest["T"].isin((25, 41))]
    short_manifest = manifest[manifest.split.eq("frozen_short") & manifest["T"].eq(13)]
    assert len(long_manifest) == 60 and len(short_manifest) == 30
    extra = pd.read_csv(out / "long_internal_missing.tsv", sep="\t")
    assert len(extra) == 120
    edison_long = pd.read_csv(root / "revision_e6" / "external_e4test_edison_summary.tsv", sep="\t")
    edison_short = pd.read_csv(root / "revision_e6" / "external_frozen_short_edison_summary.tsv", sep="\t")
    assert len(edison_long) == 60 and len(edison_short) == 30

    def summarize_edison(frame: pd.DataFrame, truth: pd.DataFrame, length: int) -> dict:
        frame = frame[frame.config.eq("poisson")].merge(truth, on="system_id", validate="one_to_one")
        assert len(frame) == 30
        planned_changed = int(frame.changed.sum())
        planned_stationary = len(frame) - planned_changed
        frame = frame[frame.status.eq("ok") & frame.cp_score.notna()]
        changed = frame[frame.changed]
        stationary = frame[~frame.changed]
        within = exact = 0
        for item in changed.itertuples(index=False):
            if item.cp_score < 0.08:
                continue
            times = pd.read_csv(root / "revision_e6" / "input" /
                                f"{item.system_id}.tsv", sep="\t").time.to_numpy()
            estimated = int(item.primary_cp_index)
            true = int(item.true_cp_index)
            assert 0 <= estimated < len(times)
            exact += estimated == true
            within += abs(times[estimated] - times[true]) <= TOLERANCE
        return {"T": length, "method": "EDISON", "n_changed": len(changed),
                "within_1_5_actual_time": within, "exact": exact,
                "n_stationary": len(stationary),
                "correct_no_change": int((stationary.cp_score < 0.08).sum()),
                "planned_changed": planned_changed,
                "planned_stationary": planned_stationary}

    rows.append(summarize_edison(edison_short, short_manifest, 13))
    for length in (25, 41):
        internal = pd.read_csv(root / "revision_e4_single" /
                               f"changed_T{length}_r0.35_w0.35.tsv", sep="\t")
        null = pd.read_csv(root / "revision_e4_single" /
                           f"null_T{length}.tsv", sep="\t")
        internal = pd.concat([internal, null], ignore_index=True)
        for code, method in (("ct", "DiSCGRN"), ("euler", "Euler--DP"),
                             ("dplsq", "DPLSQ--DP"), ("two_stage", "Two-stage")):
            source = extra if code in ("dplsq", "two_stage") else internal
            subset = source[source["T"].eq(length) & source.method.eq(code)]
            changed = subset[subset.changed]
            stationary = subset[~subset.changed]
            assert len(changed) == len(stationary) == 15, (length, method, len(changed), len(stationary))
            within, exact = 0, 0
            for row in changed.itertuples(index=False):
                if not bool(row.selected_change):
                    continue
                system_id = f"e4test_T{length}_change_{row.seed}"
                truth = long_manifest[long_manifest.system_id.eq(system_id)].iloc[0]
                true_index = int(truth.true_cp_index)
                estimated_index = int(row.estimated_index)
                times = pd.read_csv(root / "revision_e6" / "input" /
                                    f"{system_id}.tsv", sep="\t").time.to_numpy()
                exact += estimated_index == true_index
                within += abs(times[estimated_index] - times[true_index]) <= TOLERANCE
            rows.append({"T": length, "method": method, "n_changed": len(changed),
                         "within_1_5_actual_time": within, "exact": exact,
                         "n_stationary": len(stationary),
                         "correct_no_change": int((~stationary.selected_change.astype(bool)).sum()),
                         "planned_changed": 15, "planned_stationary": 15})
        rows.append(summarize_edison(edison_long, long_manifest[long_manifest["T"].eq(length)], length))
    result = pd.DataFrame(rows)
    order = {"DiSCGRN": 0, "Euler--DP": 1, "DPLSQ--DP": 2,
             "Two-stage": 3, "EDISON": 4}
    result["order"] = result.method.map(order)
    result = result.sort_values(["T", "order"]).drop(columns="order").reset_index(drop=True)
    assert result.groupby("T").size().eq(5).all()
    result["balanced_accuracy"] = 0.5 * (
        result.within_1_5_actual_time / result.n_changed
        + result.correct_no_change / result.n_stationary)
    result.to_csv(out / "across_lengths_actual_time.tsv", sep="\t", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
