#!/usr/bin/env python3
"""Second-round CT refinement and untouched-seed final validation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from run_optimized_benchmark import (
    METHODS,
    fit_path,
    make_dataset,
    one_step_rmse,
    score_result,
    trajectory_groups,
)


GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUTPUT = Path("results/final_validation")
MINIMUM_TRANSITIONS = 5
MAX_ITER = 80


def ct_path(simulation, alpha: float, max_iter: int = MAX_ITER):
    return scan_one_change_candidates(
        trajectory_groups(simulation), simulation.times, "ct", alpha,
        minimum_transitions=MINIMUM_TRANSITIONS, max_iter=max_iter,
        ct_initialiser="euler", ct_ridge=0.0,
    )


def calibrate_ct() -> pd.DataFrame:
    alpha_grid = [0.0002, 0.00035, 0.0005, 0.00075, 0.001, 0.002, 0.004]
    beta_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    datasets = [(seed, True, make_dataset(GOLD, seed, True)) for seed in [4101, 4102, 4103]]
    datasets += [(seed, False, make_dataset(GOLD, seed, False)) for seed in [5101, 5102, 5103]]
    rows = []
    for alpha in alpha_grid:
        fitted = []
        for seed, changed, (_, first, second, true_index, simulation) in datasets:
            started = time.perf_counter()
            path = ct_path(simulation, alpha)
            fitted.append((changed, first, second, true_index, simulation, path, time.perf_counter() - started))
        for beta in beta_grid:
            changed_scores, null_scores, rmses = [], [], []
            for changed, first, second, true_index, simulation, path, runtime in fitted:
                result = select_one_change(path, beta, relative_beta=True)
                true_time = None if true_index is None else float(simulation.times[true_index])
                score = score_result(result, true_time, first, second)
                (changed_scores if changed else null_scores).append(score)
                rmses.append(one_step_rmse(result, simulation, "ct"))
            row = {
                "method": "ct", "initialiser": "euler", "alpha": alpha,
                "beta_ratio": beta, "ridge": 0.0,
                "mean_cp_f1": np.mean([x["cp_f1"] for x in changed_scores]),
                "mean_change_aupr": np.mean([x["change_aupr"] for x in changed_scores]),
                "mean_network_aupr": np.mean([x["network_aupr"] for x in changed_scores]),
                "null_specificity": np.mean([x["cp_f1"] for x in null_scores]),
                "mean_rmse": np.mean(rmses),
            }
            row["selection_score"] = (
                row["mean_cp_f1"] + 0.35 * row["mean_change_aupr"]
                + 0.20 * row["mean_network_aupr"] + 0.50 * row["null_specificity"]
            )
            rows.append(row)
            print("REFINE", json.dumps(row), flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUTPUT / "ct_refinement_grid.tsv", sep="\t", index=False)
    selected = grid.sort_values(
        ["selection_score", "mean_rmse", "beta_ratio", "alpha"],
        ascending=[False, True, True, False],
    ).iloc[[0]]
    selected.to_csv(OUTPUT / "ct_selected_parameters.tsv", sep="\t", index=False)
    return selected


def final_test(ct_selected: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv("results/optimized_benchmark/selected_parameters.tsv", sep="\t")
    params = {row.method: row for row in baseline.itertuples(index=False)}
    ct = ct_selected.iloc[0]
    rows = []
    for changed, seeds in [(True, range(94001, 94011)), (False, range(95001, 95011))]:
        for seed in seeds:
            _, first, second, true_index, simulation = make_dataset(GOLD, seed, changed)
            for method in METHODS:
                started = time.perf_counter()
                if method == "ct":
                    path = ct_path(simulation, float(ct.alpha))
                    beta = float(ct.beta_ratio)
                else:
                    p = params[method]
                    path = fit_path(method, simulation, p.alpha, p.ridge, MAX_ITER, MINIMUM_TRANSITIONS)
                    beta = p.beta_ratio
                result = select_one_change(path, beta, relative_beta=True)
                score = score_result(
                    result, None if true_index is None else float(simulation.times[true_index]), first, second,
                )
                row = {
                    "seed": seed, "changed": changed, "method": method, "label": METHODS[method],
                    "true_change_time": None if true_index is None else float(simulation.times[true_index]),
                    "estimated_change_time": result.change_time,
                    "one_step_rmse": one_step_rmse(result, simulation, method),
                    "runtime_seconds": time.perf_counter() - started,
                    **score,
                }
                rows.append(row)
                print("FINAL", json.dumps(row), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "final_test_by_seed.tsv", sep="\t", index=False)
    summary = []
    for method, group in frame.groupby("method"):
        changed = group[group.changed]
        null = group[~group.changed]
        summary.append({
            "method": method, "label": METHODS[method],
            "cp_f1_mean": changed.cp_f1.mean(),
            "cp_exact_rate": np.mean(changed.cp_mae.fillna(np.inf) == 0),
            "network_aupr_mean": changed.network_aupr.mean(),
            "change_aupr_mean": changed.change_aupr.mean(),
            "rmse_mean": changed.one_step_rmse.mean(),
            "null_specificity": null.cp_f1.mean(),
            "runtime_seconds_mean": group.runtime_seconds.mean(),
        })
    pd.DataFrame(summary).to_csv(OUTPUT / "final_test_summary.tsv", sep="\t", index=False)
    return frame


def paired_intervals(frame: pd.DataFrame) -> None:
    rng = np.random.default_rng(20260929)
    changed = frame[frame.changed]
    records = []
    for comparator in ["dplsq", "euler", "two_stage"]:
        for metric, higher_better in [("cp_f1", True), ("network_aupr", True), ("change_aupr", True), ("one_step_rmse", False)]:
            ct = changed[changed.method == "ct"].sort_values("seed")[metric].to_numpy()
            other = changed[changed.method == comparator].sort_values("seed")[metric].to_numpy()
            difference = ct - other if higher_better else other - ct
            samples = np.array([np.mean(rng.choice(difference, len(difference), replace=True)) for _ in range(10000)])
            records.append({
                "comparison": f"ct_minus_{comparator}", "metric": metric,
                "positive_means_ct_better": True, "mean_advantage": difference.mean(),
                "ci_low": np.quantile(samples, 0.025), "ci_high": np.quantile(samples, 0.975),
            })
    pd.DataFrame(records).to_csv(OUTPUT / "paired_bootstrap_intervals.tsv", sep="\t", index=False)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = calibrate_ct()
    frame = final_test(selected)
    paired_intervals(frame)


if __name__ == "__main__":
    main()
