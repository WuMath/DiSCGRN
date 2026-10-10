#!/usr/bin/env python3
"""Tune CT exact-refinement depth, then validate once on new untouched seeds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.paths import results_root
from run_optimized_benchmark import METHODS, fit_path, make_dataset, one_step_rmse, score_result, trajectory_groups

GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = results_root() / "iteration_refinement"
MINIMUM_TRANSITIONS = 5


def ct_path(simulation, alpha, iterations):
    return scan_one_change_candidates(
        trajectory_groups(simulation), simulation.times, "ct", alpha,
        minimum_transitions=MINIMUM_TRANSITIONS, max_iter=iterations,
        ct_initialiser="euler",
    )


def calibrate():
    data = [(True, make_dataset(GOLD, seed, True)) for seed in [4101, 4102, 4103]]
    data += [(False, make_dataset(GOLD, seed, False)) for seed in [5101, 5102, 5103]]
    rows = []
    for alpha in [0.00035, 0.0005, 0.00075]:
        for iterations in [5, 10, 20, 40, 80]:
            fitted = []
            for changed, (_, first, second, index, simulation) in data:
                path = ct_path(simulation, alpha, iterations)
                fitted.append((changed, first, second, index, simulation, path))
            for beta in [0.25, 0.30, 0.35, 0.40]:
                changed_scores, null_scores, rmses = [], [], []
                for changed, first, second, index, simulation, path in fitted:
                    result = select_one_change(path, beta, relative_beta=True)
                    score = score_result(result, None if index is None else float(simulation.times[index]), first, second)
                    (changed_scores if changed else null_scores).append(score)
                    rmses.append(one_step_rmse(result, simulation, "ct"))
                row = {
                    "alpha": alpha, "beta_ratio": beta, "max_iter": iterations,
                    "mean_cp_f1": np.mean([x["cp_f1"] for x in changed_scores]),
                    "mean_change_aupr": np.mean([x["change_aupr"] for x in changed_scores]),
                    "mean_network_aupr": np.mean([x["network_aupr"] for x in changed_scores]),
                    "null_specificity": np.mean([x["cp_f1"] for x in null_scores]),
                    "mean_rmse": np.mean(rmses),
                }
                row["selection_score"] = row["mean_cp_f1"] + .35*row["mean_change_aupr"] + .2*row["mean_network_aupr"] + .5*row["null_specificity"]
                rows.append(row)
                print("ITER", json.dumps(row), flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "iteration_grid.tsv", sep="\t", index=False)
    selected = grid.sort_values(
        ["selection_score", "mean_rmse", "max_iter", "beta_ratio"],
        ascending=[False, True, True, True],
    ).iloc[[0]]
    selected.to_csv(OUT / "selected_parameters.tsv", sep="\t", index=False)
    return selected.iloc[0]


def validate(selected):
    baseline = pd.read_csv(results_root() / "optimized_benchmark/selected_parameters.tsv", sep="\t")
    params = {row.method: row for row in baseline.itertuples(index=False)}
    rows = []
    for changed, seeds in [(True, range(96001, 96016)), (False, range(97001, 97016))]:
        for seed in seeds:
            _, first, second, index, simulation = make_dataset(GOLD, seed, changed)
            for method in METHODS:
                started = time.perf_counter()
                if method == "ct":
                    path = ct_path(simulation, selected.alpha, int(selected.max_iter))
                    beta = selected.beta_ratio
                else:
                    p = params[method]
                    path = fit_path(method, simulation, p.alpha, p.ridge, 80, MINIMUM_TRANSITIONS)
                    beta = p.beta_ratio
                result = select_one_change(path, beta, relative_beta=True)
                score = score_result(result, None if index is None else float(simulation.times[index]), first, second)
                rows.append({
                    "seed": seed, "changed": changed, "method": method,
                    "one_step_rmse": one_step_rmse(result, simulation, method),
                    "runtime_seconds": time.perf_counter()-started, **score,
                })
                print("VALID", json.dumps(rows[-1]), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "validation_by_seed.tsv", sep="\t", index=False)
    summary = []
    for method, group in frame.groupby("method"):
        changed = group[group.changed]; null = group[~group.changed]
        summary.append({
            "method": method, "cp_f1": changed.cp_f1.mean(),
            "network_aupr": changed.network_aupr.mean(), "change_aupr": changed.change_aupr.mean(),
            "rmse": changed.one_step_rmse.mean(), "null_specificity": null.cp_f1.mean(),
            "runtime_seconds": group.runtime_seconds.mean(),
        })
    pd.DataFrame(summary).to_csv(OUT / "validation_summary.tsv", sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-only", action="store_true",
                        help="Run the 60-setting development grid without validation systems")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = calibrate()
    if not args.calibration_only:
        validate(selected)


if __name__ == "__main__":
    main()
