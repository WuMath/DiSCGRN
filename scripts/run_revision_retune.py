#!/usr/bin/env python3
"""Leakage-controlled CT re-tuning after the objective-scale correction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from run_optimized_benchmark import make_dataset, one_step_rmse, score_result, trajectory_groups

GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = Path("results/revision_retune")
ALPHAS = (0.00025, 0.00040, 0.00063)
BETAS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


def _path_job(args):
    seed, changed, alpha = args
    _, first, second, index, simulation = make_dataset(GOLD, seed, changed)
    path = scan_one_change_candidates(
        trajectory_groups(simulation), simulation.times, "ct", alpha,
        minimum_transitions=5, max_iter=320, ct_initialiser="euler",
        ct_learning_rate=0.03,
    )
    rows = []
    for beta in BETAS:
        result = select_one_change(path, beta, relative_beta=True)
        true_time = None if index is None else float(simulation.times[index])
        rows.append({
            "seed": seed, "changed": changed, "alpha": alpha, "beta_ratio": beta,
            "rmse": one_step_rmse(result, simulation, "ct"),
            **score_result(result, true_time, first, second),
        })
    return rows


def calibration():
    jobs = [(seed, changed, alpha)
            for changed, seeds in ((True, range(4101, 4104)), (False, range(5101, 5104)))
            for seed in seeds for alpha in ALPHAS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [r for batch in pool.map(_path_job, jobs) for r in batch]
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "calibration_by_seed.tsv", sep="\t", index=False)
    summaries = []
    for (alpha, beta), g in raw.groupby(["alpha", "beta_ratio"]):
        changed, null = g[g.changed], g[~g.changed]
        summaries.append({
            "alpha": alpha, "beta_ratio": beta,
            "mean_cp_f1": changed.cp_f1.mean(),
            "mean_network_aupr": changed.network_aupr.mean(),
            "mean_change_aupr": changed.change_aupr.mean(),
            "null_specificity": null.cp_f1.mean(),
            "mean_rmse": g.rmse.mean(),
        })
    summary = pd.DataFrame(summaries)
    summary["selection_score"] = (
        summary.mean_cp_f1 + 0.5 * summary.mean_network_aupr
        + 0.5 * summary.mean_change_aupr + 0.5 * summary.null_specificity
    )
    summary.to_csv(OUT / "calibration_summary.tsv", sep="\t", index=False)
    eligible = summary[(summary.mean_cp_f1 >= 0.99) & (summary.null_specificity >= 0.99)]
    if eligible.empty:
        eligible = summary
    selected = eligible.sort_values(
        ["selection_score", "mean_rmse"], ascending=[False, True]
    ).iloc[[0]]
    selected.to_csv(OUT / "selected_parameters.tsv", sep="\t", index=False)
    return float(selected.iloc[0].alpha), float(selected.iloc[0].beta_ratio)


def validation(alpha, beta):
    jobs = [(seed, changed, alpha)
            for changed, seeds in ((True, range(99401, 99416)), (False, range(99501, 99516)))
            for seed in seeds]
    with ProcessPoolExecutor(max_workers=4) as pool:
        batches = list(pool.map(_path_job, jobs))
    rows = []
    for batch in batches:
        rows.append(next(r for r in batch if np.isclose(r["beta_ratio"], beta)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "validation_30systems.tsv", sep="\t", index=False)
    changed, null = frame[frame.changed], frame[~frame.changed]
    summary = pd.DataFrame([{
        "alpha": alpha, "beta_ratio": beta, "n_changed": len(changed), "n_null": len(null),
        "cp_f1_mean": changed.cp_f1.mean(),
        "cp_exact_rate": np.mean(changed.cp_mae == 0),
        "network_aupr_mean": changed.network_aupr.mean(),
        "change_aupr_mean": changed.change_aupr.mean(),
        "rmse_mean": frame.rmse.mean(),
        "null_specificity": null.cp_f1.mean(),
    }])
    summary.to_csv(OUT / "validation_summary.tsv", sep="\t", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    alpha, beta = calibration()
    validation(alpha, beta)


if __name__ == "__main__":
    main()
