#!/usr/bin/env python3
"""Frozen-parameter comparison on 30 untouched changed/null systems."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.paths import results_root
from run_optimized_benchmark import fit_path, make_dataset, one_step_rmse, score_result, trajectory_groups

GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = results_root() / "revision_comparison_h80"
def _job(args):
    seed, changed, method, ct_ridge, params = args
    _, first, second, index, simulation = make_dataset(GOLD, seed, changed)
    alpha, beta, ridge, max_iter = params
    if method == "ct":
        ridge = ct_ridge
    if method == "ct":
        path = scan_one_change_candidates(
            trajectory_groups(simulation), simulation.times, "ct", alpha,
            minimum_transitions=5, max_iter=max_iter, ct_initialiser="euler",
            ct_learning_rate=0.03, ct_ridge=ridge,
        )
    else:
        path = fit_path(method, simulation, alpha, ridge, max_iter, 5)
    result = select_one_change(path, beta, relative_beta=True)
    true_time = None if index is None else float(simulation.times[index])
    return {
        "seed": seed, "changed": changed, "method": method,
        "alpha": alpha, "beta_ratio": beta, "ridge": ridge,
        "max_iter": max_iter,
        "rmse": one_step_rmse(result, simulation, method),
        **score_result(result, true_time, first, second),
    }


def paired_ci(values, rng, n=20000):
    draws = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return values.mean(), *np.quantile(draws, [0.025, 0.975])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-ridge", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frozen-parameters", type=Path,
                        help="Existing development-only selected_parameters.tsv")
    args = parser.parse_args()
    params = {method: (alpha, rho, 0.0, budget)
              for method, (alpha, rho, budget) in
              load_frozen_params(args.frozen_parameters).items()}
    if args.ct_ridge < 0 or not np.isfinite(args.ct_ridge):
        raise ValueError("--ct-ridge must be finite and nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    jobs = [(seed, changed, method, args.ct_ridge, settings)
            for changed, seeds in ((True, range(99601, 99616)), (False, range(99701, 99716)))
            for seed in seeds for method, settings in params.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "comparison_30systems.tsv", sep="\t", index=False)
    summaries = []
    for method, g in frame.groupby("method"):
        changed, null = g[g.changed], g[~g.changed]
        summaries.append({
            "method": method, "cp_f1_mean": changed.cp_f1.mean(),
            "cp_exact_rate": np.mean(changed.cp_mae == 0),
            "network_aupr_mean": changed.network_aupr.mean(),
            "change_aupr_mean": changed.change_aupr.mean(),
            "rmse_mean": g.rmse.mean(), "null_specificity": null.cp_f1.mean(),
        })
    pd.DataFrame(summaries).to_csv(args.output_dir / "comparison_summary.tsv", sep="\t", index=False)

    changed = frame[frame.changed].pivot(index="seed", columns="method")
    rng = np.random.default_rng(20260930)
    comparisons = []
    for metric in ("network_aupr", "change_aupr", "rmse"):
        for comparator in ("euler", "dplsq", "two_stage"):
            # Positive means CT is better for AUPR; negative means CT is better for RMSE.
            delta = (changed[metric]["ct"] - changed[metric][comparator]).to_numpy()
            mean, low, high = paired_ci(delta, rng)
            comparisons.append({"metric": metric, "comparison": f"ct_vs_{comparator}",
                                "mean_difference": mean, "ci_low": low, "ci_high": high})
    pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_bootstrap.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
