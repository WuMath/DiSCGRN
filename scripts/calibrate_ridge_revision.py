#!/usr/bin/env python3
"""Development-only elastic-net ridge screen with frozen DiSCGRN alpha and rho.

The six original calibration seeds are reused. Independent frozen-test seeds
are never consulted when selecting the ridge coefficient.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.experiment_params import load_frozen_params
from run_optimized_benchmark import make_dataset, one_step_rmse, score_result, trajectory_groups


GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridges", default="0,0.0002,0.0004,0.001")
    parser.add_argument("--frozen-parameters", type=Path,
                        help="Existing development-only selected_parameters.tsv")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    ridges = [float(item) for item in args.ridges.split(",")]
    if not ridges or any(value < 0 or not np.isfinite(value) for value in ridges):
        raise ValueError("Ridge grid must contain finite nonnegative values")
    alpha, rho, budget = load_frozen_params(args.frozen_parameters)["ct"]
    args.output_dir.mkdir(parents=True)
    rows = []
    for seed, changed in [(seed, True) for seed in (4101, 4102, 4103)] + [
        (seed, False) for seed in (5101, 5102, 5103)
    ]:
        _, first, second, true_index, simulation = make_dataset(GOLD, seed, changed)
        for ridge in ridges:
            path = scan_one_change_candidates(
                trajectory_groups(simulation), simulation.times, "ct", alpha,
                minimum_transitions=5, max_iter=budget,
                ct_initialiser="euler", ct_learning_rate=0.03, ct_ridge=ridge,
            )
            result = select_one_change(path, rho, relative_beta=True)
            true_time = None if true_index is None else float(simulation.times[true_index])
            rows.append({"seed": seed, "changed": changed, "alpha": alpha,
                         "rho": rho, "max_iter": budget, "ridge": ridge,
                         "rmse": one_step_rmse(result, simulation, "ct"),
                         **score_result(result, true_time, first, second)})
            pd.DataFrame(rows).to_csv(args.output_dir / "ridge_calibration_by_seed.tsv",
                                      sep="\t", index=False)
            print(f"seed={seed} ridge={ridge:g} network_aupr={rows[-1]['network_aupr']:.4f}",
                  flush=True)
    grid = pd.DataFrame(rows)
    summary = []
    for ridge, group in grid.groupby("ridge", sort=True):
        changed = group[group.changed]
        null = group[~group.changed]
        record = {"ridge": ridge, "alpha": alpha, "rho": rho,
                  "network_aupr": changed.network_aupr.mean(),
                  "change_aupr": changed.change_aupr.mean(),
                  "cp_f1": changed.cp_f1.mean(),
                  "null_specificity": null.cp_f1.mean(),
                  "rmse": group.rmse.mean()}
        record["selection_score"] = (record["cp_f1"] + 0.35 * record["change_aupr"]
                                     + 0.20 * record["network_aupr"]
                                     + 0.50 * record["null_specificity"])
        summary.append(record)
    summary = pd.DataFrame(summary).sort_values(
        ["selection_score", "rmse", "ridge"], ascending=[False, True, True],
        kind="stable")
    summary.to_csv(args.output_dir / "ridge_calibration_summary.tsv", sep="\t", index=False)
    summary.head(1).to_csv(args.output_dir / "selected_ridge.tsv", sep="\t", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
