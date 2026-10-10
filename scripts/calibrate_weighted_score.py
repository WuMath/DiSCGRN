#!/usr/bin/env python3
"""Recalibrate rho on six development systems under the N-weighted score.

Alpha and H are frozen from the stage-estimation development analysis. Every
candidate path is fitted once per system and method, then the prespecified
rho grid is evaluated without fitting on any independent test system.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.experiment_params import FIT_SPECS, RHO_GRID, SCORE_RULE
from ctcpgrn.paths import results_root
from run_optimized_benchmark import fit_path, make_dataset, one_step_rmse, score_result, trajectory_groups


GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = results_root() / "weighted_score_calibration"
DEVELOPMENT = ((True, 4101), (True, 4102), (True, 4103),
               (False, 5101), (False, 5102), (False, 5103))


def candidate_path(method: str, simulation, alpha: float, budget: int):
    if method == "ct":
        return scan_one_change_candidates(
            trajectory_groups(simulation), simulation.times, "ct", alpha,
            minimum_transitions=5, max_iter=budget,
            ct_initialiser="euler", ct_learning_rate=0.03,
        )
    return fit_path(method, simulation, alpha, 0.0, budget, 5)


def main() -> None:
    if results_root() == Path("results").resolve():
        raise RuntimeError("Set CTCPGRN_RESULTS_ROOT to a new, isolated result directory")
    if not GOLD.is_file():
        raise FileNotFoundError(GOLD)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, (alpha, budget) in FIT_SPECS.items():
        for changed, seed in DEVELOPMENT:
            _, first, second, true_index, simulation = make_dataset(GOLD, seed, changed)
            path = candidate_path(method, simulation, alpha, budget)
            for rho in RHO_GRID:
                selected = select_one_change(path, rho, relative_beta=True)
                truth_time = None if true_index is None else float(simulation.times[true_index])
                metrics = score_result(selected, truth_time, first, second)
                rows.append({
                    "score_rule": SCORE_RULE, "method": method, "seed": seed,
                    "changed": changed, "alpha": alpha, "max_iter": budget,
                    "beta_ratio": rho, "stationary_cost": path[0].objective,
                    "selected_change_index": selected.change_index,
                    "one_step_rmse": one_step_rmse(selected, simulation, method),
                    **metrics,
                })
        print(f"calibration paths fitted: {method}", flush=True)
    systems = pd.DataFrame(rows)
    systems.to_csv(OUT / "rho_systems.tsv", sep="\t", index=False)
    summary = []
    for (method, rho), group in systems.groupby(["method", "beta_ratio"], sort=True):
        changed = group[group.changed]
        null = group[~group.changed]
        row = {
            "score_rule": SCORE_RULE, "method": method,
            "alpha": FIT_SPECS[method][0], "max_iter": FIT_SPECS[method][1],
            "beta_ratio": rho,
            "mean_cp_f1": float(changed.cp_f1.mean()),
            "mean_change_aupr": float(changed.change_aupr.mean()),
            "mean_network_aupr": float(changed.network_aupr.mean()),
            "null_specificity": float(null.cp_f1.mean()),
            "mean_rmse": float(group.one_step_rmse.mean()),
        }
        row["selection_score"] = (
            row["mean_cp_f1"] + 0.35 * row["mean_change_aupr"]
            + 0.20 * row["mean_network_aupr"] + 0.50 * row["null_specificity"]
        )
        summary.append(row)
    grid = pd.DataFrame(summary)
    if not np.isfinite(grid["selection_score"]).all():
        raise RuntimeError("Calibration score contains non-finite values")
    grid.to_csv(OUT / "rho_grid.tsv", sep="\t", index=False)
    selected = (grid.sort_values(
        ["method", "selection_score", "mean_rmse", "beta_ratio"],
        ascending=[True, False, True, True], kind="stable",
    ).drop_duplicates("method", keep="first").sort_values("method"))
    selected.to_csv(OUT / "selected_parameters.tsv", sep="\t", index=False)
    weight_rows = []
    base_weights = (1.0, 0.35, 0.20, 0.50)
    metrics = ("mean_cp_f1", "mean_change_aupr", "mean_network_aupr", "null_specificity")
    for multipliers in product((0.5, 1.0, 1.5), repeat=4):
        for method, candidates in grid.groupby("method"):
            candidates = candidates.copy()
            candidates["perturbed_score"] = sum(
                base * multiplier * candidates[metric]
                for base, multiplier, metric in zip(base_weights, multipliers, metrics)
            )
            best = candidates.sort_values(
                ["perturbed_score", "mean_rmse", "beta_ratio"],
                ascending=[False, True, True], kind="stable",
            ).iloc[0]
            weight_rows.append({
                "score_rule": SCORE_RULE, "method": method,
                **{f"weight_multiplier_{index}": value for index, value in enumerate(multipliers, 1)},
                "selected_beta_ratio": float(best.beta_ratio),
                "selection_score": float(best.perturbed_score),
            })
    pd.DataFrame(weight_rows).to_csv(OUT / "weight_sensitivity.tsv", sep="\t", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "score_rule": SCORE_RULE,
        "development_changed_seeds": [seed for changed, seed in DEVELOPMENT if changed],
        "development_stationary_seeds": [seed for changed, seed in DEVELOPMENT if not changed],
        "fixed_fit_settings": FIT_SPECS,
        "rho_grid": RHO_GRID,
        "selection_score": "cp_f1 + 0.35*change_aupr + 0.20*network_aupr + 0.50*null_specificity",
        "tie_break": ["lower_mean_rmse", "lower_rho"],
    }, indent=2) + "\n", encoding="utf-8")
    print(selected[["method", "alpha", "beta_ratio", "max_iter", "selection_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
