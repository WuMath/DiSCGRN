#!/usr/bin/env python3
"""Fully crossed, paired alpha x H x rho sensitivity on one development cohort.

Each system is generated once. For each (alpha, H), candidate interval fits are
cached in memory while all rho values are evaluated. No independent test seeds
or real-data samples are used. Completed systems are checkpointed for resume.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_weighted_score import GOLD, candidate_path
from ctcpgrn.baselines import select_one_change
from ctcpgrn.experiment_params import RHO_GRID, SCORE_RULE
from ctcpgrn.paths import results_root
from run_optimized_benchmark import make_dataset, one_step_rmse, score_result


ALPHAS = (0.0003, 0.0004, 0.0005)
BUDGETS = (40, 80, 160, 320)
BASE_WEIGHTS = (1.0, 0.35, 0.20, 0.50)
METRIC_COLUMNS = ("mean_cp_f1", "mean_change_aupr", "mean_network_aupr", "null_specificity")
SYSTEM_COLUMNS = (
    "seed", "changed", "alpha", "max_iter", "beta_ratio", "cp_f1",
    "change_aupr", "network_aupr", "one_step_rmse", "stationary_cost",
    "selected_change_index",
)


def _evaluate_system(task):
    changed, seed = task
    _, first, second, true_index, simulation = make_dataset(GOLD, seed, changed)
    truth_time = None if true_index is None else float(simulation.times[true_index])
    rows = []
    for alpha, budget in product(ALPHAS, BUDGETS):
        path = candidate_path("ct", simulation, alpha, budget)
        for rho in RHO_GRID:
            selected = select_one_change(path, rho, relative_beta=True)
            metrics = score_result(selected, truth_time, first, second)
            rows.append({
                "seed": seed, "changed": changed, "alpha": alpha,
                "max_iter": budget, "beta_ratio": rho,
                "cp_f1": metrics["cp_f1"],
                "change_aupr": metrics["change_aupr"],
                "network_aupr": metrics["network_aupr"],
                "one_step_rmse": one_step_rmse(selected, simulation, "ct"),
                "stationary_cost": path[0].objective,
                "selected_change_index": selected.change_index,
            })
    return seed, rows


def _grid(frame):
    rows = []
    for (alpha, budget, rho), group in frame.groupby(
        ["alpha", "max_iter", "beta_ratio"], sort=True
    ):
        changed = group[group.changed]
        stationary = group[~group.changed]
        row = {
            "alpha": alpha, "max_iter": budget, "beta_ratio": rho,
            "mean_cp_f1": changed.cp_f1.mean(),
            "mean_change_aupr": changed.change_aupr.mean(),
            "mean_network_aupr": changed.network_aupr.mean(),
            "null_specificity": stationary.cp_f1.mean(),
            "mean_rmse": group.one_step_rmse.mean(),
            "n_changed": changed.seed.nunique(),
            "n_stationary": stationary.seed.nunique(),
        }
        row["selection_score"] = sum(
            weight * row[column] for weight, column in zip(BASE_WEIGHTS, METRIC_COLUMNS)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _decomposition(grid, response):
    levels = [ALPHAS, BUDGETS, tuple(RHO_GRID)]
    indexed = grid.set_index(["alpha", "max_iter", "beta_ratio"])[response]
    y = np.array([indexed.loc[key] for key in product(*levels)]).reshape(
        len(ALPHAS), len(BUDGETS), len(RHO_GRID)
    )
    mean = y.mean()
    a = y.mean(axis=(1, 2), keepdims=True) - mean
    h = y.mean(axis=(0, 2), keepdims=True) - mean
    r = y.mean(axis=(0, 1), keepdims=True) - mean
    ah = y.mean(axis=2, keepdims=True) - mean - a - h
    ar = y.mean(axis=1, keepdims=True) - mean - a - r
    hr = y.mean(axis=0, keepdims=True) - mean - h - r
    ahr = y - mean - a - h - r - ah - ar - hr
    components = {
        "alpha": a, "H": h, "rho": r,
        "alpha:H": ah, "alpha:rho": ar, "H:rho": hr,
        "alpha:H:rho": ahr,
    }
    total = float(np.square(y - mean).sum())
    rows = []
    for name, component in components.items():
        ss = float(np.square(np.broadcast_to(component, y.shape)).sum())
        rows.append({
            "response": response, "component": name, "sum_of_squares": ss,
            "variance_share": ss / total if total > 0 else np.nan,
            "rms_effect": np.sqrt(ss / y.size),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--systems-per-class", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.systems_per_class <= 15:
        parser.error("systems-per-class must be between 1 and 15")
    if not GOLD.is_file():
        raise FileNotFoundError(GOLD)
    out = results_root() / "joint_parameter_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    systems_path = out / "system_scores.tsv"
    tasks = [(True, 98101 + i) for i in range(args.systems_per_class)]
    tasks += [(False, 98201 + i) for i in range(args.systems_per_class)]
    completed = set()
    if systems_path.exists():
        existing = pd.read_csv(systems_path, sep="\t")
        expected = len(ALPHAS) * len(BUDGETS) * len(RHO_GRID)
        counts = existing.groupby("seed").size()
        completed = set(counts[counts == expected].index)
        if len(existing) != len(completed) * expected:
            raise RuntimeError("Partial checkpoint detected; inspect system_scores.tsv before resuming")
    pending = [task for task in tasks if task[1] not in completed]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_evaluate_system, task): task for task in pending}
        for future in as_completed(futures):
            seed, rows = future.result()
            pd.DataFrame(rows, columns=SYSTEM_COLUMNS).to_csv(
                systems_path, sep="\t", index=False,
                mode="a", header=not systems_path.exists(),
            )
            print(f"completed system {seed}: {len(rows)} settings", flush=True)
    frame = pd.read_csv(systems_path, sep="\t")
    expected_seeds = {seed for _, seed in tasks}
    if set(frame.seed) != expected_seeds:
        raise RuntimeError("Incomplete system set; resume the command")
    grid = _grid(frame)
    grid.to_csv(out / "grid.tsv", sep="\t", index=False)
    decomposition = []
    for response in ("selection_score", "mean_network_aupr", "mean_cp_f1", "null_specificity"):
        decomposition.extend(_decomposition(grid, response))
    pd.DataFrame(decomposition).to_csv(out / "interaction_decomposition.tsv", sep="\t", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "score_rule": SCORE_RULE,
        "changed_seeds": [seed for changed, seed in tasks if changed],
        "stationary_seeds": [seed for changed, seed in tasks if not changed],
        "alpha_grid": ALPHAS, "H_grid": BUDGETS, "rho_grid": tuple(RHO_GRID),
        "candidate_fits_cached_across_rho": True,
        "boundary_search": "zero or one change, at least five transitions per segment",
        "interaction": "balanced factorial sum-of-squares decomposition of grid means",
        "independent_test_systems_used": False,
    }, indent=2) + "\n", encoding="utf-8")
    best = grid.sort_values(
        ["selection_score", "mean_rmse", "alpha", "max_iter", "beta_ratio"],
        ascending=[False, True, True, True, True], kind="stable",
    ).iloc[0]
    print("joint grid shape:", len(grid), "combinations")
    print("best descriptive setting:", best[[
        "alpha", "max_iter", "beta_ratio", "selection_score", "mean_rmse"
    ]].to_dict())


if __name__ == "__main__":
    main()
