#!/usr/bin/env python3
"""E3: optimisation ablation, convergence audit, and regularisation path.

The true change point is supplied in this diagnostic experiment.  This removes
change-point selection as a confounder and asks whether exact continuous-time
segment fitting itself has converged and recovers the two regime networks.
All summaries use 15 independently generated systems.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import fit_by_method, slice_groups
from ctcpgrn.metrics import edge_change_metrics, edge_ranking_metrics
from ctcpgrn.paths import results_root
from run_optimized_benchmark import make_dataset, trajectory_groups

GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = results_root() / "revision_e3"
SEEDS = tuple(range(99101, 99116))


def _score(first, second, left, right):
    m1 = edge_ranking_metrics(first, left.matrix)
    m2 = edge_ranking_metrics(second, right.matrix)
    change = edge_change_metrics(first, second, left.matrix, right.matrix)
    return {
        "network_aupr": (m1["aupr"] + m2["aupr"]) / 2,
        "network_auroc": (m1["auroc"] + m2["auroc"]) / 2,
        "change_aupr": change["change_aupr"],
        "change_auroc": change["change_auroc"],
        "objective": left.objective + right.objective,
        "data_loss": left.data_loss + right.data_loss,
        "iterations": left.n_iterations + right.n_iterations,
        "both_converged": bool(left.converged and right.converged),
    }


def _fit_known_change(seed, method, alpha, max_iter, warm_start=True, initialiser="euler", learning_rate=0.03):
    _, first, second, index, simulation = make_dataset(GOLD, seed, True)
    groups = trajectory_groups(simulation)
    left_groups = slice_groups(groups, simulation.times[0], simulation.times[index])
    right_groups = slice_groups(groups, simulation.times[index], simulation.times[-1])
    left = fit_by_method(
        left_groups, method, alpha, None, max_iter,
        ct_warm_start=warm_start, ct_initialiser=initialiser,
        ct_learning_rate=learning_rate,
    )
    right = fit_by_method(
        right_groups, method, alpha, None, max_iter,
        ct_warm_start=warm_start, ct_initialiser=initialiser,
        ct_learning_rate=learning_rate,
    )
    return _score(first, second, left, right)


def _ablation_seed(seed):
    rows = []
    specifications = [
        ("Euler-Lasso", "euler", 0.0005, 1, True, "euler", 0.03),
        ("CT-EulerInit-H80", "ct", 0.001, 80, True, "euler", 0.03),
        ("CT-EulerInit-H320", "ct", 0.001, 320, True, "euler", 0.03),
        ("CT-EulerInit-H1000", "ct", 0.001, 1000, True, "euler", 0.03),
        ("CT-DPLSQInit-H320", "ct", 0.001, 320, True, "dplsq", 0.03),
        ("CT-ZeroInit-H320", "ct", 0.001, 320, False, "euler", 0.03),
        ("CT-EulerInit-H320-LR0.10", "ct", 0.001, 320, True, "euler", 0.10),
        ("CT-EulerInit-H320-LR0.30", "ct", 0.001, 320, True, "euler", 0.30),
        ("CT-a0004-EulerInit-H80", "ct", 0.0004, 80, True, "euler", 0.03),
        ("CT-a0004-EulerInit-H160", "ct", 0.0004, 160, True, "euler", 0.03),
        ("CT-a0004-EulerInit-H320", "ct", 0.0004, 320, True, "euler", 0.03),
    ]
    for label, method, alpha, max_iter, warm, initialiser, learning_rate in specifications:
        row = _fit_known_change(seed, method, alpha, max_iter, warm, initialiser, learning_rate)
        rows.append({"seed": seed, "label": label, "method": method,
                     "alpha": alpha, "max_iter": max_iter,
                     "warm_start": warm, "initialiser": initialiser,
                     "learning_rate": learning_rate, **row})
    return rows


def _path_seed(seed):
    rows = []
    # Twelve logarithmically spaced values, as requested by the reviewer.
    for alpha in np.geomspace(1e-4, 1.6e-2, 12):
        row = _fit_known_change(seed, "ct", float(alpha), 320, True, "euler")
        rows.append({"seed": seed, "alpha": float(alpha), **row})
    return rows


def _budget_extension_seed(seed):
    rows = []
    for max_iter in (40, 640, 1000):
        row = _fit_known_change(seed, "ct", 0.0004, max_iter, True, "euler")
        rows.append({"seed": seed, "label": f"CT-a0004-EulerInit-H{max_iter}",
                     "alpha": 0.0004, "max_iter": max_iter, **row})
    return rows


def _summarise(frame, keys):
    metrics = ["network_aupr", "network_auroc", "change_aupr", "change_auroc",
               "objective", "data_loss", "iterations", "both_converged"]
    return frame.groupby(keys, as_index=False)[metrics].agg(["mean", "std"]).reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--part", choices=("all", "ablation", "path", "budget-extension"), default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part in {"all", "ablation"}:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [row for seed_rows in pool.map(_ablation_seed, SEEDS) for row in seed_rows]
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "ablation_15systems.tsv", sep="\t", index=False)
        summary = frame.groupby("label", as_index=False).agg(
            network_aupr_mean=("network_aupr", "mean"),
            network_aupr_sd=("network_aupr", "std"),
            change_aupr_mean=("change_aupr", "mean"),
            objective_mean=("objective", "mean"),
            data_loss_mean=("data_loss", "mean"),
            convergence_rate=("both_converged", "mean"),
            iterations_mean=("iterations", "mean"),
        )
        summary.to_csv(OUT / "ablation_15systems_summary.tsv", sep="\t", index=False)
    if args.part in {"all", "path"}:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [row for seed_rows in pool.map(_path_seed, SEEDS) for row in seed_rows]
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "regularisation_path_15systems.tsv", sep="\t", index=False)
        summary = frame.groupby("alpha", as_index=False).agg(
            network_aupr_mean=("network_aupr", "mean"),
            network_aupr_sd=("network_aupr", "std"),
            change_aupr_mean=("change_aupr", "mean"),
            objective_mean=("objective", "mean"),
            data_loss_mean=("data_loss", "mean"),
            convergence_rate=("both_converged", "mean"),
        )
        summary.to_csv(OUT / "regularisation_path_15systems_summary.tsv", sep="\t", index=False)
    if args.part == "budget-extension":
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [row for seed_rows in pool.map(_budget_extension_seed, SEEDS) for row in seed_rows]
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "budget_extension_15systems.tsv", sep="\t", index=False)
        summary = frame.groupby("label", as_index=False).agg(
            network_aupr_mean=("network_aupr", "mean"),
            network_aupr_sd=("network_aupr", "std"),
            change_aupr_mean=("change_aupr", "mean"),
            objective_mean=("objective", "mean"),
            data_loss_mean=("data_loss", "mean"),
            convergence_rate=("both_converged", "mean"),
            iterations_mean=("iterations", "mean"),
        )
        summary.to_csv(OUT / "budget_extension_15systems_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
