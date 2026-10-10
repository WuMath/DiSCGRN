#!/usr/bin/env python3
"""Leakage-controlled optimisation benchmark for CT-CP-GRN.

The calibration set contains both changed and stationary systems.  Hyperparameters
are selected once, then frozen for independent changed and stationary test seeds.
Biological replicates are retained as separate trajectories.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import (
    ScanResult,
    fit_by_method,
    scan_one_change_candidates,
    select_one_change,
    slice_groups,
)
from ctcpgrn.io import LongitudinalGroup, read_dream_signed_network, signed_edge_list_to_matrix
from ctcpgrn.metrics import change_point_metrics, edge_change_metrics, edge_ranking_metrics
from ctcpgrn.paths import results_root
from ctcpgrn.simulation import (
    make_irregular_times,
    parameterise_signed_network,
    perturb_network,
    simulate_piecewise_linear_ode,
)


METHODS = {"ct": "CT-CP-GRN", "dplsq": "DPLSQ-DP", "euler": "Euler-DP", "two_stage": "Two-stage"}


def make_dataset(gold_path: Path, seed: int, changed: bool):
    signs, genes = signed_edge_list_to_matrix(read_dream_signed_network(gold_path))
    rng = np.random.default_rng(seed)
    first = parameterise_signed_network(signs, rng, magnitude_range=(0.12, 0.45), stability_margin=0.30)
    times = make_irregular_times(13, 12.0, 0.50, seed + 100)
    if changed:
        second = perturb_network(first, rng, rewiring_fraction=0.35, sign_flip_fraction=0.10, weight_scale=0.35)
        true_index = 6
        simulation = simulate_piecewise_linear_ode(
            [first, second], [float(times[true_index])], times,
            n_replicates=5, noise_std=0.03, seed=seed + 200,
        )
    else:
        second = first.copy()
        true_index = None
        simulation = simulate_piecewise_linear_ode(
            [first], [], times, n_replicates=5, noise_std=0.03, seed=seed + 200,
        )
    return genes, first, second, true_index, simulation


def trajectory_groups(simulation) -> list[LongitudinalGroup]:
    """Keep every replicate trajectory; do not collapse to a mean trajectory."""
    groups = []
    for replicate, values in enumerate(simulation.observations):
        groups.append(
            LongitudinalGroup(
                name=f"replicate_{replicate}",
                times=simulation.times.copy(),
                values=values.copy(),
                weights=np.ones_like(values),
                n_replicates=np.ones(len(simulation.times), dtype=float),
            )
        )
    return groups


def mean_group(simulation) -> LongitudinalGroup:
    values = simulation.observations.mean(axis=0)
    return LongitudinalGroup(
        "mean", simulation.times.copy(), values, np.ones_like(values),
        np.full(len(simulation.times), simulation.observations.shape[0], dtype=float),
    )


def score_result(result: ScanResult, true_time: float | None, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    truth = [] if true_time is None else [true_time]
    estimate = [] if result.change_time is None else [result.change_time]
    cp = change_point_metrics(truth, estimate, tolerance=1.5)
    if len(result.fits) == 2:
        before, after = result.fits[0].matrix, result.fits[1].matrix
    else:
        before = after = result.fits[0].matrix
    first_metrics = edge_ranking_metrics(first, before)
    second_metrics = edge_ranking_metrics(second, after)
    if true_time is None:
        change = {"change_aupr": float("nan"), "change_auroc": float("nan")}
    else:
        change = edge_change_metrics(first, second, before, after)
    return {
        **cp,
        "network_aupr": float((first_metrics["aupr"] + second_metrics["aupr"]) / 2),
        "network_auroc": float((first_metrics["auroc"] + second_metrics["auroc"]) / 2),
        **change,
    }


def two_stage_path(simulation, alpha: float, max_iter: int, minimum_transitions: int) -> list[ScanResult]:
    group = mean_group(simulation)
    stationary_fit = fit_by_method(trajectory_groups(simulation), "ct", alpha, None, max_iter)
    stationary_sse = float(np.sum((group.values - group.values.mean(axis=0)) ** 2))
    path = [ScanResult(None, None, [stationary_fit], stationary_sse)]
    best_boundary, best_sse = None, float("inf")
    for boundary in range(minimum_transitions, len(group.times) - minimum_transitions):
        sse = float(np.sum((group.values[: boundary + 1] - group.values[: boundary + 1].mean(axis=0)) ** 2))
        sse += float(np.sum((group.values[boundary:] - group.values[boundary:].mean(axis=0)) ** 2))
        if sse < best_sse:
            best_boundary, best_sse = boundary, sse
    groups = trajectory_groups(simulation)
    left = slice_groups(groups, group.times[0], group.times[best_boundary])
    right = slice_groups(groups, group.times[best_boundary], group.times[-1])
    fits = [fit_by_method(left, "ct", alpha, None, max_iter), fit_by_method(right, "ct", alpha, None, max_iter)]
    path.append(ScanResult(best_boundary, float(group.times[best_boundary]), fits, best_sse))
    return path


def fit_path(method: str, simulation, alpha: float, ridge: float, max_iter: int, minimum_transitions: int):
    if method == "two_stage":
        return two_stage_path(simulation, alpha, max_iter, minimum_transitions)
    return scan_one_change_candidates(
        trajectory_groups(simulation), simulation.times, method, alpha,
        minimum_transitions=minimum_transitions, max_iter=max_iter,
        ct_ridge=ridge if method == "ct" else 0.0,
    )


def one_step_rmse(result: ScanResult, simulation, method: str) -> float:
    from ctcpgrn.simulation import exact_step

    squared = []
    for trajectory in simulation.observations:
        for index in range(len(simulation.times) - 1):
            fit_index = 0 if result.change_index is None or index < result.change_index else 1
            fit = result.fits[fit_index]
            dt = float(simulation.times[index + 1] - simulation.times[index])
            if method in {"ct", "two_stage"}:
                prediction = exact_step(fit.matrix, fit.baseline, trajectory[index], dt)
            else:
                prediction = trajectory[index] + dt * (fit.matrix @ trajectory[index] + fit.baseline)
            squared.extend((prediction - trajectory[index + 1]) ** 2)
    return float(np.sqrt(np.mean(squared)))


def calibrate(args, gold: Path, output: Path) -> pd.DataFrame:
    alpha_grid = [0.0002, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016]
    beta_grid = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40]
    ridge_grid = [0.0, 0.0001]  # only applied to CT
    changed_seeds = [4101, 4102, 4103]
    null_seeds = [5101, 5102, 5103]
    datasets = [(seed, True, make_dataset(gold, seed, True)) for seed in changed_seeds]
    datasets += [(seed, False, make_dataset(gold, seed, False)) for seed in null_seeds]
    rows = []
    for method in METHODS:
        ridges = ridge_grid if method == "ct" else [0.0]
        for alpha in alpha_grid:
            for ridge in ridges:
                paths = []
                for seed, changed, (_, first, second, true_index, simulation) in datasets:
                    start = time.perf_counter()
                    path = fit_path(method, simulation, alpha, ridge, args.max_iter, args.minimum_transitions)
                    paths.append((seed, changed, first, second, true_index, simulation, path, time.perf_counter() - start))
                for beta in beta_grid:
                    changed_scores, null_scores, rmses, runtimes = [], [], [], []
                    for seed, changed, first, second, true_index, simulation, path, runtime in paths:
                        result = select_one_change(path, beta, relative_beta=True)
                        true_time = None if true_index is None else float(simulation.times[true_index])
                        metrics = score_result(result, true_time, first, second)
                        runtimes.append(runtime)
                        rmses.append(one_step_rmse(result, simulation, method))
                        (changed_scores if changed else null_scores).append(metrics)
                    row = {
                        "method": method, "alpha": alpha, "beta_ratio": beta, "ridge": ridge,
                        "mean_cp_f1": float(np.mean([x["cp_f1"] for x in changed_scores])),
                        "mean_change_aupr": float(np.mean([x["change_aupr"] for x in changed_scores])),
                        "mean_network_aupr": float(np.mean([x["network_aupr"] for x in changed_scores])),
                        "null_specificity": float(np.mean([x["cp_f1"] for x in null_scores])),
                        "mean_rmse": float(np.mean(rmses)),
                        "mean_runtime_seconds": float(np.mean(runtimes)),
                    }
                    row["selection_score"] = (
                        row["mean_cp_f1"] + 0.35 * row["mean_change_aupr"]
                        + 0.20 * row["mean_network_aupr"] + 0.50 * row["null_specificity"]
                    )
                    rows.append(row)
                    print("CAL", json.dumps(row), flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(output / "calibration_grid.tsv", sep="\t", index=False)
    selected = (
        grid.sort_values(
            ["method", "selection_score", "mean_rmse", "beta_ratio", "alpha"],
            ascending=[True, False, True, True, False],
        ).groupby("method", as_index=False).first()
    )
    selected.to_csv(output / "selected_parameters.tsv", sep="\t", index=False)
    return selected


def independent_test(args, gold: Path, output: Path, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for changed, seeds in [(True, range(92001, 92006)), (False, range(93001, 93006))]:
        for seed in seeds:
            _, first, second, true_index, simulation = make_dataset(gold, seed, changed)
            for params in selected.itertuples(index=False):
                start = time.perf_counter()
                path = fit_path(params.method, simulation, params.alpha, params.ridge, args.max_iter, args.minimum_transitions)
                result = select_one_change(path, params.beta_ratio, relative_beta=True)
                metrics = score_result(
                    result, None if true_index is None else float(simulation.times[true_index]), first, second,
                )
                rows.append({
                    "seed": seed, "changed": changed, "method": params.method,
                    "label": METHODS[params.method], "alpha": params.alpha,
                    "beta_ratio": params.beta_ratio, "ridge": params.ridge,
                    "true_change_time": None if true_index is None else float(simulation.times[true_index]),
                    "estimated_change_time": result.change_time,
                    "one_step_rmse": one_step_rmse(result, simulation, params.method),
                    "runtime_seconds": time.perf_counter() - start,
                    **metrics,
                })
                print("TEST", json.dumps(rows[-1]), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "independent_test_by_seed.tsv", sep="\t", index=False)
    summaries = []
    for method, group in frame.groupby("method"):
        changed_group = group[group.changed]
        null_group = group[~group.changed]
        summaries.append({
            "method": method, "label": METHODS[method],
            "cp_f1_mean": changed_group.cp_f1.mean(),
            "cp_exact_rate": np.mean(changed_group.cp_mae.fillna(np.inf) == 0),
            "network_aupr_mean": changed_group.network_aupr.mean(),
            "change_aupr_mean": changed_group.change_aupr.mean(),
            "rmse_mean": changed_group.one_step_rmse.mean(),
            "null_specificity": null_group.cp_f1.mean(),
            "runtime_seconds_mean": group.runtime_seconds.mean(),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "independent_test_summary.tsv", sep="\t", index=False)
    return frame


def bootstrap_ct(args, gold: Path, output: Path, selected: pd.DataFrame) -> None:
    params = selected[selected.method == "ct"].iloc[0]
    _, first, second, true_index, simulation = make_dataset(gold, 91001, True)
    rng = np.random.default_rng(20260929)
    rows = []
    for repetition in range(args.bootstrap):
        indices = rng.integers(0, simulation.observations.shape[0], size=simulation.observations.shape[0])
        boot = type(simulation)(
            simulation.times, simulation.observations[indices], simulation.clean_states[indices],
            simulation.networks, simulation.baselines, simulation.change_points,
        )
        path = fit_path("ct", boot, float(params.alpha), float(params.ridge), args.bootstrap_max_iter, args.minimum_transitions)
        result = select_one_change(path, float(params.beta_ratio), relative_beta=True)
        rows.append({"repetition": repetition, "change_index": result.change_index, "change_time": result.change_time})
        if (repetition + 1) % 20 == 0:
            print(f"BOOTSTRAP {repetition + 1}/{args.bootstrap}", flush=True)
    pd.DataFrame(rows).to_csv(output / "bootstrap_200.tsv", sep="\t", index=False)
    counts = Counter("none" if x["change_time"] is None else f"{x['change_time']:.6f}" for x in rows)
    (output / "bootstrap_summary.json").write_text(
        json.dumps({"true_time": float(simulation.times[true_index]), "counts": counts}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv"))
    parser.add_argument("--output-dir", type=Path, default=results_root() / "optimized_benchmark")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--minimum-transitions", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap-max-iter", type=int, default=40)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--calibration-only", action="store_true",
                        help="Select baseline settings without running test or bootstrap systems")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = calibrate(args, args.gold, args.output_dir)
    if args.calibration_only:
        return
    independent_test(args, args.gold, args.output_dir, selected)
    if not args.skip_bootstrap:
        bootstrap_ct(args, args.gold, args.output_dir, selected)


if __name__ == "__main__":
    main()
