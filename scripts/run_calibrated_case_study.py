#!/usr/bin/env python3
"""Calibrate and run one complete DREAM4 non-stationary case study."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ctcpgrn.baselines import ScanResult, fit_by_method, scan_one_change, two_stage_change
from ctcpgrn.io import LongitudinalGroup, read_dream_signed_network, signed_edge_list_to_matrix
from ctcpgrn.metrics import change_point_metrics, edge_change_metrics, edge_ranking_metrics
from ctcpgrn.simulation import make_irregular_times, parameterise_signed_network, perturb_network, simulate_piecewise_linear_ode


METHODS = {"ct": "CT-CP-GRN", "euler": "Euler-DP", "dplsq": "DPLSQ-TV", "two_stage": "Two-stage"}

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_dataset(gold_path: Path, seed: int):
    signs, genes = signed_edge_list_to_matrix(read_dream_signed_network(gold_path))
    rng = np.random.default_rng(seed)
    first = parameterise_signed_network(signs, rng, magnitude_range=(0.12, 0.45), stability_margin=0.30)
    second = perturb_network(first, rng, rewiring_fraction=0.35, sign_flip_fraction=0.10, weight_scale=0.35)
    times = make_irregular_times(13, 12.0, 0.50, seed + 100)
    true_index = 6
    simulation = simulate_piecewise_linear_ode([first, second], [float(times[true_index])], times, n_replicates=5, noise_std=0.03, seed=seed + 200)
    return genes, first, second, true_index, simulation


def group_from_observations(times: np.ndarray, observations: np.ndarray, name: str = "dream") -> LongitudinalGroup:
    means = observations.mean(axis=0)
    variance = observations.var(axis=0, ddof=1)
    weights = 1.0 / np.maximum(variance, 1e-4)
    weights /= weights.mean()
    return LongitudinalGroup(name, times, means, weights, np.full(len(times), observations.shape[0]))


def run_method(method: str, group: LongitudinalGroup, times: np.ndarray, alpha: float, beta: float, max_iter: int) -> ScanResult:
    if method == "two_stage":
        return two_stage_change(group, times, alpha, beta, 3, max_iter)
    return scan_one_change([group], times, method, alpha, beta, 3, max_iter=max_iter)


def score_result(result: ScanResult, true_time: float, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    cp = change_point_metrics([true_time], [] if result.change_time is None else [result.change_time], tolerance=1.5)
    if len(result.fits) == 2:
        before, after = result.fits[0].matrix, result.fits[1].matrix
    else:
        before = after = result.fits[0].matrix
    first_metrics = edge_ranking_metrics(first, before)
    second_metrics = edge_ranking_metrics(second, after)
    change = edge_change_metrics(first, second, before, after)
    return {
        **cp,
        "network_aupr": float((first_metrics["aupr"] + second_metrics["aupr"]) / 2),
        "network_auroc": float((first_metrics["auroc"] + second_metrics["auroc"]) / 2),
        **change,
    }


def predict(fit, method: str, x: np.ndarray, dt: float) -> np.ndarray:
    if method in {"ct", "two_stage"}:
        from ctcpgrn.simulation import exact_step
        return exact_step(fit.matrix, fit.baseline, x, dt)
    return x + dt * (fit.matrix @ x + fit.baseline)


def blocked_cv(group: LongitudinalGroup, result: ScanResult, method: str, alpha: float, max_iter: int) -> list[float]:
    n_transitions = len(group.times) - 1
    blocks = [np.arange(start, min(start + 2, n_transitions)) for start in (1, 5, 9)]
    rmses = []
    boundary = result.change_index
    for fold, heldout in enumerate(blocks):
        training_by_regime: list[list[LongitudinalGroup]] = [[], []] if boundary is not None else [[]]
        for idx in range(n_transitions):
            if idx in heldout:
                continue
            regime = 0 if boundary is None or idx < boundary else 1
            training_by_regime[regime].append(LongitudinalGroup(f"tr{fold}_{idx}", group.times[idx:idx+2], group.values[idx:idx+2], group.weights[idx:idx+2], group.n_replicates[idx:idx+2]))
        fits = [fit_by_method(groups, "ct" if method == "two_stage" else method, alpha, None, max_iter) for groups in training_by_regime]
        errors = []
        for idx in heldout:
            regime = 0 if boundary is None or idx < boundary else 1
            prediction = predict(fits[regime], method, group.values[idx], group.times[idx + 1] - group.times[idx])
            errors.extend((prediction - group.values[idx + 1]) ** 2)
        rmses.append(float(np.sqrt(np.mean(errors))))
    return rmses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/calibrated_case_study"))
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alpha_grid = [0.0005, 0.002, 0.008]
    beta_grid = [0.0, 0.002, 0.01, 0.04]
    calibration_seeds = [4101, 4102, 4103]

    calibration_rows = []
    for method in METHODS:
        for alpha in alpha_grid:
            for beta in beta_grid:
                scores = []
                for seed in calibration_seeds:
                    _, first, second, true_index, sim = make_dataset(args.gold, seed)
                    group = group_from_observations(sim.times, sim.observations)
                    result = run_method(method, group, sim.times, alpha, beta, args.max_iter)
                    metrics = score_result(result, sim.times[true_index], first, second)
                    scores.append(metrics)
                row = {
                    "method": method, "alpha": alpha, "beta": beta,
                    "mean_cp_f1": float(np.mean([x["cp_f1"] for x in scores])),
                    "mean_change_aupr": float(np.mean([x["change_aupr"] for x in scores])),
                    "mean_network_aupr": float(np.mean([x["network_aupr"] for x in scores])),
                }
                row["selection_score"] = row["mean_cp_f1"] + 0.25 * row["mean_change_aupr"] + 0.10 * row["mean_network_aupr"]
                calibration_rows.append(row)
                print("CAL", json.dumps(row))
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(args.output_dir / "calibration_grid.tsv", sep="\t", index=False)
    # Deterministic tie-break: retain the equally accurate, less penalised
    # candidate so a boundary is not suppressed merely by arbitrary ordering.
    selected = calibration.sort_values(
        ["method", "selection_score", "beta", "alpha"],
        ascending=[True, False, True, False],
    ).groupby("method", as_index=False).first()
    selected.to_csv(args.output_dir / "selected_parameters.tsv", sep="\t", index=False)

    genes, first, second, true_index, simulation = make_dataset(args.gold, 91001)
    group = group_from_observations(simulation.times, simulation.observations)
    test_rows, fitted = [], {}
    for row in selected.itertuples(index=False):
        result = run_method(row.method, group, simulation.times, row.alpha, row.beta, args.max_iter)
        metrics = score_result(result, simulation.times[true_index], first, second)
        cv = blocked_cv(group, result, row.method, row.alpha, args.max_iter)
        test_row = {
            "method": row.method, "label": METHODS[row.method], "alpha": row.alpha, "beta": row.beta,
            "true_change_time": float(simulation.times[true_index]),
            "estimated_change_time": result.change_time,
            "blocked_cv_rmse_mean": float(np.mean(cv)),
            "blocked_cv_rmse_sd": float(np.std(cv, ddof=1)),
            **metrics,
        }
        test_rows.append(test_row)
        fitted[row.method] = result
        print("TEST", json.dumps(test_row))
    pd.DataFrame(test_rows).to_csv(args.output_dir / "test_results.tsv", sep="\t", index=False)

    ct_selected = selected[selected.method == "ct"].iloc[0]
    rng = np.random.default_rng(20260929)
    bootstrap_rows = []
    for repetition in range(args.bootstrap):
        indices = rng.integers(0, simulation.observations.shape[0], size=simulation.observations.shape[0])
        boot_group = group_from_observations(simulation.times, simulation.observations[indices], f"boot{repetition}")
        result = run_method("ct", boot_group, simulation.times, float(ct_selected.alpha), float(ct_selected.beta), max(10, args.max_iter // 2))
        bootstrap_rows.append({"repetition": repetition, "change_index": result.change_index, "change_time": result.change_time})
        if (repetition + 1) % 20 == 0:
            print(f"BOOTSTRAP {repetition + 1}/{args.bootstrap}", flush=True)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output_dir / "bootstrap_200.tsv", sep="\t", index=False)
    counts = Counter("none" if pd.isna(x) else f"{x:.6f}" for x in bootstrap.change_time)
    summary = {
        "network": str(args.gold), "test_seed": 91001, "genes": genes,
        "times": simulation.times.tolist(), "true_change_index": true_index,
        "true_change_time": float(simulation.times[true_index]),
        "calibration_seeds": calibration_seeds, "alpha_grid": alpha_grid, "beta_grid": beta_grid,
        "bootstrap_repetitions": args.bootstrap, "bootstrap_counts": counts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
