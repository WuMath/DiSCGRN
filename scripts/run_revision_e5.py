#!/usr/bin/env python3
"""Mandatory E5 reanalysis requested by the manuscript review.

All resampling units are independent simulated systems.  Blocked-CV first
averages the three folds within each system and then performs paired bootstrap
over systems.  Change-point stability is evaluated over all 15 validation
systems rather than a selected favourable example.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.io import LongitudinalGroup
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.paths import results_root
from ctcpgrn.simulation import exact_step
from run_optimized_benchmark import make_dataset, trajectory_groups

GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = results_root() / "revision_e5"
_FROZEN = load_frozen_params()
METHODS = {method: {"alpha": _FROZEN[method][0], "beta": _FROZEN[method][1]}
           for method in ("ct", "euler", "dplsq")}
BLOCKS = ((1, 2), (5, 6), (9, 10))


def _fit(method: str, groups: list[LongitudinalGroup], times: np.ndarray):
    params = METHODS[method]
    path = scan_one_change_candidates(
        groups,
        times,
        method,
        params["alpha"],
        minimum_transitions=5,
        max_iter=80,
        ct_initialiser="euler" if method == "ct" else "dplsq",
    )
    return select_one_change(path, params["beta"], relative_beta=True)


def _training_groups(simulation, heldout: tuple[int, ...]):
    groups = []
    for replicate, trajectory in enumerate(simulation.observations):
        for index in range(len(simulation.times) - 1):
            if index in heldout:
                continue
            groups.append(
                LongitudinalGroup(
                    f"r{replicate}_t{index}",
                    simulation.times[index : index + 2],
                    trajectory[index : index + 2],
                    np.ones((2, trajectory.shape[1])),
                    np.ones(2),
                )
            )
    return groups


def _blocked_system(seed: int):
    _, _, _, _, simulation = make_dataset(GOLD, seed, True)
    rows = []
    for method in METHODS:
        for fold, heldout in enumerate(BLOCKS):
            result = _fit(method, _training_groups(simulation, heldout), simulation.times)
            errors = []
            for trajectory in simulation.observations:
                for index in heldout:
                    regime = 0 if result.change_index is None or index < result.change_index else 1
                    fit = result.fits[regime]
                    dt = float(simulation.times[index + 1] - simulation.times[index])
                    if method == "ct":
                        prediction = exact_step(fit.matrix, fit.baseline, trajectory[index], dt)
                    else:
                        prediction = trajectory[index] + dt * (fit.matrix @ trajectory[index] + fit.baseline)
                    errors.extend((prediction - trajectory[index + 1]) ** 2)
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "fold": fold,
                    "heldout_transitions": ",".join(map(str, heldout)),
                    "estimated_change_time": result.change_time,
                    "rmse": float(np.sqrt(np.mean(errors))),
                }
            )
    return rows


def paired_interval(values: np.ndarray, rng: np.random.Generator, n_boot: int = 20_000):
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[indices].mean(axis=1)
    return float(values.mean()), *map(float, np.quantile(samples, [0.025, 0.975]))


def blocked_cv():
    seeds = list(range(98001, 98016))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [row for system_rows in pool.map(_blocked_system, seeds) for row in system_rows]
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / "blocked_cv_15systems_folds.tsv", sep="\t", index=False)
    systems = folds.groupby(["seed", "method"], as_index=False).rmse.mean()
    systems.to_csv(OUT / "blocked_cv_15systems_system_means.tsv", sep="\t", index=False)

    wide = systems.pivot(index="seed", columns="method", values="rmse")
    rng = np.random.default_rng(20260930)
    comparisons = []
    for comparator in ("euler", "dplsq"):
        log_ratio = np.log(wide.ct / wide[comparator]).to_numpy()
        mean_log, low_log, high_log = paired_interval(log_ratio, rng)
        difference = (wide.ct - wide[comparator]).to_numpy()
        mean_diff, low_diff, high_diff = paired_interval(difference, rng)
        comparisons.append(
            {
                "comparison": f"ct_vs_{comparator}",
                "n_systems": len(wide),
                "mean_ct_minus_comparator_rmse": mean_diff,
                "difference_ci_low": low_diff,
                "difference_ci_high": high_diff,
                "mean_log_rmse_ratio": mean_log,
                "log_ratio_ci_low": low_log,
                "log_ratio_ci_high": high_log,
                "geometric_rmse_ratio": float(np.exp(mean_log)),
                "ratio_ci_low": float(np.exp(low_log)),
                "ratio_ci_high": float(np.exp(high_log)),
            }
        )
    pd.DataFrame(comparisons).to_csv(OUT / "blocked_cv_15systems_paired.tsv", sep="\t", index=False)
    summary = systems.groupby("method").rmse.agg(["mean", "std", "median"]).reset_index()
    summary.to_csv(OUT / "blocked_cv_15systems_summary.tsv", sep="\t", index=False)


def _stability_system(seed: int):
    _, _, _, true_index, simulation = make_dataset(GOLD, seed, True)
    rng = np.random.default_rng(seed + 20260930)
    draws = [tuple(sorted(rng.integers(0, 5, size=5).tolist())) for _ in range(200)]
    unique_draws = list(dict.fromkeys(draws))
    fitted = {}
    for draw in unique_draws:
        observations = simulation.observations[np.asarray(draw)]
        boot = type(simulation)(
            simulation.times,
            observations,
            simulation.clean_states[np.asarray(draw)],
            simulation.networks,
            simulation.baselines,
            simulation.change_points,
        )
        fitted[draw] = _fit("ct", trajectory_groups(boot), boot.times).change_index
    true_time = float(simulation.times[true_index])
    records = []
    for repetition, draw in enumerate(draws):
        index = fitted[draw]
        time = np.nan if index is None else float(simulation.times[index])
        records.append(
            {
                "seed": seed,
                "repetition": repetition,
                "resample": ",".join(map(str, draw)),
                "change_index": index,
                "change_time": time,
                "true_index": true_index,
                "true_time": true_time,
            }
        )
    return records


def stability():
    seeds = list(range(96001, 96016))
    with ProcessPoolExecutor(max_workers=4) as pool:
        records = [row for system_rows in pool.map(_stability_system, seeds) for row in system_rows]
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "bootstrap_15systems_200.tsv", sep="\t", index=False)
    frame["selected"] = frame.change_index.notna()
    frame["exact"] = frame.change_index == frame.true_index
    frame["within_one_index"] = (frame.change_index - frame.true_index).abs() <= 1
    summary = frame.groupby("seed").agg(
        selection_probability=("selected", "mean"),
        exact_probability=("exact", "mean"),
        within_one_probability=("within_one_index", "mean"),
        unique_resamples=("resample", "nunique"),
    ).reset_index()
    summary.to_csv(OUT / "bootstrap_15systems_summary.tsv", sep="\t", index=False)


def calibration_sensitivity():
    grid = pd.read_csv(results_root() / "weighted_score_calibration/rho_grid.tsv", sep="\t")
    grid = grid[grid.method.eq("ct")].copy()
    rows = []
    weights = {
        "cp": (0.5, 1.0, 1.5),
        "change": (0.175, 0.35, 0.525),
        "network": (0.10, 0.20, 0.30),
        "specificity": (0.25, 0.50, 0.75),
    }
    for cp_w in weights["cp"]:
        for change_w in weights["change"]:
            for network_w in weights["network"]:
                for specificity_w in weights["specificity"]:
                    score = (
                        cp_w * grid.mean_cp_f1
                        + change_w * grid.mean_change_aupr
                        + network_w * grid.mean_network_aupr
                        + specificity_w * grid.null_specificity
                    )
                    candidates = grid.assign(perturbed_score=score)
                    best = candidates.sort_values(
                        ["perturbed_score", "mean_rmse", "beta_ratio"],
                        ascending=[False, True, True], kind="stable",
                    ).iloc[0]
                    rows.append(
                        {
                            "cp_weight": cp_w,
                            "change_weight": change_w,
                            "network_weight": network_w,
                            "specificity_weight": specificity_w,
                            "alpha": best.alpha,
                            "beta_ratio": best.beta_ratio,
                            "max_iter": int(best.max_iter),
                            "score": float(score.max()),
                            "n_tied_at_1e_12": int(np.isclose(score, score.max(), atol=1e-12).sum()),
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "calibration_weight_sensitivity.tsv", sep="\t", index=False)
    counts = result.groupby(["alpha", "beta_ratio", "max_iter"]).size().reset_index(name="selected_count")
    counts.sort_values("selected_count", ascending=False).to_csv(
        OUT / "calibration_weight_selection_counts.tsv", sep="\t", index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocked", action="store_true")
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not any((args.blocked, args.stability, args.calibration)):
        args.blocked = args.stability = args.calibration = True
    if args.blocked:
        blocked_cv()
    if args.stability:
        stability()
    if args.calibration:
        calibration_sensitivity()
    (OUT / "run_manifest.json").write_text(
        json.dumps({"blocked": args.blocked, "stability": args.stability, "calibration": args.calibration}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
