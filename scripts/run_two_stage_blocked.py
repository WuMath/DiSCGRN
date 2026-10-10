#!/usr/bin/env python3
"""Add frozen Two-stage to the existing 15-system blocked prediction test.

Only endpoints of retained training transitions enter the mean-segmentation
score. Original candidate boundaries, folds, seeds and frozen parameters are
preserved. Historical three-method output files are never overwritten.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ctcpgrn.baselines import fit_by_method, slice_groups
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.paths import results_root
from ctcpgrn.simulation import exact_step
from run_optimized_benchmark import make_dataset
from run_revision_e5 import BLOCKS, GOLD, _training_groups, paired_interval


OUT = results_root() / "revision_e5"
SEEDS = tuple(range(98001, 98016))


def fit_training_fold(simulation, heldout: tuple[int, ...]):
    """Apply the original Two-stage decision using train-visible timepoints."""
    alpha, rho, budget = load_frozen_params()["two_stage"]
    groups = _training_groups(simulation, heldout)
    visible = sorted({index for transition in range(len(simulation.times) - 1)
                      if transition not in heldout for index in (transition, transition + 1)})
    means = simulation.observations[:, visible, :].mean(axis=0)
    stationary_sse = float(np.sum((means - means.mean(axis=0)) ** 2))
    best_boundary, best_sse = None, float("inf")
    for boundary in range(5, len(simulation.times) - 5):
        left = means[np.asarray(visible) <= boundary]
        right = means[np.asarray(visible) >= boundary]
        sse = float(np.sum((left - left.mean(axis=0)) ** 2)
                    + np.sum((right - right.mean(axis=0)) ** 2))
        if sse < best_sse:
            best_boundary, best_sse = boundary, sse
    if best_sse + rho * stationary_sse < stationary_sse:
        boundary_time = float(simulation.times[best_boundary])
        left_groups = slice_groups(groups, simulation.times[0], boundary_time)
        right_groups = slice_groups(groups, boundary_time, simulation.times[-1])
        fits = [fit_by_method(part, "ct", alpha, None, budget)
                for part in (left_groups, right_groups)]
        return best_boundary, boundary_time, fits
    fit = fit_by_method(groups, "ct", alpha, None, budget)
    return None, float("nan"), [fit]


def run_system(seed: int):
    _, _, _, _, simulation = make_dataset(GOLD, seed, True)
    rows = []
    for fold, heldout in enumerate(BLOCKS):
        boundary, change_time, fits = fit_training_fold(simulation, heldout)
        errors = []
        for trajectory in simulation.observations:
            for index in heldout:
                regime = 0 if boundary is None or index < boundary else 1
                fit = fits[regime]
                dt = float(simulation.times[index + 1] - simulation.times[index])
                prediction = exact_step(fit.matrix, fit.baseline, trajectory[index], dt)
                errors.extend((prediction - trajectory[index + 1]) ** 2)
        rows.append({"seed": seed, "method": "two_stage", "fold": fold,
                     "heldout_transitions": ",".join(map(str, heldout)),
                     "estimated_change_time": change_time,
                     "rmse": float(np.sqrt(np.mean(errors)))})
    return rows


def main():
    source = OUT / "blocked_cv_15systems_system_means.tsv"
    old_systems = pd.read_csv(source, sep="\t")
    assert set(old_systems.method) == {"ct", "euler", "dplsq"}
    assert set(old_systems.seed) == set(SEEDS)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [row for system_rows in pool.map(run_system, SEEDS) for row in system_rows]
    folds = pd.DataFrame(rows)
    assert len(folds) == 45 and np.isfinite(folds.rmse).all()
    new_systems = folds.groupby(["seed", "method"], as_index=False).rmse.mean()
    systems = pd.concat([old_systems, new_systems], ignore_index=True)
    wide = systems.pivot(index="seed", columns="method", values="rmse")
    assert wide.shape == (15, 4) and np.isfinite(wide.to_numpy()).all()
    difference = (wide.ct - wide.two_stage).to_numpy()
    log_ratio = np.log(wide.ct / wide.two_stage).to_numpy()
    rng = np.random.default_rng(20260930)
    mean_log, low_log, high_log = paired_interval(log_ratio, rng)
    mean_diff, low_diff, high_diff = paired_interval(difference, rng)
    paired = pd.DataFrame([{
        "comparison": "ct_vs_two_stage", "n_systems": len(wide),
        "mean_ct_minus_comparator_rmse": mean_diff,
        "difference_ci_low": low_diff, "difference_ci_high": high_diff,
        "mean_log_rmse_ratio": mean_log, "log_ratio_ci_low": low_log,
        "log_ratio_ci_high": high_log,
        "geometric_rmse_ratio": float(np.exp(mean_log)),
        "ratio_ci_low": float(np.exp(low_log)), "ratio_ci_high": float(np.exp(high_log)),
    }])
    summary = systems.groupby("method").rmse.agg(["mean", "std", "median"]).reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT / "blocked_cv_two_stage_folds.tsv", sep="\t", index=False)
    systems.to_csv(OUT / "blocked_cv_four_methods_system_means.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "blocked_cv_four_methods_summary.tsv", sep="\t", index=False)
    paired.to_csv(OUT / "blocked_cv_two_stage_paired.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
