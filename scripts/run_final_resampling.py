#!/usr/bin/env python3
"""Blocked-time validation and 200 replicate bootstraps for final CT settings."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.io import LongitudinalGroup
from ctcpgrn.simulation import exact_step
from run_optimized_benchmark import make_dataset, trajectory_groups

GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = Path("results/final_validation")
BOOTSTRAP_SEED = 96001
PARAMETERS = {
    "ct": {"alpha": 0.0005, "beta": 0.30},
    "dplsq": {"alpha": 0.001, "beta": 0.10},
    "euler": {"alpha": 0.0005, "beta": 0.20},
}


def fit_training_path(method, groups, times):
    p = PARAMETERS[method]
    return scan_one_change_candidates(
        groups, times, method, p["alpha"], minimum_transitions=5, max_iter=80,
        ct_initialiser="euler" if method == "ct" else "dplsq",
    )


def blocked_cv():
    rows = []
    blocks = [[1, 2], [5, 6], [9, 10]]
    for seed in range(98001, 98006):
        _, _, _, _, simulation = make_dataset(GOLD, seed, True)
        n_transitions = len(simulation.times) - 1
        for method in PARAMETERS:
            for fold, heldout in enumerate(blocks):
                training = []
                for replicate, trajectory in enumerate(simulation.observations):
                    for index in range(n_transitions):
                        if index in heldout:
                            continue
                        training.append(LongitudinalGroup(
                            f"r{replicate}_t{index}", simulation.times[index:index+2],
                            trajectory[index:index+2], np.ones((2, trajectory.shape[1])), np.ones(2),
                        ))
                path = fit_training_path(method, training, simulation.times)
                result = select_one_change(path, PARAMETERS[method]["beta"], relative_beta=True)
                errors = []
                for trajectory in simulation.observations:
                    for index in heldout:
                        regime = 0 if result.change_index is None or index < result.change_index else 1
                        fit = result.fits[regime]
                        dt = float(simulation.times[index+1] - simulation.times[index])
                        if method == "ct":
                            prediction = exact_step(fit.matrix, fit.baseline, trajectory[index], dt)
                        else:
                            prediction = trajectory[index] + dt * (fit.matrix @ trajectory[index] + fit.baseline)
                        errors.extend((prediction - trajectory[index+1]) ** 2)
                rows.append({
                    "seed": seed, "method": method, "fold": fold,
                    "heldout_transitions": ",".join(map(str, heldout)),
                    "estimated_change_time": result.change_time,
                    "rmse": float(np.sqrt(np.mean(errors))),
                })
                print("BLOCK", json.dumps(rows[-1]), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "blocked_cv.tsv", sep="\t", index=False)
    frame.groupby("method", as_index=False).rmse.agg(["mean", "std"]).to_csv(OUT / "blocked_cv_summary.tsv", sep="\t")


def bootstrap_worker(indices):
    _, _, _, _, simulation = make_dataset(GOLD, BOOTSTRAP_SEED, True)
    observations = simulation.observations[np.asarray(indices)]
    boot = type(simulation)(simulation.times, observations, simulation.clean_states[np.asarray(indices)], simulation.networks, simulation.baselines, simulation.change_points)
    path = scan_one_change_candidates(
        trajectory_groups(boot), boot.times, "ct", 0.0005,
        minimum_transitions=5, max_iter=80, ct_initialiser="euler",
    )
    result = select_one_change(path, 0.30, relative_beta=True)
    return result.change_index, result.change_time


def bootstrap():
    rng = np.random.default_rng(20260929)
    draws = [tuple(sorted(rng.integers(0, 5, size=5).tolist())) for _ in range(200)]
    unique = list(dict.fromkeys(draws))
    with ProcessPoolExecutor(max_workers=4) as pool:
        fitted = dict(zip(unique, pool.map(bootstrap_worker, unique)))
    rows = []
    for repetition, draw in enumerate(draws):
        index, change_time = fitted[draw]
        rows.append({"repetition": repetition, "resample": ",".join(map(str, draw)), "change_index": index, "change_time": change_time})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "bootstrap_200.tsv", sep="\t", index=False)
    true_time = make_dataset(GOLD, BOOTSTRAP_SEED, True)[-1].times[6]
    summary = {
        "n": 200, "seed": BOOTSTRAP_SEED, "unique_resamples": len(unique), "true_time": float(true_time),
        "exact_rate": float(np.mean(np.isclose(frame.change_time.fillna(np.inf), true_time))),
        "within_1_5_rate": float(np.mean(np.abs(frame.change_time.fillna(np.inf) - true_time) <= 1.5)),
        "no_change_rate": float(frame.change_time.isna().mean()),
        "counts": pd.to_numeric(frame.change_time, errors="coerce").round(6).fillna("none").value_counts().to_dict(),
    }
    (OUT / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("BOOT", json.dumps(summary), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    blocked_cv()
    bootstrap()
