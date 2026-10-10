#!/usr/bin/env python3
"""Paired T=25/41 strong-change and null benchmark for an elastic-net CT fit."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.io import read_dream_signed_network, signed_edge_list_to_matrix
from ctcpgrn.simulation import (make_irregular_times, parameterise_signed_network,
                                perturb_network, simulate_piecewise_linear_ode)
from run_optimized_benchmark import score_result, trajectory_groups


GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")


def dataset(seed: int, changed: bool, n_times: int):
    signs, _ = signed_edge_list_to_matrix(read_dream_signed_network(GOLD))
    rng = np.random.default_rng(seed)
    first = parameterise_signed_network(signs, rng, magnitude_range=(0.12, 0.45),
                                        stability_margin=0.30)
    times = make_irregular_times(n_times, float(n_times - 1), 0.50, seed + 100)
    if changed:
        second = perturb_network(first, rng, rewiring_fraction=0.35,
                                 sign_flip_fraction=0.10, weight_scale=0.35)
        true_index = int(rng.choice(np.arange(5, n_times - 5)))
        networks, change_times = [first, second], [float(times[true_index])]
    else:
        second, true_index, networks, change_times = first.copy(), None, [first], []
    simulation = simulate_piecewise_linear_ode(
        networks, change_times, times, n_replicates=5, noise_std=0.03,
        seed=seed + 200)
    return first, second, true_index, simulation


def run_one(job):
    seed, changed, n_times, alpha, rho, budget, ridge = job
    first, second, true_index, simulation = dataset(seed, changed, n_times)
    path = scan_one_change_candidates(
        trajectory_groups(simulation), simulation.times, "ct", alpha,
        minimum_transitions=5, max_iter=budget, ct_initialiser="euler",
        ct_ridge=ridge)
    result = select_one_change(path, rho, relative_beta=True)
    truth_time = None if true_index is None else float(simulation.times[true_index])
    return {"seed": seed, "changed": changed, "T": n_times, "method": "ct",
            "ridge": ridge, "alpha": alpha, "rho": rho,
            "true_index": true_index, "estimated_index": result.change_index,
            "selected_change": result.change_index is not None,
            **score_result(result, truth_time, first, second)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-parameters", type=Path, required=True)
    parser.add_argument("--ridge", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.ridge < 0 or not np.isfinite(args.ridge):
        raise ValueError("Ridge must be finite and nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    alpha, rho, budget = load_frozen_params(args.frozen_parameters)["ct"]
    jobs = []
    for ti, n_times in ((1, 25), (2, 41)):
        jobs.extend((130000 + 10000 * ti + j, False, n_times,
                     alpha, rho, budget, args.ridge) for j in range(1, 16))
        base = 140000 + 10000 * ti + 1000 * 2 + 100 * 2
        jobs.extend((base + j, True, n_times,
                     alpha, rho, budget, args.ridge) for j in range(1, 16))
    args.output_dir.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, jobs))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "paired_60systems.tsv", sep="\t", index=False)
    summary = frame.groupby(["T", "changed"], as_index=False).agg(
        n_systems=("seed", "size"), network_aupr=("network_aupr", "mean"),
        change_aupr=("change_aupr", "mean"), cp_f1=("cp_f1", "mean"))
    summary.to_csv(args.output_dir / "paired_60systems_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
