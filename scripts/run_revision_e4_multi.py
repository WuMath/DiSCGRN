#!/usr/bin/env python3
"""E4 multi-change experiment using segment-neighbourhood dynamic programming."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import _n_transitions, _segment_cost, fit_by_method, slice_groups
from ctcpgrn.io import read_dream_signed_network, signed_edge_list_to_matrix
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.paths import results_root
from ctcpgrn.simulation import (make_irregular_times, parameterise_signed_network,
                                perturb_network, simulate_piecewise_linear_ode)
from run_optimized_benchmark import trajectory_groups

GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = results_root() / "revision_e4_multi"
MINLEN = 5
_FROZEN = load_frozen_params()
PARAMS = {method: _FROZEN[method] for method in ("ct", "euler")}


def valid_partitions(T, k):
    if k == 0:
        return [()]
    return [c for c in combinations(range(MINLEN, T - MINLEN), k)
            if all(b - a >= MINLEN for a, b in zip((0,) + c, c + (T - 1,)))]


def make_system(seed, T, k):
    signs, _ = signed_edge_list_to_matrix(read_dream_signed_network(GOLD))
    rng = np.random.default_rng(seed)
    networks = [parameterise_signed_network(signs, rng, magnitude_range=(0.12, 0.45), stability_margin=0.30)]
    for _ in range(k):
        networks.append(perturb_network(networks[-1], rng, rewiring_fraction=0.35,
                                        sign_flip_fraction=0.10, weight_scale=0.35))
    candidates = valid_partitions(T, k)
    boundaries = candidates[int(rng.integers(len(candidates)))]
    times = make_irregular_times(T, float(T - 1), 0.50, seed + 100)
    simulation = simulate_piecewise_linear_ode(
        networks, [float(times[i]) for i in boundaries], times,
        n_replicates=5, noise_std=0.03, seed=seed + 200,
    )
    return boundaries, simulation


def infer(simulation, method, alpha, rho, max_iter):
    groups, times, T = trajectory_groups(simulation), simulation.times, len(simulation.times)
    costs = {}
    for start in range(0, T - MINLEN):
        for end in range(start + MINLEN, T):
            segment = slice_groups(groups, times[start], times[end])
            fit = fit_by_method(segment, method, alpha, None, max_iter,
                                ct_initialiser="euler" if method == "ct" else "dplsq")
            costs[(start, end)] = _segment_cost(fit, _n_transitions(segment))
    stationary = costs[(0, T - 1)]
    candidates = []
    for k in range(4):
        best_cost, best_partition = np.inf, None
        for partition in valid_partitions(T, k):
            endpoints = (0,) + partition + (T - 1,)
            raw = sum(costs[(a, b)] for a, b in zip(endpoints[:-1], endpoints[1:]))
            penalised = raw + k * rho * stationary
            if penalised < best_cost:
                best_cost, best_partition = penalised, partition
        candidates.append((best_cost, best_partition))
    return min(candidates, key=lambda x: x[0])[1]


def hausdorff(a, b, T):
    if not a and not b:
        return 0.0
    if not a or not b:
        return float(T - 1)
    a, b = np.asarray(a), np.asarray(b)
    return float(max(np.max(np.min(abs(a[:, None] - b[None, :]), axis=1)),
                     np.max(np.min(abs(b[:, None] - a[None, :]), axis=1))))


def _job(args):
    seed, T, true_k, method = args
    truth, simulation = make_system(seed, T, true_k)
    alpha, rho, max_iter = PARAMS[method]
    estimate = infer(simulation, method, alpha, rho, max_iter)
    return {
        "seed": seed, "T": T, "method": method, "true_k": true_k,
        "estimated_k": len(estimate), "count_correct": len(estimate) == true_k,
        "hausdorff_index": hausdorff(truth, estimate, T),
        "true_boundaries": ",".join(map(str, truth)),
        "estimated_boundaries": ",".join(map(str, estimate)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for T in (25, 41):
        for true_k in range(4):
            target = OUT / f"T{T}_K{true_k}.tsv"
            if target.exists():
                continue
            base = 180000 + 10000 * (T == 41) + 1000 * true_k
            jobs = [(base + j, T, true_k, method) for j in range(1, 16) for method in PARAMS]
            with ProcessPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(_job, jobs))
            pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
            print(f"completed {target}", flush=True)
    frame = pd.concat([pd.read_csv(p, sep="\t") for p in OUT.glob("T*_K*.tsv")], ignore_index=True)
    frame.to_csv(OUT / "e4_multi_all.tsv", sep="\t", index=False)
    summary = frame.groupby(["T", "true_k", "method"], as_index=False).agg(
        count_accuracy=("count_correct", "mean"),
        hausdorff_mean=("hausdorff_index", "mean"),
        hausdorff_median=("hausdorff_index", "median"),
    )
    summary.to_csv(OUT / "e4_multi_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
