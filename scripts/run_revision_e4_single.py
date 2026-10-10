#!/usr/bin/env python3
"""E4 single-change power study with longer sequences and weaker changes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.io import read_dream_signed_network, signed_edge_list_to_matrix
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.paths import results_root
from ctcpgrn.simulation import (make_irregular_times, parameterise_signed_network,
                                perturb_network, simulate_piecewise_linear_ode)
from run_optimized_benchmark import score_result, trajectory_groups

GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = results_root() / "revision_e4_single"
TS = (13, 25, 41)
REWIRE = (0.05, 0.15, 0.35)
WEIGHT = (0.10, 0.20, 0.35)
_FROZEN = load_frozen_params()
PARAMS = {method: _FROZEN[method] for method in ("ct", "euler")}


def _dataset(seed, changed, n_times, rewire, weight):
    signs, _ = signed_edge_list_to_matrix(read_dream_signed_network(GOLD))
    rng = np.random.default_rng(seed)
    first = parameterise_signed_network(signs, rng, magnitude_range=(0.12, 0.45), stability_margin=0.30)
    times = make_irregular_times(n_times, float(n_times - 1), 0.50, seed + 100)
    if changed:
        second = perturb_network(first, rng, rewiring_fraction=rewire,
                                 sign_flip_fraction=0.10, weight_scale=weight)
        allowed = np.arange(5, n_times - 5)
        index = int(rng.choice(allowed))
        networks, cps = [first, second], [float(times[index])]
    else:
        second, index, networks, cps = first.copy(), None, [first], []
    simulation = simulate_piecewise_linear_ode(networks, cps, times, n_replicates=5,
                                                noise_std=0.03, seed=seed + 200)
    return first, second, index, simulation


def _job(args):
    seed, changed, n_times, rewire, weight = args
    first, second, index, simulation = _dataset(seed, changed, n_times, rewire, weight)
    rows = []
    for method, (alpha, beta, max_iter) in PARAMS.items():
        path = scan_one_change_candidates(
            trajectory_groups(simulation), simulation.times, method, alpha,
            minimum_transitions=5, max_iter=max_iter,
            ct_initialiser="euler" if method == "ct" else "dplsq",
        )
        result = select_one_change(path, beta, relative_beta=True)
        error = np.nan if index is None or result.change_index is None else abs(result.change_index - index)
        true_time = None if index is None else float(simulation.times[index])
        rows.append({
            "seed": seed, "changed": changed, "T": n_times,
            "rewiring_fraction": rewire, "weight_scale": weight, "method": method,
            "true_index": index, "estimated_index": result.change_index,
            "selected_change": result.change_index is not None,
            "exact": bool(error == 0) if changed else result.change_index is None,
            "within_one": bool(error <= 1) if changed else result.change_index is None,
            "index_error": error,
            **score_result(result, true_time, first, second),
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # One stationary cell per T; strength parameters are irrelevant without change.
    for ti, n_times in enumerate(TS):
        target = OUT / f"null_T{n_times}.tsv"
        if not target.exists():
            jobs = [(130000 + 10000 * ti + j, False, n_times, 0.35, 0.35) for j in range(1, 16)]
            with ProcessPoolExecutor(max_workers=4) as pool:
                rows = [r for batch in pool.map(_job, jobs) for r in batch]
            pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
            print(f"completed {target}", flush=True)
        for ri, rewire in enumerate(REWIRE):
            for wi, weight in enumerate(WEIGHT):
                target = OUT / f"changed_T{n_times}_r{rewire:.2f}_w{weight:.2f}.tsv"
                if target.exists():
                    continue
                base = 140000 + 10000 * ti + 1000 * ri + 100 * wi
                jobs = [(base + j, True, n_times, rewire, weight) for j in range(1, 16)]
                with ProcessPoolExecutor(max_workers=4) as pool:
                    rows = [r for batch in pool.map(_job, jobs) for r in batch]
                pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
                print(f"completed {target}", flush=True)
    files = list(OUT.glob("null_*.tsv")) + list(OUT.glob("changed_*.tsv"))
    frame = pd.concat([pd.read_csv(p, sep="\t") for p in files], ignore_index=True)
    frame.to_csv(OUT / "e4_single_all.tsv", sep="\t", index=False)
    changed = frame[frame.changed]
    summary = changed.groupby(["T", "rewiring_fraction", "weight_scale", "method"], as_index=False).agg(
        detection_rate=("selected_change", "mean"), exact_rate=("exact", "mean"),
        within_one_rate=("within_one", "mean"), median_index_error=("index_error", "median"),
        network_aupr=("network_aupr", "mean"), change_aupr=("change_aupr", "mean"),
    )
    null = frame[~frame.changed].groupby(["T", "method"], as_index=False).agg(
        specificity=("exact", "mean")
    )
    summary.to_csv(OUT / "e4_power_summary.tsv", sep="\t", index=False)
    null.to_csv(OUT / "e4_null_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
