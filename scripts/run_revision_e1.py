#!/usr/bin/env python3
"""E1: interval-scale and irregularity scan with frozen parameters.

Each of 20 cells contains 15 changed and 15 stationary systems.  Results are
written cell-by-cell so the experiment can be resumed safely.
"""

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
from run_optimized_benchmark import one_step_rmse, score_result, trajectory_groups

GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = results_root() / "revision_e1"
TOTAL_TIMES = (3.0, 6.0, 12.0, 24.0, 48.0)
IRREGULARITIES = (0.0, 0.25, 0.50, 0.75)
_FROZEN = load_frozen_params()
PARAMS = {"ct": (_FROZEN["ct"][0], _FROZEN["ct"][1], 320),
          "euler": _FROZEN["euler"]}


def _dataset(seed, changed, total_time, irregularity):
    signs, _ = signed_edge_list_to_matrix(read_dream_signed_network(GOLD))
    rng = np.random.default_rng(seed)
    first = parameterise_signed_network(signs, rng, magnitude_range=(0.12, 0.45), stability_margin=0.30)
    times = make_irregular_times(13, total_time, irregularity, seed + 100)
    if changed:
        second = perturb_network(first, rng, rewiring_fraction=0.35,
                                 sign_flip_fraction=0.10, weight_scale=0.35)
        networks, cps, index = [first, second], [float(times[6])], 6
    else:
        second, networks, cps, index = first.copy(), [first], [], None
    simulation = simulate_piecewise_linear_ode(networks, cps, times, n_replicates=5,
                                                noise_std=0.03, seed=seed + 200)
    radii = [max(abs(np.linalg.eigvals(a))) for a in networks]
    kappa = float(np.mean(radii) * np.mean(np.diff(times)))
    return first, second, index, simulation, kappa


def _job(args):
    seed, changed, total_time, irregularity = args
    first, second, index, simulation, kappa = _dataset(seed, changed, total_time, irregularity)
    rows = []
    for method, (alpha, beta, max_iter) in PARAMS.items():
        path = scan_one_change_candidates(
            trajectory_groups(simulation), simulation.times, method, alpha,
            minimum_transitions=5, max_iter=max_iter,
            ct_initialiser="euler" if method == "ct" else "dplsq",
        )
        result = select_one_change(path, beta, relative_beta=True)
        true_time = None if index is None else float(simulation.times[index])
        rows.append({
            "seed": seed, "changed": changed, "total_time": total_time,
            "irregularity": irregularity, "kappa": kappa, "method": method,
            "rmse": one_step_rmse(result, simulation, method),
            **score_result(result, true_time, first, second),
        })
    return rows


def paired_ci(values, seed):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return values.mean(), *np.quantile(draws, (0.025, 0.975))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ti, total_time in enumerate(TOTAL_TIMES):
        for ii, irregularity in enumerate(IRREGULARITIES):
            target = OUT / f"cell_time{total_time:g}_irr{irregularity:.2f}.tsv"
            if target.exists():
                continue
            base = 110000 + 1000 * ti + 100 * ii
            jobs = [(base + j, changed, total_time, irregularity)
                    for changed in (True, False) for j in range(1, 16)]
            with ProcessPoolExecutor(max_workers=4) as pool:
                rows = [row for batch in pool.map(_job, jobs) for row in batch]
            pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
            print(f"completed {target}", flush=True)
    frame = pd.concat([pd.read_csv(p, sep="\t") for p in sorted(OUT.glob("cell_*.tsv"))], ignore_index=True)
    frame.to_csv(OUT / "e1_all_systems.tsv", sep="\t", index=False)
    summaries = []
    for (total_time, irregularity), g in frame.groupby(["total_time", "irregularity"]):
        changed = g[g.changed].pivot(index="seed", columns="method")
        delta = (changed.network_aupr.ct - changed.network_aupr.euler).to_numpy()
        mean, low, high = paired_ci(delta, int(total_time * 100 + irregularity * 10))
        null = g[~g.changed]
        summaries.append({
            "total_time": total_time, "irregularity": irregularity,
            "kappa_mean": g.kappa.mean(), "delta_network_aupr": mean,
            "ci_low": low, "ci_high": high,
            "ct_cp_detection": g[(g.changed) & (g.method == "ct")].cp_f1.mean(),
            "euler_cp_detection": g[(g.changed) & (g.method == "euler")].cp_f1.mean(),
            "ct_null_specificity": null[null.method == "ct"].cp_f1.mean(),
            "euler_null_specificity": null[null.method == "euler"].cp_f1.mean(),
        })
    pd.DataFrame(summaries).to_csv(OUT / "e1_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
