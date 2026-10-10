#!/usr/bin/env python3
"""Frozen-method robustness on synthetic topology variants and process noise.

The two extra signed topologies are generated from DREAM4 network 1. They are
not additional independently sourced DREAM4 networks. No parameters are tuned
on these systems. Process noise is added to the latent state after each exact
affine transition, making the deterministic fitted model misspecified.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.io import read_dream_signed_network, signed_edge_list_to_matrix
from ctcpgrn.paths import results_root
from ctcpgrn.simulation import (
    SimulatedDataset, exact_step, make_irregular_times,
    parameterise_signed_network, perturb_network, simulate_piecewise_linear_ode,
)
from run_optimized_benchmark import score_result, trajectory_groups


GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = results_root() / "robustness"
_FROZEN = load_frozen_params()
METHODS = {method: _FROZEN[method][:2] for method in ("ct", "euler")}
TOPOLOGIES = ("base", "degree_rewired", "random_same_density")
GENERATORS = ("affine", "process_noise")


def _topology_variants() -> dict[str, np.ndarray]:
    base, _ = signed_edge_list_to_matrix(read_dream_signed_network(GOLD))
    rng = np.random.default_rng(20261003)
    rewired = base.copy()
    accepted = 0
    for _ in range(10000):
        edges = np.argwhere(rewired != 0)
        (a, b), (c, d) = edges[rng.choice(len(edges), size=2, replace=False)]
        if len({a, b, c, d}) < 4 or rewired[a, d] != 0 or rewired[c, b] != 0:
            continue
        rewired[a, d], rewired[c, b] = rewired[a, b], rewired[c, d]
        rewired[a, b] = rewired[c, d] = 0
        accepted += 1
        if accepted == 12:
            break
    if accepted != 12:
        raise RuntimeError(f"Only {accepted} degree-preserving edge swaps")
    p = len(base)
    random = np.zeros_like(base)
    candidates = np.argwhere(~np.eye(p, dtype=bool))
    chosen = candidates[rng.choice(len(candidates), size=np.count_nonzero(base), replace=False)]
    signs = np.sign(base[base != 0]).copy()
    rng.shuffle(signs)
    random[chosen[:, 0], chosen[:, 1]] = signs
    return {"base": base, "degree_rewired": rewired,
            "random_same_density": random}


def _simulate_process_noise(networks, cps, times, seed):
    rng = np.random.default_rng(seed + 200)
    p = len(networks[0])
    baselines = [rng.uniform(0.05, 0.25, size=p) for _ in networks]
    clean = np.zeros((5, len(times), p))
    observed = np.zeros_like(clean)
    for replicate in range(5):
        state = rng.uniform(0.5, 1.5, size=p)
        clean[replicate, 0] = state
        observed[replicate, 0] = state + rng.normal(0, 0.03, size=p)
        for i in range(len(times) - 1):
            stage = int(np.searchsorted(cps, (times[i] + times[i + 1]) / 2, side="right"))
            state = exact_step(networks[stage], baselines[stage], state,
                               float(times[i + 1] - times[i]))
            state = state + rng.normal(0, 0.03, size=p)
            clean[replicate, i + 1] = state
            observed[replicate, i + 1] = state + rng.normal(0, 0.03, size=p)
    return SimulatedDataset(np.asarray(times), observed, clean, networks, baselines, cps)


def _job(item):
    topology, generator, seed, changed = item
    signs = _topology_variants()[topology]
    rng = np.random.default_rng(seed)
    first = parameterise_signed_network(signs, rng, magnitude_range=(0.12, 0.45),
                                        stability_margin=0.30)
    times = make_irregular_times(13, 12.0, 0.50, seed + 100)
    if changed:
        second = perturb_network(first, rng, rewiring_fraction=0.35,
                                 sign_flip_fraction=0.10, weight_scale=0.35)
        index = 6
        networks, cps = [first, second], [float(times[index])]
    else:
        second, index, networks, cps = first.copy(), None, [first], []
    if generator == "affine":
        simulation = simulate_piecewise_linear_ode(networks, cps, times,
                                                    n_replicates=5, noise_std=0.03,
                                                    seed=seed + 200)
    else:
        simulation = _simulate_process_noise(networks, cps, times, seed)
    rows = []
    for method, (alpha, rho) in METHODS.items():
        path = scan_one_change_candidates(
            trajectory_groups(simulation), times, method, alpha,
            minimum_transitions=5, max_iter=80,
            ct_initialiser="euler" if method == "ct" else "dplsq",
        )
        selected = select_one_change(path, rho, relative_beta=True)
        metrics = score_result(selected, None if index is None else float(times[index]),
                               first, second)
        rows.append({"topology": topology, "generator": generator,
                     "seed": seed, "changed": changed, "method": method,
                     "selected_index": selected.change_index,
                     "alpha": alpha, "rho": rho, "H": 80, **metrics})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    topologies = _topology_variants()
    for name, matrix in topologies.items():
        np.savetxt(OUT / f"topology_{name}.tsv", matrix, fmt="%d", delimiter="\t")
    with (OUT / "protocol.json").open("w") as f:
        json.dump({"base_topology": str(GOLD), "extra_topologies": "synthetic variants of network 1",
                   "degree_preserving_swaps": 12, "process_noise_sd_per_interval": 0.03,
                   "measurement_noise_sd": 0.03, "n_changed_per_condition": 15,
                   "n_stationary_per_condition": 15, "T": 13, "replicates": 5,
                   "frozen_methods": METHODS}, f, indent=2)
    jobs = []
    for ti, topology in enumerate(TOPOLOGIES):
        for gi, generator in enumerate(GENERATORS):
            base = 701000 + 1000 * ti + 100 * gi
            jobs += [(topology, generator, base + j, True) for j in range(1, 16)]
            jobs += [(topology, generator, base + 50 + j, False) for j in range(1, 16)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [row for block in pool.map(_job, jobs) for row in block]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "robustness_systems.tsv", sep="\t", index=False)
    summaries = []
    for (topology, generator, method), group in frame.groupby(["topology", "generator", "method"]):
        changed, null = group[group.changed], group[~group.changed]
        summaries.append({"topology": topology, "generator": generator,
                          "method": method, "n_changed": len(changed),
                          "n_null": len(null),
                          "network_aupr": changed.network_aupr.mean(),
                          "within_tolerance": changed.cp_f1.mean(),
                          "specificity": null.cp_f1.mean()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "robustness_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
