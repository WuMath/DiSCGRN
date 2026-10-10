#!/usr/bin/env python3
"""Run the preregistered non-stationary DREAM benchmark with checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ctcpgrn.io import LongitudinalGroup, read_dream_signed_network, signed_edge_list_to_matrix
from ctcpgrn.metrics import change_point_metrics, edge_change_metrics, edge_ranking_metrics
from ctcpgrn.segmentation import segment_shared_change_points
from ctcpgrn.simulation import (
    make_irregular_times,
    parameterise_signed_network,
    perturb_network,
    simulate_piecewise_linear_ode,
)


def canonical_gold_files(root: Path) -> list[Path]:
    files = sorted(root.glob("**/*goldstandard_signed.tsv"))
    return [path for path in files if "oscillating" not in str(path)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--dream-root", type=Path, default=Path("data/raw/dream4/DREAM4 in-silico challenge"))
    parser.add_argument("--output", type=Path, default=Path("results/simulation/results.jsonl"))
    parser.add_argument("--quick", action="store_true", help="Run one 10-gene instance for pipeline verification")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    files = canonical_gold_files(args.dream_root)
    if args.quick:
        files = [next(path for path in files if "size10_1/" in str(path))]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    settings = [(0.35, 0.05, 1, 0.2)] if args.quick else [
        (irr, noise, changes, rewire)
        for irr in config["irregularity_levels"]
        for noise in config["noise_levels"]
        for changes in config["change_counts"]
        for rewire in config["rewiring_fractions"]
    ]
    with args.output.open("a", encoding="utf-8") as handle:
        for file_index, gold_path in enumerate(files):
            signs, genes = signed_edge_list_to_matrix(read_dream_signed_network(gold_path))
            for setting_index, (irregularity, noise, n_changes, rewiring) in enumerate(settings):
                seed = int(config["seed"] + 10_000 * file_index + setting_index)
                rng = np.random.default_rng(seed)
                first = parameterise_signed_network(signs, rng)
                networks = [first]
                for _ in range(n_changes):
                    networks.append(perturb_network(networks[-1], rng, rewiring_fraction=rewiring))
                times = make_irregular_times(config["n_timepoints"], config["total_time"], irregularity, seed)
                cp_indices = np.linspace(0, len(times) - 1, n_changes + 2, dtype=int)[1:-1]
                true_cp = [float(times[index]) for index in cp_indices]
                simulated = simulate_piecewise_linear_ode(
                    networks,
                    true_cp,
                    times,
                    n_replicates=config["n_replicates"],
                    noise_std=noise,
                    seed=seed + 1,
                )
                means = simulated.observations.mean(axis=0)
                variances = simulated.observations.var(axis=0, ddof=1)
                group = LongitudinalGroup(
                    name="dream",
                    times=times,
                    values=means,
                    weights=1.0 / np.maximum(variances, 1e-4),
                    n_replicates=np.full(len(times), config["n_replicates"]),
                )
                lambda_l1 = config["lambda_grid"][2]
                beta = config["beta_grid"][2]
                result = segment_shared_change_points(
                    [group], times, lambda_l1, beta,
                    minimum_transitions=config["minimum_transitions"],
                    max_changes=config["max_changes"],
                    fitter_kwargs={"max_iter": 100 if args.quick else 500},
                )
                record = {
                    "network": str(gold_path), "n_genes": len(genes), "seed": seed,
                    "irregularity": irregularity, "noise": noise, "n_changes": n_changes,
                    "rewiring_fraction": rewiring, "true_change_points": true_cp,
                    "estimated_change_points": result.change_point_times,
                    **change_point_metrics(true_cp, result.change_point_times, tolerance=config["total_time"] / config["n_timepoints"]),
                }
                for regime, fit_map in enumerate(result.fits):
                    truth_index = min(regime, len(networks) - 1)
                    for key, value in edge_ranking_metrics(networks[truth_index], fit_map["dream"].matrix).items():
                        record[f"regime_{regime}_{key}"] = value
                if len(result.fits) >= 2 and len(networks) >= 2:
                    record.update(edge_change_metrics(networks[0], networks[1], result.fits[0]["dream"].matrix, result.fits[1]["dream"].matrix))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()

