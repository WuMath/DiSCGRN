#!/usr/bin/env python3
"""Freeze inputs for stage-wise GENIE3 and GRNBoost2 on independent systems."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.paths import results_root
from run_optimized_benchmark import make_dataset
from run_revision_comparison import GOLD


OUT = results_root() / "tree_grn_benchmark"
SEEDS = range(99601, 99616)


def mean_split(observations: np.ndarray, rho: float) -> int | None:
    """Match the frozen Two-stage mean segmentation without fitting its networks."""
    mean = observations.mean(axis=0)
    stationary = float(np.sum((mean - mean.mean(axis=0)) ** 2))
    best_boundary, best_cost = None, float("inf")
    for boundary in range(5, len(mean) - 5):
        left, right = mean[: boundary + 1], mean[boundary:]
        cost = float(np.sum((left - left.mean(axis=0)) ** 2)
                     + np.sum((right - right.mean(axis=0)) ** 2))
        if cost < best_cost:
            best_boundary, best_cost = boundary, cost
    return best_boundary if best_cost + rho * stationary < stationary else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    alpha, rho, budget = load_frozen_params()["two_stage"]
    manifest = []
    for seed in SEEDS:
        genes, first, second, true_index, simulation = make_dataset(GOLD, seed, True)
        estimated = mean_split(simulation.observations, rho)
        assert true_index is not None and len(genes) == 10
        np.savez_compressed(OUT / f"system_{seed}.npz", observations=simulation.observations,
                            first=first, second=second, times=simulation.times,
                            genes=np.asarray(genes), true_index=true_index,
                            mean_split_index=-1 if estimated is None else estimated)
        manifest.append({"seed": seed, "true_index": true_index,
                         "mean_split_index": estimated,
                         "n_replicates": simulation.observations.shape[0],
                         "n_timepoints": simulation.observations.shape[1]})
    pd.DataFrame(manifest).to_csv(OUT / "input_manifest.tsv", sep="\t", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "systems": list(SEEDS), "source": "DREAM4 signed topology 1 simulated trajectories",
        "segmentation": "frozen Two-stage mean SSE; boundary observation belongs to both stages",
        "two_stage_alpha": alpha, "two_stage_rho": rho, "two_stage_budget": budget,
        "regulators": "all 10 genes; no gold-standard regulator information",
        "input": "all five replicate observations pooled within each selected stage; timestamps not used by tree learners",
        "oracle_variant": "true boundary supplied only for a separately labeled network-inference diagnostic",
    }, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
