#!/usr/bin/env python3
"""Small end-to-end test: simulate, segment, and score."""

from __future__ import annotations

import json

import numpy as np

from ctcpgrn.io import LongitudinalGroup
from ctcpgrn.metrics import change_point_metrics, edge_ranking_metrics
from ctcpgrn.segmentation import segment_shared_change_points
from ctcpgrn.simulation import parameterise_signed_network, perturb_network, simulate_piecewise_linear_ode


def main() -> None:
    rng = np.random.default_rng(7)
    topology = np.zeros((4, 4))
    topology[1, 0], topology[2, 1], topology[3, 2], topology[0, 3] = 1, -1, 1, -1
    first = parameterise_signed_network(topology, rng)
    second = perturb_network(first, rng, rewiring_fraction=0.75, sign_flip_fraction=0.50, weight_scale=0.50)
    times = np.linspace(0, 8, 17)
    simulation = simulate_piecewise_linear_ode([first, second], [4.0], times, n_replicates=20, noise_std=0.005, seed=4)
    means = simulation.observations.mean(axis=0)
    variances = simulation.observations.var(axis=0, ddof=1)
    group = LongitudinalGroup(
        name="synthetic",
        times=times,
        values=means,
        weights=1.0 / np.maximum(variances, 1e-4),
        n_replicates=np.full(len(times), 20),
    )
    result = segment_shared_change_points(
        [group],
        candidate_times=times,
        lambda_l1=0.002,
        beta_change=0.0001,
        minimum_transitions=2,
        max_changes=1,
        fitter_kwargs={"max_iter": 80, "learning_rate": 0.02, "tolerance": 1e-6},
    )
    metrics = change_point_metrics([4.0], result.change_point_times, tolerance=1.0)
    metrics.update(edge_ranking_metrics(first, result.fits[0]["synthetic"].matrix))
    print(json.dumps({"change_points": result.change_point_times, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
