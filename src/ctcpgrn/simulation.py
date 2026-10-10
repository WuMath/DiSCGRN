"""Non-stationary DREAM4 simulation with known signed networks and change points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass
class SimulatedDataset:
    times: np.ndarray
    observations: np.ndarray
    clean_states: np.ndarray
    networks: list[np.ndarray]
    baselines: list[np.ndarray]
    change_points: list[float]


def parameterise_signed_network(
    signed_adjacency: np.ndarray,
    rng: np.random.Generator,
    magnitude_range: tuple[float, float] = (0.08, 0.35),
    stability_margin: float = 0.25,
) -> np.ndarray:
    """Turn a signed topology into a stable continuous-time coefficient matrix."""

    signs = np.sign(signed_adjacency)
    magnitudes = rng.uniform(*magnitude_range, size=signs.shape)
    matrix = signs * magnitudes
    np.fill_diagonal(matrix, 0.0)
    spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(matrix))))
    np.fill_diagonal(matrix, -(spectral_abscissa + stability_margin))
    return matrix


def perturb_network(
    matrix: np.ndarray,
    rng: np.random.Generator,
    rewiring_fraction: float = 0.15,
    sign_flip_fraction: float = 0.05,
    weight_scale: float = 0.25,
) -> np.ndarray:
    """Create a new regime through deletions/additions, sign flips and weight shifts."""

    result = matrix.copy()
    p = matrix.shape[0]
    offdiag = np.argwhere(~np.eye(p, dtype=bool))
    edges = np.argwhere((np.abs(matrix) > 0) & (~np.eye(p, dtype=bool)))
    nonedges = np.argwhere((np.abs(matrix) == 0) & (~np.eye(p, dtype=bool)))
    n_rewire = max(1, int(round(rewiring_fraction * len(edges)))) if len(edges) else 0
    if n_rewire:
        deleted = edges[rng.choice(len(edges), size=min(n_rewire, len(edges)), replace=False)]
        for i, j in deleted:
            result[i, j] = 0.0
        if len(nonedges):
            added = nonedges[rng.choice(len(nonedges), size=min(n_rewire, len(nonedges)), replace=False)]
            reference = np.abs(matrix[~np.eye(p, dtype=bool)])
            reference = reference[reference > 0]
            low, high = (float(np.quantile(reference, 0.25)), float(np.quantile(reference, 0.75)))
            for i, j in added:
                result[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(low, high)
    remaining = np.argwhere((np.abs(result) > 0) & (~np.eye(p, dtype=bool)))
    n_flip = int(round(sign_flip_fraction * len(remaining)))
    if n_flip:
        for i, j in remaining[rng.choice(len(remaining), size=n_flip, replace=False)]:
            result[i, j] *= -1
    result[~np.eye(p, dtype=bool)] *= rng.lognormal(mean=0.0, sigma=weight_scale, size=len(offdiag))
    np.fill_diagonal(result, 0.0)
    spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(result))))
    np.fill_diagonal(result, -(spectral_abscissa + 0.25))
    return result


def exact_step(matrix: np.ndarray, baseline: np.ndarray, state: np.ndarray, delta_t: float) -> np.ndarray:
    p = len(state)
    augmented = np.zeros((p + 1, p + 1), dtype=float)
    augmented[:p, :p] = matrix
    augmented[:p, p] = baseline
    transition = expm(delta_t * augmented)
    return (transition @ np.r_[state, 1.0])[:p]


def simulate_piecewise_linear_ode(
    networks: list[np.ndarray],
    change_points: list[float],
    times: np.ndarray,
    n_replicates: int = 5,
    noise_std: float = 0.05,
    seed: int = 1,
) -> SimulatedDataset:
    """Simulate exact piecewise-linear ODE trajectories with measurement noise."""

    if len(networks) != len(change_points) + 1:
        raise ValueError("Number of networks must equal number of change points plus one")
    rng = np.random.default_rng(seed)
    p = networks[0].shape[0]
    baselines = [rng.uniform(0.05, 0.25, size=p) for _ in networks]
    clean = np.zeros((n_replicates, len(times), p), dtype=float)
    observed = np.zeros_like(clean)
    for replicate in range(n_replicates):
        state = rng.uniform(0.5, 1.5, size=p)
        clean[replicate, 0] = state
        observed[replicate, 0] = state + rng.normal(0, noise_std, size=p)
        for t in range(len(times) - 1):
            midpoint = (times[t] + times[t + 1]) / 2
            regime = int(np.searchsorted(change_points, midpoint, side="right"))
            state = exact_step(networks[regime], baselines[regime], state, times[t + 1] - times[t])
            clean[replicate, t + 1] = state
            observed[replicate, t + 1] = state + rng.normal(0, noise_std, size=p)
    return SimulatedDataset(
        times=np.asarray(times, dtype=float),
        observations=observed,
        clean_states=clean,
        networks=networks,
        baselines=baselines,
        change_points=list(change_points),
    )


def make_irregular_times(n_timepoints: int, total_time: float, irregularity: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    intervals = rng.lognormal(mean=0.0, sigma=irregularity, size=n_timepoints - 1)
    intervals *= total_time / intervals.sum()
    return np.r_[0.0, np.cumsum(intervals)]

