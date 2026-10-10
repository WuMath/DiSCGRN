"""Fast one-change benchmark implementations used in the case study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Lasso

from .io import LongitudinalGroup
from .model import FitResult, fit_exact_segment


@dataclass
class ScanResult:
    change_index: int | None
    change_time: float | None
    fits: list[FitResult]
    objective: float


def _n_transitions(groups: list[LongitudinalGroup]) -> int:
    return int(sum(max(len(group.times) - 1, 0) for group in groups))


def _segment_cost(fit: FitResult, n_transitions: int) -> float:
    """Sum the fitted mean-loss-plus-alpha-L1 objective over transitions.

    ``fit.objective`` is the segment mean prediction loss plus the same alpha
    times the off-diagonal L1 norm used during fitting. Its network penalty
    must therefore receive the same transition-count multiplier as the mean
    data loss. The separate relative change-point penalty is added later.
    """
    if n_transitions <= 0:
        raise ValueError("A candidate segment needs at least one transition")
    return float(n_transitions * fit.objective)


def slice_groups(groups: list[LongitudinalGroup], lower: float, upper: float) -> list[LongitudinalGroup]:
    result = []
    for group in groups:
        keep = (group.times >= lower) & (group.times <= upper)
        indices = np.flatnonzero(keep)
        if len(indices) < 2:
            continue
        # Retain only genuinely consecutive observations inside each supplied
        # group. Blocked-CV represents isolated training transitions as
        # separate two-point groups, preventing artificial gap bridging.
        result.append(LongitudinalGroup(group.name, group.times[indices], group.values[indices], group.weights[indices], group.n_replicates[indices]))
    return result


def fit_linear_segment(groups: list[LongitudinalGroup], alpha: float, mode: str) -> FitResult:
    predictors, responses, sample_weights = [], [], []
    for group in groups:
        for index in range(len(group.times) - 1):
            dt = float(group.times[index + 1] - group.times[index])
            if mode == "euler":
                predictors.append(np.r_[dt * group.values[index], dt])
                responses.append(group.values[index + 1] - group.values[index])
                sample_weights.append(group.weights[index + 1])
            elif mode == "dplsq":
                predictors.append(np.r_[group.values[index], 1.0])
                responses.append((group.values[index + 1] - group.values[index]) / dt)
                sample_weights.append(np.ones(group.values.shape[1]))
            else:
                raise ValueError(mode)
    x = np.asarray(predictors)
    y = np.asarray(responses)
    weights = np.asarray(sample_weights)
    p = y.shape[1]
    coefficients = np.zeros((p, p + 1))
    for target in range(p):
        model = Lasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=100_000,
            tol=1e-6,
            selection="cyclic",
        )
        model.fit(x, y[:, target], sample_weight=weights[:, target])
        coefficients[target] = model.coef_
    prediction = x @ coefficients.T
    data_loss = float(np.mean(weights * (y - prediction) ** 2))
    matrix, baseline = coefficients[:, :p], coefficients[:, p]
    penalty = alpha * float(np.sum(np.abs(matrix[~np.eye(p, dtype=bool)])))
    return FitResult(matrix, baseline, data_loss, data_loss + penalty, 1, True)


def fit_by_method(
    groups: list[LongitudinalGroup],
    method: str,
    alpha: float,
    allowed_edges: np.ndarray | None,
    max_iter: int,
    ct_warm_start: bool = True,
    ct_initialiser: str = "dplsq",
    ct_ridge: float = 0.0,
    ct_learning_rate: float = 0.03,
) -> FitResult:
    if method == "ct":
        initial_matrix = None
        initial_baseline = None
        if ct_warm_start:
            if ct_initialiser not in {"dplsq", "euler"}:
                raise ValueError(f"Unknown CT initialiser: {ct_initialiser}")
            initial = fit_linear_segment(groups, alpha, ct_initialiser)
            initial_matrix = initial.matrix.copy()
            initial_baseline = initial.baseline.copy()
            if allowed_edges is not None:
                initial_matrix[~allowed_edges] = 0.0
        return fit_exact_segment(
            groups,
            alpha,
            allowed_edges=allowed_edges,
            max_iter=max_iter,
            learning_rate=ct_learning_rate,
            initial_matrix=initial_matrix,
            initial_baseline=initial_baseline,
            enforce_negative_diagonal=True,
            ridge=ct_ridge,
        )
    if method in {"euler", "dplsq"}:
        fit = fit_linear_segment(groups, alpha, method)
        if allowed_edges is not None:
            fit.matrix[~allowed_edges] = 0.0
        return fit
    raise ValueError(method)


def scan_one_change(
    groups: list[LongitudinalGroup],
    candidate_times: np.ndarray,
    method: str,
    alpha: float,
    beta: float,
    minimum_transitions: int = 3,
    allowed_edges: np.ndarray | None = None,
    max_iter: int = 40,
    relative_beta: bool = False,
    ct_warm_start: bool = True,
    ct_initialiser: str = "dplsq",
    ct_ridge: float = 0.0,
    ct_learning_rate: float = 0.03,
) -> ScanResult:
    path = scan_one_change_candidates(
        groups=groups,
        candidate_times=candidate_times,
        method=method,
        alpha=alpha,
        minimum_transitions=minimum_transitions,
        allowed_edges=allowed_edges,
        max_iter=max_iter,
        ct_warm_start=ct_warm_start,
        ct_initialiser=ct_initialiser,
        ct_ridge=ct_ridge,
        ct_learning_rate=ct_learning_rate,
    )
    return select_one_change(path, beta=beta, relative_beta=relative_beta)


def scan_one_change_candidates(
    groups: list[LongitudinalGroup],
    candidate_times: np.ndarray,
    method: str,
    alpha: float,
    minimum_transitions: int = 3,
    allowed_edges: np.ndarray | None = None,
    max_iter: int = 40,
    ct_warm_start: bool = True,
    ct_initialiser: str = "dplsq",
    ct_ridge: float = 0.0,
    ct_learning_rate: float = 0.03,
) -> list[ScanResult]:
    """Fit the stationary model and every admissible one-change candidate once."""
    times = np.asarray(candidate_times)
    stationary = fit_by_method(
        groups, method, alpha, allowed_edges, max_iter,
        ct_warm_start=ct_warm_start, ct_initialiser=ct_initialiser, ct_ridge=ct_ridge,
        ct_learning_rate=ct_learning_rate,
    )
    stationary_result = ScanResult(
        None,
        None,
        [stationary],
        _segment_cost(stationary, _n_transitions(groups)),
    )
    candidates = [stationary_result]
    for boundary in range(minimum_transitions, len(times) - minimum_transitions):
        left_groups = slice_groups(groups, times[0], times[boundary])
        right_groups = slice_groups(groups, times[boundary], times[-1])
        if not left_groups or not right_groups:
            continue
        left = fit_by_method(
            left_groups, method, alpha, allowed_edges, max_iter,
            ct_warm_start=ct_warm_start, ct_initialiser=ct_initialiser, ct_ridge=ct_ridge,
            ct_learning_rate=ct_learning_rate,
        )
        right = fit_by_method(
            right_groups, method, alpha, allowed_edges, max_iter,
            ct_warm_start=ct_warm_start, ct_initialiser=ct_initialiser, ct_ridge=ct_ridge,
            ct_learning_rate=ct_learning_rate,
        )
        raw_objective = (
            _segment_cost(left, _n_transitions(left_groups))
            + _segment_cost(right, _n_transitions(right_groups))
        )
        candidates.append(ScanResult(boundary, float(times[boundary]), [left, right], raw_objective))
    return candidates


def select_one_change(path: list[ScanResult], beta: float, relative_beta: bool = False) -> ScanResult:
    """Select zero or one change from a pre-fitted candidate path."""
    if not path or path[0].change_index is not None:
        raise ValueError("Candidate path must start with the stationary fit")
    stationary = path[0]
    change_penalty = beta * stationary.objective if relative_beta else beta
    best = stationary
    best_penalised = stationary.objective
    for candidate in path[1:]:
        penalised = candidate.objective + change_penalty
        if penalised < best_penalised:
            best = candidate
            best_penalised = penalised
    return ScanResult(best.change_index, best.change_time, best.fits, float(best_penalised))


def two_stage_change(
    group: LongitudinalGroup,
    candidate_times: np.ndarray,
    alpha: float,
    beta: float,
    minimum_transitions: int,
    max_iter: int,
) -> ScanResult:
    y = group.values
    stationary_sse = float(np.sum((y - y.mean(axis=0)) ** 2))
    best_boundary, best_sse = None, stationary_sse
    for boundary in range(minimum_transitions, len(candidate_times) - minimum_transitions):
        sse = float(np.sum((y[: boundary + 1] - y[: boundary + 1].mean(axis=0)) ** 2))
        sse += float(np.sum((y[boundary:] - y[boundary:].mean(axis=0)) ** 2)) + beta
        if sse < best_sse:
            best_boundary, best_sse = boundary, sse
    if best_boundary is None:
        fit = fit_exact_segment([group], alpha, max_iter=max_iter, learning_rate=0.03)
        return ScanResult(None, None, [fit], best_sse)
    left = slice_groups([group], candidate_times[0], candidate_times[best_boundary])
    right = slice_groups([group], candidate_times[best_boundary], candidate_times[-1])
    fits = [fit_exact_segment(left, alpha, max_iter=max_iter, learning_rate=0.03), fit_exact_segment(right, alpha, max_iter=max_iter, learning_rate=0.03)]
    return ScanResult(best_boundary, float(candidate_times[best_boundary]), fits, best_sse)
