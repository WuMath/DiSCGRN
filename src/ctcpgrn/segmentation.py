"""Exact dynamic programming over candidate continuous-time change points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import LongitudinalGroup
from .model import FitResult, fit_exact_segment


@dataclass
class SegmentationResult:
    change_point_indices: list[int]
    change_point_times: list[float]
    segments: list[tuple[int, int]]
    fits: list[dict[str, FitResult]]
    objective: float


def _slice_groups(groups: list[LongitudinalGroup], lower: float, upper: float) -> list[LongitudinalGroup]:
    sliced = []
    for group in groups:
        keep = (group.times >= lower) & (group.times <= upper)
        if np.sum(keep) < 2:
            continue
        sliced.append(
            LongitudinalGroup(
                name=group.name,
                times=group.times[keep],
                values=group.values[keep],
                weights=group.weights[keep],
                n_replicates=group.n_replicates[keep],
            )
        )
    return sliced


def segment_shared_change_points(
    groups: list[LongitudinalGroup],
    candidate_times: np.ndarray,
    lambda_l1: float,
    beta_change: float,
    minimum_transitions: int = 3,
    max_changes: int = 3,
    allowed_edges: np.ndarray | None = None,
    fitter_kwargs: dict | None = None,
) -> SegmentationResult:
    """Find the globally optimal shared partition conditional on segment costs.

    Candidate boundaries are indices on the union time grid. A segment (a,b)
    covers [candidate_times[a], candidate_times[b]], hence b-a transitions.
    The expensive ODE fits are cached once and the outer partition is solved
    exactly for every segment count up to ``max_changes + 1``.
    """

    fitter_kwargs = fitter_kwargs or {}
    times = np.asarray(candidate_times, dtype=float)
    n_transitions = len(times) - 1
    costs: dict[tuple[int, int], float] = {}
    fits: dict[tuple[int, int], dict[str, FitResult]] = {}
    for start in range(n_transitions):
        for end in range(start + minimum_transitions, n_transitions + 1):
            interval_groups = _slice_groups(groups, times[start], times[end])
            # Every biological group must contribute at least one transition;
            # otherwise a boundary could be favoured merely because a strain
            # has no observations in that interval.
            if len(interval_groups) != len(groups):
                continue
            group_fits = {
                group.name: fit_exact_segment(
                    [group],
                    lambda_l1=lambda_l1,
                    allowed_edges=allowed_edges,
                    **fitter_kwargs,
                )
                for group in interval_groups
            }
            # Each fit.objective is a mean over the group's observed
            # transitions. Weight it by that group's transition count so
            # intervals are compared on the same cumulative scale.
            costs[(start, end)] = sum(
                (len(group.times) - 1) * group_fits[group.name].objective
                for group in interval_groups
            )
            fits[(start, end)] = group_fits

    max_segments = max_changes + 1
    dp = np.full((max_segments + 1, n_transitions + 1), np.inf)
    previous = np.full((max_segments + 1, n_transitions + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for n_segments in range(1, max_segments + 1):
        for end in range(1, n_transitions + 1):
            for start in range(0, end):
                if (start, end) not in costs or not np.isfinite(dp[n_segments - 1, start]):
                    continue
                candidate = dp[n_segments - 1, start] + costs[(start, end)]
                if candidate < dp[n_segments, end]:
                    dp[n_segments, end] = candidate
                    previous[n_segments, end] = start

    penalised = np.asarray([dp[k, n_transitions] + beta_change * (k - 1) for k in range(1, max_segments + 1)])
    best_segments = int(np.nanargmin(penalised)) + 1
    if not np.isfinite(penalised[best_segments - 1]):
        raise RuntimeError("No feasible segmentation; reduce minimum_transitions")
    intervals: list[tuple[int, int]] = []
    end = n_transitions
    for k in range(best_segments, 0, -1):
        start = int(previous[k, end])
        intervals.append((start, end))
        end = start
    intervals.reverse()
    change_indices = [start for start, _ in intervals[1:]]
    return SegmentationResult(
        change_point_indices=change_indices,
        change_point_times=[float(times[index]) for index in change_indices],
        segments=intervals,
        fits=[fits[interval] for interval in intervals],
        objective=float(penalised[best_segments - 1]),
    )
