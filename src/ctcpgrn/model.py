"""Exact continuous-time segment fitting using differentiable matrix exponentials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .io import LongitudinalGroup


@dataclass
class FitResult:
    matrix: np.ndarray
    baseline: np.ndarray
    data_loss: float
    objective: float
    n_iterations: int
    converged: bool


def _transition(matrix: torch.Tensor, baseline: torch.Tensor, state: torch.Tensor, delta_t: float) -> torch.Tensor:
    p = state.numel()
    top = torch.cat([matrix, baseline[:, None]], dim=1)
    bottom = torch.zeros((1, p + 1), dtype=matrix.dtype, device=matrix.device)
    augmented = torch.cat([top, bottom], dim=0)
    state_augmented = torch.cat([state, torch.ones(1, dtype=matrix.dtype, device=matrix.device)])
    return (torch.matrix_exp(augmented * delta_t) @ state_augmented)[:p]


def _weighted_loss(matrix: torch.Tensor, baseline: torch.Tensor, groups: list[LongitudinalGroup]) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for group in groups:
        values = torch.as_tensor(group.values, dtype=matrix.dtype, device=matrix.device)
        weights = torch.as_tensor(group.weights, dtype=matrix.dtype, device=matrix.device)
        for idx in range(len(group.times) - 1):
            prediction = _transition(matrix, baseline, values[idx], float(group.times[idx + 1] - group.times[idx]))
            losses.append(torch.mean(weights[idx + 1] * (values[idx + 1] - prediction) ** 2))
    if not losses:
        raise ValueError("A segment needs at least one observed transition")
    return torch.stack(losses).mean()


def _soft_threshold(values: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.sign(values) * torch.clamp(torch.abs(values) - threshold, min=0.0)


def fit_exact_segment(
    groups: list[LongitudinalGroup],
    lambda_l1: float,
    allowed_edges: np.ndarray | None = None,
    penalise_diagonal: bool = False,
    learning_rate: float = 0.05,
    max_iter: int = 500,
    tolerance: float = 1e-7,
    initial_matrix: np.ndarray | None = None,
    initial_baseline: np.ndarray | None = None,
    enforce_negative_diagonal: bool = False,
    ridge: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> FitResult:
    """Fit one stationary segment by proximal gradient descent.

    The smooth term is differentiated exactly through ``torch.matrix_exp``.
    L1 sparsity is imposed by a proximal soft-thresholding update, so zeros are
    true optimisation outputs rather than a post-hoc display threshold. An
    optional off-diagonal L2 term gives an elastic-net objective.
    """

    p = groups[0].values.shape[1]
    matrix = torch.zeros((p, p), dtype=dtype) if initial_matrix is None else torch.tensor(initial_matrix, dtype=dtype)
    baseline = torch.zeros(p, dtype=dtype) if initial_baseline is None else torch.tensor(initial_baseline, dtype=dtype)
    allowed = torch.ones((p, p), dtype=torch.bool) if allowed_edges is None else torch.as_tensor(allowed_edges, dtype=torch.bool)
    offdiag = ~torch.eye(p, dtype=torch.bool)
    penalty_mask = allowed & (offdiag if not penalise_diagonal else torch.ones_like(offdiag))
    ridge_mask = allowed & offdiag
    matrix = matrix * allowed
    previous_objective: float | None = None
    step = learning_rate
    converged = False

    for iteration in range(1, max_iter + 1):
        matrix = matrix.detach().requires_grad_(True)
        baseline = baseline.detach().requires_grad_(True)
        smooth = (_weighted_loss(matrix, baseline, groups)
                  + 0.5 * ridge * torch.sum(matrix[ridge_mask] ** 2))
        smooth.backward()
        grad_matrix = matrix.grad.detach()
        grad_baseline = baseline.grad.detach()

        accepted = False
        local_step = step
        with torch.no_grad():
            for _ in range(20):
                candidate_matrix = matrix.detach() - local_step * grad_matrix
                candidate_matrix[penalty_mask] = _soft_threshold(
                    candidate_matrix[penalty_mask], local_step * lambda_l1
                )
                candidate_matrix[~allowed] = 0.0
                if enforce_negative_diagonal:
                    diagonal = torch.diagonal(candidate_matrix)
                    diagonal.copy_(torch.clamp(diagonal, max=-1e-6))
                candidate_baseline = baseline.detach() - local_step * grad_baseline
                candidate_loss = _weighted_loss(candidate_matrix, candidate_baseline, groups)
                candidate_objective = (
                    candidate_loss
                    + lambda_l1 * torch.sum(torch.abs(candidate_matrix[penalty_mask]))
                    + 0.5 * ridge * torch.sum(candidate_matrix[ridge_mask] ** 2)
                )
                if torch.isfinite(candidate_objective) and (
                    previous_objective is None or float(candidate_objective) <= previous_objective + 1e-12
                ):
                    accepted = True
                    break
                local_step *= 0.5
        if not accepted:
            break
        matrix, baseline = candidate_matrix, candidate_baseline
        objective_value = float(candidate_objective)
        relative_change = (
            float("inf")
            if previous_objective is None
            else abs(previous_objective - objective_value) / max(1.0, abs(previous_objective))
        )
        previous_objective = objective_value
        step = min(local_step * 1.05, learning_rate)
        if relative_change < tolerance:
            converged = True
            break

    with torch.no_grad():
        final_loss = float(_weighted_loss(matrix, baseline, groups))
        final_objective = (
            final_loss
            + lambda_l1 * float(torch.sum(torch.abs(matrix[penalty_mask])))
            + 0.5 * ridge * float(torch.sum(matrix[ridge_mask] ** 2))
        )
    return FitResult(
        matrix=matrix.detach().cpu().numpy(),
        baseline=baseline.detach().cpu().numpy(),
        data_loss=final_loss,
        objective=final_objective,
        n_iterations=iteration,
        converged=converged,
    )
