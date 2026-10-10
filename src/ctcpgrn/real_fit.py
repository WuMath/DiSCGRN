"""Mask-aware fits for the independent real-data DiSCGRN application.

This module does not alter the frozen simulation pipeline. Its fitting loss
matches the paper's interval-specific affine-ODE transition loss, while the
training transitions are supplied explicitly to prevent held-out bridging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import expm


@dataclass(frozen=True)
class Transitions:
    start: np.ndarray
    end: np.ndarray
    dt: np.ndarray
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray

    def take(self, mask: np.ndarray) -> "Transitions":
        return Transitions(self.start[mask], self.end[mask], self.dt[mask],
                           self.x[mask], self.y[mask], self.weights[mask])

    @property
    def n(self) -> int:
        return int(len(self.dt))


@dataclass(frozen=True)
class Fit:
    matrix: np.ndarray
    baseline: np.ndarray
    data_loss: float
    objective: float
    iterations: int
    converged: bool


def from_archive(archive: np.lib.npyio.NpzFile, strain: str, split: str = "all") -> Transitions:
    times = np.asarray(archive[f"{strain}_times"], dtype=float)
    values = np.asarray(archive[f"{strain}_values"], dtype=float)
    weights = np.asarray(archive[f"{strain}_weights"], dtype=float)
    if split == "all":
        mask = np.ones(len(times) - 1, dtype=bool)
    elif split in {"train", "test"}:
        mask = np.asarray(archive[f"{strain}_{split}_transition_mask"], dtype=bool)
    else:
        raise ValueError(split)
    if len(mask) != len(times) - 1 or np.any(np.diff(times) <= 0):
        raise ValueError(f"Invalid time grid or transition mask for {strain}")
    return Transitions(times[:-1][mask], times[1:][mask], np.diff(times)[mask],
                       values[:-1][mask], values[1:][mask], weights[1:][mask])


def weighted_mse(prediction: np.ndarray, transitions: Transitions) -> float:
    return float(np.mean(transitions.weights * (prediction - transitions.y) ** 2))


def predict(fit: Fit, transitions: Transitions, method: str) -> np.ndarray:
    if method == "euler":
        return transitions.x + transitions.dt[:, None] * (
            transitions.x @ fit.matrix.T + fit.baseline
        )
    p = fit.matrix.shape[0]
    augmented = np.zeros((p + 1, p + 1), dtype=float)
    augmented[:p, :p] = fit.matrix
    augmented[:p, p] = fit.baseline
    state = np.concatenate([transitions.x, np.ones((transitions.n, 1))], axis=1)
    result = np.empty_like(transitions.y)
    for dt in np.unique(transitions.dt):
        keep = transitions.dt == dt
        result[keep] = (state[keep] @ expm(augmented * float(dt)).T)[:, :p]
    return result


def fit_euler(transitions: Transitions, alpha: float, allowed: np.ndarray) -> Fit:
    """Fit the same weighted mean-loss and sparsity penalty using Euler propagation."""
    if transitions.n < 2:
        raise ValueError("At least two training transitions are required")
    n, p = transitions.x.shape
    matrix = np.zeros((p, p), dtype=float)
    baseline = np.zeros(p, dtype=float)
    design = transitions.dt[:, None] * np.column_stack((transitions.x, np.ones(n)))
    response = transitions.y - transitions.x
    offdiag = ~np.eye(p, dtype=bool)
    penalised = allowed & offdiag
    # This is a conservative global Lipschitz bound for every target-specific
    # weighted regression, keeping the objective identical to fit_ct.
    lipschitz = 2.0 * np.max(transitions.weights) * np.linalg.norm(design, 2) ** 2 / (n * p)
    step = 1.0 / max(lipschitz, 1e-12)
    converged = False
    for iteration in range(1, 1201):
        residual = (design[:, :p] @ matrix.T + design[:, p, None] * baseline - response)
        weighted = transitions.weights * residual
        gradient = (2.0 / (n * p)) * (weighted.T @ design)
        candidate = matrix - step * gradient[:, :p]
        candidate[penalised] = np.sign(candidate[penalised]) * np.maximum(
            np.abs(candidate[penalised]) - step * alpha, 0.0)
        candidate[~allowed] = 0.0
        candidate[np.diag_indices(p)] = np.minimum(candidate.diagonal(), -1e-6)
        candidate_b = baseline - step * gradient[:, p]
        change = max(float(np.max(np.abs(candidate - matrix))),
                     float(np.max(np.abs(candidate_b - baseline))))
        matrix, baseline = candidate, candidate_b
        if change < 1e-7:
            converged = True
            break
    fitted = Fit(matrix, baseline, np.nan, np.nan, iteration, converged)
    data_loss = weighted_mse(predict(fitted, transitions, "euler"), transitions)
    penalty = alpha * float(np.abs(matrix[penalised]).sum())
    return Fit(matrix, baseline, data_loss, data_loss + penalty, iteration, converged)


def _torch_batches(transitions: Transitions) -> list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]]:
    result = []
    for dt in np.unique(transitions.dt):
        keep = transitions.dt == dt
        x = np.column_stack((transitions.x[keep], np.ones(int(keep.sum()))))
        result.append((float(dt), torch.as_tensor(x, dtype=torch.float64),
                       torch.as_tensor(transitions.y[keep], dtype=torch.float64),
                       torch.as_tensor(transitions.weights[keep], dtype=torch.float64)))
    return result


def _torch_loss(matrix: torch.Tensor, baseline: torch.Tensor,
                batches: list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]],
                n: int, p: int) -> torch.Tensor:
    top = torch.cat((matrix, baseline[:, None]), dim=1)
    augmented = torch.cat((top, torch.zeros((1, p + 1), dtype=matrix.dtype)), dim=0)
    total = torch.zeros((), dtype=matrix.dtype)
    for dt, x, y, weights in batches:
        prediction = (x @ torch.matrix_exp(augmented * dt).T)[:, :p]
        total = total + torch.sum(weights * (y - prediction) ** 2)
    return total / (n * p)


def fit_ct(transitions: Transitions, alpha: float, allowed: np.ndarray,
           max_iter: int = 40, learning_rate: float = 0.03,
           initial: Fit | None = None) -> Fit:
    """Proximal exact matrix-exponential fit with an Euler sparse warm start."""
    if initial is None:
        initial = fit_euler(transitions, alpha, allowed)
    p = transitions.x.shape[1]
    batches = _torch_batches(transitions)
    mask = torch.as_tensor(allowed, dtype=torch.bool)
    offdiag = ~torch.eye(p, dtype=torch.bool)
    penalty_mask = mask & offdiag
    matrix = torch.as_tensor(initial.matrix.copy(), dtype=torch.float64)
    baseline = torch.as_tensor(initial.baseline.copy(), dtype=torch.float64)
    matrix[~mask] = 0
    with torch.no_grad():
        diagonal = torch.diagonal(matrix)
        diagonal.copy_(torch.clamp(diagonal, max=-1e-6))
        previous = float("inf")
        for _ in range(12):
            initial_loss = _torch_loss(matrix, baseline, batches, transitions.n, p)
            initial_objective = initial_loss + alpha * torch.sum(torch.abs(matrix[penalty_mask]))
            if torch.isfinite(initial_objective):
                previous = float(initial_objective)
                break
            matrix *= 0.5
            baseline *= 0.5
            diagonal = torch.diagonal(matrix)
            diagonal.copy_(torch.clamp(diagonal, max=-1e-6))
        if not np.isfinite(previous):
            raise FloatingPointError("Non-finite continuous-time warm-start objective")
    step = learning_rate
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        matrix = matrix.detach().requires_grad_(True)
        baseline = baseline.detach().requires_grad_(True)
        loss = _torch_loss(matrix, baseline, batches, transitions.n, p)
        loss.backward()
        with torch.no_grad():
            accepted = False
            local_step = step
            for _ in range(18):
                candidate = matrix.detach() - local_step * matrix.grad
                candidate[penalty_mask] = torch.sign(candidate[penalty_mask]) * torch.clamp(
                    torch.abs(candidate[penalty_mask]) - local_step * alpha, min=0.0
                )
                candidate[~mask] = 0.0
                diagonal = torch.diagonal(candidate)
                diagonal.copy_(torch.clamp(diagonal, max=-1e-6))
                candidate_b = baseline.detach() - local_step * baseline.grad
                data_loss = _torch_loss(candidate, candidate_b, batches, transitions.n, p)
                objective = data_loss + alpha * torch.sum(torch.abs(candidate[penalty_mask]))
                if torch.isfinite(objective) and float(objective) <= previous + 1e-10:
                    accepted = True
                    break
                local_step *= 0.5
            if not accepted:
                break
            change = abs(previous - float(objective)) if np.isfinite(previous) else float("inf")
            matrix, baseline = candidate, candidate_b
            previous = float(objective)
            step = min(local_step * 1.05, learning_rate)
            if change < 1e-7:
                converged = True
                break
    with torch.no_grad():
        final_data = float(_torch_loss(matrix, baseline, batches, transitions.n, p))
        final_obj = final_data + alpha * float(torch.sum(torch.abs(matrix[penalty_mask])))
    return Fit(matrix.detach().cpu().numpy(), baseline.detach().cpu().numpy(), final_data,
               final_obj, iteration, converged)


def fit_method(transitions: Transitions, method: str, alpha: float,
               allowed: np.ndarray, max_iter: int) -> Fit:
    if method == "euler":
        return fit_euler(transitions, alpha, allowed)
    if method == "ct":
        return fit_ct(transitions, alpha, allowed, max_iter=max_iter)
    raise ValueError(method)
