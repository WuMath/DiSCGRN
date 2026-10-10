"""Global affine-ODE fitting with sparse exponential actions.

The statistical model is the same coupled x'(t) = A x(t) + b used by the
dense DiSCGRN implementation. Only the numerical representation of A and
the evaluation of exp(A*dt) @ x differ. Candidate edges are fixed before
fitting. An L1 penalty selects nonzero entries; an optional L2 term shrinks
the selected coefficients without changing the continuous-time dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, Lasso

from ctcpgrn.sparse_expm_action import sparse_affine_expm_action


@dataclass(frozen=True)
class GlobalTransitions:
    x: np.ndarray
    y: np.ndarray
    dt: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or self.x.shape != self.y.shape or self.x.shape != self.weights.shape:
            raise ValueError("x, y, and weights must be matching transition-by-gene arrays")
        if self.dt.shape != (len(self.x),) or not np.all(np.isfinite(self.dt)) or np.any(self.dt <= 0):
            raise ValueError("Transition intervals must be positive and finite")


@dataclass(frozen=True)
class SparseGlobalFit:
    edge_values: np.ndarray
    baseline: np.ndarray
    objective: float
    data_loss: float
    iterations: int
    converged: bool
    kkt_inf: float
    n_nonzero_offdiag: int
    trace: list[dict]


def euler_lasso_initialisation(
    transitions: GlobalTransitions,
    offdiag_rows: np.ndarray,
    offdiag_cols: np.ndarray,
    penalty: float,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Targetwise Euler elastic-net warm start for the same sparse edge support."""
    n, p = transitions.x.shape
    values = np.zeros(len(offdiag_rows) + p, dtype=float)
    values[len(offdiag_rows):] = -0.02
    baseline = np.zeros(p, dtype=float)
    for target in range(p):
        edge_positions = np.flatnonzero(offdiag_rows == target)
        parents = offdiag_cols[edge_positions]
        interval = transitions.dt
        response = transitions.y[:, target] - transitions.x[:, target]
        parent_design = interval[:, None] * transitions.x[:, parents]
        nuisance = np.column_stack((interval * transitions.x[:, target], interval))
        root_weight = np.sqrt(transitions.weights[:, target])
        weighted_response = root_weight * response
        weighted_parent = root_weight[:, None] * parent_design
        weighted_nuisance = root_weight[:, None] * nuisance
        nuisance_for_response = np.linalg.lstsq(weighted_nuisance, weighted_response, rcond=None)[0]
        nuisance_for_parents = np.linalg.lstsq(weighted_nuisance, weighted_parent, rcond=None)[0]
        residual_response = weighted_response - weighted_nuisance @ nuisance_for_response
        residual_parent = weighted_parent - weighted_nuisance @ nuisance_for_parents
        if ridge:
            solver = ElasticNet(alpha=(penalty + ridge) / 2,
                                l1_ratio=penalty / (penalty + ridge),
                                fit_intercept=False, max_iter=2000, tol=1e-5,
                                selection="cyclic")
        else:
            solver = Lasso(alpha=penalty / 2, fit_intercept=False,
                           max_iter=2000, tol=1e-5, selection="cyclic")
        solver.fit(residual_parent, residual_response)
        coefficients = solver.coef_
        values[edge_positions] = coefficients
        remainder = weighted_response - weighted_parent @ coefficients
        self_and_baseline = np.linalg.lstsq(weighted_nuisance, remainder, rcond=None)[0]
        self_weight = min(float(self_and_baseline[0]), -1e-6)
        values[len(offdiag_rows) + target] = self_weight
        if self_weight != self_and_baseline[0]:
            baseline[target] = float(np.dot(weighted_nuisance[:, 1],
                                            remainder - weighted_nuisance[:, 0] * self_weight)
                                     / np.dot(weighted_nuisance[:, 1], weighted_nuisance[:, 1]))
        else:
            baseline[target] = float(self_and_baseline[1])
    return values, baseline


def _batches(transitions: GlobalTransitions) -> list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]]:
    output = []
    for interval in np.unique(transitions.dt):
        keep = transitions.dt == interval
        output.append((float(interval),
                       torch.as_tensor(transitions.x[keep], dtype=torch.float64),
                       torch.as_tensor(transitions.y[keep], dtype=torch.float64),
                       torch.as_tensor(transitions.weights[keep], dtype=torch.float64)))
    return output


def fit_global_sparse_stage(
    transitions: GlobalTransitions,
    offdiag_rows: np.ndarray,
    offdiag_cols: np.ndarray,
    penalty: float,
    max_iter: int = 80,
    learning_rate: float = 0.5,
    taylor_order: int = 32,
    kkt_tolerance: float = 1e-3,
    objective_tolerance: float = 1e-7,
    initial: tuple[np.ndarray, np.ndarray] | None = None,
    optimizer: str = "proximal",
    parameter_scale: float = 1.0,
    active_rounds: int = 8,
    active_add_per_round: int = 200,
    progress_callback: Callable[[np.ndarray, np.ndarray, dict], None] | None = None,
    ridge: float = 0.0,
) -> SparseGlobalFit:
    """Proximal-gradient fit of one shared global network for a fixed stage.

    The smooth loss is mean over transitions of the weighted sum of squared
    errors across genes. This equals the existing mean-over-gene loss times
    the number of genes. Consequently, ``penalty`` is on the same multiplied
    scale and must not be numerically compared with the old alpha without
    accounting for that factor.
    """
    n, p = transitions.x.shape
    offdiag_rows = np.asarray(offdiag_rows, dtype=np.int64)
    offdiag_cols = np.asarray(offdiag_cols, dtype=np.int64)
    if len(offdiag_rows) != len(offdiag_cols) or np.any(offdiag_rows == offdiag_cols):
        raise ValueError("Only unique off-diagonal candidate indices may be passed")
    if len(set(zip(offdiag_rows.tolist(), offdiag_cols.tolist()))) != len(offdiag_rows):
        raise ValueError("Duplicate candidate edge")
    if penalty < 0 or ridge < 0 or not np.isfinite(ridge) or max_iter < 1:
        raise ValueError("Invalid penalty or iteration budget")
    edge_rows = torch.as_tensor(np.r_[offdiag_rows, np.arange(p)], dtype=torch.long)
    edge_cols = torch.as_tensor(np.r_[offdiag_cols, np.arange(p)], dtype=torch.long)
    if initial is None:
        init_values, init_baseline = euler_lasso_initialisation(
            transitions, offdiag_rows, offdiag_cols, penalty, ridge)
    else:
        init_values, init_baseline = initial
    values = torch.as_tensor(init_values.copy(), dtype=torch.float64)
    baseline = torch.as_tensor(init_baseline.copy(), dtype=torch.float64)
    n_offdiag = len(offdiag_rows)
    batches = _batches(transitions)

    def data_loss(edge: torch.Tensor, intercept: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64)
        for interval, x, y, weight in batches:
            predicted = sparse_affine_expm_action(
                x, edge_rows, edge_cols, edge, intercept, interval,
                taylor_order=taylor_order)
            total = total + torch.sum(weight * (predicted - y) ** 2)
        return total / n

    def objective(edge: torch.Tensor, intercept: torch.Tensor) -> torch.Tensor:
        offdiag = edge[:n_offdiag]
        return (data_loss(edge, intercept) + penalty * torch.sum(torch.abs(offdiag))
                + 0.5 * ridge * torch.sum(offdiag * offdiag))

    def kkt_residual(edge: torch.Tensor, intercept: torch.Tensor) -> float:
        current_edge = edge.detach().requires_grad_(True)
        current_intercept = intercept.detach().requires_grad_(True)
        data_loss(current_edge, current_intercept).backward()
        gradient = current_edge.grad.detach()
        offdiag = current_edge.detach()[:n_offdiag]
        offdiag_gradient = gradient[:n_offdiag]
        selected = torch.abs(offdiag) > 1e-8
        offdiag_gradient = offdiag_gradient + ridge * offdiag
        residual = torch.where(
            selected,
            torch.abs(offdiag_gradient + penalty * torch.sign(offdiag)),
            torch.clamp(torch.abs(offdiag_gradient) - penalty, min=0.0),
        )
        diagonal = current_edge.detach()[n_offdiag:]
        diagonal_gradient = gradient[n_offdiag:]
        diagonal_residual = torch.where(
            diagonal < -1e-6 - 1e-8,
            torch.abs(diagonal_gradient),
            torch.clamp(diagonal_gradient, min=0.0),
        )
        return max(float(residual.max()), float(diagonal_residual.max()),
                   float(torch.abs(current_intercept.grad).max()))

    if optimizer not in {"proximal", "lbfgsb", "active_lbfgsb"}:
        raise ValueError("optimizer must be 'proximal', 'lbfgsb', or 'active_lbfgsb'")
    if parameter_scale <= 0 or not np.isfinite(parameter_scale):
        raise ValueError("parameter_scale must be positive and finite")
    if optimizer == "proximal" and parameter_scale != 1.0:
        raise ValueError("parameter_scale is supported only with lbfgsb")
    if active_rounds < 1 or active_add_per_round < 1:
        raise ValueError("Active-set budgets must be positive")
    if optimizer == "active_lbfgsb":
        current = values.numpy().copy()
        current_baseline = baseline.numpy().copy()
        active = np.flatnonzero(np.abs(current[:n_offdiag]) > 1e-8)
        sign = np.sign(current[active])
        trace = []
        total_iterations = 0
        final_kkt = float("inf")
        converged = False

        def full_gradient(edge_array: np.ndarray, intercept_array: np.ndarray):
            e = torch.as_tensor(edge_array, dtype=torch.float64).requires_grad_(True)
            b = torch.as_tensor(intercept_array, dtype=torch.float64).requires_grad_(True)
            loss = data_loss(e, b)
            loss.backward()
            return float(loss.detach()), e.grad.detach().numpy().copy(), b.grad.detach().numpy().copy()

        for outer in range(1, active_rounds + 1):
            count = len(active)
            current_magnitude = np.maximum(sign * current[active] * parameter_scale, 0.0)
            initial_vector = np.r_[current_magnitude,
                                   np.maximum((-current[n_offdiag:] - 1e-6)
                                              * parameter_scale, 0.0),
                                   current_baseline * parameter_scale]
            active_tensor = torch.as_tensor(active, dtype=torch.long)
            sign_tensor = torch.as_tensor(sign, dtype=torch.float64)

            def active_objective(vector: np.ndarray):
                point = torch.as_tensor(vector, dtype=torch.float64).requires_grad_(True)
                magnitudes = point[:count]
                diagonal = -point[count:count + p] / parameter_scale - 1e-6
                offdiag = torch.zeros(n_offdiag, dtype=torch.float64)
                offdiag = offdiag.index_copy(
                    0, active_tensor, sign_tensor * magnitudes / parameter_scale)
                edge = torch.cat((offdiag, diagonal))
                intercept = point[count + p:] / parameter_scale
                objective_value = (data_loss(edge, intercept)
                                   + penalty * torch.sum(magnitudes) / parameter_scale
                                   + 0.5 * ridge * torch.sum(magnitudes ** 2)
                                   / parameter_scale ** 2)
                objective_value.backward()
                return float(objective_value.detach()), point.grad.detach().numpy().copy()

            result = minimize(active_objective, initial_vector, method="L-BFGS-B",
                              jac=True,
                              bounds=[(0.0, None)] * (count + p) + [(None, None)] * p,
                              options={"maxiter": max_iter, "ftol": objective_tolerance,
                                       "gtol": kkt_tolerance / parameter_scale,
                                       "maxls": 40, "maxcor": 20})
            total_iterations += int(result.nit)
            current[:] = 0.0
            current[active] = sign * result.x[:count] / parameter_scale
            current[n_offdiag:] = -result.x[count:count + p] / parameter_scale - 1e-6
            current_baseline = result.x[count + p:] / parameter_scale
            data_value, gradient, baseline_gradient = full_gradient(current, current_baseline)
            offdiag = current[:n_offdiag]
            selected = np.abs(offdiag) > 1e-8
            regularised_gradient = gradient[:n_offdiag] + ridge * offdiag
            residual = np.where(selected,
                                np.abs(regularised_gradient
                                       + penalty * np.sign(offdiag)),
                                np.maximum(np.abs(regularised_gradient) - penalty, 0.0))
            diagonal = current[n_offdiag:]
            diagonal_residual = np.where(
                diagonal < -1e-6 - 1e-8, np.abs(gradient[n_offdiag:]),
                np.maximum(gradient[n_offdiag:], 0.0))
            final_kkt = max(float(residual.max()), float(diagonal_residual.max()),
                            float(np.abs(baseline_gradient).max()))
            score = (data_value + penalty * np.abs(offdiag).sum()
                     + 0.5 * ridge * np.square(offdiag).sum())
            trace.append({"round": outer, "iteration": total_iterations,
                          "objective": score,
                          "nonzero_offdiag": int(selected.sum()),
                          "active_size": count, "kkt_inf": final_kkt,
                          "kkt_inactive_max": float(residual[~selected].max())
                          if np.any(~selected) else 0.0,
                          "optimizer_success": bool(result.success),
                          "optimizer_message": str(result.message)})
            if progress_callback is not None:
                progress_callback(current.copy(), current_baseline.copy(), trace[-1].copy())
            if final_kkt < kkt_tolerance:
                converged = True
                break
            # Drop coefficients that reached their orthant boundary, then
            # activate the strongest omitted KKT violations.
            active = np.flatnonzero(selected)
            sign = np.sign(current[active])
            inactive = np.flatnonzero(~selected & (residual > kkt_tolerance))
            if len(inactive):
                ranked = inactive[np.argsort(-residual[inactive], kind="stable")]
                additions = ranked[:active_add_per_round]
                active = np.r_[active, additions]
                sign = np.r_[sign, -np.sign(gradient[additions])]
            elif int(result.nit) == 0:
                break
        final_data = data_value
        final_offdiag = current[:n_offdiag]
        final_objective = (final_data + penalty * np.abs(final_offdiag).sum()
                           + 0.5 * ridge * np.square(final_offdiag).sum())
        return SparseGlobalFit(current, current_baseline, final_objective,
                               final_data, total_iterations, converged, final_kkt,
                               int(np.count_nonzero(np.abs(current[:n_offdiag]) > 1e-8)),
                               trace)
    if optimizer == "lbfgsb":
        # A = A_plus - A_minus, with both nonnegative. This is an exact
        # smooth reformulation of the L1 objective at its optimum.
        edge_init = values.numpy() * parameter_scale
        initial_vector = np.r_[np.maximum(edge_init[:n_offdiag], 0),
                               np.maximum(-edge_init[:n_offdiag], 0),
                               -edge_init[n_offdiag:] - 1e-6 * parameter_scale,
                               baseline.numpy() * parameter_scale]
        initial_vector[2 * n_offdiag:2 * n_offdiag + p] = np.maximum(
            initial_vector[2 * n_offdiag:2 * n_offdiag + p], 0)
        trace = []
        last = {"value": None}

        def evaluate(vector: np.ndarray) -> tuple[float, np.ndarray]:
            point = torch.as_tensor(vector, dtype=torch.float64).requires_grad_(True)
            plus = point[:n_offdiag]
            minus = point[n_offdiag:2 * n_offdiag]
            diagonal = (-point[2 * n_offdiag:2 * n_offdiag + p]
                        - 1e-6 * parameter_scale)
            intercept = point[2 * n_offdiag + p:] / parameter_scale
            edge = torch.cat((plus - minus, diagonal)) / parameter_scale
            loss = (data_loss(edge, intercept)
                    + (penalty / parameter_scale) * torch.sum(plus + minus)
                    + 0.5 * ridge * torch.sum((plus - minus) ** 2)
                    / parameter_scale ** 2)
            loss.backward()
            value = float(loss.detach())
            gradient = point.grad.detach().numpy().copy()
            last["value"] = value
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("Non-finite objective or gradient")
            return value, gradient

        def record(vector: np.ndarray) -> None:
            trace.append({"iteration": len(trace) + 1,
                          "objective": last["value"],
                          "nonzero_offdiag": int(np.count_nonzero(
                              np.abs(vector[:n_offdiag] -
                                     vector[n_offdiag:2 * n_offdiag]) > 1e-8))})

        bounds = ([(0.0, None)] * (2 * n_offdiag + p) +
                  [(None, None)] * p)
        result = minimize(evaluate, initial_vector, method="L-BFGS-B",
                          jac=True, bounds=bounds, callback=record,
                          options={"maxiter": max_iter, "ftol": objective_tolerance,
                                   "gtol": kkt_tolerance, "maxls": 40,
                                   "maxcor": 20})
        point = result.x
        final_edge = np.r_[point[:n_offdiag] - point[n_offdiag:2 * n_offdiag],
                           -point[2 * n_offdiag:2 * n_offdiag + p]
                           - 1e-6 * parameter_scale] / parameter_scale
        final_baseline = point[2 * n_offdiag + p:] / parameter_scale
        final_edge_tensor = torch.as_tensor(final_edge, dtype=torch.float64)
        final_baseline_tensor = torch.as_tensor(final_baseline, dtype=torch.float64)
        with torch.no_grad():
            final_data = float(data_loss(final_edge_tensor, final_baseline_tensor))
        final_kkt = kkt_residual(final_edge_tensor, final_baseline_tensor)
        final_offdiag = final_edge[:n_offdiag]
        final_objective = (final_data + penalty * np.abs(final_offdiag).sum()
                           + 0.5 * ridge * np.square(final_offdiag).sum())
        trace.append({"iteration": int(result.nit), "objective": final_objective,
                      "kkt_inf": final_kkt, "optimizer_success": bool(result.success),
                      "optimizer_message": str(result.message)})
        return SparseGlobalFit(final_edge, final_baseline, final_objective,
                               final_data, int(result.nit),
                               bool(result.success and final_kkt < kkt_tolerance),
                               final_kkt,
                               int(np.count_nonzero(np.abs(final_edge[:n_offdiag]) > 1e-8)),
                               trace)

    with torch.no_grad():
        previous = float(objective(values, baseline))
    if not np.isfinite(previous):
        raise FloatingPointError("Non-finite Euler warm-start objective")
    trace = [{"iteration": 0, "objective": previous,
              "nonzero_offdiag": int(torch.count_nonzero(values[:n_offdiag]))}]
    step = learning_rate
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        values = values.detach().requires_grad_(True)
        baseline = baseline.detach().requires_grad_(True)
        smooth = data_loss(values, baseline) + 0.5 * ridge * torch.sum(
            values[:n_offdiag] ** 2)
        smooth.backward()
        gradient = values.grad.detach()
        offdiag = values.detach()[:n_offdiag]
        selected = torch.abs(offdiag) > 1e-8
        offdiag_residual = torch.where(
            selected,
            torch.abs(gradient[:n_offdiag] + penalty * torch.sign(offdiag)),
            torch.clamp(torch.abs(gradient[:n_offdiag]) - penalty, min=0.0),
        )
        diagonal = values.detach()[n_offdiag:]
        diagonal_residual = torch.where(
            diagonal < -1e-6 - 1e-8,
            torch.abs(gradient[n_offdiag:]),
            torch.clamp(gradient[n_offdiag:], min=0.0),
        )
        current_kkt = max(float(offdiag_residual.max()),
                          float(diagonal_residual.max()),
                          float(torch.abs(baseline.grad).max()))
        accepted = False
        local_step = step
        with torch.no_grad():
            for _ in range(16):
                candidate = values.detach() - local_step * values.grad
                offdiag = candidate[:n_offdiag]
                candidate[:n_offdiag] = torch.sign(offdiag) * torch.clamp(
                    torch.abs(offdiag) - local_step * penalty, min=0.0)
                candidate[n_offdiag:] = torch.clamp(candidate[n_offdiag:], max=-1e-6)
                candidate_baseline = baseline.detach() - local_step * baseline.grad
                try:
                    score = float(objective(candidate, candidate_baseline))
                except ValueError:
                    score = float("inf")
                if np.isfinite(score) and score <= previous + 1e-9:
                    accepted = True
                    break
                local_step *= 0.5
            if not accepted:
                break
            relative_change = abs(previous - score) / max(abs(previous), 1.0)
            values, baseline, previous = candidate, candidate_baseline, score
            step = min(local_step * 1.05, learning_rate)
            trace.append({"iteration": iteration, "objective": previous,
                          "nonzero_offdiag": int(torch.count_nonzero(values[:n_offdiag])),
                          "previous_point_kkt_inf": current_kkt,
                          "relative_objective_change": relative_change})
            if relative_change < objective_tolerance and current_kkt < kkt_tolerance:
                converged = True
                break
    with torch.no_grad():
        final_data = float(data_loss(values, baseline))
    final_kkt = kkt_residual(values, baseline)
    converged = converged and final_kkt < kkt_tolerance
    final_values = values.detach().numpy()
    return SparseGlobalFit(final_values, baseline.detach().numpy(), previous,
                           final_data, iteration, converged, final_kkt,
                           int(np.count_nonzero(np.abs(final_values[:n_offdiag]) > 1e-8)), trace)
