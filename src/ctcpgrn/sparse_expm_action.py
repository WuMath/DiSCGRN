"""Sparse matrix-exponential action for the unchanged global affine ODE.

This evaluates exp([[A,b],[0,0]] * dt) @ [x,1] without materializing the
dense exponential. A is still one coupled global generator; no TF is treated
as an exogenous input. The truncated Taylor action is scaled so that each
    substep has a conservative infinity-norm bound of 4.0. Numerical accuracy
must be checked against the dense action before scientific use.
"""

from __future__ import annotations

import math

import torch


def sparse_affine_expm_action(
    states: torch.Tensor,
    edge_rows: torch.Tensor,
    edge_cols: torch.Tensor,
    edge_values: torch.Tensor,
    baseline: torch.Tensor,
    delta_t: float,
    *,
    taylor_order: int = 32,
    substep_norm_limit: float = 4.0,
    max_substeps: int = 256,
) -> torch.Tensor:
    """Propagate a batch of row-oriented states through a sparse global ODE.

    The sparse COO indices must be unique. ``edge_values`` includes the
    diagonal coefficients. This routine is differentiable with respect to
    edge values and baseline through PyTorch's sparse COO @ dense operation.
    """
    if states.ndim != 2:
        raise ValueError("states must be a 2-D batch-by-gene tensor")
    p = states.shape[1]
    if baseline.shape != (p,) or edge_rows.shape != edge_cols.shape or edge_rows.shape != edge_values.shape:
        raise ValueError("Incompatible global generator dimensions")
    if delta_t < 0 or taylor_order < 1 or substep_norm_limit <= 0:
        raise ValueError("Invalid propagation interval or Taylor settings")
    if delta_t == 0:
        return states
    device = edge_values.device
    rows = torch.cat((edge_rows, torch.arange(p, device=device, dtype=torch.long)))
    cols = torch.cat((edge_cols, torch.full((p,), p, device=device, dtype=torch.long)))
    values = torch.cat((edge_values, baseline))
    with torch.no_grad():
        row_sum = torch.zeros(p + 1, dtype=values.dtype, device=device)
        row_sum.scatter_add_(0, rows, torch.abs(values.detach()))
        n_substeps = max(1, math.ceil(float(row_sum.max()) * delta_t / substep_norm_limit))
    if n_substeps > max_substeps:
        raise ValueError(f"Need {n_substeps} exponential substeps; limit is {max_substeps}")
    scaled = values * (delta_t / n_substeps)
    generator = torch.sparse_coo_tensor(
        torch.stack((rows, cols)), scaled, (p + 1, p + 1),
        dtype=values.dtype, device=device,
    ).coalesce()
    ones = torch.ones((states.shape[0], 1), dtype=states.dtype, device=states.device)
    result = torch.cat((states, ones), dim=1).T.contiguous()
    for _ in range(n_substeps):
        term = result
        propagated = result
        for degree in range(1, taylor_order + 1):
            term = torch.sparse.mm(generator, term) / degree
            propagated = propagated + term
        result = propagated
    return result[:p].T
