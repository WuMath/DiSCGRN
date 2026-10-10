#!/usr/bin/env python3
"""Inspect numerical scale and KKT residuals of saved global ODE checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from ctcpgrn.sparse_expm_action import sparse_affine_expm_action
from run_global_three_stage_sparse import STAGES, load_stage_transitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--penalty", type=float, required=True)
    parser.add_argument("--expression", type=Path,
                        default=Path("data/raw/gse74243/GSE74243_matrix_RMAlog2expressionvalues.txt.gz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    genes, stages = load_stage_transitions(args.candidate_dir, args.expression)
    gene_index = {gene: i for i, gene in enumerate(genes)}
    candidates = pd.read_csv(args.candidate_dir / "candidate_tf_target_pairs.tsv.gz", sep="\t")
    diagnostics = []
    for stage, _, _ in STAGES:
        table = stages[stage]
        part = candidates[candidates.stage.eq(stage)]
        rows = part.target.map(gene_index).to_numpy(dtype=np.int64)
        cols = part.regulator.map(gene_index).to_numpy(dtype=np.int64)
        p = len(genes)
        n_edges = len(rows)
        generator = sparse.load_npz(args.fit_dir / f"{stage}_generator.npz").tocsr()
        baseline = np.load(args.fit_dir / f"{stage}_baseline.npy")
        initial_edges = np.r_[np.asarray(generator[rows, cols]).ravel(),
                              generator.diagonal()]
        edge = torch.as_tensor(initial_edges, dtype=torch.float64).requires_grad_(True)
        intercept = torch.as_tensor(baseline, dtype=torch.float64).requires_grad_(True)
        edge_rows = torch.as_tensor(np.r_[rows, np.arange(p)], dtype=torch.long)
        edge_cols = torch.as_tensor(np.r_[cols, np.arange(p)], dtype=torch.long)
        loss = torch.zeros((), dtype=torch.float64)
        for interval in np.unique(table.dt):
            keep = table.dt == interval
            x = torch.as_tensor(table.x[keep], dtype=torch.float64)
            y = torch.as_tensor(table.y[keep], dtype=torch.float64)
            weights = torch.as_tensor(table.weights[keep], dtype=torch.float64)
            prediction = sparse_affine_expm_action(
                x, edge_rows, edge_cols, edge, intercept, float(interval))
            loss = loss + torch.sum(weights * (prediction - y) ** 2)
        (loss / len(table.dt)).backward()
        g = edge.grad.detach().numpy()
        gb = intercept.grad.detach().numpy()
        e = initial_edges[:n_edges]
        selected = np.abs(e) > 1e-8
        offdiag_residual = np.where(selected,
                                    np.abs(g[:n_edges] + args.penalty * np.sign(e)),
                                    np.maximum(np.abs(g[:n_edges]) - args.penalty, 0))
        diagonal = initial_edges[n_edges:]
        diagonal_residual = np.where(diagonal < -1e-6 - 1e-8,
                                     np.abs(g[n_edges:]),
                                     np.maximum(g[n_edges:], 0))
        row_abs_sum = np.asarray(abs(generator).sum(axis=1)).ravel() + abs(baseline)
        max_interval = float(np.max(table.dt))
        diagnostics.append({
            "stage": stage, "n_transitions": len(table.dt),
            "n_state_genes": p, "n_candidate_edges": n_edges,
            "n_selected_edges": int(selected.sum()),
            "dt_min": float(np.min(table.dt)),
            "dt_median": float(np.median(table.dt)),
            "dt_max": max_interval,
            "max_augmented_row_abs_sum": float(row_abs_sum.max()),
            "max_interval_times_row_sum": float(max_interval * row_abs_sum.max()),
            "kkt_offdiag_max": float(offdiag_residual.max()),
            "kkt_offdiag_p99": float(np.quantile(offdiag_residual, .99)),
            "kkt_diagonal_max": float(diagonal_residual.max()),
            "kkt_baseline_max": float(np.abs(gb).max()),
            "smooth_gradient_offdiag_max": float(np.abs(g[:n_edges]).max()),
            "smooth_gradient_diagonal_max": float(np.abs(g[n_edges:]).max()),
            "smooth_gradient_baseline_max": float(np.abs(gb).max()),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(args.output, sep="\t", index=False)
    print(pd.DataFrame(diagnostics).to_string(index=False))


if __name__ == "__main__":
    main()
