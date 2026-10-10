#!/usr/bin/env python3
"""Exploratory fixed-window, pairwise, and jointly regularized GRN analysis.

The three fixed windows partition all transitions in the full GSE74243 panel.
They are not estimated change points. Outputs are kept separate from the
original zero/one-change and blocked-prediction experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ctcpgrn.real_fit import Fit, Transitions, _torch_batches, _torch_loss, fit_ct, from_archive, predict


STRAINS = ("AJ", "B6", "C3H")
STAGES = (("prenatal", 11.5, 20.0), ("early_postnatal", 20.0, 31.0),
          ("late_postnatal", 31.0, 76.0))
FOCUSED_REGULATORS = ("Hif3a", "Nr4a1", "Nkx2-1", "Meox2", "Nr1d1", "Snai2")
FOCUSED_TARGETS = ("Ager", "Sftpb", "Sftpa1", "Scgb1a1", "Cyp2f2",
                   "Dpt", "Igfbp7", "Lyz2", "H2-Ab1")


def stage_table(table: Transitions, left: float, right: float) -> Transitions:
    return table.take((table.start >= left) & (table.end <= right))


def subset_table(table: Transitions, indices: np.ndarray) -> Transitions:
    return Transitions(table.start, table.end, table.dt, table.x[:, indices],
                       table.y[:, indices], table.weights[:, indices])


def edge_mask(allowed: np.ndarray) -> np.ndarray:
    mask = allowed.copy()
    np.fill_diagonal(mask, False)
    return mask


def joint_fit(tables: list[Transitions], initial: list[Fit], allowed: np.ndarray,
              alpha: float, gamma: float, max_iter: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit two or three CT generators together with a group-lasso edge penalty.

    Smooth loss is weighted by the number of transitions in each stage. The
    nonsmooth penalty is weighted stage-wise L1 plus gamma times the L2 norm
    of each edge's cross-stage coefficient vector. Its prox is sequential
    soft thresholding and group shrinkage, applied only to eligible off-diagonal
    entries. The group penalty encourages selection of a common candidate set,
    but does not force equal coefficients or nonzero weights in every stage.
    """
    n_stages = len(tables)
    if n_stages not in (2, 3) or len(initial) != n_stages:
        raise ValueError("joint_fit expects matching two or three stages")
    p = allowed.shape[0]
    batches = [_torch_batches(t) for t in tables]
    fractions = torch.as_tensor([t.n / sum(u.n for u in tables) for t in tables], dtype=torch.float64)
    mask = torch.as_tensor(allowed, dtype=torch.bool)
    penalized = torch.as_tensor(edge_mask(allowed), dtype=torch.bool)
    matrices = torch.as_tensor(np.stack([f.matrix for f in initial]), dtype=torch.float64)
    baselines = torch.as_tensor(np.stack([f.baseline for f in initial]), dtype=torch.float64)

    def smooth(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return sum(fractions[k] * _torch_loss(a[k], b[k], batches[k], tables[k].n, p)
                   for k in range(n_stages))

    def total(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        sparse = sum(fractions[k] * alpha * torch.abs(a[k][penalized]).sum()
                     for k in range(n_stages))
        grouped = gamma * torch.linalg.vector_norm(a[:, penalized], dim=0).sum()
        return smooth(a, b) + sparse + grouped

    with torch.no_grad():
        previous = float(total(matrices, baselines))
    step = 0.03
    converged = False
    accepted_steps = 0
    for _ in range(max_iter):
        matrices = matrices.detach().requires_grad_(True)
        baselines = baselines.detach().requires_grad_(True)
        smooth(matrices, baselines).backward()
        with torch.no_grad():
            accepted = False
            trial = step
            for _ in range(20):
                a = matrices.detach() - trial * matrices.grad
                b = baselines.detach() - trial * baselines.grad
                for k in range(n_stages):
                    values = a[k][penalized]
                    a[k][penalized] = torch.sign(values) * torch.clamp(
                        torch.abs(values) - trial * alpha * fractions[k], min=0.0)
                    a[k][~mask] = 0.0
                    a[k].diagonal().clamp_(max=-1e-6)
                group = a[:, penalized]
                norms = torch.linalg.vector_norm(group, dim=0)
                group = group * torch.clamp(1.0 - trial * gamma / norms.clamp_min(1e-15), min=0.0)
                a[:, penalized] = group
                score = float(total(a, b))
                if np.isfinite(score) and score <= previous + 1e-10:
                    accepted = True
                    break
                trial *= 0.5
            if not accepted:
                break
            improvement = abs(previous - score)
            matrices, baselines, previous = a, b, score
            accepted_steps += 1
            step = min(trial * 1.05, 0.03)
            if improvement < 1e-7:
                converged = True
                break
    return matrices.detach().numpy(), baselines.detach().numpy(), {
        "objective": previous, "iterations": accepted_steps, "converged": converged}


def fit_to_struct(a: np.ndarray, b: np.ndarray) -> Fit:
    return Fit(a, b, np.nan, np.nan, 0, False)


def selected_edges(matrices: np.ndarray, genes: list[str], allowed: np.ndarray,
                   threshold: float, strain: str, mode: str) -> list[dict]:
    rows = []
    for stage_index, (stage, _, _) in enumerate(STAGES):
        for target, regulator in zip(*np.where(edge_mask(allowed))):
            weight = float(matrices[stage_index, target, regulator])
            rows.append({"mode": mode, "strain": strain, "stage": stage,
                         "regulator": genes[regulator], "target": genes[target],
                         "weight": weight, "selected": abs(weight) > threshold})
    return rows


def pairwise_summary(a: np.ndarray, genes: list[str], allowed: np.ndarray,
                     threshold: float, strain: str, mode: str) -> list[dict]:
    rows = []
    mask = edge_mask(allowed)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        x, y = a[left][mask], a[right][mask]
        sx, sy = np.abs(x) > threshold, np.abs(y) > threshold
        union = int(np.count_nonzero(sx | sy))
        rows.append({"mode": mode, "strain": strain, "left_stage": STAGES[left][0],
                     "right_stage": STAGES[right][0], "left_edges": int(sx.sum()),
                     "right_edges": int(sy.sum()), "shared_edges": int((sx & sy).sum()),
                     "gained_edges": int((~sx & sy).sum()),
                     "lost_edges": int((sx & ~sy).sum()),
                     "jaccard": float((sx & sy).sum() / union) if union else np.nan,
                     "mean_abs_weight_change_on_union": float(
                         np.mean(np.abs(y[sx | sy] - x[sx | sy]))) if union else np.nan})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("data/processed/real_study_20261004/model/full"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/real_data_20261006/three_stage_analysis"))
    parser.add_argument("--alpha", type=float, default=0.003)
    parser.add_argument("--gamma-grid", type=float, nargs="+", default=[0.0, 0.001, 0.003, 0.01])
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--edge-threshold", type=float, default=1e-8)
    parser.add_argument("--focused-panel", action="store_true",
                        help="Use the six displayed regulators and nine marker targets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.input_dir / "gene_panel.tsv", sep="\t")
    full_genes = panel["Gene Symbol"].astype(str).tolist()
    if args.focused_panel:
        requested = set(FOCUSED_REGULATORS + FOCUSED_TARGETS)
        absent = requested.difference(full_genes)
        if absent:
            raise ValueError(f"Focused-panel genes absent: {sorted(absent)}")
        indices = np.asarray([i for i, gene in enumerate(full_genes) if gene in requested])
        panel = panel.iloc[indices].reset_index(drop=True)
    else:
        indices = np.arange(len(full_genes))
    genes = panel["Gene Symbol"].astype(str).tolist()
    allowed = np.zeros((len(genes), len(genes)), dtype=bool)
    allowed[:, panel["is_candidate_regulator"].astype(bool).to_numpy()] = True
    np.fill_diagonal(allowed, True)
    counts, edges, pairs, diagnostics = [], [], [], []
    pair_joint_edges, pair_joint_summary = [], []
    payload = {"genes": np.asarray(genes, dtype=str)}
    with np.load(args.input_dir / "trajectories.npz", allow_pickle=False) as archive:
        for strain in STRAINS:
            all_transitions = subset_table(from_archive(archive, strain, "all"), indices)
            tables = [stage_table(all_transitions, left, right) for _, left, right in STAGES]
            assert sum(t.n for t in tables) == all_transitions.n, "Stage boundaries omit transitions"
            for (stage, left, right), table in zip(STAGES, tables):
                counts.append({"strain": strain, "stage": stage, "start": left,
                               "end": right, "n_transitions": table.n})
                if table.n < 3:
                    raise ValueError(f"Too few transitions in {strain}/{stage}: {table.n}")
            print(f"[{strain}] independent stage fits: {[t.n for t in tables]}", flush=True)
            independent = [fit_ct(t, args.alpha, allowed, max_iter=args.max_iter) for t in tables]
            base_a = np.stack([f.matrix for f in independent])
            base_b = np.stack([f.baseline for f in independent])
            payload[f"{strain}_independent_A"] = base_a
            payload[f"{strain}_independent_b"] = base_b
            edges += selected_edges(base_a, genes, allowed, args.edge_threshold, strain, "independent")
            pairs += pairwise_summary(base_a, genes, allowed, args.edge_threshold, strain, "independent")
            for k, (fit, table) in enumerate(zip(independent, tables)):
                diagnostics.append({"mode": "independent", "strain": strain,
                                    "stage": STAGES[k][0], "gamma": 0.0,
                                    "n_transitions": table.n, "train_rmse": float(np.sqrt(
                                        np.mean((predict(fit, table, "ct") - table.y) ** 2))),
                                    "iterations": fit.iterations, "converged": fit.converged})
            for gamma in args.gamma_grid:
                if gamma < 0:
                    raise ValueError("gamma must be nonnegative")
                print(f"[{strain}] joint three-stage gamma={gamma}", flush=True)
                a, b, info = joint_fit(tables, independent, allowed, args.alpha,
                                       gamma, args.max_iter)
                mode = f"joint_gamma_{gamma:g}"
                payload[f"{strain}_{mode}_A"] = a
                payload[f"{strain}_{mode}_b"] = b
                edges += selected_edges(a, genes, allowed, args.edge_threshold, strain, mode)
                pairs += pairwise_summary(a, genes, allowed, args.edge_threshold, strain, mode)
                for k, table in enumerate(tables):
                    fit = fit_to_struct(a[k], b[k])
                    diagnostics.append({"mode": mode, "strain": strain,
                                        "stage": STAGES[k][0], "gamma": gamma,
                                        "n_transitions": table.n, "train_rmse": float(np.sqrt(
                                            np.mean((predict(fit, table, "ct") - table.y) ** 2))),
                                        **info})
            for left, right in ((0, 1), (0, 2), (1, 2)):
                left_name, right_name = STAGES[left][0], STAGES[right][0]
                for gamma in args.gamma_grid:
                    a, b, info = joint_fit([tables[left], tables[right]],
                                           [independent[left], independent[right]],
                                           allowed, args.alpha, gamma, args.max_iter)
                    present_left = np.abs(a[0][edge_mask(allowed)]) > args.edge_threshold
                    present_right = np.abs(a[1][edge_mask(allowed)]) > args.edge_threshold
                    union = int(np.count_nonzero(present_left | present_right))
                    pair_joint_summary.append({
                        "strain": strain, "left_stage": left_name, "right_stage": right_name,
                        "gamma": gamma, "left_edges": int(present_left.sum()),
                        "right_edges": int(present_right.sum()),
                        "shared_edges": int((present_left & present_right).sum()),
                        "gained_edges": int((~present_left & present_right).sum()),
                        "lost_edges": int((present_left & ~present_right).sum()),
                        "jaccard": float((present_left & present_right).sum() / union) if union else np.nan,
                        "objective": info["objective"], "iterations": info["iterations"],
                        "converged": info["converged"],
                    })
                    payload[f"{strain}_pair_{left_name}_{right_name}_gamma_{gamma:g}_A"] = a
                    payload[f"{strain}_pair_{left_name}_{right_name}_gamma_{gamma:g}_b"] = b
                    for local, stage_name in enumerate((left_name, right_name)):
                        for target, regulator in zip(*np.where(edge_mask(allowed))):
                            weight = float(a[local, target, regulator])
                            pair_joint_edges.append({
                                "strain": strain, "left_stage": left_name,
                                "right_stage": right_name, "gamma": gamma,
                                "stage": stage_name, "regulator": genes[regulator],
                                "target": genes[target], "weight": weight,
                                "selected": abs(weight) > args.edge_threshold,
                            })
    pd.DataFrame(counts).to_csv(args.output_dir / "stage_transition_counts.tsv", sep="\t", index=False)
    pd.DataFrame(edges).to_csv(args.output_dir / "stage_edges.tsv", sep="\t", index=False)
    pd.DataFrame(pairs).to_csv(args.output_dir / "pairwise_stage_comparison.tsv", sep="\t", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_dir / "fit_diagnostics.tsv", sep="\t", index=False)
    pd.DataFrame(pair_joint_edges).to_csv(args.output_dir / "pairwise_joint_edges.tsv", sep="\t", index=False)
    pd.DataFrame(pair_joint_summary).to_csv(args.output_dir / "pairwise_joint_summary.tsv", sep="\t", index=False)
    np.savez_compressed(args.output_dir / "stage_networks.npz", **payload)
    (args.output_dir / "analysis_config.json").write_text(json.dumps({
        "stages": STAGES, "alpha": args.alpha, "gamma_grid": args.gamma_grid,
        "max_iter": args.max_iter, "edge_threshold": args.edge_threshold,
        "focused_panel": args.focused_panel, "genes": genes,
        "interpretation": "fixed analysis windows; gamma sensitivity, not validated hyperparameter selection",
    }, indent=2) + "\n")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
