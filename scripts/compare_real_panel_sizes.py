#!/usr/bin/env python3
"""Exploratory, anchor-free comparison of nested GSE74243 gene panels.

The same three outer time blocks are used at every panel size. Genes and
normalization were selected within each outer training set by
prepare_real_data_inputs.py. Inner validation selects alpha and rho, whereas
the outer held-out blocks are used only for the reported prediction errors.
Panel-size selection from these outer results remains exploratory and needs
an independent final validation set before making confirmatory claims.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_real_data_analysis import (
    STRAINS, _bootstrap_data, fit_one, load_bundle, scan, select, tune,
)
from ctcpgrn.real_fit import Transitions, predict


def predict_strain(chosen, strain: str, table: Transitions) -> np.ndarray:
    if chosen.cp is None:
        return predict(chosen.fits[strain][0], table, "ct")
    left = table.start < chosen.cp
    result = np.empty_like(table.y)
    if left.any():
        result[left] = predict(chosen.fits[strain][0], table.take(left), "ct")
    if (~left).any():
        result[~left] = predict(chosen.fits[strain][1], table.take(~left), "ct")
    return result


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error)))) if error.size else float("nan")


def score_and_presence(chosen, n_genes: int) -> tuple[np.ndarray, np.ndarray]:
    matrices = [fit.matrix for strain in STRAINS for fit in chosen.fits[strain]]
    score = np.mean([np.abs(matrix) for matrix in matrices], axis=0)
    present = np.any([np.abs(matrix) > 1e-8 for matrix in matrices], axis=0)
    np.fill_diagonal(score, 0.0)
    np.fill_diagonal(present, False)
    assert score.shape == (n_genes, n_genes)
    return score, present


def evidence_table(bundle: Path, score: np.ndarray, output: Path,
                   permutations: int, seed: int) -> list[dict]:
    edge = pd.read_csv(bundle / "evaluation/edge_evidence.tsv", sep="\t")
    edge["score"] = score[edge.target_index.to_numpy(), edge.regulator_index.to_numpy()]
    edge["in_core50"] = edge.target_index.lt(50) & edge.regulator_index.lt(50)
    edge = edge.sort_values("score", ascending=False, kind="stable")
    edge.to_csv(output / "ranked_edges_with_evidence.tsv", sep="\t", index=False)
    rng = np.random.default_rng(seed)
    rows = []
    for scope, part in (("core50", edge[edge.in_core50]), ("all", edge)):
        parts = [group for _, group in part.groupby("regulator", sort=False)]
        for k in (20, 50):
            observed = int(part.head(k).collectri_reported.sum())
            null = np.empty(permutations, dtype=int)
            for repetition in range(permutations):
                randomized = pd.concat([
                    group.assign(score=rng.permutation(group.score.to_numpy()))
                    for group in parts
                ], ignore_index=True)
                null[repetition] = int(randomized.nlargest(k, "score").collectri_reported.sum())
            rows.append({"scope": scope, "top_k": k,
                         "reported_pairs_in_universe": int(part.collectri_reported.sum()),
                         "top_k_reported": observed,
                         "matched_null_mean": float(null.mean()),
                         "permutation_p_ge": float((1 + (null >= observed).sum()) / (permutations + 1))})
    pd.DataFrame(rows).to_csv(output / "collectri_support.tsv", sep="\t", index=False)
    return rows


def conditional_bootstrap(bundle: Path, chosen, genes: list[str], allowed: np.ndarray,
                          alpha: float, max_iter: int, repetitions: int,
                          seed: int, score: np.ndarray, full_present: np.ndarray,
                          output: Path) -> dict:
    eligible = allowed.copy()
    np.fill_diagonal(eligible, False)
    core = np.zeros_like(eligible)
    core[:50, :50] = eligible[:50, :50]
    top = sorted(zip(*np.where(core)), key=lambda ij: (-score[ij], ij))[:20]
    counts = np.zeros_like(eligible, dtype=int)
    rows = []
    for repetition in range(repetitions):
        data, bootstrap_allowed = _bootstrap_data(bundle / "model/full", seed + repetition)
        if not np.array_equal(bootstrap_allowed, allowed):
            raise ValueError("Bootstrap changed the candidate edge universe")
        fitted = {}
        for strain in STRAINS:
            table = data[strain]
            parts = ([table] if chosen.cp is None else
                     [table.take(table.start < chosen.cp), table.take(table.start >= chosen.cp)])
            fitted[strain] = [fit_one(part, "discgrn", alpha, allowed, max_iter) for part in parts]
        presence = np.any([np.abs(fit.matrix) > 1e-8
                           for strain in STRAINS for fit in fitted[strain]], axis=0) & eligible
        counts += presence
        intersection = int((presence & full_present & core).sum())
        union = int(((presence | full_present) & core).sum())
        rows.append({"bootstrap": repetition + 1,
                     "core50_jaccard": intersection / union if union else 1.0,
                     "n_core50_edges": int((presence & core).sum()),
                     "top20_core50_frequency": float(np.mean([presence[ij] for ij in top]))})
        print(f"[bootstrap] {bundle.name} {repetition + 1}/{repetitions}", flush=True)
    pd.DataFrame(rows).to_csv(output / "conditional_bootstrap_by_repeat.tsv", sep="\t", index=False)
    edge_rows = [{"regulator": genes[j], "target": genes[i], "score": float(score[i, j]),
                  "selection_frequency": float(counts[i, j] / repetitions),
                  "in_core50": bool(core[i, j])}
                 for i, j in zip(*np.where(eligible))]
    pd.DataFrame(edge_rows).sort_values("score", ascending=False).to_csv(
        output / "conditional_bootstrap_edges.tsv", sep="\t", index=False)
    return {"core50_jaccard_mean": float(np.mean([row["core50_jaccard"] for row in rows])),
            "top20_core50_frequency_mean": float(np.mean([row["top20_core50_frequency"] for row in rows]))}


def run_size(n_genes: int, input_root: Path, output_root: Path,
             alphas: list[float], rhos: list[float], max_iter: int,
             minimum_transitions: int, bootstrap: int,
             permutations: int, seed: int) -> dict:
    bundle = input_root / f"genes_{n_genes:03d}"
    output = output_root / f"genes_{n_genes:03d}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Existing results would be overwritten: {output}")
    output.mkdir(parents=True)
    panel = pd.read_csv(bundle / "model/full/gene_panel.tsv", sep="\t")
    if panel.pre_specified_anchor.any() or len(panel) != n_genes:
        raise ValueError("Panel must be unanchored and have the requested size")
    fold_rows, grid_rows = [], []
    start = time.perf_counter()
    for fold in sorted((bundle / "model/cv").glob("fold_*")):
        genes, train, allowed = load_bundle(fold, "train")
        _, heldout, _ = load_bundle(fold, "test")
        if len(genes) != n_genes:
            raise ValueError("Gene count mismatch")
        best, grid = tune(train, allowed, "discgrn", alphas, rhos,
                          max_iter, minimum_transitions)
        grid_rows.extend({"fold": fold.name, **entry} for entry in grid)
        path = scan(train, "discgrn", best["alpha"], allowed,
                    max_iter, minimum_transitions)
        chosen = select(path, best["rho"], "discgrn")
        for strain in STRAINS:
            table = heldout[strain]
            residual = predict_strain(chosen, strain, table) - table.y
            fold_rows.append({"fold": fold.name, "strain": strain,
                              "n_test_transitions": table.n,
                              "alpha": best["alpha"], "rho": best["rho"],
                              "change_time": chosen.cp,
                              "all_rmse": rmse(residual),
                              "core50_rmse": rmse(residual[:, :50]),
                              "added_rmse": rmse(residual[:, 50:])})
        print(f"[cv] {n_genes} {fold.name} alpha={best['alpha']} "
              f"rho={best['rho']} cp={chosen.cp}", flush=True)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "blocked_cv_by_fold_strain.tsv", sep="\t", index=False)
    pd.DataFrame(grid_rows).to_csv(output / "inner_calibration_grid.tsv", sep="\t", index=False)
    alpha = float(folds.alpha.median())
    rho = float(folds.rho.median())
    genes, data, allowed = load_bundle(bundle / "model/full", "all")
    path = scan(data, "discgrn", alpha, allowed, max_iter, minimum_transitions)
    chosen = select(path, rho, "discgrn")
    score, present = score_and_presence(chosen, n_genes)
    regulators = panel.loc[panel.is_candidate_regulator, "Gene Symbol"].astype(str).tolist()
    tf_rows = []
    for regulator in regulators:
        j = genes.index(regulator)
        tf_rows.append({"regulator": regulator,
                        "total_abs_weight": float(score[:, j].sum()),
                        "selected_targets": int(present[:, j].sum()),
                        "core50_abs_weight": float(score[:50, j].sum())})
    pd.DataFrame(tf_rows).sort_values("total_abs_weight", ascending=False).to_csv(
        output / "discovered_regulator_ranking.tsv", sep="\t", index=False)
    evidence = evidence_table(bundle, score, output, permutations, seed)
    stability = conditional_bootstrap(bundle, chosen, genes, allowed,
                                      alpha, max_iter, bootstrap, seed,
                                      score, present, output)
    result = {"n_genes": n_genes, "n_regulators": len(regulators),
              "n_fold_strain_cells": len(folds),
              "core50_rmse_mean": float(folds.core50_rmse.mean()),
              "core50_rmse_median": float(folds.core50_rmse.median()),
              "all_rmse_mean": float(folds.all_rmse.mean()),
              "added_rmse_mean": float(folds.added_rmse.mean()) if n_genes > 50 else None,
              "selected_alpha": alpha, "selected_rho": rho,
              "full_change_time": chosen.cp,
              "full_selected_core50_edges": int(present[:50, :50].sum()),
              "full_selected_all_edges": int(present.sum()),
              "collectri": evidence, "stability": stability,
              "runtime_seconds": time.perf_counter() - start}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path("data/processed/panel_size_20261007"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/panel_size_20261007"))
    parser.add_argument("--sizes", default="50,100,150")
    parser.add_argument("--alphas", default="0.001,0.003,0.01,0.03")
    parser.add_argument("--rhos", default="0.1,0.3,0.5")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--minimum-transitions", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_root.mkdir(parents=True, exist_ok=True)
    sizes = [int(value) for value in args.sizes.split(",")]
    alphas = [float(value) for value in args.alphas.split(",")]
    rhos = [float(value) for value in args.rhos.split(",")]
    summaries = [run_size(size, args.input_root, args.output_root,
                          alphas, rhos, args.max_iter,
                          args.minimum_transitions, args.bootstrap,
                          args.permutations, args.seed)
                 for size in sizes]
    pd.DataFrame([{key: value for key, value in row.items()
                   if key not in {"collectri", "stability"}}
                  | {"core50_jaccard_mean": row["stability"]["core50_jaccard_mean"],
                     "top20_core50_frequency_mean": row["stability"]["top20_core50_frequency_mean"]}
                  for row in summaries]).to_csv(args.output_root / "panel_comparison.tsv", sep="\t", index=False)
    (args.output_root / "run_configuration.json").write_text(json.dumps({
        "sizes": sizes, "alphas": alphas, "rhos": rhos,
        "max_iter": args.max_iter, "minimum_transitions": args.minimum_transitions,
        "bootstrap": args.bootstrap, "permutations": args.permutations,
        "seed": args.seed,
        "interpretation": "Exploratory panel-size comparison; outer folds cannot also serve as independent confirmation of the selected size."
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
