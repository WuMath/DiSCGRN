#!/usr/bin/env python3
"""Independent real-data application of DiSCGRN and matched comparators.

The frozen simulation results are never read or rewritten. Every outer blocked
fold has an inner training-only validation block for method-specific tuning.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from ctcpgrn.real_fit import Fit, Transitions, fit_method, from_archive, predict


STRAINS = ("AJ", "B6", "C3H")
METHODS = ("discgrn", "stationary_ct", "euler_dp", "two_stage")


@dataclass
class Candidate:
    cp: float | None
    fits: dict[str, list[Fit]]
    cost: float
    raw_cost: float
    boundary_score: float | None = None


def load_bundle(path: Path, split: str) -> tuple[list[str], dict[str, Transitions], np.ndarray]:
    with np.load(path / "trajectories.npz", allow_pickle=False) as archive:
        genes = archive["genes"].astype(str).tolist()
        transitions = {strain: from_archive(archive, strain, split) for strain in STRAINS}
    panel = pd.read_csv(path / "gene_panel.tsv", sep="\t")
    if panel["Gene Symbol"].astype(str).tolist() != genes:
        raise ValueError("Panel and NPZ gene orders differ")
    allowed = np.zeros((len(genes), len(genes)), dtype=bool)
    allowed[:, panel["is_candidate_regulator"].astype(bool).to_numpy()] = True
    np.fill_diagonal(allowed, True)
    return genes, transitions, allowed


def test_mse(candidate: Candidate, transitions: dict[str, Transitions], method: str) -> tuple[float, dict[str, float]]:
    squared = []
    by_strain = {}
    formula = "euler" if method == "euler_dp" else "ct"
    for strain in STRAINS:
        table = transitions[strain]
        if not table.n:
            by_strain[strain] = float("nan")
            continue
        predictions = np.empty_like(table.y)
        if candidate.cp is None:
            predictions[:] = predict(candidate.fits[strain][0], table, formula)
        else:
            left = table.start < candidate.cp
            if left.any():
                predictions[left] = predict(candidate.fits[strain][0], table.take(left), formula)
            if (~left).any():
                predictions[~left] = predict(candidate.fits[strain][1], table.take(~left), formula)
        error = (predictions - table.y) ** 2
        squared.append(error.ravel())
        by_strain[strain] = float(np.sqrt(error.mean()))
    return float(np.concatenate(squared).mean()), by_strain


def segment_cost(fits: dict[str, list[Fit]], data: dict[str, Transitions],
                 cp: float | None, alpha: float) -> float:
    total = 0.0
    for strain in STRAINS:
        table = data[strain]
        parts = [table] if cp is None else [table.take(table.start < cp), table.take(table.start >= cp)]
        for part, fit in zip(parts, fits[strain]):
            # The per-transition mean objective used during fitting is
            # accumulated over each candidate interval for model selection.
            total += part.n * fit.objective
    return float(total)


def candidate_times(data: dict[str, Transitions], min_transitions: int) -> list[float]:
    all_times = np.unique(np.concatenate([np.r_[x.start, x.end] for x in data.values()]))
    result = []
    for cp in all_times[1:-1]:
        if all((x.start < cp).sum() >= min_transitions and
               (x.start >= cp).sum() >= min_transitions for x in data.values()):
            result.append(float(cp))
    return result


def fit_one(table: Transitions, method: str, alpha: float, allowed: np.ndarray,
            max_iter: int) -> Fit:
    underlying = "euler" if method == "euler_dp" else "ct"
    return fit_method(table, underlying, alpha, allowed, max_iter)


def scan(data: dict[str, Transitions], method: str, alpha: float,
         allowed: np.ndarray, max_iter: int, minimum_transitions: int) -> list[Candidate]:
    if any(x.n < 2 for x in data.values()):
        raise ValueError("Each strain must contain at least two training transitions")
    stationary = {strain: [fit_one(data[strain], method, alpha, allowed, max_iter)] for strain in STRAINS}
    cost0 = segment_cost(stationary, data, None, alpha)
    path = [Candidate(None, stationary, cost0, cost0)]
    if method == "stationary_ct":
        return path
    possible = candidate_times(data, minimum_transitions)
    if method == "two_stage":
        # Split by sample mean before fitting networks, using training states
        # only. The boundary is independent of the subsequent ODE fit.
        shifts = []
        baseline_shift = 0.0
        for table in data.values():
            baseline_shift += float(np.sum((table.x - table.x.mean(axis=0)) ** 2))
        path[0].boundary_score = baseline_shift
        for cp in possible:
            score = 0.0
            for table in data.values():
                left = table.start < cp
                for states in (table.x[left], table.x[~left]):
                    score += float(np.sum((states - states.mean(axis=0)) ** 2))
            shifts.append(score)
        best_index = int(np.argmin(shifts)) if possible else None
        best_shift = shifts[best_index] if best_index is not None else None
        possible = [possible[best_index]] if best_index is not None else []
    for cp in possible:
        fitted = {}
        for strain in STRAINS:
            table = data[strain]
            fitted[strain] = [
                fit_one(table.take(table.start < cp), method, alpha, allowed, max_iter),
                fit_one(table.take(table.start >= cp), method, alpha, allowed, max_iter),
            ]
        cost = segment_cost(fitted, data, cp, alpha)
        path.append(Candidate(cp, fitted, cost, cost,
                              best_shift if method == "two_stage" else None))
    return path


def select(path: list[Candidate], rho: float, method: str) -> Candidate:
    if method == "stationary_ct":
        return path[0]
    stationary = path[0]
    if len(path) == 1:
        return stationary
    if method == "two_stage":
        # Select zero or one change using expression means only, before the
        # stage-specific ODE networks are considered.
        return path[1] if path[1].boundary_score + rho * path[0].boundary_score < path[0].boundary_score else path[0]
    best = min(path[1:], key=lambda item: (item.raw_cost, item.cp))
    changed = best.raw_cost + rho * stationary.raw_cost
    return best if changed < stationary.raw_cost else stationary


def make_inner_split(data: dict[str, Transitions]) -> tuple[dict[str, Transitions], dict[str, Transitions]]:
    """Hold out a short interior block from the outer-training transitions."""
    common = sorted(set.intersection(*(set(x.end) for x in data.values())))
    if len(common) < 6:
        raise ValueError("Too few common training endpoints for inner validation")
    center = len(common) // 2
    held = set(common[max(1, center - 1): min(len(common) - 1, center + 1)])
    train, validation = {}, {}
    for strain, table in data.items():
        test_mask = np.isin(table.end, list(held))
        # Exclude a transition that begins at an inner-validation endpoint.
        train_mask = ~np.isin(table.start, list(held)) & ~test_mask
        train[strain] = table.take(train_mask)
        validation[strain] = table.take(test_mask)
    return train, validation


def tune(data: dict[str, Transitions], allowed: np.ndarray, method: str,
         alphas: list[float], rhos: list[float], max_iter: int,
         minimum_transitions: int) -> tuple[dict, list[dict]]:
    inner_train, inner_test = make_inner_split(data)
    rows = []
    for alpha in alphas:
        path = scan(inner_train, method, alpha, allowed, max_iter, minimum_transitions)
        for rho in ([0.0] if method == "stationary_ct" else rhos):
            chosen = select(path, rho, method)
            mse, _ = test_mse(chosen, inner_test, method)
            rows.append({"method": method, "alpha": alpha, "rho": rho,
                         "inner_mse": mse, "inner_rmse": float(np.sqrt(mse)),
                         "inner_cp": chosen.cp})
    best = min(rows, key=lambda row: (row["inner_mse"], row["alpha"], row["rho"]))
    return best, rows


def write_cv(input_dir: Path, output: Path, alphas: list[float], rhos: list[float],
             max_iter: int, minimum_transitions: int) -> pd.DataFrame:
    records, grid_records = [], []
    for fold_dir in sorted((input_dir / "model/cv").glob("fold_*")):
        genes, train, allowed = load_bundle(fold_dir, "train")
        _, test, _ = load_bundle(fold_dir, "test")
        print(f"[cv] {fold_dir.name}: {len(genes)} genes", flush=True)
        for method in METHODS:
            start = time.perf_counter()
            best, grid = tune(train, allowed, method, alphas, rhos, max_iter, minimum_transitions)
            grid_records.extend({"fold": fold_dir.name, **item} for item in grid)
            path = scan(train, method, best["alpha"], allowed, max_iter, minimum_transitions)
            chosen = select(path, best["rho"], method)
            pooled_mse, by_strain = test_mse(chosen, test, method)
            for strain in STRAINS:
                records.append({"fold": fold_dir.name, "method": method, "strain": strain,
                                "n_test_transitions": test[strain].n,
                                "rmse": by_strain[strain], "pooled_rmse": float(np.sqrt(pooled_mse)),
                                "alpha": best["alpha"], "rho": best["rho"],
                                "change_time": chosen.cp,
                                "inner_rmse": best["inner_rmse"],
                                "runtime_seconds": time.perf_counter() - start})
            print(f"[cv] {fold_dir.name} {method} RMSE={np.sqrt(pooled_mse):.4f} "
                  f"alpha={best['alpha']} rho={best['rho']} cp={chosen.cp}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "blocked_cv_by_fold_strain.tsv", sep="\t", index=False)
    pd.DataFrame(grid_records).to_csv(output / "inner_calibration_grid.tsv", sep="\t", index=False)
    summary = frame.groupby("method", as_index=False).agg(
        mean_rmse=("rmse", "mean"), median_rmse=("rmse", "median"),
    )
    runtime = frame.drop_duplicates(["fold", "method"]).groupby("method")["runtime_seconds"].sum()
    summary["total_runtime_seconds"] = summary["method"].map(runtime)
    summary.to_csv(output / "blocked_cv_summary.tsv", sep="\t", index=False)
    return frame


def full_parameters(cv: pd.DataFrame, method: str) -> tuple[float, float]:
    settings = cv[cv.method.eq(method)][["fold", "alpha", "rho"]].drop_duplicates()
    alpha = float(settings.alpha.median())
    rho = float(settings.rho.median())
    return alpha, rho


def write_full(input_dir: Path, output: Path, cv: pd.DataFrame,
               max_iter: int, minimum_transitions: int) -> dict:
    genes, data, allowed = load_bundle(input_dir / "model/full", "all")
    selected = {}
    path_rows = []
    network_rows = []
    fit_rows = []
    network_payload = {"genes": np.asarray(genes, dtype=str)}
    for method in METHODS:
        alpha, rho = full_parameters(cv, method)
        path = scan(data, method, alpha, allowed, max_iter, minimum_transitions)
        chosen = select(path, rho, method)
        selected[method] = {"alpha": alpha, "rho": rho, "change_time": chosen.cp,
                            "stationary_cost": path[0].raw_cost,
                            "selected_cost": chosen.raw_cost,
                            "n_candidates": len(path) - 1}
        for item in path:
            selection_cost = (item.boundary_score + rho * path[0].boundary_score
                              if method == "two_stage" and item.cp is not None else
                              item.boundary_score if method == "two_stage" else
                              item.raw_cost + (rho * path[0].raw_cost if item.cp is not None else 0.0))
            path_rows.append({"method": method, "alpha": alpha, "rho": rho,
                              "change_time": item.cp, "cost": item.raw_cost,
                              "penalised_cost": selection_cost,
                              "boundary_score": item.boundary_score,
                              "selected": item is chosen})
        for strain in STRAINS:
            for stage_index, fit in enumerate(chosen.fits[strain]):
                network_payload[f"{method}_{strain}_stage{stage_index}"] = fit.matrix
                network_payload[f"{method}_{strain}_baseline{stage_index}"] = fit.baseline
                fit_rows.append({"method": method, "strain": strain, "stage": stage_index + 1,
                                 "iterations": fit.iterations, "converged": fit.converged,
                                 "data_loss": fit.data_loss, "objective": fit.objective})
                for target_index, regulator_index in zip(*np.where(allowed)):
                    if target_index == regulator_index:
                        continue
                    network_rows.append({
                        "method": method, "strain": strain, "stage": stage_index + 1,
                        "regulator": genes[regulator_index], "target": genes[target_index],
                        "edge_weight": float(fit.matrix[target_index, regulator_index]),
                    })
    np.savez_compressed(output / "full_networks.npz", **network_payload)
    pd.DataFrame(path_rows).to_csv(output / "candidate_change_costs.tsv", sep="\t", index=False)
    pd.DataFrame(network_rows).to_csv(output / "full_network_edges.tsv", sep="\t", index=False)
    pd.DataFrame(fit_rows).to_csv(output / "full_fit_diagnostics.tsv", sep="\t", index=False)
    (output / "full_selected_models.json").write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def _bootstrap_data(bundle: Path, seed: int) -> tuple[dict[str, Transitions], np.ndarray]:
    """Resample biological expression samples within each strain/time."""
    rng = np.random.default_rng(seed)
    metadata = pd.read_csv(bundle / "sample_metadata.tsv", sep="\t")
    expression = pd.read_csv(bundle / "sample_expression_zscore.tsv.gz", sep="\t")
    genes = expression["gene"].astype(str).tolist()
    panel = pd.read_csv(bundle / "gene_panel.tsv", sep="\t")
    allowed = np.zeros((len(genes), len(genes)), dtype=bool)
    allowed[:, panel["is_candidate_regulator"].astype(bool).to_numpy()] = True
    np.fill_diagonal(allowed, True)
    result = {}
    for strain in STRAINS:
        group = metadata[metadata.strain.eq(strain)]
        times, values, weights = [], [], []
        for time_point, subset in group.groupby("time", sort=True):
            sample_cols = subset["sample"].tolist()
            sampled = rng.choice(sample_cols, size=len(sample_cols), replace=True)
            matrix = expression[sampled].to_numpy(dtype=float)
            times.append(float(time_point))
            values.append(matrix.mean(axis=1))
            variance = matrix.var(axis=1, ddof=1) if len(sampled) > 1 else np.ones(len(genes))
            weights.append(np.maximum(variance, 1e-3))
        times_arr = np.asarray(times)
        value_arr = np.stack(values)
        variance_arr = np.stack(weights)
        # Bootstrap weights are fixed to the original fitted observation
        # weights to avoid zero variance after duplicate draws.
        with np.load(bundle / "trajectories.npz", allow_pickle=False) as archive:
            original_weights = np.asarray(archive[f"{strain}_weights"], dtype=float)
        result[strain] = Transitions(times_arr[:-1], times_arr[1:], np.diff(times_arr),
                                     value_arr[:-1], value_arr[1:], original_weights[1:])
    return result, allowed


def write_stability(input_dir: Path, output: Path, full: dict, max_iter: int,
                    minimum_transitions: int, n_bootstrap: int, seed: int) -> pd.DataFrame:
    bundle = input_dir / "model/full"
    params = full["discgrn"]
    rows = []
    genes = pd.read_csv(bundle / "gene_panel.tsv", sep="\t")["Gene Symbol"].astype(str).tolist()
    allowed_edges = None
    edge_counts = None
    for repetition in range(n_bootstrap):
        data, allowed = _bootstrap_data(bundle, seed + repetition)
        if edge_counts is None:
            allowed_edges = allowed.copy()
            np.fill_diagonal(allowed_edges, False)
            edge_counts = np.zeros_like(allowed_edges, dtype=int)
        path = scan(data, "discgrn", params["alpha"], allowed,
                    max_iter, minimum_transitions)
        result = select(path, params["rho"], "discgrn")
        present = np.zeros_like(allowed_edges)
        for strain in STRAINS:
            for fit in result.fits[strain]:
                present |= (np.abs(fit.matrix) > 1e-8) & allowed_edges
        edge_counts += present
        rows.append({"bootstrap": repetition, "change_time": result.cp,
                     "n_candidates": len(path) - 1,
                     "n_edges_present_any_strain_or_stage": int(present.sum())})
        if (repetition + 1) % 10 == 0:
            print(f"[stability] {repetition + 1}/{n_bootstrap}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "bootstrap_change_points.tsv", sep="\t", index=False)
    edge_rows = [
        {"regulator": genes[j], "target": genes[i],
         "n_present": int(edge_counts[i, j]),
         "appearance_frequency": float(edge_counts[i, j] / n_bootstrap)}
        for i, j in zip(*np.where(allowed_edges))
    ]
    pd.DataFrame(edge_rows).sort_values(
        ["appearance_frequency", "regulator", "target"],
        ascending=[False, True, True],
    ).to_csv(output / "bootstrap_edge_frequency.tsv", sep="\t", index=False)
    counts = Counter("no_change" if pd.isna(value) else str(value) for value in frame.change_time)
    (output / "bootstrap_change_summary.json").write_text(json.dumps({
        "n_bootstrap": n_bootstrap, "counts": counts,
        "selection_frequency": float(frame.change_time.notna().mean()),
    }, indent=2) + "\n")
    return frame


def write_evidence(input_dir: Path, output: Path, full: dict,
                   n_permutations: int, seed: int) -> dict:
    edge = pd.read_csv(input_dir / "evaluation/edge_evidence.tsv", sep="\t")
    ko = pd.read_csv(input_dir / "evaluation/nkx21_panel_knockout_response.tsv", sep="\t")
    with np.load(output / "full_networks.npz", allow_pickle=False) as archive:
        genes = archive["genes"].astype(str).tolist()
        score_matrices = []
        for strain in STRAINS:
            stages = [archive[key] for key in archive.files if key.startswith(f"discgrn_{strain}_stage")]
            score_matrices.extend(np.abs(stage) for stage in stages)
        score_matrix = np.mean(score_matrices, axis=0)
        # Perturbation at P5 should be compared with the network active then.
        postnatal = []
        for strain in STRAINS:
            stage = 0 if full["discgrn"]["change_time"] is None or 25.0 < full["discgrn"]["change_time"] else 1
            postnatal.append(archive[f"discgrn_{strain}_stage{stage}"])
        postnatal_matrix = np.mean(postnatal, axis=0)
    edge["discgrn_score"] = [score_matrix[int(row.target_index), int(row.regulator_index)]
                             for row in edge.itertuples(index=False)]
    edge["p5_edge_weight"] = [postnatal_matrix[int(row.target_index), int(row.regulator_index)]
                               for row in edge.itertuples(index=False)]
    edge = edge.sort_values("discgrn_score", ascending=False, kind="stable").reset_index(drop=True)
    edge["rank"] = np.arange(1, len(edge) + 1)
    edge.to_csv(output / "ranked_edges_with_evidence.tsv", sep="\t", index=False)
    rng = np.random.default_rng(seed)
    top_rows = []
    for k in (10, 20, 50):
        observed = int(edge.head(k).collectri_reported.sum())
        null = np.empty(n_permutations, dtype=int)
        groups = [part for _, part in edge.groupby("regulator", sort=False)]
        for i in range(n_permutations):
            sampled = pd.concat([part.assign(discgrn_score=rng.permutation(part.discgrn_score.to_numpy()))
                                 for part in groups], ignore_index=True)
            null[i] = int(sampled.nlargest(k, "discgrn_score").collectri_reported.sum())
        top_rows.append({"k": k, "collectri_supported": observed,
                         "support_fraction": observed / k,
                         "matched_null_mean": float(null.mean()),
                         "permutation_p_ge": float((1 + np.sum(null >= observed)) / (n_permutations + 1))})
    pd.DataFrame(top_rows).to_csv(output / "collectri_topk_support.tsv", sep="\t", index=False)
    nkx = edge[edge.regulator.eq("Nkx2-1")].merge(
        ko[["gene", "mutant_minus_control_log2_fpkm1", "abs_response", "welch_p", "bh_q"]],
        left_on="target", right_on="gene", how="left",
    )
    nkx.to_csv(output / "nkx21_network_chip_knockout.tsv", sep="\t", index=False)
    valid_chip = nkx[["p5_edge_weight", "chip_lung_score"]].dropna()
    valid_ko = nkx[["p5_edge_weight", "abs_response"]].dropna()
    chip_r = stats.spearmanr(np.abs(valid_chip.p5_edge_weight), valid_chip.chip_lung_score)
    ko_r = stats.spearmanr(np.abs(valid_ko.p5_edge_weight), valid_ko.abs_response)
    signs = nkx.dropna(subset=["mutant_minus_control_log2_fpkm1"])
    signs = signs[signs.p5_edge_weight.ne(0)]
    concordance = np.sign(signs.p5_edge_weight) == -np.sign(signs.mutant_minus_control_log2_fpkm1)
    result = {"collectri_topk": top_rows,
              "nkx_lung_chip_supported": int(nkx.chip_lung_support.sum()),
              "nkx_candidates": len(nkx),
              "nkx_chip_score_spearman": float(chip_r.statistic),
              "nkx_chip_score_p": float(chip_r.pvalue),
              "nkx_ko_response_spearman": float(ko_r.statistic),
              "nkx_ko_response_p": float(ko_r.pvalue),
              "nkx_sign_concordance": float(concordance.mean()) if len(concordance) else float("nan"),
              "nkx_n_sign_evaluable": len(concordance)}
    (output / "evidence_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_single_cell(input_dir: Path, output: Path) -> dict:
    pseudobulk = pd.read_csv(input_dir / "evaluation/single_cell/stage_pseudobulk_log2cpm.tsv.gz", sep="\t")
    if not {"E17.5", "P7"}.issubset(pseudobulk.columns):
        raise ValueError("Need E17.5 and P7 single-cell stage pseudobulk")
    sc = pseudobulk.set_index("gene")["P7"] - pseudobulk.set_index("gene")["E17.5"]
    bundle = input_dir / "model/full"
    expression = pd.read_csv(bundle / "sample_expression_zscore.tsv.gz", sep="\t").set_index("gene")
    metadata = pd.read_csv(bundle / "sample_metadata.tsv", sep="\t")
    bulk = []
    for strain in STRAINS:
        early = metadata[metadata.strain.eq(strain) & metadata.time.eq(17.5)]["sample"]
        late = metadata[metadata.strain.eq(strain) & metadata.time.eq(27.0)]["sample"]
        if len(early) and len(late):
            bulk.append(expression[late].mean(axis=1) - expression[early].mean(axis=1))
    bulk_delta = pd.concat(bulk, axis=1).mean(axis=1)
    paired = pd.DataFrame({"bulk_z_delta": bulk_delta, "sc_log2cpm_delta": sc}).dropna()
    paired.rename_axis("gene").reset_index().to_csv(output / "bulk_single_cell_stage_delta.tsv", sep="\t", index=False)
    r = stats.spearmanr(paired.bulk_z_delta, paired.sc_log2cpm_delta)
    summary = {"comparison": "E17.5_to_P7", "n_genes": len(paired),
               "spearman_r": float(r.statistic), "spearman_p": float(r.pvalue),
               "interpretation": "Gene-level expression trend only; not a cell-composition estimate."}
    (output / "single_cell_concordance.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_budget_sensitivity(input_dir: Path, output: Path, full: dict,
                             reference_iter: int = 80, longer_iter: int = 320) -> pd.DataFrame:
    _, data, allowed = load_bundle(input_dir / "model/full", "all")
    offdiag = allowed.copy()
    np.fill_diagonal(offdiag, False)
    rows = []
    for strain in STRAINS:
        reference = fit_one(data[strain], "discgrn", full["discgrn"]["alpha"], allowed, reference_iter)
        longer = fit_one(data[strain], "discgrn", full["discgrn"]["alpha"], allowed, longer_iter)
        left = (np.abs(reference.matrix) > 1e-8) & offdiag
        right = (np.abs(longer.matrix) > 1e-8) & offdiag
        rows.append({
            "strain": strain, "reference_iter": reference_iter, "longer_iter": longer_iter,
            "reference_objective": reference.objective, "longer_objective": longer.objective,
            "relative_objective_reduction": (reference.objective - longer.objective) / reference.objective,
            "reference_edges": int(left.sum()), "longer_edges": int(right.sum()),
            "edge_jaccard": float((left & right).sum() / (left | right).sum()),
            "absolute_weight_spearman": float(stats.spearmanr(
                np.abs(reference.matrix[offdiag]), np.abs(longer.matrix[offdiag])).statistic),
            "longer_converged": longer.converged,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "optimization_budget_sensitivity.tsv", sep="\t", index=False)
    return frame


def make_figures(output: Path) -> None:
    cv_path = output / "blocked_cv_by_fold_strain.tsv"
    if cv_path.is_file():
        cv = pd.read_csv(cv_path, sep="\t")
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        labels = [f"{f}/{s}" for f in sorted(cv.fold.unique()) for s in STRAINS]
        x = np.arange(len(labels))
        for method, marker, label, offset in (("discgrn", "o", "DiSCGRN", -0.18),
                                              ("stationary_ct", "s", "Stationary CT", -0.06),
                                              ("euler_dp", "^", "Euler–DP", 0.06),
                                              ("two_stage", "d", "Two-stage", 0.18)):
            values = [float(cv[cv.fold.eq(f) & cv.strain.eq(s) & cv.method.eq(method)].rmse.iloc[0])
                      for f in sorted(cv.fold.unique()) for s in STRAINS]
            ax.scatter(x + offset, values, marker=marker, label=label, s=40)
        ax.set_xticks(x, labels, rotation=50, ha="right")
        ax.set_ylabel("Held-out conditional one-step RMSE (z units)")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(output / "blocked_prediction.pdf")
        plt.close(fig)
    path = output / "candidate_change_costs.tsv"
    if path.is_file():
        frame = pd.read_csv(path, sep="\t")
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        method_frame = frame[frame.method.eq("discgrn")]
        row = method_frame[method_frame.change_time.notna()]
        stationary = float(method_frame[method_frame.change_time.isna()].cost.iloc[0])
        ax.plot(row.change_time, row.cost, marker="o", label="One change, unpenalised")
        ax.plot(row.change_time, row.penalised_cost, marker="s", label="One change, penalised")
        ax.axhline(stationary, color="black", linestyle="--", label="No change")
        ax.set_xlabel("Candidate change time (computational day)")
        ax.set_ylabel("DiSCGRN training selection cost")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / "change_point_cost_profiles.pdf")
        plt.close(fig)
    path = output / "nkx21_network_chip_knockout.tsv"
    if path.is_file():
        frame = pd.read_csv(path, sep="\t")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(frame.p5_edge_weight.abs(), frame.abs_response,
                   c=frame.chip_lung_support.map({True: "#2166ac", False: "#bdbdbd"}), alpha=0.8)
        ax.set_xlabel("Absolute predicted NKX2-1 edge weight")
        ax.set_ylabel("Absolute P5 AT1 knockout response")
        fig.tight_layout()
        fig.savefig(output / "nkx21_orthogonal_evidence.pdf")
        plt.close(fig)
    path = output / "bootstrap_edge_frequency.tsv"
    if path.is_file():
        frame = pd.read_csv(path, sep="\t")
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.hist(frame.appearance_frequency, bins=np.linspace(0, 1, 11),
                edgecolor="white", color="#377eb8")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Bootstrap appearance frequency")
        ax.set_ylabel("Candidate regulator–target pairs")
        fig.tight_layout()
        fig.savefig(output / "bootstrap_edge_frequency_distribution.pdf")
        plt.close(fig)
    path = output / "bulk_single_cell_stage_delta.tsv"
    if path.is_file():
        frame = pd.read_csv(path, sep="\t")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(frame.bulk_z_delta, frame.sc_log2cpm_delta, s=25, alpha=0.75)
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.axvline(0, color="0.7", linewidth=0.8)
        ax.set_xlabel("Bulk E17.5 to P7 expression difference (z units)")
        ax.set_ylabel("Single-cell pseudobulk difference (log2 CPM)")
        fig.tight_layout()
        fig.savefig(output / "bulk_single_cell_trend.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/real_study_20261004"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/real_data_20261006"))
    parser.add_argument("--part", choices=("cv", "full", "stability", "evidence", "single-cell", "sensitivity", "figures", "all"), default="all")
    parser.add_argument("--alphas", default="0.01,0.03,0.07")
    parser.add_argument("--rhos", default="0.10,0.30,0.50")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--minimum-transitions", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=30)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(value) for value in args.alphas.split(",")]
    rhos = [float(value) for value in args.rhos.split(",")]
    if args.part in {"cv", "all"}:
        write_cv(args.input_dir, args.output_dir, alphas, rhos,
                 args.max_iter, args.minimum_transitions)
    if args.part in {"full", "all"}:
        cv = pd.read_csv(args.output_dir / "blocked_cv_by_fold_strain.tsv", sep="\t")
        write_full(args.input_dir, args.output_dir, cv, args.max_iter, args.minimum_transitions)
    if args.part in {"stability", "all"}:
        full = json.loads((args.output_dir / "full_selected_models.json").read_text())
        write_stability(args.input_dir, args.output_dir, full, args.max_iter,
                        args.minimum_transitions, args.bootstrap, args.seed)
    if args.part in {"evidence", "all"}:
        full = json.loads((args.output_dir / "full_selected_models.json").read_text())
        write_evidence(args.input_dir, args.output_dir, full, args.permutations, args.seed)
    if args.part in {"single-cell", "all"}:
        write_single_cell(args.input_dir, args.output_dir)
    if args.part in {"sensitivity", "all"}:
        full = json.loads((args.output_dir / "full_selected_models.json").read_text())
        write_budget_sensitivity(args.input_dir, args.output_dir, full, args.max_iter, 320)
    if args.part in {"figures", "all"}:
        make_figures(args.output_dir)
    (args.output_dir / f"run_configuration_{args.part}.json").write_text(json.dumps({
        "input_dir": str(args.input_dir), "part": args.part, "alphas": alphas,
        "rhos": rhos, "max_iter": args.max_iter,
        "minimum_transitions": args.minimum_transitions,
        "bootstrap": args.bootstrap, "permutations": args.permutations,
        "seed": args.seed,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
