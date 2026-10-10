#!/usr/bin/env python3
"""Tune DiSCGRN on BEELINE runs 1--5 and evaluate on untouched runs 6--10.

The split is shared across all cell-count variants to avoid using one BoolODE
parameter draw for both tuning and assessment. Other methods are never refit.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.io import LongitudinalGroup
from score_five_method_grn_benchmark import beeline_truth, evaluate


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results/beeline_extended_valid_20261009"
FROZEN = PROJECT / "results/runs/grn_accuracy_20261009/five_method_reconstruction/per_run_metrics.tsv"
DEFAULT_OUTPUT = PROJECT / "results/runs/grn_accuracy_20261009/beeline_tuned_holdout_20261009"
DATASETS = ("dyn-LI-100", "dyn-LI-200", "dyn-LI-500", "dyn-LL-100",
            "dyn-LL-200", "dyn-LL-500", "dyn-CY-200")
ALPHAS = (0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0064, 0.0128)
RHOS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0)
BASE_BUDGETS = (80, 160)
REFINE_BUDGET = 320
METRICS = ("aupr", "precision", "recall", "f1", "f0_1")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_id(dataset: str, number: int) -> str:
    return f"{dataset}-{number}"


def job_path(output: Path, dataset: str, number: int, alpha: float, budget: int) -> Path:
    return output / "grid_jobs" / dataset / f"{run_id(dataset, number)}_a{alpha:g}_h{budget}.json"


def score_selected(result, n_times: int) -> np.ndarray:
    weights = ([n_times - 1] if len(result.fits) == 1 else
               [result.change_index, n_times - 1 - result.change_index])
    scores = sum(weight * np.abs(fit.matrix)
                 for weight, fit in zip(weights, result.fits)) / sum(weights)
    np.fill_diagonal(scores, 0.0)
    return scores


def fit_job(job: tuple[str, int, float, int, str]) -> list[dict]:
    dataset, number, alpha, budget, output_str = job
    output = Path(output_str)
    cached = job_path(output, dataset, number, alpha, budget)
    if cached.is_file():
        return json.loads(cached.read_text())
    binned = pd.read_csv(SOURCE / dataset / run_id(dataset, number) / "binned_input.csv", index_col=0)
    genes = binned.columns.astype(str).tolist()
    values = binned.to_numpy(float)
    times = binned.index.to_numpy(float)
    group = LongitudinalGroup("pseudobulk", times, values,
                              np.ones_like(values), np.ones(len(times)))
    path = scan_one_change_candidates([group], times, "ct", alpha,
                                      minimum_transitions=5, max_iter=budget,
                                      ct_initialiser="euler")
    truth_file = PROJECT / "data/raw/beeline/Synthetic" / dataset.rsplit("-", 1)[0] / run_id(dataset, number) / "GroundTruth.csv"
    truth = beeline_truth(genes, pd.read_csv(truth_file))
    rows = []
    for rho in RHOS:
        selected = select_one_change(path, rho, relative_beta=True)
        metrics = evaluate(truth, score_selected(selected, len(times)), 0.10)
        rows.append({"dataset": dataset, "run_id": run_id(dataset, number),
                     "run_number": number, "alpha": alpha, "rho": rho,
                     "H": budget, "change_index": selected.change_index,
                     "converged_all_segments": all(fit.converged for fit in selected.fits),
                     **{key: metrics[key] for key in METRICS}})
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def dispatch(jobs: list[tuple[str, int, float, int, str]], workers: int) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fit_job, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if index % 20 == 0 or index == len(jobs):
                print(f"Completed {index}/{len(jobs)} alpha-H-run fits", flush=True)
    return pd.DataFrame(rows)


def rank_configs(frame: pd.DataFrame, max_selected_rho: float = 1.0) -> pd.DataFrame:
    frame = frame[frame.rho <= max_selected_rho + 1e-12].copy()
    summary = frame.groupby(["dataset", "alpha", "H", "rho"], as_index=False).agg(
        n=("aupr", "size"), **{key: (key, "mean") for key in METRICS})
    if not (summary.n == 5).all():
        raise ValueError("Every development configuration must contain five instances")
    return summary.sort_values(["dataset", "aupr", "f0_1", "f1", "H", "rho", "alpha"],
                               ascending=[True, False, False, False, True, False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phase", choices=("base", "refine", "extend", "rho-cap", "holdout", "all"), default="all")
    parser.add_argument("--max-selected-rho", type=float, default=1.0,
                        help="Restrict parameter selection and development alpha refinement to rho at or below this value")
    parser.add_argument("--li-only", action="store_true",
                        help="Report only dyn-LI-100/200/500 using existing cached fits")
    args = parser.parse_args()
    if not 0 <= args.max_selected_rho <= 1:
        raise ValueError("max-selected-rho must lie in [0,1]")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    analysis_dir = (output if args.max_selected_rho == 1.0 else
                    output / f"rho_le_{str(args.max_selected_rho).replace('.', 'p')}")
    active_datasets = DATASETS[:3] if args.li_only else DATASETS
    if args.li_only:
        if args.phase != "holdout":
            raise ValueError("--li-only uses existing fits and requires --phase holdout")
        analysis_dir = analysis_dir / "li_only"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    protocol = {"development_run_numbers": [1, 2, 3, 4, 5],
                "holdout_run_numbers": [6, 7, 8, 9, 10],
                "shared_split_across_cell_counts": True,
                "selection_primary": "development mean AUPR",
                "selection_ties": "F0.1, F1, smaller H, larger rho, larger alpha",
                "alpha_grid": ALPHAS, "rho_grid": RHOS,
                "base_budgets": BASE_BUDGETS, "refine_budget": REFINE_BUDGET,
                "minimum_transitions": 5, "initialiser": "euler",
                "top_edge_fraction": 0.10,
                "comparators": "reused frozen score matrices; no refit"}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Existing output has a different protocol")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    if args.phase in ("base", "all"):
        jobs = [(dataset, number, alpha, budget, str(output))
                for dataset in DATASETS for number in range(1, 6)
                for alpha in ALPHAS for budget in BASE_BUDGETS]
        base = dispatch(jobs, args.workers)
        base.sort_values(["dataset", "alpha", "H", "rho", "run_number"]).to_csv(
            output / "base_development_metrics.tsv", sep="\t", index=False)
        rank_configs(base).to_csv(output / "base_development_summary.tsv", sep="\t", index=False)
        if args.phase == "base":
            return

    if args.phase in ("refine", "all"):
        base = pd.read_csv(output / "base_development_metrics.tsv", sep="\t")
        ranked = rank_configs(base)
        chosen_alphas = {}
        for dataset in DATASETS:
            chosen_alphas[dataset] = ranked.loc[ranked.dataset == dataset, "alpha"].drop_duplicates().head(2).tolist()
        (output / "refined_alpha_choices.json").write_text(json.dumps(chosen_alphas, indent=2) + "\n")
        jobs = [(dataset, number, alpha, REFINE_BUDGET, str(output))
                for dataset in DATASETS for number in range(1, 6)
                for alpha in chosen_alphas[dataset]]
        refined = dispatch(jobs, args.workers)
        refined.sort_values(["dataset", "alpha", "H", "rho", "run_number"]).to_csv(
            output / "refined_development_metrics.tsv", sep="\t", index=False)
        if args.phase == "refine":
            return

    if args.phase == "extend":
        prior = pd.read_csv(output / "all_development_summary.tsv", sep="\t")
        extension_jobs = []
        for dataset in DATASETS:
            top_alphas = prior.loc[prior.dataset == dataset, "alpha"].drop_duplicates().head(2)
            extension_jobs.extend((dataset, number, float(alpha), 640, str(output))
                                  for number in range(1, 6) for alpha in top_alphas)
        for dataset in ("dyn-LI-100", "dyn-LI-200", "dyn-LI-500"):
            extension_jobs.extend((dataset, number, alpha, budget, str(output))
                                  for number in range(1, 6)
                                  for alpha in (0.0256, 0.0512) for budget in (160, 320))
        extension_protocol = {
            "purpose": "development-only extension after original alpha/H boundary check",
            "H_640_alpha_choice": "top two distinct alphas per condition from original development grid",
            "wider_alpha_conditions": ["dyn-LI-100", "dyn-LI-200", "dyn-LI-500"],
            "wider_alphas": [0.0256, 0.0512], "wider_alpha_budgets": [160, 320],
            "rho_grid": RHOS, "holdout_used_for_choice": False,
        }
        (output / "extension_protocol.json").write_text(json.dumps(extension_protocol, indent=2) + "\n")
        extension = dispatch(extension_jobs, args.workers)
        extension.sort_values(["dataset", "alpha", "H", "rho", "run_number"]).to_csv(
            output / "extension_development_metrics.tsv", sep="\t", index=False)

    if args.phase == "rho-cap":
        base = pd.read_csv(output / "base_development_metrics.tsv", sep="\t")
        shortlist = rank_configs(base, args.max_selected_rho)
        existing = [pd.read_csv(output / "refined_development_metrics.tsv", sep="\t")]
        extension_path = output / "extension_development_metrics.tsv"
        if extension_path.exists():
            existing.append(pd.read_csv(extension_path, sep="\t"))
        available = pd.concat(existing, ignore_index=True)[["dataset", "alpha", "H", "run_number"]]
        available_keys = set(map(tuple, available.itertuples(index=False, name=None)))
        extra_jobs = []
        for dataset in DATASETS:
            leading = shortlist.loc[shortlist.dataset == dataset, "alpha"].drop_duplicates().head(2)
            for alpha in leading:
                for budget in (320, 640):
                    for number in range(1, 6):
                        key = (dataset, float(alpha), budget, number)
                        if key not in available_keys:
                            extra_jobs.append((dataset, number, float(alpha), budget, str(output)))
        (analysis_dir / "selection_protocol.json").write_text(json.dumps(
            {"max_selected_rho": args.max_selected_rho,
             "development_run_numbers": [1, 2, 3, 4, 5],
             "holdout_run_numbers": [6, 7, 8, 9, 10],
             "shortlist": "top two alphas per condition under rho cap on base grid",
             "extra_budgets": [320, 640], "extra_job_count": len(extra_jobs),
             "no_test_labels_used_for_selection": True}, indent=2) + "\n")
        if extra_jobs:
            added = dispatch(extra_jobs, args.workers)
            added.to_csv(analysis_dir / "added_development_metrics.tsv", sep="\t", index=False)
        else:
            print("Rho-cap shortlist already fully evaluated at H=320,640", flush=True)

    base = pd.read_csv(output / "base_development_metrics.tsv", sep="\t")
    refined = pd.read_csv(output / "refined_development_metrics.tsv", sep="\t")
    pieces = [base, refined]
    extension_path = output / "extension_development_metrics.tsv"
    if extension_path.exists():
        pieces.append(pd.read_csv(extension_path, sep="\t"))
    added_path = analysis_dir / "added_development_metrics.tsv"
    if added_path.exists():
        pieces.append(pd.read_csv(added_path, sep="\t"))
    ranked = rank_configs(pd.concat(pieces, ignore_index=True), args.max_selected_rho)
    ranked = ranked[ranked.dataset.isin(active_datasets)].copy()
    ranked.to_csv(analysis_dir / "all_development_summary.tsv", sep="\t", index=False)
    selected = ranked.groupby("dataset", sort=False).head(1).copy()
    selected.to_csv(analysis_dir / "selected_parameters.tsv", sep="\t", index=False)
    if args.phase == "holdout":
        print("Using pre-existing development selection; no test labels enter selection", flush=True)

    jobs = [(row.dataset, number, float(row.alpha), int(row.H), str(output))
            for row in selected.itertuples(index=False) for number in range(6, 11)]
    test_grid = dispatch(jobs, args.workers)
    selected_test = test_grid.merge(selected[["dataset", "alpha", "rho", "H"]],
                                    on=["dataset", "alpha", "rho", "H"], how="inner")
    if len(selected_test) != 5 * len(active_datasets):
        raise ValueError(f"Expected {5 * len(active_datasets)} held-out DiSCGRN runs; obtained {len(selected_test)}")
    selected_test.to_csv(analysis_dir / "discgrn_holdout_metrics.tsv", sep="\t", index=False)

    frozen = pd.read_csv(FROZEN, sep="\t")
    frozen = frozen[frozen.domain == "BEELINE"].copy()
    frozen = frozen[frozen.dataset.isin(active_datasets)].copy()
    frozen = frozen[frozen.run_id.str.rsplit("-", n=1).str[-1].astype(int) >= 6].copy()
    frozen["evaluation"] = "frozen_parameter"
    tuned = selected_test[["dataset", "run_id", *METRICS]].copy()
    tuned["domain"] = "BEELINE"
    tuned["method"] = "DiSCGRN_tuned"
    tuned["evaluation"] = "development_selected"
    combined = pd.concat([frozen[["domain", "dataset", "run_id", "method", "evaluation", *METRICS]],
                          tuned[["domain", "dataset", "run_id", "method", "evaluation", *METRICS]]],
                         ignore_index=True)
    combined.to_csv(analysis_dir / "comparison_holdout_per_run.tsv", sep="\t", index=False)
    summary = combined.groupby(["dataset", "method"], as_index=False).agg(
        n=("aupr", "size"), **{key: (key, "mean") for key in METRICS})
    summary.to_csv(analysis_dir / "comparison_holdout_by_condition.tsv", sep="\t", index=False)
    macro = summary.groupby("method", as_index=False).agg(
        n_conditions=("dataset", "size"), **{key: (key, "mean") for key in METRICS})
    macro.to_csv(analysis_dir / "comparison_holdout_macro.tsv", sep="\t", index=False)
    paired = []
    for metric in METRICS:
        pivot = combined.pivot(index=["dataset", "run_id"], columns="method", values=metric)
        for dataset in active_datasets:
            segment = pivot.loc[dataset]
            for comparator in ("DiSCGRN", "GENIE3", "GRNBoost2", "dynGENIE3", "SCODE"):
                delta = (segment["DiSCGRN_tuned"] - segment[comparator]).to_numpy()
                rng = np.random.default_rng(20261009 + sum(map(ord, dataset + comparator + metric)))
                boot = rng.choice(delta, size=(5000, 5), replace=True).mean(axis=1)
                paired.append({"dataset": dataset, "metric": metric,
                               "comparison": f"DiSCGRN_tuned - {comparator}",
                               "n_paired": 5, "mean_difference": float(delta.mean()),
                               "ci_low": float(np.quantile(boot, 0.025)),
                               "ci_high": float(np.quantile(boot, 0.975))})
        pivot = pivot.reset_index()
        pivot["run_number"] = pivot.run_id.str.rsplit("-", n=1).str[-1].astype(int)
        for comparator in ("DiSCGRN", "GENIE3", "GRNBoost2", "dynGENIE3", "SCODE"):
            by_draw = (pivot["DiSCGRN_tuned"] - pivot[comparator]).groupby(pivot.run_number).mean().to_numpy()
            rng = np.random.default_rng(20261009 + sum(map(ord, comparator + metric + "macro")))
            boot = rng.choice(by_draw, size=(5000, 5), replace=True).mean(axis=1)
            paired.append({"dataset": f"{len(active_datasets)}-condition macro", "metric": metric,
                           "comparison": f"DiSCGRN_tuned - {comparator}",
                           "n_paired": 5, "mean_difference": float(by_draw.mean()),
                           "ci_low": float(np.quantile(boot, 0.025)),
                           "ci_high": float(np.quantile(boot, 0.975))})
    pd.DataFrame(paired).to_csv(analysis_dir / "comparison_holdout_paired.tsv", sep="\t", index=False)
    print("Selected parameters and held-out macro metrics:", flush=True)
    print(selected[["dataset", "alpha", "rho", "H", "aupr", "f0_1"]].to_string(index=False), flush=True)
    print(macro.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
