#!/usr/bin/env python3
"""Matched-input, exploratory BEELINE BoolODE network ranking benchmark.

Single-cell snapshots are quantile-binned along supplied pseudotime. The same
binned expression matrix is passed to all four methods. This is not an
individual longitudinal experiment or the original BEELINE evaluation setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.io import LongitudinalGroup


METHODS = ("DiSCGRN", "EDISON", "GENIE3", "GRNBoost2")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_run(directory: Path, bins: int, dropout_fraction: float = 0.0,
                dropout_probability: float = 0.5) -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict]:
    expr_path = directory / "ExpressionData.csv"
    time_path = directory / "PseudoTime.csv"
    truth_path = directory / "GroundTruth.csv"
    expression = pd.read_csv(expr_path, index_col=0)
    pseudotime = pd.read_csv(time_path, index_col=0).iloc[:, 0]
    truth = pd.read_csv(truth_path)
    if expression.index.has_duplicates or expression.columns.has_duplicates:
        raise ValueError(f"Duplicate cell/gene names in {directory}")
    if not expression.index.equals(pseudotime.index):
        pseudotime = pseudotime.reindex(expression.index)
    if pseudotime.isna().any() or expression.isna().any().any():
        raise ValueError(f"Missing expression or pseudotime in {directory}")
    genes = expression.columns.astype(str).tolist()
    x = expression.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(pseudotime.to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite values in {directory}")
    if len(x) < bins * 4:
        raise ValueError(f"Too few cells for {bins} bins: {len(x)}")
    dropout_meta = {}
    if dropout_fraction:
        if not 0 < dropout_fraction < 1 or not 0 < dropout_probability <= 1:
            raise ValueError("Dropout fraction and probability must lie in (0,1)")
        # The same per-cell random draw is used at all severity levels of a run.
        run_number = int(directory.name.rsplit("-", 1)[-1])
        seed = 20261009 + run_number
        rng = np.random.default_rng(seed)
        cutoff = np.quantile(x, dropout_fraction, axis=0)
        selected = (x <= cutoff) & (rng.random(x.shape) < dropout_probability)
        newly_zeroed = int(np.count_nonzero(selected & (x != 0)))
        x = x.copy()
        x[selected] = 0.0
        dropout_meta = {"dropout_fraction": dropout_fraction,
                        "dropout_probability": dropout_probability,
                        "dropout_seed": seed, "newly_zeroed_values": newly_zeroed}
    deviations = x.std(axis=0, ddof=0)
    if np.any(deviations <= 0):
        raise ValueError(f"Constant expression gene in {directory}")
    x = (x - x.mean(axis=0)) / deviations
    order = np.argsort(pseudotime.to_numpy(dtype=float), kind="stable")
    groups = np.array_split(order, bins)
    means = np.vstack([x[group].mean(axis=0) for group in groups])
    raw_times = np.array([pseudotime.to_numpy(dtype=float)[group].mean() for group in groups])
    if not np.all(np.diff(raw_times) > 0):
        raise ValueError(f"Pseudotime bin means not strictly increasing in {directory}")
    times = 12 * (raw_times - raw_times[0]) / (raw_times[-1] - raw_times[0])
    binned = pd.DataFrame(means, index=times, columns=genes)
    meta = {"n_cells": len(x), "n_genes": len(genes), "n_bins": bins,
            "min_bin_cells": min(map(len, groups)), "max_bin_cells": max(map(len, groups)),
            "expression_sha256": digest(expr_path), "pseudotime_sha256": digest(time_path),
            "truth_sha256": digest(truth_path), **dropout_meta}
    return binned, genes, truth, meta


def fit_discgrn(binned: pd.DataFrame, alpha: float, rho: float, budget: int) -> tuple[np.ndarray, dict]:
    values = binned.to_numpy(dtype=float, copy=True)
    times = binned.index.to_numpy(dtype=float)
    group = LongitudinalGroup("pseudobulk", times, values,
                              np.ones_like(values), np.ones(len(times)))
    path = scan_one_change_candidates([group], times, "ct", alpha,
                                      minimum_transitions=5, max_iter=budget,
                                      ct_initialiser="euler")
    selected = select_one_change(path, rho, relative_beta=True)
    weights = np.array([len(times) - 1], dtype=float) if len(selected.fits) == 1 else np.array(
        [selected.change_index, len(times) - 1 - selected.change_index], dtype=float)
    scores = sum(weight * np.abs(fit.matrix) for weight, fit in zip(weights, selected.fits)) / weights.sum()
    np.fill_diagonal(scores, 0.0)
    return scores, {"change_index": selected.change_index,
                    "change_pseudotime": selected.change_time,
                    "converged_segments": [bool(fit.converged) for fit in selected.fits],
                    "iterations": [int(fit.n_iterations) for fit in selected.fits],
                    "n_candidate_models": len(path)}


def fit_edison(binned: pd.DataFrame, run_dir: Path, rscript: str) -> tuple[np.ndarray, dict]:
    input_path = run_dir / "binned_input.csv"
    output_path = run_dir / "edison_scores.csv"
    binned.to_csv(input_path)
    command = [rscript, str(Path(__file__).with_name("run_beeline_edison.R")),
               str(input_path), str(output_path)]
    process = subprocess.run(command, text=True, capture_output=True, timeout=1200)
    (run_dir / "edison.log").write_text(process.stdout + "\n" + process.stderr)
    if process.returncode:
        raise RuntimeError(f"EDISON exited {process.returncode}: {process.stderr[-500:]}")
    frame = pd.read_csv(output_path, index_col=0).loc[binned.columns, binned.columns]
    scores = frame.to_numpy(dtype=float)
    return scores, {"rscript": rscript, "edison_stdout": process.stdout[-500:]}


def fit_tree(method: str, binned: pd.DataFrame, seed: int, run_dir: Path,
             tree_python: str | None) -> tuple[np.ndarray, dict]:
    if tree_python is None:
        from run_tree_grn_benchmark import fit_stage

        scores, _ = fit_stage(method.lower(), binned.to_numpy(dtype=float),
                              binned.columns.tolist(), seed)
        return scores, {}
    input_path = run_dir / "binned_input.csv"
    output_path = run_dir / f"{method}_scores.csv"
    command = [tree_python, str(Path(__file__).with_name("run_beeline_tree_once.py")),
               "--input", str(input_path), "--output", str(output_path),
               "--method", method.lower(), "--seed", str(seed)]
    process = subprocess.run(command, text=True, capture_output=True, timeout=1200)
    (run_dir / f"{method}_tree.log").write_text(process.stdout + "\n" + process.stderr)
    if process.returncode:
        raise RuntimeError(f"{method} exited {process.returncode}: {process.stderr[-500:]}")
    scores = pd.read_csv(output_path, index_col=0).loc[binned.columns, binned.columns].to_numpy(dtype=float)
    return scores, {"tree_python": tree_python}


def score_network(scores: np.ndarray, genes: list[str], truth: pd.DataFrame) -> dict:
    position = {gene: index for index, gene in enumerate(genes)}
    labels = np.zeros_like(scores, dtype=int)
    if not {"from", "to"}.issubset(truth.columns):
        raise ValueError("GroundTruth.csv must have from and to columns")
    for source_value, target_value in truth[["from", "to"]].itertuples(index=False, name=None):
        source, target = str(source_value), str(target_value)
        if source in position and target in position and source != target:
            labels[position[target], position[source]] = 1
    mask = ~np.eye(len(genes), dtype=bool)
    y = labels[mask]
    q = np.asarray(scores, dtype=float)[mask]
    if not np.isfinite(q).all() or y.min() == y.max():
        raise ValueError("Nonfinite scores or only one reference class")
    n_positive = int(y.sum())
    # Stable ordering makes ties deterministic; AP itself handles ties jointly.
    top = np.argsort(-q, kind="stable")[:n_positive]
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y, q)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(precision), where=denominator > 0)
    ap = float(average_precision_score(y, q))
    prevalence = float(y.mean())
    return {"aupr_ap": ap, "aupr_lift": ap / prevalence,
            "auroc": float(roc_auc_score(y, q)),
            "max_f1": float(f1.max()),
            "precision_at_k": float(y[top].mean()),
            "k_true_edges": n_positive,
            "n_tested_edges": int(len(y)),
            "positive_prevalence": prevalence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/raw/beeline"))
    parser.add_argument("--pattern", default="Synthetic/dyn-LL/dyn-LL-200-*")
    parser.add_argument("--dataset-label", default=None,
                        help="Label for this homogeneous dataset series (for example dyn-LI-100).")
    parser.add_argument("--output-root", type=Path, default=Path("results/beeline_matched_20261007"))
    parser.add_argument("--bins", type=int, default=13)
    parser.add_argument("--dropout-fraction", type=float, default=0.0,
                        help="Eligible lowest-expression fraction per gene; zero disables dropout.")
    parser.add_argument("--dropout-probability", type=float, default=0.5)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--alpha", type=float, default=0.0004)
    parser.add_argument("--rho", type=float, default=0.30)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--rscript", default="/opt/miniconda3/bin/Rscript")
    parser.add_argument("--tree-python", default=None,
                        help="Python executable with Arboreto; defaults to current interpreter.")
    args = parser.parse_args()
    directories = [path for path in args.input_root.glob(args.pattern)
                   if path.is_dir() and (path / "GroundTruth.csv").is_file()]
    directories.sort(key=lambda path: (path.parent.name, int(path.name.rsplit("-", 1)[-1])))
    if args.max_runs:
        directories = directories[:args.max_runs]
    if not directories:
        raise FileNotFoundError(f"No BEELINE runs under {args.input_root / args.pattern}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {"source": "BEELINE derivative Zenodo record 7682713 (Pratapa et al. 2020 BoolODE data)",
                "pattern": args.pattern, "run_ids": [str(path.relative_to(args.input_root)) for path in directories],
                "methods": args.methods, "input": "all methods receive the same quantile-pseudotime bin means",
                "binning": "cell-wise per-gene z-score, equal-cell-count bins, mean pseudotime rescaled to [0,12]",
                "discgrn": {"alpha": args.alpha, "rho": args.rho, "max_iter": args.budget,
                             "initialiser": "euler", "minimum_transitions": 5},
                "truth": "static directed non-self edges; signs ignored for common unsigned ranking",
                "dynamic_to_static": "transition-count-weighted mean of absolute fitted stage coefficients",
                "interpretation": "exploratory; pseudotime bins are not longitudinal trajectories"}
    if args.dataset_label is not None:
        protocol["dataset_label"] = args.dataset_label
    if args.tree_python is not None:
        protocol["tree_python"] = args.tree_python
    if args.dropout_fraction:
        protocol["dropout"] = {"lowest_expression_fraction": args.dropout_fraction,
                               "probability": args.dropout_probability,
                               "seed_rule": "20261009 + run number"}
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(f"Existing output has a different protocol: {protocol_path}")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    all_rows = []
    for directory in directories:
        run_id = directory.name
        run_dir = args.output_root / run_id
        run_dir.mkdir(exist_ok=True)
        binned, genes, truth, input_meta = prepare_run(
            directory, args.bins, args.dropout_fraction, args.dropout_probability)
        binned.to_csv(run_dir / "binned_input.csv")
        (run_dir / "input_manifest.json").write_text(json.dumps(input_meta, indent=2) + "\n")
        for method in args.methods:
            metric_path = run_dir / f"{method}_metrics.json"
            if metric_path.exists():
                row = json.loads(metric_path.read_text())
                if row.get("status") == "ok" and "max_f1" not in row:
                    scores = pd.read_csv(run_dir / f"{method}_scores.csv", index_col=0)
                    row.update(score_network(scores.loc[genes, genes].to_numpy(dtype=float), genes, truth))
                    metric_path.write_text(json.dumps(row, indent=2) + "\n")
            else:
                start = time.perf_counter()
                row = {"run_id": run_id, "method": method,
                       "dataset": args.dataset_label or directory.parent.name, **input_meta}
                try:
                    if method == "DiSCGRN":
                        scores, detail = fit_discgrn(binned, args.alpha, args.rho, args.budget)
                    elif method == "EDISON":
                        scores, detail = fit_edison(binned, run_dir, args.rscript)
                    else:
                        scores, detail = fit_tree(method, binned,
                                                  seed=20261007 + int(run_id.rsplit("-", 1)[-1]),
                                                  run_dir=run_dir, tree_python=args.tree_python)
                    pd.DataFrame(scores, index=genes, columns=genes).to_csv(run_dir / f"{method}_scores.csv")
                    row.update(score_network(scores, genes, truth))
                    row.update({"status": "ok", "detail": detail})
                except Exception as exc:
                    row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                row["runtime_seconds"] = time.perf_counter() - start
                metric_path.write_text(json.dumps(row, indent=2) + "\n")
            if "error" in row:
                # Keep the TSV one physical line per method/run; the full R log remains on disk.
                row["error"] = " ".join(str(row["error"]).split())
                metric_path.write_text(json.dumps(row, indent=2) + "\n")
            print(run_id, method, row["status"], row.get("aupr_ap", row.get("error")), flush=True)
            all_rows.append({key: value for key, value in row.items() if key not in {"detail"}})
        pd.DataFrame(all_rows).to_csv(args.output_root / "all_metrics.tsv", sep="\t", index=False)
    frame = pd.DataFrame(all_rows)
    summary = frame.groupby("method", sort=False).agg(
        n_planned=("run_id", "count"), n_evaluable=("status", lambda x: int((x == "ok").sum())),
        aupr_ap_mean=("aupr_ap", "mean"), auroc_mean=("auroc", "mean"),
        precision_at_k_mean=("precision_at_k", "mean")).reset_index()
    summary.to_csv(args.output_root / "summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
