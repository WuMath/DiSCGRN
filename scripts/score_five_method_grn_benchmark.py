#!/usr/bin/env python3
"""Uniform five-metric directed-edge evaluation for DREAM-derived and BEELINE data.

AUPR uses all directed non-self edges. Binary metrics use the predeclared top
10% of candidate edges, with stable tie-breaking; no gold-standard-dependent
threshold is selected. DREAM static scores are evaluated on each true stage
before averaging the two stage metrics within a system.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "results/runs/grn_accuracy_20261009"
DREAM = PROJECT / "results/runs/weighted_score_20261004/tree_grn_long"
BEELINE = PROJECT / "results/beeline_extended_valid_20261009"
METHODS = ("DiSCGRN", "GENIE3", "GRNBoost2", "dynGENIE3", "SCODE")
METRICS = ("aupr", "precision", "recall", "f1", "f0_1")
BEELINE_DATASETS = ("dyn-LI-100", "dyn-LI-200", "dyn-LI-500")


def evaluate(truth: np.ndarray, scores: np.ndarray, top_fraction: float) -> dict:
    if truth.shape != scores.shape or truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        raise ValueError("Truth and score must be matching square matrices")
    mask = ~np.eye(len(truth), dtype=bool)
    y = (truth[mask] != 0).astype(int)
    q = np.asarray(scores, float)[mask]
    if not np.isfinite(q).all() or y.min() == y.max():
        raise ValueError("Invalid edge scores or reference labels")
    k = max(1, math.ceil(top_fraction * len(y)))
    selected = np.argsort(-q, kind="stable")[:k]
    tp = int(y[selected].sum())
    precision = tp / k
    recall = tp / int(y.sum())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    beta_sq = 0.01
    f0_1 = (0.0 if beta_sq * precision + recall == 0 else
            (1 + beta_sq) * precision * recall / (beta_sq * precision + recall))
    return {"aupr": float(average_precision_score(y, q)),
            "precision": precision, "recall": recall,
            "f1": f1, "f0_1": f0_1, "n_candidates": len(y),
            "n_gold_edges": int(y.sum()), "n_called_edges": k, "true_positives": tp}


def dream_scores(method: str, seed: int, root: Path) -> tuple[np.ndarray, np.ndarray]:
    if method == "DiSCGRN":
        archive = np.load(root / "discgrn_dream_refit" / f"system_{seed}_fit.npz")
        return np.abs(archive["before"]), np.abs(archive["after"])
    if method in ("GENIE3", "GRNBoost2"):
        token = "genie3" if method == "GENIE3" else "grnboost2"
        archive = np.load(DREAM / f"system_{seed}_{token}_global_scores.npz")
        return archive["before"], archive["after"]
    if method == "dynGENIE3":
        score = np.load(PROJECT / "results/runs/weighted_score_20261004/dyngenie3_long" /
                        f"system_{seed}_scores.npz")["scores"]
        return score, score
    if method == "SCODE":
        score = np.load(root / "scode/dream" / f"system_{seed}" / "scores.npz")["scores"]
        return score, score
    raise ValueError(method)


def dream_rows(root: Path, top_fraction: float) -> tuple[list[dict], list[dict]]:
    manifest = pd.read_csv(DREAM / "input_manifest.tsv", sep="\t")
    rows, stage_rows = [], []
    for item in manifest.itertuples(index=False):
        seed = int(item.seed)
        archive = np.load(DREAM / f"system_{seed}.npz")
        for method in METHODS:
            before, after = dream_scores(method, seed, root)
            per_stage = []
            for stage, truth, score in ((1, archive["first"], before),
                                        (2, archive["second"], after)):
                metrics = evaluate(truth, score, top_fraction)
                per_stage.append(metrics)
                stage_rows.append({"domain": "DREAM_derived", "dataset": f"T{int(item.T)}",
                                   "run_id": f"system_{seed}", "changed": bool(item.changed),
                                   "stage": stage, "method": method, **metrics})
            rows.append({"domain": "DREAM_derived", "dataset": f"T{int(item.T)}",
                         "run_id": f"system_{seed}", "changed": bool(item.changed),
                         "method": method,
                         **{key: float(np.mean([part[key] for part in per_stage]))
                            for key in METRICS},
                         "n_candidates": per_stage[0]["n_candidates"],
                         "n_called_edges": per_stage[0]["n_called_edges"]})
    return rows, stage_rows


def beeline_truth(genes: list[str], gold: pd.DataFrame) -> np.ndarray:
    position = {gene: i for i, gene in enumerate(genes)}
    truth = np.zeros((len(genes), len(genes)), dtype=int)
    if not {"from", "to"}.issubset(gold.columns):
        raise ValueError("BEELINE gold standard lacks from/to fields")
    for source, target in gold[["from", "to"]].itertuples(index=False, name=None):
        source, target = str(source), str(target)
        if source in position and target in position and source != target:
            truth[position[target], position[source]] = 1
    return truth


def beeline_score_path(method: str, label: str, run_id: str, root: Path) -> Path:
    if method in ("DiSCGRN", "GENIE3", "GRNBoost2"):
        return BEELINE / label / run_id / f"{method}_scores.csv"
    if method == "dynGENIE3":
        return root / "dyngenie3_beeline" / label / run_id / "dynGENIE3_scores.csv"
    if method == "SCODE":
        return root / "scode/beeline" / label / run_id / "SCODE_scores.csv"
    raise ValueError(method)


def beeline_rows(root: Path, top_fraction: float,
                 datasets: tuple[str, ...] = BEELINE_DATASETS) -> list[dict]:
    rows = []
    for input_path in sorted(BEELINE.glob("dyn-*/dyn-*-*/binned_input.csv")):
        label, run_id = input_path.parent.parent.name, input_path.parent.name
        if label not in datasets:
            continue
        binned = pd.read_csv(input_path, index_col=0)
        genes = binned.columns.astype(str).tolist()
        network = label.rsplit("-", 1)[0]
        gold = pd.read_csv(PROJECT / "data/raw/beeline/Synthetic" /
                           network / run_id / "GroundTruth.csv")
        truth = beeline_truth(genes, gold)
        for method in METHODS:
            score_path = beeline_score_path(method, label, run_id, root)
            frame = pd.read_csv(score_path, index_col=0)
            if set(frame.index) != set(genes) or set(frame.columns) != set(genes):
                raise ValueError(f"Gene set differs: {score_path}")
            scores = frame.loc[genes, genes].to_numpy(float)
            metrics = evaluate(truth, scores, top_fraction)
            rows.append({"domain": "BEELINE", "dataset": label, "run_id": run_id,
                         "changed": None, "method": method, **metrics})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--beeline-datasets", nargs="+", default=list(BEELINE_DATASETS),
                        help="BEELINE conditions to include in the scored benchmark")
    args = parser.parse_args()
    if not 0 < args.top_fraction < 1:
        raise ValueError("top-fraction must be strictly between zero and one")
    output = args.output_root or args.root / "five_method_reconstruction"
    output.mkdir(parents=True, exist_ok=True)
    dream, stages = dream_rows(args.root, args.top_fraction)
    combined = pd.DataFrame(dream + beeline_rows(args.root, args.top_fraction,
                                                 tuple(args.beeline_datasets)))
    if combined.duplicated(["domain", "dataset", "run_id", "method"]).any():
        raise ValueError("Duplicate run-method records")
    combined.to_csv(output / "per_run_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(stages).to_csv(output / "dream_stage_metrics.tsv", sep="\t", index=False)
    summary = combined.groupby(["domain", "dataset", "method"], as_index=False).agg(
        n=("aupr", "size"), **{key: (key, "mean") for key in METRICS})
    summary.to_csv(output / "summary_by_dataset.tsv", sep="\t", index=False)
    macro = summary.groupby(["domain", "method"], as_index=False).agg(
        n_conditions=("dataset", "size"), **{key: (key, "mean") for key in METRICS})
    macro.to_csv(output / "macro_summary.tsv", sep="\t", index=False)
    paired_rows = []
    for (domain, dataset), group in combined.groupby(["domain", "dataset"]):
        for metric in METRICS:
            pivot = group.pivot(index="run_id", columns="method", values=metric)
            for method in METHODS[1:]:
                pair = pivot[["DiSCGRN", method]].dropna()
                delta = (pair["DiSCGRN"] - pair[method]).to_numpy()
                rng = np.random.default_rng(20261009 +
                                            sum(map(ord, domain + dataset + method + metric)))
                boot = rng.choice(delta, size=(5000, len(delta)), replace=True).mean(axis=1)
                paired_rows.append({"domain": domain, "dataset": dataset,
                                    "metric": metric, "comparison": f"DiSCGRN - {method}",
                                    "n_paired": len(delta), "mean_difference": delta.mean(),
                                    "ci_low": np.quantile(boot, 0.025),
                                    "ci_high": np.quantile(boot, 0.975)})
    pd.DataFrame(paired_rows).to_csv(output / "paired_differences.tsv", sep="\t", index=False)
    protocol = {"methods": list(METHODS), "metrics": list(METRICS),
                "selection_rule": f"top ceil({args.top_fraction} * n_directed_nonself_candidates)",
                "tie_rule": "stable descending score; matrix row-major off-diagonal order",
                "aupr": "sklearn average_precision_score over all directed non-self edges",
                "dream_stage_aggregation": "arithmetic mean of stage-specific metrics per system",
                "beeline_truth": "static directed GroundTruth.csv",
                "beeline_datasets": list(args.beeline_datasets),
                "direction": "matrix[target, source]",
                "note": "top-fraction is fixed independently of the gold-standard edge count"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"DREAM systems: {len(dream)//len(METHODS)}, BEELINE runs: "
          f"{(len(combined)-len(dream))//len(METHODS)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
