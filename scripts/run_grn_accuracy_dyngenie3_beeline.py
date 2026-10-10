#!/usr/bin/env python3
"""Run official dynGENIE3 on the frozen 13-bin BEELINE pseudotrajectories."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from run_dyngenie3_long import load_official


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "third_party/dynGENIE3/dynGENIE3.py"
INPUT = PROJECT / "results/beeline_extended_valid_20261009"
OUTPUT = PROJECT / "results/runs/grn_accuracy_20261009/dyngenie3_beeline"


def score_network(scores: np.ndarray, genes: list[str], truth: pd.DataFrame) -> dict:
    position = {gene: i for i, gene in enumerate(genes)}
    labels = np.zeros_like(scores, dtype=int)
    for source, target in truth[["from", "to"]].itertuples(index=False, name=None):
        source, target = str(source), str(target)
        if source in position and target in position and source != target:
            labels[position[target], position[source]] = 1
    mask = ~np.eye(len(genes), dtype=bool)
    y, q = labels[mask], scores[mask]
    if y.min() == y.max() or not np.isfinite(q).all():
        raise ValueError("Invalid reference classes or nonfinite scores")
    n_positive = int(y.sum())
    precision, recall, _ = precision_recall_curve(y, q)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(precision), where=denominator > 0)
    return {"aupr_ap": float(average_precision_score(y, q)),
            "auroc": float(roc_auc_score(y, q)),
            "precision_at_k": float(y[np.argsort(-q, kind="stable")[:n_positive]].mean()),
            "max_f1": float(f1.max()),
            "k_true_edges": n_positive, "n_tested_edges": int(y.size),
            "positive_prevalence": float(y.mean()),
            "aupr_lift": float(average_precision_score(y, q) / y.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--input-root", type=Path, default=INPUT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--ntrees", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    inference = load_official(args.source)
    jobs = sorted(args.input_root.glob("dyn-*/dyn-*-*/binned_input.csv"))
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        raise FileNotFoundError("No frozen BEELINE binned inputs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {"method": "official dynGENIE3 Python implementation",
                "source_url": "https://github.com/vahuynh/dynGENIE3",
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "input": "one frozen 13-bin pseudotime trajectory per run",
                "tree_method": "RF", "K": "sqrt", "ntrees": args.ntrees,
                "alpha": "from_data", "nthreads": 1,
                "score": "directed non-self edge AP against BEELINE static gold"}
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output has a different protocol")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    for path in jobs:
        label, run_id = path.parent.parent.name, path.parent.name
        target = args.output_root / label / run_id
        target.mkdir(parents=True, exist_ok=True)
        metric_path = target / "dynGENIE3_metrics.json"
        if metric_path.exists():
            record = json.loads(metric_path.read_text())
        else:
            record = {"dataset": label, "run_id": run_id, "status": "failed"}
            log = io.StringIO()
            start = time.perf_counter()
            try:
                frame = pd.read_csv(path, index_col=0)
                genes = frame.columns.astype(str).tolist()
                network = label.rsplit("-", 1)[0]
                truth = pd.read_csv(PROJECT / "data/raw/beeline/Synthetic" /
                                    network / run_id / "GroundTruth.csv")
                seed = 20261009 + int(run_id.rsplit("-", 1)[-1]) + sum(map(ord, label))
                np.random.seed(seed)
                with contextlib.redirect_stdout(log):
                    vim = inference([frame.to_numpy(float, copy=True)],
                                    [frame.index.to_numpy(float, copy=True)],
                                    alpha="from_data", tree_method="RF", K="sqrt",
                                    ntrees=args.ntrees, nthreads=1)[0]
                scores = np.asarray(vim, float).T
                if scores.shape != (len(genes), len(genes)):
                    raise ValueError("Unexpected network matrix dimensions")
                np.fill_diagonal(scores, 0.0)
                metrics = score_network(scores, genes, truth)
                pd.DataFrame(scores, index=genes, columns=genes).to_csv(
                    target / "dynGENIE3_scores.csv")
                record.update(metrics)
                record["status"] = "ok"
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            record["runtime_seconds"] = time.perf_counter() - start
            (target / "dynGENIE3_stdout.txt").write_text(log.getvalue())
            metric_path.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(run_id, record["status"], record.get("aupr_ap", record.get("error")), flush=True)
    pd.DataFrame(records).to_csv(args.output_root / "all_metrics.tsv", sep="\t", index=False)
    print(f"Completed {sum(r['status'] == 'ok' for r in records)}/{len(records)} jobs")


if __name__ == "__main__":
    main()
