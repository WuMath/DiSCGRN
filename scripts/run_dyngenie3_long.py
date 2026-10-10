#!/usr/bin/env python3
"""Run the authors' dynGENIE3 Python code on the 60 frozen longitudinal systems.

The method produces one network for a whole series. Its directed scores are
evaluated against both true stages, with the same two-stage mean AP used for
the existing whole-series GENIE3 and GRNBoost2 comparisons.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def load_official(source: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("official_dynGENIE3", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dynGENIE3


def stage_ap(truth: np.ndarray, scores: np.ndarray) -> float:
    mask = ~np.eye(len(truth), dtype=bool)
    y = (truth[mask] != 0).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Stage truth has only one edge class")
    return float(average_precision_score(y, scores[mask]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Official vahuynh/dynGENIE3 Python source dynGENIE3.py")
    parser.add_argument("--input-root", type=Path,
                        default=Path("results/runs/weighted_score_20261004/tree_grn_long"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/runs/weighted_score_20261004/dyngenie3_long"))
    parser.add_argument("--limit", type=int, default=0, help="First N systems for smoke test")
    args = parser.parse_args()
    inference = load_official(args.source)
    manifest = pd.read_csv(args.input_root / "input_manifest.tsv", sep="\t")
    if len(manifest) != 60:
        raise ValueError("Expected the frozen inventory of 60 long systems")
    if args.limit:
        manifest = manifest.head(args.limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {"method": "dynGENIE3 official Python implementation",
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "source_url": "https://github.com/vahuynh/dynGENIE3",
                "tree_method": "RF", "K": "sqrt", "ntrees": 1000,
                "alpha": "from_data", "nthreads": 1,
                "input": "all five unaveraged trajectories and actual observation times",
                "output": "single directed whole-series network; static score evaluated on each true stage"}
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output has a different dynGENIE3 protocol")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for item in manifest.itertuples(index=False):
        seed = int(item.seed)
        metric_path = args.output_root / f"system_{seed}_metrics.json"
        if metric_path.exists():
            record = json.loads(metric_path.read_text())
        else:
            data = np.load(args.input_root / f"system_{seed}.npz")
            observations = data["observations"]
            times = data["times"]
            np.random.seed(seed)
            start = time.perf_counter()
            log = io.StringIO()
            record = {"seed": seed, "T": int(item.T), "changed": bool(item.changed)}
            try:
                with contextlib.redirect_stdout(log):
                    vim = inference([trajectory.copy() for trajectory in observations],
                                    [times.copy() for _ in observations],
                                    alpha="from_data", tree_method="RF", K="sqrt",
                                    ntrees=1000, nthreads=1)[0]
                scores = np.asarray(vim, dtype=float).T  # source x target -> target x source
                if scores.shape != data["first"].shape or not np.isfinite(scores).all():
                    raise ValueError("Nonfinite or wrong-sized dynGENIE3 score matrix")
                np.fill_diagonal(scores, 0.0)
                ap1, ap2 = stage_ap(data["first"], scores), stage_ap(data["second"], scores)
                np.savez_compressed(args.output_root / f"system_{seed}_scores.npz", scores=scores)
                record.update({"status": "ok", "stage1_ap": ap1, "stage2_ap": ap2,
                               "network_aupr": (ap1 + ap2) / 2})
            except Exception as exc:
                record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            record["runtime_seconds"] = time.perf_counter() - start
            (args.output_root / f"system_{seed}_stdout.txt").write_text(log.getvalue())
            metric_path.write_text(json.dumps(record, indent=2) + "\n")
        print(seed, record["status"], record.get("network_aupr", record.get("error")), flush=True)
        rows.append(record)
        pd.DataFrame(rows).to_csv(args.output_root / "system_aupr.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
