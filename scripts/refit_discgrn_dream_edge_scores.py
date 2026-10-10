#!/usr/bin/env python3
"""Refit frozen DiSCGRN DREAM systems and preserve both inferred stage matrices."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.experiment_params import load_frozen_params
from ctcpgrn.io import LongitudinalGroup


PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "results/runs/weighted_score_20261004/tree_grn_long"
HISTORICAL = PROJECT / "results/runs/weighted_score_20261004/revision_e6/e6_all_systems.tsv"
OUTPUT = PROJECT / "results/runs/grn_accuracy_20261009/discgrn_dream_refit"


def stage_ap(truth: np.ndarray, estimate: np.ndarray) -> float:
    mask = ~np.eye(len(truth), dtype=bool)
    return float(average_precision_score((truth[mask] != 0).astype(int),
                                         np.abs(estimate[mask])))


def fit_one(item: tuple[int, int, bool], output: Path, alpha: float,
            rho: float, budget: int, expected: float | None) -> dict:
    seed, n_times, changed = item
    stem = f"system_{seed}"
    result_path = output / f"{stem}_fit.npz"
    metric_path = output / f"{stem}_metrics.json"
    if metric_path.exists() and result_path.exists():
        return json.loads(metric_path.read_text())
    data = np.load(INPUT / f"{stem}.npz")
    values, times = data["observations"], data["times"]
    groups = [LongitudinalGroup(f"replicate_{i}", times.copy(), trajectory.copy(),
                                np.ones_like(trajectory), np.ones(len(times)))
              for i, trajectory in enumerate(values)]
    start = time.perf_counter()
    path = scan_one_change_candidates(groups, times, "ct", alpha,
                                      minimum_transitions=5, max_iter=budget,
                                      ct_initialiser="euler")
    result = select_one_change(path, rho, relative_beta=True)
    if len(result.fits) == 1:
        before = after = result.fits[0].matrix
    else:
        before, after = result.fits[0].matrix, result.fits[1].matrix
    ap = (stage_ap(data["first"], before) + stage_ap(data["second"], after)) / 2
    record = {"seed": seed, "T": n_times, "changed": changed, "status": "ok",
              "selected_change_index": result.change_index,
              "aupr_ap": ap, "historical_aupr_ap": expected,
              "aupr_absolute_difference": None if expected is None else abs(ap - expected),
              "runtime_seconds": time.perf_counter() - start}
    np.savez_compressed(result_path, before=before, after=after)
    metric_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    frozen_params = PROJECT / "results/runs/weighted_score_20261004/weighted_score_calibration/selected_parameters.tsv"
    params = load_frozen_params(frozen_params)["ct"]
    manifest = pd.read_csv(INPUT / "input_manifest.tsv", sep="\t")
    frozen = None
    if HISTORICAL.is_file():
        frozen = pd.read_csv(HISTORICAL, sep="\t")
        frozen = frozen[frozen.method == "DiSCGRN"].set_index("seed")
    jobs = []
    for item in manifest.itertuples(index=False):
        seed = int(item.seed)
        expected = None if frozen is None else float(frozen.loc[seed, "network_aupr"])
        jobs.append(((seed, int(item.T), bool(item.changed)), expected))
    if args.limit:
        jobs = jobs[:args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {"input": str(INPUT), "historical_check": str(HISTORICAL),
                "alpha": params[0], "rho": params[1], "max_iter": params[2],
                "minimum_transitions": 5, "initialiser": "euler",
                "estimator": "ct", "n_replicates": 5}
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fit_one, item, args.output_root, *params, expected): item
                   for item, expected in jobs}
        for future in as_completed(futures):
            record = future.result()
            rows.append(record)
            delta = record["aupr_absolute_difference"]
            print(record["seed"], f"AP={record['aupr_ap']:.6f}",
                  "historical_delta=NA" if delta is None else f"historical_delta={delta:.3g}",
                  flush=True)
            pd.DataFrame(rows).sort_values("seed").to_csv(
                args.output_root / "all_metrics.tsv", sep="\t", index=False)
    if frozen is not None and len(rows) == len(manifest) and max(row["aupr_absolute_difference"] for row in rows) > 1e-8:
        raise ValueError("Refitted network AP diverges from frozen historical result")


if __name__ == "__main__":
    main()
