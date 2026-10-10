#!/usr/bin/env python3
"""Stationary GRN ranking on original DREAM4 size-10 time-series observations.

These are GeneNetWeaver's nonlinear challenge data, not trajectories generated
by this project's affine simulator. There is no change-point truth in these
files. Network 1 selects each method's alpha; networks 2--5 are untouched
topology-held-out evaluations. Time is scaled by 50 uniformly across networks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import fit_by_method
from ctcpgrn.io import LongitudinalGroup, read_dream_signed_network, sha256sum, signed_edge_list_to_matrix
from ctcpgrn.metrics import edge_ranking_metrics
from ctcpgrn.paths import results_root


ROOT = Path("data/raw/dream4")


def load_series(index: int) -> tuple[list[LongitudinalGroup], np.ndarray, str]:
    path = ROOT / "timeseries" / f"insilico_size10_{index}_timeseries.tsv"
    gold_path = ROOT / "goldstandards" / f"insilico_size10_{index}_goldstandard_signed.tsv"
    table = pd.read_csv(path, sep="\t")
    truth, genes = signed_edge_list_to_matrix(read_dream_signed_network(gold_path))
    if list(table.columns) != ["Time", *genes]:
        raise ValueError(f"Gene order or columns differ from gold standard: {path}")
    times = table["Time"].to_numpy(float)
    starts = np.r_[0, np.flatnonzero(np.diff(times) <= 0) + 1]
    ends = np.r_[starts[1:], len(times)]
    groups = []
    for replicate, (start, end) in enumerate(zip(starts, ends)):
        local = times[start:end] / 50.0
        values = table.iloc[start:end, 1:].to_numpy(float)
        if len(local) != 21 or np.any(np.diff(local) <= 0):
            raise ValueError(f"Expected a 21-point ordered trajectory: {path}, replicate {replicate}")
        groups.append(LongitudinalGroup(f"replicate_{replicate}", local, values,
                                        np.ones_like(values), np.ones(len(local))))
    if len(groups) != 5:
        raise ValueError(f"Expected five original DREAM4 trajectories: {path}")
    return groups, truth, sha256sum(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=results_root() / "original_dream4_timeseries")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument("--max-iter", type=int, default=80)
    args = parser.parse_args()
    if any(value <= 0 for value in args.alphas):
        raise ValueError("Every alpha must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups_dev, truth_dev, dev_sha = load_series(1)
    dev_rows = []
    for method in ("ct", "euler"):
        for alpha in args.alphas:
            fit = fit_by_method(groups_dev, method, alpha, None, args.max_iter,
                                ct_initialiser="euler")
            metrics = edge_ranking_metrics(truth_dev, fit.matrix)
            dev_rows.append({"network": 1, "method": method, "alpha": alpha,
                             "max_iter": args.max_iter, "converged": fit.converged,
                             **metrics})
            print(f"[development] {method} alpha={alpha:g} AUPR={metrics['aupr']:.4f}", flush=True)
    development = pd.DataFrame(dev_rows)
    development.to_csv(args.output_dir / "network1_alpha_development.tsv", sep="\t", index=False)
    selected = {}
    for method in ("ct", "euler"):
        candidates = development[development.method.eq(method)].sort_values(
            ["aupr", "alpha"], ascending=[False, True])
        selected[method] = float(candidates.iloc[0].alpha)
    (args.output_dir / "selected_alpha.json").write_text(json.dumps(selected, indent=2) + "\n")
    test_rows = []
    checksums = {"1": dev_sha}
    for index in (2, 3, 4, 5):
        groups, truth, checksums[str(index)] = load_series(index)
        for method in ("ct", "euler"):
            fit = fit_by_method(groups, method, selected[method], None, args.max_iter,
                                ct_initialiser="euler")
            metrics = edge_ranking_metrics(truth, fit.matrix)
            test_rows.append({"network": index, "method": method,
                              "alpha": selected[method], "max_iter": args.max_iter,
                              "converged": fit.converged, **metrics})
    test = pd.DataFrame(test_rows)
    test.to_csv(args.output_dir / "heldout_network_results.tsv", sep="\t", index=False)
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "source": "official DREAM4 challenge time-series files",
        "original_time_divisor": 50, "network1_role": "alpha development only",
        "networks2to5_role": "held-out evaluation", "alpha_grid": args.alphas,
        "max_iter": args.max_iter, "sha256_timeseries": checksums,
        "note": "stationary directed-edge ranking; no change-point benchmark",
    }, indent=2) + "\n")
    print(test.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
