#!/usr/bin/env python3
"""Run frozen DiSCGRN on exactly the standardized mean series used by E6 R methods."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.io import LongitudinalGroup
from ctcpgrn.metrics import edge_ranking_metrics
from ctcpgrn.paths import results_root


ROOT = results_root() / "revision_e6"
INPUT = ROOT / "input"
OUT = results_root() / "matched_mean"


def _job(row):
    system_id, t, changed, true_index, alpha, rho = row
    frame = pd.read_csv(INPUT / f"{system_id}.tsv", sep="\t")
    times = frame.time.to_numpy(dtype=float)
    x = frame.drop(columns="time").to_numpy(dtype=float)
    # R's scale() centers each gene and uses sample standard deviation.
    sd = x.std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError(f"Constant gene in {system_id}")
    x = (x - x.mean(axis=0)) / sd
    group = LongitudinalGroup("replicate_mean_zscore", times, x,
                              np.ones_like(x), np.ones(len(times)))
    path = scan_one_change_candidates(
        [group], times, "ct", alpha, minimum_transitions=5,
        max_iter=80, ct_initialiser="euler", ct_learning_rate=0.03,
    )
    selected = select_one_change(path, rho, relative_beta=True)
    with np.load(INPUT / f"{system_id}_truth.npz") as truth:
        a1, a2 = truth["A1"], truth["A2"]
    if selected.change_index is None:
        estimated = (selected.fits[0].matrix, selected.fits[0].matrix)
    else:
        estimated = (selected.fits[0].matrix, selected.fits[1].matrix)
    network_aupr = np.mean([
        edge_ranking_metrics(true, estimate)["aupr"]
        for true, estimate in zip((a1, a2), estimated)
    ])
    return {
        "system_id": system_id, "T": t, "changed": changed,
        "true_cp_index": true_index, "estimated_cp_index": selected.change_index,
        "network_aupr": network_aupr,
        "input": "replicate_mean_gene_zscore", "alpha": alpha,
        "rho": rho, "H": 80,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated", action="store_true",
                        help="Use parameters selected on 16 calibration systems")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.calibrated:
        raise ValueError("Use --calibrated after recalibrating rho on mean-input development systems")
    selected = pd.read_csv(OUT / "matched_mean_selected.tsv", sep="\t").iloc[0]
    alpha, rho = float(selected.alpha), float(selected.rho)
    stem = "matched_mean_calibrated"
    manifest = pd.read_csv(INPUT / "manifest.tsv", sep="\t")
    manifest = manifest[manifest.split == "e4test"]
    jobs = [(r.system_id, int(r.T), bool(r.changed), r.true_cp_index, alpha, rho)
            for r in manifest.itertuples()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(_job, jobs))
    mean = pd.DataFrame(rows)
    mean.to_csv(OUT / f"{stem}_systems.tsv", sep="\t", index=False)

    original = pd.read_csv(ROOT / "e6_all_systems.tsv", sep="\t")
    rng = np.random.default_rng(20261003)
    comparisons = []
    for t in (25, 41):
        for comparator in ("EDISON", "VARDetect"):
            comp = original[(original["T"] == t) & (original.method == comparator)]
            pair = mean[mean["T"] == t].merge(
                comp[["system_id", "network_aupr"]], on="system_id",
                suffixes=("_mean", "_comparator"), validate="one_to_one",
            ).dropna(subset=["network_aupr_mean", "network_aupr_comparator"])
            delta = (pair.network_aupr_mean - pair.network_aupr_comparator).to_numpy()
            draws = delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(axis=1)
            comparisons.append({
                "T": t, "comparator": comparator, "n_evaluable": len(delta),
                "n_planned": 30, "mean_ct_mean_aupr": pair.network_aupr_mean.mean(),
                "mean_comparator_aupr": pair.network_aupr_comparator.mean(),
                "mean_difference": delta.mean(),
                "ci_low": np.quantile(draws, 0.025),
                "ci_high": np.quantile(draws, 0.975),
            })
    summary = pd.DataFrame(comparisons)
    summary.to_csv(OUT / f"{stem}_paired.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
