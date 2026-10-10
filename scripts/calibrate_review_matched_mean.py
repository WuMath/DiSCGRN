#!/usr/bin/env python3
"""Development-only calibration for DiSCGRN on standardized replicate means.

Uses the 16 external-method calibration systems, never the 60 e4test systems.
The selection score and rho grid follow the original calibration structure.
This is a post hoc input-matching analysis, not a replacement for the paper's
pre-frozen full-replicate method.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.experiment_params import RHO_GRID
from ctcpgrn.io import LongitudinalGroup
from ctcpgrn.paths import results_root
from run_optimized_benchmark import score_result


INPUT = results_root() / "revision_e6/input"
OUT = results_root() / "matched_mean"
ALPHAS = (0.0008,)  # Retain the mean-input development fit scale; recalibrate rho only.
RHOS = RHO_GRID


def standardized_group(system_id):
    frame = pd.read_csv(INPUT / f"{system_id}.tsv", sep="\t")
    times = frame.time.to_numpy(dtype=float)
    x = frame.drop(columns="time").to_numpy(dtype=float)
    sd = x.std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError(f"Constant gene in {system_id}")
    x = (x - x.mean(axis=0)) / sd
    return times, LongitudinalGroup("replicate_mean_zscore", times, x,
                                    np.ones_like(x), np.ones(len(times)))


def _job(item):
    system_id, t, changed, true_time, alpha = item
    times, group = standardized_group(system_id)
    path = scan_one_change_candidates(
        [group], times, "ct", alpha, minimum_transitions=5,
        max_iter=80, ct_initialiser="euler", ct_learning_rate=0.03,
    )
    with np.load(INPUT / f"{system_id}_truth.npz") as truth:
        a1, a2 = truth["A1"], truth["A2"]
    rows = []
    for rho in RHOS:
        selected = select_one_change(path, rho, relative_beta=True)
        metrics = score_result(selected, None if not changed else float(true_time), a1, a2)
        rows.append({"system_id": system_id, "T": t, "changed": changed,
                     "alpha": alpha, "rho": rho,
                     "estimated_index": selected.change_index, **metrics})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(INPUT / "manifest.tsv", sep="\t")
    manifest = manifest[manifest.split == "calibration"]
    jobs = [(r.system_id, int(r.T), bool(r.changed), r.true_cp_time, alpha)
            for r in manifest.itertuples() for alpha in ALPHAS]
    existing_path = OUT / "matched_mean_calibration_systems.tsv"
    existing = pd.read_csv(existing_path, sep="\t") if existing_path.exists() else pd.DataFrame()
    completed = {(str(r.system_id), round(float(r.alpha), 10))
                 for r in existing.itertuples()}
    jobs = [j for j in jobs if (j[0], round(j[4], 10)) not in completed]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [row for block in pool.map(_job, jobs) for row in block]
    frame = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    frame.to_csv(OUT / "matched_mean_calibration_systems.tsv", sep="\t", index=False)
    summary = []
    for (alpha, rho), group in frame.groupby(["alpha", "rho"]):
        changed, null = group[group.changed], group[~group.changed]
        f1 = changed.cp_f1.mean()
        delta_aupr = changed.change_aupr.mean()
        network_aupr = changed.network_aupr.mean()
        specificity = null.cp_f1.mean()
        summary.append({"alpha": alpha, "rho": rho, "H": 80,
                        "n_changed": len(changed), "n_null": len(null),
                        "cp_f1": f1, "change_aupr": delta_aupr,
                        "network_aupr": network_aupr, "specificity": specificity,
                        "selection_score": f1 + 0.35 * delta_aupr
                        + 0.20 * network_aupr + 0.50 * specificity})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "matched_mean_calibration_grid.tsv", sep="\t", index=False)
    selected = summary.sort_values(["selection_score", "network_aupr", "alpha", "rho"],
                                   ascending=[False, False, True, True]).iloc[[0]]
    selected.to_csv(OUT / "matched_mean_selected.tsv", sep="\t", index=False)
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
