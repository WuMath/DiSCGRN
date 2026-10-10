#!/usr/bin/env python3
"""Post-fit segmentation-penalty sensitivity at frozen DiSCGRN settings.

All candidate networks are fit once with alpha=0.0004, H=80. Only the
post-fit score's network penalty changes; rho=0.30 stays fixed. The ratios
are prespecified diagnostics, not settings selected from these test systems.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates
from run_optimized_benchmark import make_dataset, score_result, trajectory_groups


GOLD = Path("data/raw/dream4/DREAM4 in-silico challenge/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1_goldstandard_signed.tsv")
OUT = Path("results/runs/review_response_20261003_01/a1")
ALPHA = 0.0004
RHO = 0.30
RATIOS = (0.5, 1.0, 2.0)


def _score(candidate, n_transitions: int, lam: float) -> float:
    if candidate.change_index is None:
        counts = (5 * n_transitions,)
    else:
        counts = (5 * candidate.change_index, 5 * (n_transitions - candidate.change_index))
    result = 0.0
    for fit, count in zip(candidate.fits, counts):
        offdiag = fit.matrix[~np.eye(fit.matrix.shape[0], dtype=bool)]
        result += count * fit.data_loss + lam * np.abs(offdiag).sum()
    return float(result)


def _job(item):
    seed, changed = item
    _, first, second, index, sim = make_dataset(GOLD, seed, changed)
    path = scan_one_change_candidates(
        trajectory_groups(sim), sim.times, "ct", ALPHA,
        minimum_transitions=5, max_iter=80, ct_initialiser="euler",
        ct_learning_rate=0.03,
    )
    rows = []
    for ratio in RATIOS:
        lam = ALPHA * ratio
        null_cost = _score(path[0], len(sim.times) - 1, lam)
        selected = min(
            path,
            key=lambda c: _score(c, len(sim.times) - 1, lam)
            + (0.0 if c.change_index is None else RHO * null_cost),
        )
        truth_time = None if index is None else float(sim.times[index])
        metrics = score_result(selected, truth_time, first, second)
        rows.append({
            "seed": seed, "changed": changed, "lambda_ratio": ratio,
            "lambda_seg": lam, "alpha_fit": ALPHA, "rho": RHO,
            "selected_index": selected.change_index,
            "selected_time": selected.change_time,
            "true_index": index, "true_time": truth_time,
            **metrics,
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, changed)
            for changed, seeds in ((True, range(99601, 99616)),
                                   (False, range(99701, 99716)))
            for seed in seeds]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [row for block in pool.map(_job, jobs) for row in block]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "a1_systems.tsv", sep="\t", index=False)
    summary = []
    for ratio, group in frame.groupby("lambda_ratio"):
        changed, null = group[group.changed], group[~group.changed]
        summary.append({
            "lambda_ratio": ratio, "n_changed": len(changed), "n_null": len(null),
            "stage_network_aupr": changed.network_aupr.mean(),
            "change_aupr": changed.change_aupr.mean(),
            "within_tolerance_rate": changed.cp_f1.mean(),
            "exact_rate": (changed.selected_index == changed.true_index).mean(),
            "specificity": (null.selected_index.isna()).mean(),
        })
    pd.DataFrame(summary).to_csv(OUT / "a1_summary.tsv", sep="\t", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
