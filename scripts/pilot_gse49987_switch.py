#!/usr/bin/env python3
"""Exploratory zero/one-switch feasibility check on processed GSE49987.

This intentionally uses a fixed, provisional panel and regularization values.
It is NOT an independent real-data validation or a manuscript-ready result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ctcpgrn.baselines import scan_one_change_candidates, select_one_change
from ctcpgrn.io import LongitudinalGroup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/processed/gse49987_hematopoietic/mean_trajectories.tsv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/real_data_20261007/gse49987_pilot"))
    parser.add_argument("--alpha", type=float, default=0.003)
    parser.add_argument("--rho", type=float, default=0.30)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--max-time-h", type=float, default=72.0)
    parser.add_argument("--tf-only", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.input, sep="\t")
    genes = [column for column in data if column not in ("lineage", "time_h")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, curves = [], []
    for lineage in ("erythroid", "myeloid"):
        frame = data[data.lineage.eq(lineage) & data.time_h.le(args.max_time_h)].sort_values("time_h")
        times = frame.time_h.to_numpy(dtype=float) / 2.0
        values = frame[genes].to_numpy(dtype=float)
        means, scales = values.mean(axis=0), values.std(axis=0, ddof=1)
        if np.any(scales < 1e-8):
            raise ValueError(f"Nearly constant panel gene in {lineage}")
        values = (values - means) / scales
        group = LongitudinalGroup(lineage, times, values, np.ones_like(values),
                                  np.full(len(times), 3.0))
        allowed = None
        if args.tf_only:
            regulators = {"Gata1", "Gata2", "Spi1", "Tal1", "Klf1", "Cebpa",
                          "Cebpb", "Myb", "Runx1"}
            allowed = np.zeros((len(genes), len(genes)), dtype=bool)
            allowed[:, [i for i, gene in enumerate(genes) if gene in regulators]] = True
            np.fill_diagonal(allowed, True)
        for method in ("ct", "euler"):
            path = scan_one_change_candidates(
                [group], times, method, args.alpha, minimum_transitions=5,
                max_iter=args.max_iter, ct_initialiser="euler", allowed_edges=allowed,
            )
            chosen = select_one_change(path, args.rho, relative_beta=True)
            stationary = path[0].objective
            for item in path:
                curves.append({"lineage": lineage, "method": method,
                               "candidate_time_h": None if item.change_time is None
                               else 2.0 * item.change_time,
                               "raw_objective": item.objective,
                               "penalized_objective": item.objective +
                               (args.rho * stationary if item.change_time is not None else 0.0),
                               "all_fits_converged": all(fit.converged for fit in item.fits)})
            summaries.append({"lineage": lineage, "method": method,
                              "chosen_time_h": None if chosen.change_time is None
                              else 2.0 * chosen.change_time,
                              "stationary_cost": stationary,
                              "chosen_penalized_cost": chosen.objective,
                              "fit_convergence": [fit.converged for fit in chosen.fits],
                              "nonzero_edges": [int(np.count_nonzero(
                                  fit.matrix - np.diag(np.diag(fit.matrix))))
                                  for fit in chosen.fits]})
    pd.DataFrame(curves).to_csv(args.output_dir / "candidate_costs.tsv", sep="\t", index=False)
    (args.output_dir / "pilot_summary.json").write_text(json.dumps({
        "warning": "Exploratory only: full-series standardization and provisional alpha/rho",
        "genes": genes, "time_scale": "observed hours divided by 2",
        "alpha": args.alpha, "rho": args.rho, "max_iter": args.max_iter,
        "max_time_h": args.max_time_h, "tf_only": args.tf_only,
        "results": summaries,
    }, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
