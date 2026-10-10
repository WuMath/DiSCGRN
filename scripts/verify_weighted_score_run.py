#!/usr/bin/env python3
"""Check that a new weighted-score batch is complete and isolated."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from ctcpgrn.experiment_params import FIT_SPECS, SCORE_RULE, load_frozen_params
from ctcpgrn.paths import results_root


REQUIRED = {
    "weighted_score_calibration/rho_grid.tsv": 4 * 19,
    "weighted_score_calibration/rho_systems.tsv": 4 * 6 * 19,
    "weighted_score_calibration/selected_parameters.tsv": 4,
    "weighted_score_calibration/weight_sensitivity.tsv": 4 * 81,
    "revision_comparison_h80/comparison_30systems.tsv": 4 * 30,
    "revision_e1/e1_all_systems.tsv": 20 * 30 * 2,
    "revision_e4_single/e4_single_all.tsv": 30 * 15 * 2,
    "revision_e5/blocked_cv_15systems_system_means.tsv": 15 * 3,
    "revision_e6/e6_all_systems.tsv": 60 * 4,
}
FIGURES = (
    "e1_kappa_scan.pdf", "e3_regularisation_path.pdf",
    "e4_power.pdf", "e6_external_baselines.pdf",
)


def main() -> None:
    root = results_root()
    if "CTCPGRN_RESULTS_ROOT" not in os.environ or root == Path("results").resolve():
        raise RuntimeError("Set CTCPGRN_RESULTS_ROOT to the isolated weighted-score run")
    frozen = load_frozen_params()
    checks = []
    for relative, expected in REQUIRED.items():
        path = root / relative
        observed = len(pd.read_csv(path, sep="\t")) if path.is_file() else None
        checks.append({"file": relative, "expected_rows": expected,
                       "observed_rows": observed, "ok": observed == expected})
    for figure in FIGURES:
        path = root / "revision_figures" / figure
        checks.append({"file": f"revision_figures/{figure}",
                       "expected_rows": None, "observed_rows": None,
                       "ok": path.is_file() and path.stat().st_size > 0})
    report = {
        "score_rule": SCORE_RULE,
        "results_root": str(root),
        "frozen_parameters": {key: {"alpha": alpha, "rho": rho, "H": budget}
                              for key, (alpha, rho, budget) in frozen.items()},
        "expected_methods": list(FIT_SPECS),
        "checks": checks,
        "complete": all(check["ok"] for check in checks),
    }
    target = root / "weighted_score_verification.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["complete"]:
        raise SystemExit("Batch incomplete: inspect the failed checks above")


if __name__ == "__main__":
    main()
