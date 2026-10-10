"""Frozen fit settings and development-only change-penalty calibration.

The alpha values and iteration budget are held fixed while the cumulative
segment-score revision is assessed. Only rho is reselected on development
systems; formal experiments must read the resulting file, never test metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import results_root


SCORE_RULE = "transition_weighted_alpha_v2"
FIT_SPECS: dict[str, tuple[float, int]] = {
    "ct": (0.0004, 80),
    "euler": (0.0005, 80),
    "dplsq": (0.0010, 80),
    "two_stage": (0.0010, 80),
}
RHO_GRID = tuple(round(0.05 * index, 2) for index in range(19))


def selected_path() -> Path:
    return results_root() / "weighted_score_calibration" / "selected_parameters.tsv"


def load_frozen_params(path: Path | None = None) -> dict[str, tuple[float, float, int]]:
    """Return (alpha, rho, H), rejecting absent or mismatched calibrations."""
    path = selected_path() if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Run scripts/calibrate_weighted_score.py before formal tests: {path}"
        )
    frame = pd.read_csv(path, sep="\t")
    required = {"method", "alpha", "beta_ratio", "max_iter", "score_rule"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Incomplete calibration file: {path}")
    if set(frame["method"]) != set(FIT_SPECS) or frame["method"].duplicated().any():
        raise ValueError(f"Expected exactly one row for each method in {path}")
    if not frame["score_rule"].eq(SCORE_RULE).all():
        raise ValueError(f"Calibration file does not use {SCORE_RULE}: {path}")
    result = {}
    for row in frame.itertuples(index=False):
        alpha, budget = FIT_SPECS[row.method]
        if not np.isclose(row.alpha, alpha, rtol=0, atol=1e-12) or int(row.max_iter) != budget:
            raise ValueError(f"Fit settings for {row.method} differ from the frozen protocol")
        rho = float(row.beta_ratio)
        if not 0 <= rho <= 0.9:
            raise ValueError(f"Invalid rho for {row.method}: {rho}")
        result[row.method] = (alpha, rho, budget)
    return result
