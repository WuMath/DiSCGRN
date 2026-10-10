"""Regression tests for the cumulative stage score and frozen rho file."""

from __future__ import annotations

import pandas as pd
import pytest

from ctcpgrn.baselines import ScanResult, select_one_change
from ctcpgrn.experiment_params import FIT_SPECS, SCORE_RULE, load_frozen_params


def test_relative_change_penalty_requires_prespecified_cost_reduction():
    null = ScanResult(None, None, [], 100.0)
    split = ScanResult(6, 6.0, [], 75.0)
    assert select_one_change([null, split], 0.30, relative_beta=True).change_index is None
    assert select_one_change([null, split], 0.20, relative_beta=True).change_index == 6


def test_frozen_parameters_reject_wrong_score_rule(tmp_path):
    path = tmp_path / "selected.tsv"
    rows = [{"method": method, "alpha": alpha, "beta_ratio": 0.3,
             "max_iter": budget, "score_rule": SCORE_RULE}
            for method, (alpha, budget) in FIT_SPECS.items()]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    assert set(load_frozen_params(path)) == set(FIT_SPECS)
    rows[0]["score_rule"] = "legacy_once_per_network"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="does not use"):
        load_frozen_params(path)
