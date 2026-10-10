"""The no-correlation-screen mode must retain every non-self TF-target pair."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_three_stage_1000_candidates.py"
SPEC = importlib.util.spec_from_file_location("prepare_three_stage_1000_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def test_all_tf_target_pairs_include_regulator_targets_and_no_self_edges():
    pairs = prep.all_tf_target_pairs(["TF1", "TF2"], ["TF1", "TF2", "Gene3"])
    assert len(pairs) == 3 * (2 * 3 - 2)
    assert not pairs.regulator.eq(pairs.target).any()
    for stage in ("prenatal", "early_postnatal", "late_postnatal"):
        observed = set(map(tuple, pairs.loc[pairs.stage.eq(stage), ["regulator", "target"]].to_numpy()))
        assert observed == {
            ("TF1", "TF2"), ("TF1", "Gene3"),
            ("TF2", "TF1"), ("TF2", "Gene3"),
        }
