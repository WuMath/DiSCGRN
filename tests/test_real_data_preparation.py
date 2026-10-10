"""Checks for leakage-safe real-data preparation and evidence semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_real_data_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_real_data_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def test_panel_selection_ignores_heldout_values():
    expression = pd.DataFrame({
        "Gene ID": ["g1", "g2", "g3"],
        "Gene Symbol": ["Nkx2-1", "GeneA", "GeneB"],
        "train_a": [1.0, 1.0, 1.0],
        "train_b": [2.0, 5.0, 2.0],
        "test": [100.0, 0.0, 1000.0],
    })
    panel = prep.choose_panel(expression, ["train_a", "train_b"], {"Nkx2-1"}, 2, 1)
    assert panel["Gene Symbol"].tolist() == ["Nkx2-1", "GeneA"]
    assert panel["pre_specified_anchor"].tolist() == [True, False]


def test_unanchored_panel_selects_regulators_from_training_variance():
    expression = pd.DataFrame({
        "Gene ID": ["g1", "g2", "g3"],
        "Gene Symbol": ["Nkx2-1", "RegB", "TargetA"],
        "train_a": [1.0, 1.0, 1.0],
        "train_b": [2.0, 5.0, 3.0],
        "test": [1000.0, 0.0, 0.0],
    })
    panel = prep.choose_panel(expression, ["train_a", "train_b"],
                              {"Nkx2-1", "RegB"}, 2, 1, ())
    assert panel["Gene Symbol"].tolist() == ["RegB", "TargetA"]
    assert not panel["pre_specified_anchor"].any()


def test_blocked_split_embargoes_transition_leaving_holdout():
    samples = [f"{strain}_{time}" for strain in prep.STRAINS for time in range(5)]
    metadata = pd.DataFrame({
        "sample": samples,
        "strain": [strain for strain in prep.STRAINS for _ in range(5)],
        "time": [float(time) for _ in prep.STRAINS for time in range(5)],
    })
    expression = pd.DataFrame({sample: [float(time)] for sample, time in zip(samples, metadata["time"])}, index=["Nkx2-1"])
    groups, splits = prep.groups_and_splits(expression, metadata, {2.0, 3.0})
    assert len(groups) == 3
    assert splits[splits["strain"].eq("AJ")]["split"].tolist() == ["train", "test", "test", "embargo"]


def test_bh_adjustment_and_evidence_unknowns():
    assert np.allclose(prep.bh_adjust(np.array([0.01, 0.04, 0.03])), [0.03, 0.04, 0.04])
    panel = pd.DataFrame({"Gene Symbol": ["Nkx2-1", "TargetA", "TargetB"],
                          "is_candidate_regulator": [True, False, False]})
    collectri = pd.DataFrame({"regulator": ["Nkx2-1"], "target": ["TargetA"], "sign": [1]})
    chip = pd.DataFrame({"regulator": ["Nkx2-1"], "target": ["TargetA"],
                         "average_score": [1.0], "lung_score": [2.0],
                         "has_lung_experiment": [True], "cross_tissue_support": [True],
                         "lung_support": [True]})
    evidence = prep.prepare_evidence(panel, collectri, chip).set_index("target")
    assert bool(evidence.loc["TargetA", "collectri_reported"])
    assert not bool(evidence.loc["TargetB", "collectri_reported"])
    assert "is_negative" not in evidence.columns


def test_model_bundle_loads_without_pickle(tmp_path):
    samples = [f"{strain}_{time}" for strain in prep.STRAINS for time in range(5)]
    metadata = pd.DataFrame({
        "sample": samples,
        "strain": [strain for strain in prep.STRAINS for _ in range(5)],
        "time": [float(time) for _ in prep.STRAINS for time in range(5)],
    })
    expression = pd.DataFrame({
        "Gene ID": ["g1", "g2"], "Gene Symbol": ["Nkx2-1", "TargetA"],
        **{sample: [float(time), float(time * 2 + 1)]
           for sample, time in zip(samples, metadata["time"])},
    })
    prep.write_model_bundle(tmp_path, expression, metadata, {"Nkx2-1"}, 2, 1, set())
    with np.load(tmp_path / "trajectories.npz", allow_pickle=False) as archive:
        assert archive["genes"].tolist() == ["Nkx2-1", "TargetA"]
        assert archive["AJ_values"].shape == (5, 2)
