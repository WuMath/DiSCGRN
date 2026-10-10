"""Small deterministic checks for the real-data regulator-centred case study."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_real_case_study import make_case_table, select_focus  # noqa: E402


def test_select_focus_uses_strongest_strain_not_external_evidence():
    edges = pd.DataFrame([
        {"method": "discgrn", "stage": 1, "regulator": "R", "strain": strain,
         "target": target, "edge_weight": weight}
        for strain, target, weight in [
            ("AJ", "A", 0.1), ("AJ", "B", 0.0),
            ("B6", "A", 0.0), ("B6", "B", 0.2),
            ("C3H", "A", -0.3), ("C3H", "B", 0.4),
        ]
    ])
    assert select_focus(edges, "R", 2) == ["B", "A"]


def test_case_table_preserves_strain_specific_weights():
    edges = pd.DataFrame([
        {"method": "discgrn", "stage": 1, "regulator": "R", "strain": strain,
         "target": "A", "edge_weight": weight}
        for strain, weight in [("AJ", 0.0), ("B6", -0.1), ("C3H", 0.3)]
    ])
    evidence = pd.DataFrame([{
        "regulator": "R", "target": "A", "rank": 5,
        "chip_lung_support": True, "chip_lung_score": 100,
        "mutant_minus_control_log2_fpkm1": -1.0, "welch_p": 0.03, "bh_q": 0.2,
    }])
    frequencies = pd.DataFrame([{
        "regulator": "R", "target": "A", "frequency_AJ": 0.0,
        "frequency_B6": 0.1, "frequency_C3H": 0.9,
    }])
    row = make_case_table(edges, evidence, "R", frequencies).iloc[0]
    assert row.weight_AJ == 0.0
    assert row.weight_B6 == -0.1
    assert row.weight_C3H == 0.3
    assert row.frequency_C3H == 0.9
