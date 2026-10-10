import numpy as np
import pytest

from ctcpgrn.baselines import _segment_cost
from ctcpgrn.io import developmental_day
from ctcpgrn.metrics import change_edge_labels, change_point_metrics, edge_change_metrics
from ctcpgrn.model import FitResult
from ctcpgrn.paths import results_root
from ctcpgrn.simulation import exact_step


def test_developmental_day_is_monotone_around_birth():
    assert developmental_day("e19.5") < developmental_day("P0") < developmental_day("P1")


def test_exact_step_scalar_system():
    matrix = np.array([[-1.0]])
    result = exact_step(matrix, np.array([0.0]), np.array([1.0]), 1.0)
    assert np.allclose(result, np.exp(-1), atol=1e-10)


def test_change_point_matching_is_one_to_one():
    metrics = change_point_metrics([2.0, 8.0], [2.1, 2.2, 8.2], tolerance=0.5)
    assert metrics["cp_recall"] == 1.0
    assert np.isclose(metrics["cp_precision"], 2 / 3)


def test_change_edges_require_ten_percent_of_typical_initial_weight():
    before = np.zeros((3, 3))
    before[0, 1], before[1, 2] = 0.2, 0.4
    after = before.copy()
    after[0, 1] = 0.22  # Difference 0.02 < 10% of the median 0.3.
    after[1, 2] = 0.45  # Difference 0.05 > 0.03.
    after[2, 0] = 0.2   # A newly added edge also exceeds 0.03.
    labels = change_edge_labels(before, after)
    mask = ~np.eye(3, dtype=bool)
    expected = np.zeros((3, 3), dtype=int)
    expected[1, 2] = expected[2, 0] = 1
    assert np.array_equal(labels, expected[mask])


def test_segment_score_weights_mean_loss_and_alpha_penalty_by_transitions():
    fit = FitResult(np.zeros((1, 1)), np.zeros(1), data_loss=2.0,
                    objective=2.5, n_iterations=0, converged=True)
    assert _segment_cost(fit, n_transitions=7) == 17.5


def test_identical_network_is_not_penalised_twice_by_splitting():
    fit = FitResult(np.zeros((1, 1)), np.zeros(1), data_loss=2.0,
                    objective=2.5, n_iterations=0, converged=True)
    assert _segment_cost(fit, 10) == _segment_cost(fit, 4) + _segment_cost(fit, 6)


def test_change_edge_aupr_is_not_applicable_without_positive_truth():
    truth = np.zeros((3, 3))
    estimate = np.zeros((3, 3))
    estimate[0, 1] = 0.2
    result = edge_change_metrics(truth, truth.copy(), truth, estimate)
    assert np.isnan(result["change_aupr"])
    assert np.isnan(result["change_auroc"])


def test_separate_results_root(monkeypatch, tmp_path):
    monkeypatch.setenv("CTCPGRN_RESULTS_ROOT", str(tmp_path / "fresh-run"))
    assert results_root() == tmp_path / "fresh-run"
    monkeypatch.setenv("CTCPGRN_RESULTS_ROOT", "  ")
    with pytest.raises(ValueError):
        results_root()
