import numpy as np

from ctcpgrn.real_fit import Fit, Transitions, fit_euler, predict
from scripts.run_real_data_analysis import Candidate, select


def test_real_exact_prediction_uses_actual_interval():
    table = Transitions(np.array([0.0]), np.array([2.0]), np.array([2.0]),
                        np.array([[1.0]]), np.array([[0.0]]), np.ones((1, 1)))
    fit = Fit(np.array([[-1.0]]), np.array([0.0]), 0.0, 0.0, 0, True)
    assert np.allclose(predict(fit, table, "ct"), [[np.exp(-2)]], atol=1e-12)


def test_euler_allowed_sources_are_constrained_during_fit():
    x = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, -1.0], [4.0, 2.0]])
    y = x + np.column_stack([x[:, 0], x[:, 1]])
    table = Transitions(np.arange(4.0), np.arange(1.0, 5.0), np.ones(4),
                        x, y, np.ones_like(y))
    allowed = np.array([[True, False], [False, True]])
    fit = fit_euler(table, 0.001, allowed)
    assert fit.matrix[0, 1] == 0
    assert fit.matrix[1, 0] == 0


def test_two_stage_can_select_zero_changes_when_mean_improvement_is_small():
    path = [Candidate(None, {}, 1.0, 1.0, 10.0),
            Candidate(2.0, {}, 0.5, 0.5, 9.0)]
    assert select(path, 0.2, "two_stage").cp is None
    assert select(path, 0.05, "two_stage").cp == 2.0
