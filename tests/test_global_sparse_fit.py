import numpy as np
from scipy.linalg import expm

from ctcpgrn.global_sparse_fit import GlobalTransitions, fit_global_sparse_stage


def test_lbfgsb_global_stage_reaches_stationary_solution_on_identifiable_example():
    rng = np.random.default_rng(7)
    generator = np.array([[-0.4, 0.25], [-0.1, -0.3]])
    baseline = np.array([0.03, -0.02])
    initial = rng.normal(size=(80, 2))
    intervals = rng.uniform(0.1, 0.8, size=len(initial))
    observed = np.empty_like(initial)
    augmented = np.block([[generator, baseline[:, None]], [np.zeros((1, 3))]])
    for index, interval in enumerate(intervals):
        observed[index] = (expm(augmented * interval)
                           @ np.r_[initial[index], 1.0])[:2]
    transitions = GlobalTransitions(initial, observed, intervals,
                                    np.ones_like(initial))
    for optimizer, parameter_scale in (("lbfgsb", 1.0), ("lbfgsb", 10.0),
                                       ("active_lbfgsb", 1.0)):
        fit = fit_global_sparse_stage(
            transitions, np.array([0, 1]), np.array([1, 0]),
            penalty=1e-4, optimizer=optimizer, max_iter=300,
            kkt_tolerance=1e-3, parameter_scale=parameter_scale,
        )
        assert fit.converged
        assert fit.kkt_inf < 1e-3
        assert fit.data_loss < 1e-3


def test_elastic_net_objective_and_kkt_on_identifiable_example():
    rng = np.random.default_rng(11)
    generator = np.array([[-0.35, 0.2], [-0.15, -0.25]])
    baseline = np.array([0.02, -0.01])
    initial = rng.normal(size=(50, 2))
    intervals = rng.uniform(0.15, 0.6, size=len(initial))
    augmented = np.block([[generator, baseline[:, None]], [np.zeros((1, 3))]])
    observed = np.array([
        (expm(augmented * interval) @ np.r_[state, 1.0])[:2]
        for state, interval in zip(initial, intervals)
    ])
    transitions = GlobalTransitions(initial, observed, intervals, np.ones_like(initial))
    for optimizer in ("lbfgsb", "active_lbfgsb", "proximal"):
        fit = fit_global_sparse_stage(
            transitions, np.array([0, 1]), np.array([1, 0]),
            penalty=1e-4, ridge=0.02, optimizer=optimizer,
            max_iter=500, kkt_tolerance=1e-3,
        )
        assert fit.kkt_inf < 1e-3
        offdiag = fit.edge_values[:2]
        expected = fit.data_loss + 1e-4 * np.abs(offdiag).sum() + 0.01 * np.square(offdiag).sum()
        assert np.isclose(fit.objective, expected, rtol=1e-8, atol=1e-9)
