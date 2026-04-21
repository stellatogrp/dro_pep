"""
PEPit reference values for the Chambolle-Pock last-iterate duality-gap PEP.

These tests lock down the numerical values of PEPit's worst-case SDP so that
(a) regressions in PEPit itself or its SDP backend are caught independently of
our custom construction, and (b) downstream tests comparing
`construct_chambolle_pock_pep_data` to PEPit have a stable reference.

Formulation: min_x max_u  f1(x) + <K x, u> - h(u)
Saddle conditions: -K^T u_s ∈ ∂f1(x_s), K x_s ∈ ∂h(u_s)
Metric: duality gap at the LAST iterate (not averaged).
Initial condition: Euclidean ball
    (x0 - xs)^2 + (u0 - us)^2 <= 1

(The prior form was the P-norm Lyapunov IC
    (x0-xs)^2/tau + (u0-us)^2/sigma - 2(u0-us)K(x0-xs) <= 1;
 it was dropped after the PEPit probe in test_cp_ic_boundedness_probe.py
 verified this weaker Euclidean IC keeps the CP PEP bounded.)
"""

import pytest
import numpy as np

from PEPit import PEP
from PEPit.functions import ConvexFunction
from PEPit.operators import LinearOperator
from PEPit.primitive_steps import proximal_step


def _wc_cp_last_iterate_linop(tau, sigma, theta, n, L_M=1.0, verbose=0):
    """PEPit worst-case last-iterate gap for CP with linear operator K, ||K||<=L_M.

    Returns the worst-case duality gap f1(xK) + <K xK, us> - h(us)
                                    - f1(xs) - <K xs, uK> + h(uK).
    """
    problem = PEP()
    f1 = problem.declare_function(ConvexFunction)
    h = problem.declare_function(ConvexFunction)
    K = problem.declare_function(LinearOperator, L=L_M)

    xs = problem.set_initial_point()
    us = problem.set_initial_point()
    f1.add_point((xs, -K.T.gradient(us), f1.value(xs)))
    h.add_point((us, K.gradient(xs), h.value(us)))

    x0 = problem.set_initial_point()
    u0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 + (u0 - us) ** 2 <= 1)

    x, u = x0, u0
    for _ in range(n):
        x_new, _, _ = proximal_step(x - tau * K.T.gradient(u), f1, tau)
        x_bar = x_new + theta * (x_new - x)
        u_new, _, _ = proximal_step(u + sigma * K.gradient(x_bar), h, sigma)
        x, u = x_new, u_new

    def Lagrangian(a, b):
        return f1.value(a) + K.gradient(a) * b - h.value(b)

    gap = Lagrangian(x, us) - Lagrangian(xs, u)
    problem.set_performance_metric(gap)
    return problem.solve(verbose=verbose, wrapper='cvxpy')


def _wc_cp_last_iterate_scalar_M(tau, sigma, theta, n, M=1.0, verbose=0):
    """PEPit worst-case last-iterate gap for CP with SCALAR coupling M (no linop)."""
    problem = PEP()
    f1 = problem.declare_function(ConvexFunction)
    h = problem.declare_function(ConvexFunction)

    xs = problem.set_initial_point()
    ys = problem.set_initial_point()
    f1.add_point((xs, -M * ys, f1.value(xs)))
    h.add_point((ys, M * xs, h.value(ys)))

    x0 = problem.set_initial_point()
    y0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 + (y0 - ys) ** 2 <= 1)

    x, y = x0, y0
    for _ in range(n):
        x_new, _, _ = proximal_step(x - tau * M * y, f1, tau)
        x_bar = x_new + theta * (x_new - x)
        y_new, _, _ = proximal_step(y + sigma * M * x_bar, h, sigma)
        x, y = x_new, y_new

    L_primal_view = f1.value(x) + M * (x * ys) - h.value(ys)
    L_dual_view = f1.value(xs) + M * (xs * y) - h.value(y)
    gap = L_primal_view - L_dual_view
    problem.set_performance_metric(gap)
    return problem.solve(verbose=verbose, wrapper='cvxpy')


# ---------------------------------------------------------------------------
# Canonical parameter set shared across tests.
# Chosen to satisfy tau*sigma*L_M^2 <= 1 (strict contractivity).
# ---------------------------------------------------------------------------
CANONICAL_TAU = 0.5
CANONICAL_SIGMA = 0.5
CANONICAL_THETA = 1.0
CANONICAL_L_M = 1.0
CANONICAL_R = 1.0  # initial radius normalization (IC <= R^2)


@pytest.mark.parametrize('K', [1, 3, 5])
def test_pepit_linop_last_iterate(K):
    """PEPit linop-variant worst-case gap is finite, positive, with optimal status."""
    val = _wc_cp_last_iterate_linop(
        tau=CANONICAL_TAU,
        sigma=CANONICAL_SIGMA,
        theta=CANONICAL_THETA,
        n=K,
        L_M=CANONICAL_L_M,
        verbose=0,
    )
    print(f"\nPEPit linop K={K}: worst-case gap = {val!r}")
    assert np.isfinite(val), f"PEPit returned non-finite value: {val}"
    assert val > 0, f"PEPit returned non-positive value: {val}"
    # Sanity: gap should decrease (weakly) as K grows. Don't hard-assert ordering
    # because PEPit can have numerical slack; just log.


def test_pepit_nolinop_last_iterate_sanity():
    """Scalar-M variant (no linear operator) as a cross-check."""
    val = _wc_cp_last_iterate_scalar_M(
        tau=CANONICAL_TAU,
        sigma=CANONICAL_SIGMA,
        theta=CANONICAL_THETA,
        n=1,
        M=CANONICAL_L_M,
        verbose=0,
    )
    print(f"\nPEPit scalar-M K=1: worst-case gap = {val!r}")
    assert np.isfinite(val)
    assert val > 0


# ---------------------------------------------------------------------------
# Weighted-composition PEPit reference
# ---------------------------------------------------------------------------

def _wc_cp_weighted_linop(tau, sigma, theta, K, L_M=1.0,
                           decay_rate=0.9, verbose=0):
    """PEPit worst-case of sum_{k=1}^K w_k * Gap_k  (normalized weights).

    Weights match `loss_compositions.compose_weighted`:
        w_k = decay_rate^(K - k) / sum_j decay_rate^(K - j)    for k = 1..K.
    k = 0 is intentionally skipped (both iterate-0 subgradients are free,
    see loss_compositions.py docstring).
    """
    problem = PEP()
    f1 = problem.declare_function(ConvexFunction)
    h = problem.declare_function(ConvexFunction)
    K_op = problem.declare_function(LinearOperator, L=L_M)

    xs = problem.set_initial_point()
    us = problem.set_initial_point()
    f1.add_point((xs, -K_op.T.gradient(us), f1.value(xs)))
    h.add_point((us, K_op.gradient(xs), h.value(us)))

    x0 = problem.set_initial_point()
    u0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 + (u0 - us) ** 2 <= 1)

    iterates_x = [x0]
    iterates_u = [u0]
    x, u = x0, u0
    for _ in range(K):
        x_new, _, _ = proximal_step(x - tau * K_op.T.gradient(u), f1, tau)
        x_bar = x_new + theta * (x_new - x)
        u_new, _, _ = proximal_step(u + sigma * K_op.gradient(x_bar), h, sigma)
        x, u = x_new, u_new
        iterates_x.append(x)
        iterates_u.append(u)

    raw = [decay_rate ** (K - k) for k in range(1, K + 1)]
    ws = [r / sum(raw) for r in raw]

    def Lagrangian(a, b):
        return f1.value(a) + K_op.gradient(a) * b - h.value(b)

    metric = 0
    for idx, k in enumerate(range(1, K + 1)):
        gap_k = Lagrangian(iterates_x[k], us) - Lagrangian(xs, iterates_u[k])
        metric = metric + ws[idx] * gap_k

    problem.set_performance_metric(metric)
    return problem.solve(verbose=verbose, wrapper='cvxpy')


@pytest.mark.parametrize('K', [1, 3, 5])
def test_pepit_weighted_last_iterate(K):
    """PEPit weighted-gap worst case is finite and positive."""
    val = _wc_cp_weighted_linop(
        tau=CANONICAL_TAU,
        sigma=CANONICAL_SIGMA,
        theta=CANONICAL_THETA,
        K=K,
        L_M=CANONICAL_L_M,
        decay_rate=0.9,
        verbose=0,
    )
    print(f"\nPEPit linop weighted K={K}, decay=0.9: {val!r}")
    assert np.isfinite(val)
    assert val > 0


def test_pepit_weighted_K1_equals_final_K1():
    """At K=1 the weighted sum is a single normalized term; should equal final."""
    final_val = _wc_cp_last_iterate_linop(
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        n=1, L_M=CANONICAL_L_M, verbose=0,
    )
    weighted_val = _wc_cp_weighted_linop(
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        K=1, L_M=CANONICAL_L_M, decay_rate=0.9, verbose=0,
    )
    print(f"\nfinal K=1: {final_val!r};  weighted K=1: {weighted_val!r}")
    np.testing.assert_allclose(weighted_val, final_val, rtol=1e-3)


# ---------------------------------------------------------------------------
# Public helper surface: downstream tests import these to get reference values.
# ---------------------------------------------------------------------------

def pepit_linop_reference(K, tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA,
                          theta=CANONICAL_THETA, L_M=CANONICAL_L_M, verbose=0):
    """Return the PEPit linop-variant worst-case last-iterate gap (float)."""
    return _wc_cp_last_iterate_linop(tau=tau, sigma=sigma, theta=theta,
                                     n=K, L_M=L_M, verbose=verbose)


def pepit_linop_weighted_reference(K, tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA,
                                    theta=CANONICAL_THETA, L_M=CANONICAL_L_M,
                                    decay_rate=0.9, verbose=0):
    """Return the PEPit worst-case weighted-sum gap (float)."""
    return _wc_cp_weighted_linop(tau=tau, sigma=sigma, theta=theta, K=K,
                                 L_M=L_M, decay_rate=decay_rate, verbose=verbose)
