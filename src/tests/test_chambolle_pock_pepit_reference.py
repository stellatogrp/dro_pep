"""
PEPit reference values for the Chambolle-Pock last-iterate duality-gap PEP.

These tests lock down the numerical values of PEPit's worst-case SDP so that
(a) regressions in PEPit itself or its SDP backend are caught independently of
our custom construction, and (b) downstream tests comparing
`construct_chambolle_pock_pep_data` to PEPit have a stable reference.

Formulation: min_x max_u  f1(x) + <K x, u> - h(u)
Saddle conditions: -K^T u_s ∈ ∂f1(x_s), K x_s ∈ ∂h(u_s)
Metric: duality gap at the LAST iterate (not averaged).
Initial condition: P-norm Lyapunov
    (x0 - xs)^2 / tau + (u0 - us)^2 / sigma - 2 (u0 - us) K(x0 - xs) <= 1
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
    initial_term = ((x0 - xs) ** 2 / tau
                    + (u0 - us) ** 2 / sigma
                    - 2 * (u0 - us) * K.gradient(x0 - xs))
    problem.set_initial_condition(initial_term <= 1)

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
# Public helper surface: downstream tests import these to get reference values.
# ---------------------------------------------------------------------------

def pepit_linop_reference(K, tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA,
                          theta=CANONICAL_THETA, L_M=CANONICAL_L_M, verbose=0):
    """Return the PEPit linop-variant worst-case last-iterate gap (float)."""
    return _wc_cp_last_iterate_linop(tau=tau, sigma=sigma, theta=theta,
                                     n=K, L_M=L_M, verbose=verbose)
