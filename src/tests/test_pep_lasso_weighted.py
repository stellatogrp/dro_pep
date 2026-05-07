"""
Tests for ISTA/FISTA PEP with WEIGHTED composition (LPEP path).

Context / findings:
    The `final` composition PEP (objective = f(x_K) - Fs) is bounded and matches
    PEPit (see test_pep_lasso_debug.py).

    The `weighted` composition as currently implemented in
    `learning.pep_constructions.loss_compositions.compose_weighted` sums over
    k = 0..K, which for COMPOSITE problems (f1 + f2 with f2 convex but not
    necessarily smooth, e.g., Lasso's L1 term) is MATHEMATICALLY UNBOUNDED.

    Root cause: the free subgradient h_0 of f2 at x_0 is unconstrained by the
    algorithm dynamics (ISTA uses h_1, h_2, ..., h_K but NOT h_0). The convex
    interpolation constraint (0, s) for f2,
            f2(x_0) - f2(x_s) + <h_0, x_s - x_0> <= 0,
    simplifies (with f2(x_s)=0, x_s=0) to f2(x_0) <= <h_0, x_0>, and with h_0
    free, this is a vacuous upper bound. So f(x_0) is unbounded in the generic
    composite PEP. Any weighted sum including k=0 inherits the unboundedness.

    We confirmed this empirically: PEPit itself returns NumericalError on
    CLARABEL and spurious large values on SCS/MOSEK for the weighted-0..K case.

    For the non-composite Quad/GD PEP, g_s = 0 (f is its own minimum) and
    ||g_0||^2 is bounded by smoothness, so weighted 0..K is bounded there.

    This explains why:
      - LPEP (pure worst-case) for Lasso with weighted composition DIVERGES
      - LDRO-PEP for Lasso with weighted composition stays finite (sample
        constraints bound the SDP even though the pure PEP is unbounded)

    Fix (proposed, not yet applied): make `compose_weighted` sum over k = 1..K
    (skip x_0). This is also semantically correct: x_0 is given, not chosen by
    the algorithm, so including it in the training objective contributes only
    a stepsize-independent constant (and in the composite case, an unbounded
    worst-case term).

    These tests encode the INTENDED post-fix behavior (skip x_0) and should
    fail until the fix is applied.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp

jax.config.update('jax_enable_x64', True)

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction, ConvexFunction
from PEPit.primitive_steps import proximal_step

from learning.pep_constructions import (
    construct_ista_pep_data,
    construct_fista_pep_data,
)


# ============================================================================
# PEPit reference: weighted sum over k = 1..K (skip x_0)
# ============================================================================

def _normalized_weights(K, decay_rate):
    """Weights w_k = decay_rate^(K-k) for k = 1..K, normalized."""
    weights = [decay_rate ** (K - k) for k in range(1, K + 1)]
    wsum = sum(weights)
    return [w / wsum for w in weights]


def run_pepit_ista_weighted(L, mu, gamma, K, decay_rate=0.9):
    """PEPit ISTA: sum_{k=1..K} w_k (f(x_k) - Fs)."""
    problem = PEP()
    f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    f2 = problem.declare_function(ConvexFunction)
    F = f1 + f2
    xs = F.stationary_point()
    Fs = F(xs)

    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    iterates = [x0]
    fx2_vals = [f2(x0)]
    x = x0
    for _ in range(K):
        y = x - gamma * f1.gradient(x)
        x, _, fx2 = proximal_step(y, f2, gamma)
        iterates.append(x)
        fx2_vals.append(fx2)

    weights = _normalized_weights(K, decay_rate)
    metric = 0
    for idx, k in enumerate(range(1, K + 1)):
        metric = metric + weights[idx] * (f1(iterates[k]) + fx2_vals[k] - Fs)

    problem.set_performance_metric(metric)
    return problem.solve(wrapper='cvxpy', solver='CLARABEL', verbose=0)


def run_pepit_fista_weighted(L, mu, K, decay_rate=0.9):
    """PEPit FISTA: sum_{k=1..K} w_k (f(x_k) - Fs)."""
    from math import sqrt

    gamma = 1.0 / L
    problem = PEP()
    f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    f2 = problem.declare_function(ConvexFunction)
    F = f1 + f2
    xs = F.stationary_point()
    Fs = F(xs)

    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    iterates = [x0]
    fx2_vals = [f2(x0)]
    x_new = x0
    y = x0
    lam = 1.0
    for _ in range(K):
        lam_old = lam
        lam = (1 + sqrt(4 * lam_old ** 2 + 1)) / 2
        x_old = x_new
        x_new, _, fx2 = proximal_step(y - gamma * f1.gradient(y), f2, gamma)
        y = x_new + (lam_old - 1) / lam * (x_new - x_old)
        iterates.append(x_new)
        fx2_vals.append(fx2)

    weights = _normalized_weights(K, decay_rate)
    metric = 0
    for idx, k in enumerate(range(1, K + 1)):
        metric = metric + weights[idx] * (f1(iterates[k]) + fx2_vals[k] - Fs)

    problem.set_performance_metric(metric)
    return problem.solve(wrapper='cvxpy', solver='CLARABEL', verbose=0)


# ============================================================================
# Custom PEP solve (uses our construct_*_pep_data with composition='weighted')
# ============================================================================

def solve_custom_weighted(pep_data):
    A_obj, b_obj, A_vals, b_vals, c_vals = [np.array(x) for x in pep_data[:5]]
    dimG, dimF, M = A_obj.shape[0], len(b_obj), len(c_vals)

    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable(dimF)
    constraints = [G >> 0]
    for i in range(M):
        constraints.append(
            cp.trace(A_vals[i] @ G) + b_vals[i] @ F + c_vals[i] <= 0
        )
    obj = cp.trace(A_obj @ G) + b_obj @ F
    prob = cp.Problem(cp.Maximize(obj), constraints)
    try:
        prob.solve(solver='CLARABEL', verbose=False)
        return prob.value, prob.status
    except Exception:
        return None, 'failed'


# ============================================================================
# Tests
# ============================================================================

class TestISTAPEPWeighted(unittest.TestCase):
    """Weighted-composition ISTA PEP: must be bounded & match PEPit once x_0 is skipped."""

    def _compare(self, L, mu, gamma, K, decay_rate=0.9, rtol=0.2):
        pepit_tau = run_pepit_ista_weighted(L, mu, gamma, K, decay_rate)
        print(f"    PEPit weighted (skip x_0): {pepit_tau}")

        gamma_arr = jnp.array([gamma] * K)
        pep_data = construct_ista_pep_data(
            gamma_arr, mu, L, R=1.0, K_max=K, pep_obj='obj_val',
            composition_type='weighted', decay_rate=decay_rate,
        )
        custom_val, status = solve_custom_weighted(pep_data)
        print(f"    Custom weighted: {custom_val} (status={status})")

        self.assertIsNotNone(custom_val, 'Custom solve returned None (unbounded)')
        self.assertFalse(np.isinf(custom_val or np.inf), 'Custom returned inf')
        self.assertFalse(np.isnan(custom_val or np.nan), 'Custom returned NaN')
        self.assertIn(status, ['optimal', 'optimal_inaccurate'])
        np.testing.assert_allclose(
            custom_val, pepit_tau, rtol=rtol,
            err_msg=f'Custom {custom_val} vs PEPit {pepit_tau}',
        )

    def test_weighted_n1_sc(self):
        print('\nISTA K=1 strongly convex')
        self._compare(L=1.0, mu=0.1, gamma=1.0, K=1)

    def test_weighted_n2_convex(self):
        print('\nISTA K=2 convex')
        self._compare(L=1.0, mu=0.0, gamma=1.0, K=2)

    def test_weighted_n5_convex(self):
        print('\nISTA K=5 convex')
        self._compare(L=1.0, mu=0.0, gamma=1.0, K=5)

    def test_weighted_n5_large_L_aggressive_step(self):
        """Matches the empirically-failing config (L=5.64, gamma=1.5/L, K=5)."""
        print('\nISTA K=5 L=5.64 gamma=1.5/L')
        self._compare(L=5.64, mu=0.0, gamma=1.5 / 5.64, K=5)


class TestFISTAPEPWeighted(unittest.TestCase):
    """Weighted-composition FISTA PEP.

    K=1, 2: bounded and match PEPit (K=1 is equivalent to final composition
    since the sum only contains k=1=K; K=2 works because beta_0=0 gives
    y_1=x_1 so the formulations agree).

    K >= 3: UNBOUNDED with current FISTA construction — see the docstring of
    `construct_fista_pep_data` in `learning/pep_constructions/ista_fista.py`
    for a full diagnosis. Summary: our PEP lacks f1 interpolation at x_k
    for k = 2,...,K-1, whereas PEPit includes those points (verified by
    inspecting PEPit's f1.list_of_points — 10 points for K=5 weighted vs
    our 7). The missing constraints leave the SDP under-specified.
    These K>=3 tests are marked @expectedFailure pending the fix.
    """

    def _betas(self, K):
        """Raw t_k Nesterov sequence of length K+1 with beta[0]=1.0.

        This is what `construct_fista_pep_data` expects (per its docstring).
        NOT the momentum coefficients (lams[k]-1)/lams[k+1] of length K.
        """
        from math import sqrt
        lams = [1.0]
        for _ in range(K):
            lams.append((1 + sqrt(4 * lams[-1] ** 2 + 1)) / 2)
        return lams  # length K+1

    def _compare(self, L, mu, K, decay_rate=0.9, rtol=0.2):
        gamma = 1.0 / L
        betas = self._betas(K)
        pepit_tau = run_pepit_fista_weighted(L, mu, K, decay_rate)
        print(f"    PEPit weighted (skip x_0): {pepit_tau}")

        gamma_arr = jnp.array([gamma] * K)
        beta_arr = jnp.array(betas)
        pep_data = construct_fista_pep_data(
            gamma_arr, beta_arr, mu, L, R=1.0, K_max=K, pep_obj='obj_val',
            composition_type='weighted', decay_rate=decay_rate,
        )
        custom_val, status = solve_custom_weighted(pep_data)
        print(f"    Custom weighted: {custom_val} (status={status})")

        self.assertIsNotNone(custom_val, 'Custom solve returned None (unbounded)')
        self.assertIn(status, ['optimal', 'optimal_inaccurate'])
        np.testing.assert_allclose(
            custom_val, pepit_tau, rtol=rtol,
            err_msg=f'Custom {custom_val} vs PEPit {pepit_tau}',
        )

    def test_weighted_n1_convex(self):
        print('\nFISTA K=1')
        self._compare(L=1.0, mu=0.0, K=1)

    def test_weighted_n2_convex(self):
        """FISTA K=2 weighted: beta_0=0 so y_1=x_1, equivalent to ISTA K=2."""
        print('\nFISTA K=2')
        self._compare(L=1.0, mu=0.0, K=2)

    @unittest.expectedFailure
    def test_weighted_n3_convex(self):
        """FISTA K>=3 weighted: documented limitation — see
        construct_fista_pep_data docstring. Starting at K=3, y_k != x_k for
        k >= 2, and f1 interpolation at x_k (k=2..K-1) is missing from our
        PEP (PEPit has it), causing unbounded LPEP.
        """
        print('\nFISTA K=3')
        self._compare(L=1.0, mu=0.0, K=3)

    @unittest.expectedFailure
    def test_weighted_n5_convex(self):
        """FISTA K>=3 weighted: documented limitation."""
        print('\nFISTA K=5')
        self._compare(L=1.0, mu=0.0, K=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
