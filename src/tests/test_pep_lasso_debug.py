"""
Debug tests for ISTA/FISTA PEP construction with obj_val objective.

These tests verify that:
1. The PEP SDP for ISTA/FISTA is bounded (not unbounded/infeasible)
2. The obj_val objective matches PEPit (composite objective f1(xK) + f2(xK) - (f1_s + f2_s))
3. The Gram representation structure is correct
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

from learning.interpolation_conditions import (
    smooth_strongly_convex_interp,
    convex_interp,
)
from learning.pep_construction_lasso import (
    construct_ista_pep_data,
    construct_fista_pep_data,
    ista_pep_data_to_numpy,
)


# ============================================================================
# PEPit Reference Implementations
# ============================================================================

def run_pepit_ista_obj_val(L, mu, gamma, n):
    """
    Run PEPit's proximal gradient (ISTA) with obj_val metric.
    
    Performance metric: f(x_n) - f(x_s) = f1(x_n) + f2(x_n) - (f1(x_s) + f2(x_s))
    
    Args:
        L: Smoothness parameter for f1
        mu: Strong convexity parameter for f1
        gamma: Step size
        n: Number of iterations
    
    Returns:
        pepit_tau: Worst-case value from PEPit
    """
    problem = PEP()
    
    f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    f2 = problem.declare_function(ConvexFunction)
    F = f1 + f2
    
    xs = F.stationary_point()
    Fs = F(xs)  # Optimal value
    
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    
    # Run ISTA
    x = x0
    for _ in range(n):
        y = x - gamma * f1.gradient(x)
        x, _, fx2 = proximal_step(y, f2, gamma)
    
    # Objective: composite function value
    problem.set_performance_metric((f1(x) + fx2) - Fs)
    
    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL', verbose=0)
    
    return pepit_tau


def run_pepit_fista_obj_val(L, mu, n):
    """
    Run PEPit's accelerated proximal gradient (FISTA) with obj_val metric.
    
    Performance metric: f(x_n) - f(x_s)
    """
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
    
    # FISTA iterations
    x_new = x0
    y = x0
    lam = 1.0
    
    for i in range(n):
        lam_old = lam
        lam = (1 + sqrt(4 * lam_old ** 2 + 1)) / 2
        x_old = x_new
        x_new, _, fx2 = proximal_step(y - gamma * f1.gradient(y), f2, gamma)
        y = x_new + (lam_old - 1) / lam * (x_new - x_old)
    
    # Objective: function value
    problem.set_performance_metric((f1(x_new) + fx2) - Fs)
    
    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL', verbose=0)
    
    return pepit_tau


# ============================================================================
# Custom PEP Solver (using existing construction)
# ============================================================================

def solve_ista_pep_custom(gamma, mu, L, R, K, pep_obj='obj_val'):
    """
    Solve ISTA PEP using our construct_ista_pep_data.
    """
    gamma_arr = jnp.array([gamma] * K) if np.isscalar(gamma) else jnp.array(gamma)
    
    pep_data = construct_ista_pep_data(gamma_arr, mu, L, R, K, pep_obj)
    A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]
    
    # Convert to numpy
    A_obj_np = np.array(A_obj)
    b_obj_np = np.array(b_obj)
    A_vals_np = np.array(A_vals)
    b_vals_np = np.array(b_vals)
    c_vals_np = np.array(c_vals)
    
    dimG = A_obj_np.shape[0]
    dimF = len(b_obj_np)
    
    print(f"ISTA PEP dimensions: dimG={dimG}, dimF={dimF}, num_constraints={len(c_vals_np)}")
    
    # CVXPY problem
    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable(dimF)
    
    constraints = [G >> 0]
    
    for i in range(len(c_vals_np)):
        constraints.append(
            cp.trace(A_vals_np[i] @ G) + b_vals_np[i] @ F + c_vals_np[i] <= 0
        )
    
    objective = cp.trace(A_obj_np @ G) + b_obj_np @ F
    
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver='CLARABEL', verbose=False)
    
    print(f"ISTA PEP solution status: {prob.status}, value: {prob.value}")
    
    return prob.value, prob.status


def solve_fista_pep_custom(gamma, beta, mu, L, R, K, pep_obj='obj_val'):
    """
    Solve FISTA PEP using our construct_fista_pep_data.
    """
    gamma_arr = jnp.array([gamma] * K) if np.isscalar(gamma) else jnp.array(gamma)
    beta_arr = jnp.array(beta)
    
    pep_data = construct_fista_pep_data(gamma_arr, beta_arr, mu, L, R, K, pep_obj)
    A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]
    
    # Convert to numpy
    A_obj_np = np.array(A_obj)
    b_obj_np = np.array(b_obj)
    A_vals_np = np.array(A_vals)
    b_vals_np = np.array(b_vals)
    c_vals_np = np.array(c_vals)
    
    dimG = A_obj_np.shape[0]
    dimF = len(b_obj_np)
    
    print(f"FISTA PEP dimensions: dimG={dimG}, dimF={dimF}, num_constraints={len(c_vals_np)}")
    
    # CVXPY problem
    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable(dimF)
    
    constraints = [G >> 0]
    
    for i in range(len(c_vals_np)):
        constraints.append(
            cp.trace(A_vals_np[i] @ G) + b_vals_np[i] @ F + c_vals_np[i] <= 0
        )
    
    objective = cp.trace(A_obj_np @ G) + b_obj_np @ F
    
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver='CLARABEL', verbose=False)
    
    print(f"FISTA PEP solution status: {prob.status}, value: {prob.value}")
    
    return prob.value, prob.status


# ============================================================================
# Test Class
# ============================================================================

class TestISTAPEPObjVal(unittest.TestCase):
    """Test ISTA PEP construction with obj_val objective."""
    
    def test_ista_pep_bounded_n1(self):
        """Test that ISTA PEP is bounded for n=1."""
        L = 1.0
        mu = 0.1
        gamma = 1.0 / L
        K = 1
        R = 1.0
        
        value, status = solve_ista_pep_custom(gamma, mu, L, R, K, 'obj_val')
        
        self.assertIn(status, ['optimal', 'optimal_inaccurate'],
                      f"ISTA PEP status should be optimal, got {status}")
        self.assertIsNotNone(value, "ISTA PEP value should not be None")
        self.assertFalse(np.isinf(value), "ISTA PEP value should be finite")
    
    def test_ista_pep_bounded_n2(self):
        """Test that ISTA PEP is bounded for n=2."""
        L = 1.0
        mu = 0.0
        gamma = 1.0 / L
        K = 2
        R = 1.0
        
        value, status = solve_ista_pep_custom(gamma, mu, L, R, K, 'obj_val')
        
        self.assertIn(status, ['optimal', 'optimal_inaccurate'],
                      f"ISTA PEP status should be optimal, got {status}")
        self.assertIsNotNone(value, "ISTA PEP value should not be None")
        self.assertFalse(np.isinf(value), "ISTA PEP value should be finite")
    
    def test_ista_pep_matches_pepit_n1(self):
        """Test that ISTA PEP matches PEPit for n=1."""
        L = 1.0
        mu = 0.1
        gamma = 1.0 / L
        K = 1
        R = 1.0
        
        # Run PEPit
        pepit_tau = run_pepit_ista_obj_val(L, mu, gamma, K)
        print(f"PEPit obj_val: {pepit_tau}")
        
        # Run custom
        custom_tau, status = solve_ista_pep_custom(gamma, mu, L, R, K, 'obj_val')
        print(f"Custom obj_val: {custom_tau}")
        
        self.assertIn(status, ['optimal', 'optimal_inaccurate'])
        np.testing.assert_allclose(custom_tau, pepit_tau, rtol=0.1,
                                   err_msg=f"Mismatch: custom={custom_tau}, pepit={pepit_tau}")
    
    def test_ista_pep_matches_pepit_n2(self):
        """Test that ISTA PEP matches PEPit for n=2."""
        L = 1.0
        mu = 0.0
        gamma = 1.0 / L
        K = 2
        R = 1.0
        
        pepit_tau = run_pepit_ista_obj_val(L, mu, gamma, K)
        print(f"PEPit obj_val: {pepit_tau}")
        
        custom_tau, status = solve_ista_pep_custom(gamma, mu, L, R, K, 'obj_val')
        print(f"Custom obj_val: {custom_tau}")
        
        self.assertIn(status, ['optimal', 'optimal_inaccurate'])
        np.testing.assert_allclose(custom_tau, pepit_tau, rtol=0.1,
                                   err_msg=f"Mismatch: custom={custom_tau}, pepit={pepit_tau}")


class TestFISTAPEPObjVal(unittest.TestCase):
    """Test FISTA PEP construction with obj_val objective."""
    
    def test_fista_pep_bounded_n1(self):
        """Test that FISTA PEP is bounded for n=1."""
        from math import sqrt
        
        L = 1.0
        mu = 0.0
        gamma = 1.0 / L
        K = 1
        R = 1.0
        
        # Compute FISTA betas
        lams = [1.0]
        for k in range(K):
            lam_new = (1 + sqrt(4 * lams[-1] ** 2 + 1)) / 2
            lams.append(lam_new)
        betas = [(lams[k] - 1) / lams[k+1] for k in range(K)]
        
        value, status = solve_fista_pep_custom(gamma, betas, mu, L, R, K, 'obj_val')
        
        self.assertIn(status, ['optimal', 'optimal_inaccurate'],
                      f"FISTA PEP status should be optimal, got {status}")
        self.assertIsNotNone(value, "FISTA PEP value should not be None")
        self.assertFalse(np.isinf(value), "FISTA PEP value should be finite")
    
    def test_fista_pep_bounded_n2(self):
        """Test that FISTA PEP is bounded for n=2."""
        from math import sqrt
        
        L = 1.0
        mu = 0.0
        gamma = 1.0 / L
        K = 2
        R = 1.0
        
        lams = [1.0]
        for k in range(K):
            lam_new = (1 + sqrt(4 * lams[-1] ** 2 + 1)) / 2
            lams.append(lam_new)
        betas = [(lams[k] - 1) / lams[k+1] for k in range(K)]
        
        value, status = solve_fista_pep_custom(gamma, betas, mu, L, R, K, 'obj_val')
        
        self.assertIn(status, ['optimal', 'optimal_inaccurate'],
                      f"FISTA PEP status should be optimal, got {status}")
        self.assertIsNotNone(value, "FISTA PEP value should not be None")
        self.assertFalse(np.isinf(value), "FISTA PEP value should be finite")
    
    def test_fista_pep_matches_pepit_n1(self):
        """Test that FISTA PEP matches PEPit for n=1."""
        from math import sqrt
        
        L = 1.0
        mu = 0.0
        K = 1
        R = 1.0
        gamma = 1.0 / L
        
        lams = [1.0]
        for k in range(K):
            lam_new = (1 + sqrt(4 * lams[-1] ** 2 + 1)) / 2
            lams.append(lam_new)
        betas = [(lams[k] - 1) / lams[k+1] for k in range(K)]
        
        pepit_tau = run_pepit_fista_obj_val(L, mu, K)
        print(f"PEPit obj_val: {pepit_tau}")
        
        custom_tau, status = solve_fista_pep_custom(gamma, betas, mu, L, R, K, 'obj_val')
        print(f"Custom obj_val: {custom_tau}")
        
        self.assertIn(status, ['optimal', 'optimal_inaccurate'])
        np.testing.assert_allclose(custom_tau, pepit_tau, rtol=0.1,
                                   err_msg=f"Mismatch: custom={custom_tau}, pepit={pepit_tau}")


if __name__ == '__main__':
    # Run specific tests with verbose output
    print("=" * 60)
    print("Testing ISTA PEP construction with obj_val objective")
    print("=" * 60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestISTAPEPObjVal)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    print("Testing FISTA PEP construction with obj_val objective")
    print("=" * 60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFISTAPEPObjVal)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
