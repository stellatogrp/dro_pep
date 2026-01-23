"""
Comparison tests between custom PEP construction and PEPit for Chambolle-Pock.

These tests verify that our JAX-compatible PEP construction produces the same
SDP objective values as PEPit's Chambolle-Pock with gap objective.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp

jax.config.update('jax_enable_x64', True)

from tests.test_chambolle_pock import wc_chambolle_pock_last_iterate
from learning.pep_construction_chambolle_pock import (
    construct_chambolle_pock_pep_data,
    chambolle_pock_pep_data_to_numpy,
)


def solve_chambolle_pock_pep(tau, sigma, theta, M, R, K):
    """
    Solve the PEP for Chambolle-Pock using custom construction.
    
    Returns:
        optimal_value: The optimal gap objective value
    """
    pep_data = construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K)
    pep_data_np = chambolle_pock_pep_data_to_numpy(pep_data)
    
    (A_obj, b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = pep_data_np
    
    dimG = A_obj.shape[0]
    dimF = b_obj.shape[0]
    
    # Variables
    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable(dimF)
    
    constraints = [G >> 0]
    
    # Interpolation constraints
    for i in range(A_vals.shape[0]):
        constraints.append(cp.trace(A_vals[i] @ G) + b_vals[i] @ F + c_vals[i] <= 0)
    
    # Objective
    objective = cp.trace(A_obj @ G) + b_obj @ F
    
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver='CLARABEL', verbose=False)
    
    return prob.value


class TestChambollePockPEPitComparison(unittest.TestCase):
    """Tests comparing custom PEP construction with PEPit Chambolle-Pock."""
    
    def test_gap_objective_n1(self):
        """Test gap objective with n=1 iteration."""
        tau = 0.1
        sigma = 0.1
        theta = 1.0
        M = 1.0
        R = 1.0
        n = 1
        
        # Run PEPit
        pepit_gap = wc_chambolle_pock_last_iterate(
            tau=tau, sigma=sigma, theta=theta, n=n, M=M, verbose=0
        )
        
        # Run our implementation
        custom_gap = solve_chambolle_pock_pep(tau, sigma, theta, M, R, n)
        
        np.testing.assert_allclose(custom_gap, pepit_gap, rtol=1e-3,
            err_msg=f"Mismatch for n={n}: custom={custom_gap}, pepit={pepit_gap}")
    
    def test_gap_objective_n2(self):
        """Test gap objective with n=2 iterations."""
        tau = 0.1
        sigma = 0.1
        theta = 1.0
        M = 1.0
        R = 1.0
        n = 2
        
        pepit_gap = wc_chambolle_pock_last_iterate(
            tau=tau, sigma=sigma, theta=theta, n=n, M=M, verbose=0
        )
        
        custom_gap = solve_chambolle_pock_pep(tau, sigma, theta, M, R, n)
        
        np.testing.assert_allclose(custom_gap, pepit_gap, rtol=1e-3,
            err_msg=f"Mismatch for n={n}: custom={custom_gap}, pepit={pepit_gap}")
    
    def test_gap_objective_n3(self):
        """Test gap objective with n=3 iterations."""
        tau = 0.1
        sigma = 0.1
        theta = 1.0
        M = 1.0
        R = 1.0
        n = 3
        
        pepit_gap = wc_chambolle_pock_last_iterate(
            tau=tau, sigma=sigma, theta=theta, n=n, M=M, verbose=0
        )
        
        custom_gap = solve_chambolle_pock_pep(tau, sigma, theta, M, R, n)
        
        np.testing.assert_allclose(custom_gap, pepit_gap, rtol=1e-3,
            err_msg=f"Mismatch for n={n}: custom={custom_gap}, pepit={pepit_gap}")
    
    def test_different_step_sizes(self):
        """Test with different tau and sigma."""
        tau = 0.05
        sigma = 0.2
        theta = 1.0
        M = 1.0
        R = 1.0
        n = 2
        
        pepit_gap = wc_chambolle_pock_last_iterate(
            tau=tau, sigma=sigma, theta=theta, n=n, M=M, verbose=0
        )
        
        custom_gap = solve_chambolle_pock_pep(tau, sigma, theta, M, R, n)
        
        np.testing.assert_allclose(custom_gap, pepit_gap, rtol=1e-3,
            err_msg=f"Mismatch: custom={custom_gap}, pepit={pepit_gap}")
    
    def test_larger_M(self):
        """Test with larger coupling constant M."""
        tau = 0.01
        sigma = 0.01
        theta = 1.0
        M = 10.0
        R = 1.0
        n = 2
        
        pepit_gap = wc_chambolle_pock_last_iterate(
            tau=tau, sigma=sigma, theta=theta, n=n, M=M, verbose=0
        )
        
        custom_gap = solve_chambolle_pock_pep(tau, sigma, theta, M, R, n)
        
        np.testing.assert_allclose(custom_gap, pepit_gap, rtol=1e-3,
            err_msg=f"Mismatch: custom={custom_gap}, pepit={pepit_gap}")
    
    def test_gram_matrix_dimension(self):
        """Verify Gram matrix dimension is 2K + 6."""
        for K in [1, 2, 3, 5]:
            tau = 0.1
            sigma = 0.1
            theta = 1.0
            M = 1.0
            R = 1.0
            
            pep_data = construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K)
            A_obj = pep_data[0]
            dimG = A_obj.shape[0]
            
            expected_dimG = 2 * K + 6
            self.assertEqual(dimG, expected_dimG,
                f"dimG mismatch for K={K}: got {dimG}, expected {expected_dimG}")


if __name__ == '__main__':
    unittest.main()
