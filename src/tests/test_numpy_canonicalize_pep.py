"""
Unit tests for numpy_canonicalize_pep function.

Verifies that the numpy_canonicalize_pep function produces correct SDP
formulations by comparing against PEPit reference solutions.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

import clarabel
import scipy.sparse as spa

from learning.numpy_clarabel_layer import numpy_canonicalize_pep
from learning.pep_construction import construct_gd_pep_data, pep_data_to_numpy


class TestNumpyCanonicalPep:
    """Tests for numpy_canonicalize_pep against PEPit."""
    
    @pytest.fixture
    def pep_params(self):
        """Standard PEP parameters for smooth strongly convex GD."""
        return {
            'mu': 0.1,
            'L': 1.0,
            'R': 1.0,
            'K_max': 3,  # 3 steps of GD
            'pep_obj': 'obj_val',  # Minimize f(xK) - f(xs)
        }
    
    @pytest.fixture
    def gd_stepsizes(self, pep_params):
        """Optimal stepsize for smooth strongly convex GD."""
        L = pep_params['L']
        mu = pep_params['mu']
        # Classical optimal stepsize: 2 / (L + mu)
        t_optimal = 2.0 / (L + mu)
        return jnp.array([t_optimal] * pep_params['K_max'])
    
    def test_canonicalization_dimensions(self, pep_params, gd_stepsizes):
        """Test that canonicalized problem has correct dimensions."""
        # Build PEP data
        pep_data = construct_gd_pep_data(
            t=gd_stepsizes,
            mu=pep_params['mu'],
            L=pep_params['L'],
            R=pep_params['R'],
            K_max=pep_params['K_max'],
            pep_obj=pep_params['pep_obj'],
        )
        pep_data_np = pep_data_to_numpy(pep_data)
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data_np
        
        # Canonicalize
        A, b, c, cones, x_dim = numpy_canonicalize_pep(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        
        M = len(A_vals)  # Number of constraints
        V = b_obj.shape[0]  # Dimension of b_obj
        S_mat = A_obj.shape[0]  # Dimension of A_obj (square)
        S_vec = S_mat * (S_mat + 1) // 2
        
        # Check A dimensions
        assert A.shape[0] == V + M + S_vec, f"A rows: {A.shape[0]} vs expected {V + M + S_vec}"
        assert A.shape[1] == M, f"A cols: {A.shape[1]} vs expected {M}"
        
        # Check b and c dimensions
        assert len(b) == A.shape[0], f"b length: {len(b)} vs A rows: {A.shape[0]}"
        assert len(c) == M, f"c length: {len(c)} vs M: {M}"
        
        # Check cones
        assert len(cones) == 3, "Should have 3 cones: zero, nonneg, psd"
    
    def test_canonicalization_solves(self, pep_params, gd_stepsizes):
        """Test that canonicalized problem can be solved by Clarabel."""
        # Build PEP data
        pep_data = construct_gd_pep_data(
            t=gd_stepsizes,
            mu=pep_params['mu'],
            L=pep_params['L'],
            R=pep_params['R'],
            K_max=pep_params['K_max'],
            pep_obj=pep_params['pep_obj'],
        )
        pep_data_np = pep_data_to_numpy(pep_data)
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data_np
        
        # Canonicalize
        A, b, c, cones, x_dim = numpy_canonicalize_pep(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        
        # Solve with Clarabel
        P = spa.csc_matrix((x_dim, x_dim))  # No quadratic term
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        
        solver = clarabel.DefaultSolver(P, c, A, b, cones, settings)
        solution = solver.solve()
        
        # Should solve successfully
        assert solution.status in [
            clarabel.SolverStatus.Solved,
            clarabel.SolverStatus.AlmostSolved,
        ], f"Solver status: {solution.status}"
        
        # Objective should be non-positive (we're maximizing the dual)
        # Note: we negate c, so minimize(-c^T y) = maximize(c^T y)
        print(f"Clarabel objective (negated): {-solution.obj_val}")
    
    def test_matches_pepit_reference(self, pep_params, gd_stepsizes):
        """Test that our SDP matches PEPit solution for gradient descent."""
        from PEPit import PEP
        from PEPit.functions import SmoothStronglyConvexFunction
        
        mu = pep_params['mu']
        L = pep_params['L']
        R = pep_params['R']
        K_max = pep_params['K_max']
        
        # Convert stepsizes to numpy
        t = np.array(gd_stepsizes)
        
        # ==== PEPit reference solution ====
        problem = PEP()
        func = problem.declare_function(
            SmoothStronglyConvexFunction, mu=mu, L=L
        )
        
        # Initial point x0 and optimal point xs
        xs = func.stationary_point()
        x0 = problem.set_initial_point()
        
        # Initial condition: ||x0 - xs|| <= R
        problem.set_initial_condition((x0 - xs)**2 <= R**2)
        
        # Run GD iterations
        x = x0
        for k in range(K_max):
            g = func.gradient(x)
            x = x - t[k] * g
        
        # Objective: f(xK) - f(xs)
        problem.set_performance_metric(func(x) - func(xs))
        
        # Solve PEPit
        pepit_obj = problem.solve(verbose=0)
        print(f"PEPit objective: {pepit_obj}")
        
        # ==== Our SDP solution ====
        pep_data = construct_gd_pep_data(
            t=gd_stepsizes,
            mu=mu,
            L=L,
            R=R,
            K_max=K_max,
            pep_obj='obj_val',
        )
        pep_data_np = pep_data_to_numpy(pep_data)
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data_np
        
        # Canonicalize
        A, b, c, cones, x_dim = numpy_canonicalize_pep(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        
        # Solve with Clarabel
        P = spa.csc_matrix((x_dim, x_dim))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        
        solver = clarabel.DefaultSolver(P, c, A, b, cones, settings)
        solution = solver.solve()
        
        assert solution.status in [
            clarabel.SolverStatus.Solved,
            clarabel.SolverStatus.AlmostSolved,
        ], f"Clarabel failed: {solution.status}"
        
        # The Clarabel objective directly gives the PEP bound
        our_obj = solution.obj_val
        print(f"Our objective: {our_obj}")
        
        # They should match within tolerance
        np.testing.assert_allclose(
            our_obj, pepit_obj, rtol=1e-4,
            err_msg=f"PEPit: {pepit_obj}, Ours: {our_obj}"
        )
    
    def test_matches_pepit_grad_norm(self):
        """Test gradient norm objective matches PEPit."""
        from PEPit import PEP
        from PEPit.functions import SmoothStronglyConvexFunction
        
        mu = 0.1
        L = 1.0
        R = 1.0
        K_max = 2
        t = 2.0 / (L + mu)
        
        # ==== PEPit reference ====
        problem = PEP()
        func = problem.declare_function(
            SmoothStronglyConvexFunction, mu=mu, L=L
        )
        
        xs = func.stationary_point()
        x0 = problem.set_initial_point()
        problem.set_initial_condition((x0 - xs)**2 <= R**2)
        
        x = x0
        for k in range(K_max):
            g = func.gradient(x)
            x = x - t * g
        
        gK = func.gradient(x)
        problem.set_performance_metric(gK**2)
        
        pepit_obj = problem.solve(verbose=0)
        print(f"PEPit grad norm obj: {pepit_obj}")
        
        # ==== Our SDP ====
        t_vec = jnp.array([t] * K_max)
        pep_data = construct_gd_pep_data(
            t=t_vec, mu=mu, L=L, R=R, K_max=K_max, pep_obj='grad_sq_norm'
        )
        pep_data_np = pep_data_to_numpy(pep_data)
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data_np
        
        A, b, c, cones, x_dim = numpy_canonicalize_pep(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        
        P = spa.csc_matrix((x_dim, x_dim))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        
        solver = clarabel.DefaultSolver(P, c, A, b, cones, settings)
        solution = solver.solve()
        
        assert solution.status in [
            clarabel.SolverStatus.Solved,
            clarabel.SolverStatus.AlmostSolved,
        ]
        
        our_obj = solution.obj_val
        print(f"Our grad norm obj: {our_obj}")
        
        np.testing.assert_allclose(our_obj, pepit_obj, rtol=1e-4)
    
    def test_matches_pepit_opt_dist(self):
        """Test optimal distance objective matches PEPit."""
        from PEPit import PEP
        from PEPit.functions import SmoothStronglyConvexFunction
        
        mu = 0.1
        L = 1.0
        R = 1.0
        K_max = 2
        t = 2.0 / (L + mu)
        
        # ==== PEPit reference ====
        problem = PEP()
        func = problem.declare_function(
            SmoothStronglyConvexFunction, mu=mu, L=L
        )
        
        xs = func.stationary_point()
        x0 = problem.set_initial_point()
        problem.set_initial_condition((x0 - xs)**2 <= R**2)
        
        x = x0
        for k in range(K_max):
            g = func.gradient(x)
            x = x - t * g
        
        problem.set_performance_metric((x - xs)**2)
        
        pepit_obj = problem.solve(verbose=0)
        print(f"PEPit opt dist obj: {pepit_obj}")
        
        # ==== Our SDP ====
        t_vec = jnp.array([t] * K_max)
        pep_data = construct_gd_pep_data(
            t=t_vec, mu=mu, L=L, R=R, K_max=K_max, pep_obj='opt_dist_sq_norm'
        )
        pep_data_np = pep_data_to_numpy(pep_data)
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data_np
        
        A, b, c, cones, x_dim = numpy_canonicalize_pep(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        
        P = spa.csc_matrix((x_dim, x_dim))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        
        solver = clarabel.DefaultSolver(P, c, A, b, cones, settings)
        solution = solver.solve()
        
        assert solution.status in [
            clarabel.SolverStatus.Solved,
            clarabel.SolverStatus.AlmostSolved,
        ]
        
        our_obj = solution.obj_val
        print(f"Our opt dist obj: {our_obj}")
        
        np.testing.assert_allclose(our_obj, pepit_obj, rtol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
