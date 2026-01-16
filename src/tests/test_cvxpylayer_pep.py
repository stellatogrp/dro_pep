"""
Unit tests for create_full_pep_layer cvxpylayer.

Verifies that the primal PEP formulation in create_full_pep_layer produces 
correct SDP solutions by comparing against PEPit reference solutions.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp

jax.config.update('jax_enable_x64', True)

from learning.autodiff_setup import create_full_pep_layer
from learning.pep_construction import construct_gd_pep_data, pep_data_to_numpy


class TestCvxpyLayerPep:
    """Tests for create_full_pep_layer cvxpylayer against PEPit."""
    
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
    
    def _solve_primal_pep_cvxpy(self, A_obj, b_obj, A_vals, b_vals, c_vals, solver=cp.CLARABEL):
        """Solve primal PEP using CVXPY directly (not cvxpylayers).
        
        The primal PEP problem is:
            max  trace(A_obj @ G) + b_obj^T @ F
            s.t. G >> 0
                 trace(A_m @ G) + b_m^T @ F + c_m <= 0  for all m
        """
        A_obj = np.array(A_obj)
        b_obj = np.array(b_obj)
        A_vals_np = [np.array(A_vals[m]) for m in range(A_vals.shape[0])]
        b_vals_np = [np.array(b_vals[m]) for m in range(b_vals.shape[0])]
        c_vals = np.array(c_vals)
        
        M = len(A_vals_np)
        mat_shape = A_obj.shape
        vec_shape = b_obj.shape
        
        G = cp.Variable(mat_shape, symmetric=True)
        F = cp.Variable(vec_shape)
        
        constraints = [G >> 0]
        
        for m in range(M):
            Am = A_vals_np[m]
            bm = b_vals_np[m]
            cm = c_vals[m]
            constraints.append(cp.trace(Am @ G) + bm.T @ F + cm <= 0)
        
        obj = cp.trace(A_obj @ G) + b_obj.T @ F
        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(solver=solver, verbose=False)
        
        return prob.value, G.value, F.value
    
    def test_cvxpy_solves(self, pep_params, gd_stepsizes):
        """Test that CVXPY can solve the primal PEP problem."""
        pep_data = construct_gd_pep_data(
            t=gd_stepsizes,
            mu=pep_params['mu'],
            L=pep_params['L'],
            R=pep_params['R'],
            K_max=pep_params['K_max'],
            pep_obj=pep_params['pep_obj'],
        )
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        
        obj_val, G_star, F_star = self._solve_primal_pep_cvxpy(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        
        assert obj_val is not None
        assert obj_val > 0
        print(f"CVXPY objective: {obj_val}")
    
    def test_matches_pepit_reference(self, pep_params, gd_stepsizes):
        """Test that our primal PEP SDP matches PEPit solution for gradient descent."""
        from PEPit import PEP
        from PEPit.functions import SmoothStronglyConvexFunction
        
        mu = pep_params['mu']
        L = pep_params['L']
        R = pep_params['R']
        K_max = pep_params['K_max']
        
        t = np.array(gd_stepsizes)
        
        # ==== PEPit reference solution ====
        problem = PEP()
        func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        xs = func.stationary_point()
        x0 = problem.set_initial_point()
        problem.set_initial_condition((x0 - xs)**2 <= R**2)
        
        x = x0
        for k in range(K_max):
            g = func.gradient(x)
            x = x - t[k] * g
        
        problem.set_performance_metric(func(x) - func(xs))
        pepit_obj = problem.solve(verbose=0)
        print(f"PEPit objective: {pepit_obj}")
        
        # ==== Our primal PEP solution ====
        pep_data = construct_gd_pep_data(
            t=gd_stepsizes, mu=mu, L=L, R=R, K_max=K_max, pep_obj='obj_val',
        )
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        
        our_obj, _, _ = self._solve_primal_pep_cvxpy(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        print(f"Our objective: {our_obj}")
        
        np.testing.assert_allclose(
            our_obj, pepit_obj, rtol=1e-4,
            err_msg=f"PEPit: {pepit_obj}, Ours: {our_obj}"
        )
    
    def test_matches_pepit_varying_stepsizes(self):
        """Test with non-optimal varying stepsizes."""
        from PEPit import PEP
        from PEPit.functions import SmoothStronglyConvexFunction
        
        mu, L, R, K_max = 0.1, 1.0, 1.0, 3
        t = jnp.array([0.5, 0.3, 0.2])  # Non-optimal varying stepsizes
        
        # ==== PEPit reference ====
        problem = PEP()
        func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        xs = func.stationary_point()
        x0 = problem.set_initial_point()
        problem.set_initial_condition((x0 - xs)**2 <= R**2)
        
        x = x0
        for k in range(K_max):
            g = func.gradient(x)
            x = x - float(t[k]) * g
        
        problem.set_performance_metric(func(x) - func(xs))
        pepit_obj = problem.solve(verbose=0)
        print(f"PEPit (varying stepsizes): {pepit_obj}")
        
        # ==== Our solution ====
        pep_data = construct_gd_pep_data(t=t, mu=mu, L=L, R=R, K_max=K_max, pep_obj='obj_val')
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        
        our_obj, _, _ = self._solve_primal_pep_cvxpy(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        print(f"Our (varying stepsizes): {our_obj}")
        
        np.testing.assert_allclose(our_obj, pepit_obj, rtol=1e-4)
    
    def test_matches_pepit_convex_only(self):
        """Test with mu=0 (smooth but not strongly convex)."""
        from PEPit import PEP
        from PEPit.functions import SmoothConvexFunction
        
        L, R, K_max = 1.0, 1.0, 5
        mu = 0.0
        t = jnp.array([1.0 / L] * K_max)  # Standard 1/L stepsize
        
        # ==== PEPit reference ====
        problem = PEP()
        func = problem.declare_function(SmoothConvexFunction, L=L)
        xs = func.stationary_point()
        x0 = problem.set_initial_point()
        problem.set_initial_condition((x0 - xs)**2 <= R**2)
        
        x = x0
        for k in range(K_max):
            g = func.gradient(x)
            x = x - float(t[k]) * g
        
        problem.set_performance_metric(func(x) - func(xs))
        pepit_obj = problem.solve(verbose=0)
        print(f"PEPit (convex only): {pepit_obj}")
        
        # ==== Our solution ====
        pep_data = construct_gd_pep_data(t=t, mu=mu, L=L, R=R, K_max=K_max, pep_obj='obj_val')
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        
        our_obj, _, _ = self._solve_primal_pep_cvxpy(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        print(f"Our (convex only): {our_obj}")
        
        np.testing.assert_allclose(our_obj, pepit_obj, rtol=1e-4)
    
    def test_constraint_satisfaction(self, pep_params, gd_stepsizes):
        """Test that solution satisfies interpolation constraints."""
        pep_data = construct_gd_pep_data(
            t=gd_stepsizes,
            mu=pep_params['mu'],
            L=pep_params['L'],
            R=pep_params['R'],
            K_max=pep_params['K_max'],
            pep_obj=pep_params['pep_obj'],
        )
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        
        _, G_star, F_star = self._solve_primal_pep_cvxpy(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        
        A_vals = np.array(A_vals)
        b_vals = np.array(b_vals)
        c_vals = np.array(c_vals)
        M = A_vals.shape[0]
        
        # Check G >> 0 (PSD)
        eigvals = np.linalg.eigvalsh(G_star)
        assert np.min(eigvals) >= -1e-6, f"G not PSD, min eigval: {np.min(eigvals)}"
        
        # Check interpolation constraints: trace(A_m @ G) + b_m @ F + c_m <= 0
        for m in range(M):
            lhs = float(np.trace(A_vals[m] @ G_star) + np.dot(b_vals[m], F_star) + c_vals[m])
            assert lhs <= 1e-5, f"Constraint {m} violated: {lhs} > 0"
        
        print(f"All {M} interpolation constraints satisfied")
    
    def test_larger_K(self):
        """Test with larger number of iterations."""
        from PEPit import PEP
        from PEPit.functions import SmoothStronglyConvexFunction
        
        mu, L, R, K_max = 0.5, 2.0, 1.0, 8
        t = jnp.array([2.0 / (L + mu)] * K_max)
        
        # ==== PEPit reference ====
        problem = PEP()
        func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        xs = func.stationary_point()
        x0 = problem.set_initial_point()
        problem.set_initial_condition((x0 - xs)**2 <= R**2)
        
        x = x0
        for k in range(K_max):
            g = func.gradient(x)
            x = x - float(t[k]) * g
        
        problem.set_performance_metric(func(x) - func(xs))
        pepit_obj = problem.solve(verbose=0)
        print(f"PEPit (K={K_max}): {pepit_obj}")
        
        # ==== Our solution ====
        pep_data = construct_gd_pep_data(t=t, mu=mu, L=L, R=R, K_max=K_max, pep_obj='obj_val')
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        
        our_obj, _, _ = self._solve_primal_pep_cvxpy(
            A_obj, b_obj, A_vals, b_vals, c_vals
        )
        print(f"Our (K={K_max}): {our_obj}")
        
        np.testing.assert_allclose(our_obj, pepit_obj, rtol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
