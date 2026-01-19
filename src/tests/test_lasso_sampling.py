"""
Unit tests for Lasso sampling functions in learning_experiment_classes/lasso.py.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning_experiment_classes.lasso import (
    generate_A,
    generate_single_b_jax,
    generate_batch_b_jax,
    solve_lasso_cvxpy,
    solve_batch_lasso_cvxpy,
    compute_lasso_params,
)


class TestAMatrixGeneration(unittest.TestCase):
    """Tests for A matrix generation."""
    
    def test_generate_A_shape(self):
        """Test that A generation produces correct shape."""
        A = generate_A(seed=1000, m=20, n=30, p_A_nonzero=0.4)
        self.assertEqual(A.shape, (20, 30))
    
    def test_generate_A_column_norms(self):
        """Test that columns of A are normalized."""
        A = generate_A(seed=1000, m=20, n=30, p_A_nonzero=0.4)
        col_norms = np.linalg.norm(A, axis=0)
        np.testing.assert_allclose(col_norms, np.ones(30), atol=1e-6)
    
    def test_generate_A_reproducibility(self):
        """Test that A generation is reproducible with same seed."""
        A1 = generate_A(seed=1000, m=20, n=30, p_A_nonzero=0.4)
        A2 = generate_A(seed=1000, m=20, n=30, p_A_nonzero=0.4)
        np.testing.assert_array_equal(A1, A2)
    
    def test_generate_A_different_seeds(self):
        """Test that different seeds produce different matrices."""
        A1 = generate_A(seed=1000, m=20, n=30, p_A_nonzero=0.4)
        A2 = generate_A(seed=2000, m=20, n=30, p_A_nonzero=0.4)
        self.assertFalse(np.allclose(A1, A2))


class TestBVectorGeneration(unittest.TestCase):
    """Tests for b vector generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        A_np = generate_A(seed=1000, m=20, n=30, p_A_nonzero=0.4)
        self.A = jnp.array(A_np)
    
    def test_generate_single_b_shape(self):
        """Test single b generation shape."""
        key = jax.random.PRNGKey(2000)
        b = generate_single_b_jax(key, self.A, p_xsamp_nonzero=0.25, noise_eps=0.001)
        self.assertEqual(b.shape, (20,))
    
    def test_generate_batch_b_shape(self):
        """Test batch b generation shape."""
        key = jax.random.PRNGKey(2000)
        b_batch = generate_batch_b_jax(key, self.A, N=5, p_xsamp_nonzero=0.25, noise_eps=0.001)
        self.assertEqual(b_batch.shape, (5, 20))
    
    def test_generate_batch_b_different_samples(self):
        """Test that batch contains different samples."""
        key = jax.random.PRNGKey(2000)
        b_batch = generate_batch_b_jax(key, self.A, N=5, p_xsamp_nonzero=0.25, noise_eps=0.001)
        # Each row should be different
        for i in range(4):
            self.assertFalse(jnp.allclose(b_batch[i], b_batch[i+1]))
    
    def test_generate_batch_b_reproducibility(self):
        """Test that same key produces same batch."""
        key = jax.random.PRNGKey(2000)
        b1 = generate_batch_b_jax(key, self.A, N=5, p_xsamp_nonzero=0.25, noise_eps=0.001)
        b2 = generate_batch_b_jax(key, self.A, N=5, p_xsamp_nonzero=0.25, noise_eps=0.001)
        np.testing.assert_array_equal(np.array(b1), np.array(b2))


class TestLassoSolver(unittest.TestCase):
    """Tests for Lasso solving via CVXPY."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.A_np = generate_A(seed=1000, m=20, n=30, p_A_nonzero=0.4)
        key = jax.random.PRNGKey(2000)
        b_jax = generate_single_b_jax(key, jnp.array(self.A_np), 
                                       p_xsamp_nonzero=0.25, noise_eps=0.001)
        self.b_np = np.array(b_jax)
        self.lambd = 0.1
    
    def test_solve_lasso_cvxpy_returns_solution(self):
        """Test that solver returns valid solution."""
        x_opt, R = solve_lasso_cvxpy(self.A_np, self.b_np, self.lambd)
        self.assertEqual(x_opt.shape, (30,))
        self.assertIsInstance(R, float)
        self.assertGreaterEqual(R, 0)
    
    def test_solve_lasso_cvxpy_radius_is_norm(self):
        """Test that R equals norm of solution."""
        x_opt, R = solve_lasso_cvxpy(self.A_np, self.b_np, self.lambd)
        np.testing.assert_almost_equal(R, np.linalg.norm(x_opt), decimal=6)
    
    def test_solve_batch_lasso_shape(self):
        """Test batch solving returns correct shape."""
        key = jax.random.PRNGKey(2000)
        b_batch = generate_batch_b_jax(key, jnp.array(self.A_np), N=3,
                                        p_xsamp_nonzero=0.25, noise_eps=0.001)
        b_batch_np = np.array(b_batch)
        
        x_opt_batch, R_max = solve_batch_lasso_cvxpy(self.A_np, b_batch_np, self.lambd)
        self.assertEqual(x_opt_batch.shape, (3, 30))
        self.assertGreaterEqual(R_max, 0)


class TestLassoParams(unittest.TestCase):
    """Tests for L, mu computation."""
    
    def test_compute_lasso_params_overdetermined(self):
        """Test L, mu for overdetermined system (m > n)."""
        A = generate_A(seed=1000, m=30, n=20, p_A_nonzero=0.5)
        L, mu = compute_lasso_params(jnp.array(A))
        
        self.assertGreater(L, 0)
        self.assertGreaterEqual(mu, 0)
        self.assertGreaterEqual(L, mu)
    
    def test_compute_lasso_params_underdetermined(self):
        """Test L, mu for underdetermined system (m < n)."""
        A = generate_A(seed=1000, m=20, n=30, p_A_nonzero=0.5)
        L, mu = compute_lasso_params(jnp.array(A))
        
        self.assertGreater(L, 0)
        self.assertEqual(mu, 0.0)  # mu should be 0 for m < n


if __name__ == '__main__':
    unittest.main()
