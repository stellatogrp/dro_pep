"""
Unit tests for JAX interpolation conditions.

Tests the JAX implementation against the reference numpy implementation
from experiment_classes/lyap_classes/gd.py.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning.interpolation_conditions import (
    smooth_strongly_convex_interp,
    smooth_strongly_convex_interp_consecutive
)


def numpy_smooth_strongly_convex(repX, repG, repF, mu, L):
    """
    Reference numpy implementation from gd.py for comparison.
    Returns A_list, b_list as numpy arrays.
    """
    n_points = len(repX) - 1
    A_list, b_list = [], []
    
    for i in range(n_points + 1):
        for j in range(n_points + 1):
            if i != j:
                xi, xj = repX[i, :], repX[j, :]
                gi, gj = repG[i, :], repG[j, :]
                fi, fj = repF[i, :], repF[j, :]
                
                Ai = (1 / 2) * np.outer(gj, xi - xj) + (1 / 2) * np.outer(xi - xj, gj)
                Ai += 1 / 2 / (1 - (mu / L)) * (
                    (1 / L) * np.outer(gi - gj, gi - gj)
                    + mu * np.outer(xi - xj, xi - xj)
                    - (mu / L) * np.outer(gi - gj, xi - xj)
                    - (mu / L) * np.outer(xi - xj, gi - gj)
                )
                bi = fj - fi
                
                A_list.append(Ai)
                b_list.append(bi)
    
    return np.array(A_list), np.array(b_list)


class TestSmoothStronglyConvexInterp(unittest.TestCase):
    """Tests for smooth_strongly_convex_interp function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_points = 3
        self.dimG = 5
        self.dimF = 4
        self.mu = 0.1
        self.L = 1.0
        
        # Create random test data
        np.random.seed(42)
        self.repX_np = np.random.randn(self.n_points + 1, self.dimG)
        self.repG_np = np.random.randn(self.n_points + 1, self.dimG)
        self.repF_np = np.random.randn(self.n_points + 1, self.dimF)
        
        # JAX versions
        self.repX = jnp.array(self.repX_np)
        self.repG = jnp.array(self.repG_np)
        self.repF = jnp.array(self.repF_np)
    
    def test_output_shapes(self):
        """Test that output shapes are correct."""
        A_vals, b_vals = smooth_strongly_convex_interp(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        
        expected_num_constraints = (self.n_points + 1) * self.n_points
        
        self.assertEqual(A_vals.shape, (expected_num_constraints, self.dimG, self.dimG))
        self.assertEqual(b_vals.shape, (expected_num_constraints, self.dimF))
    
    def test_matches_numpy_reference(self):
        """Test that JAX implementation matches numpy reference."""
        # Get JAX result
        A_jax, b_jax = smooth_strongly_convex_interp(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        
        # Get numpy reference result
        A_np, b_np = numpy_smooth_strongly_convex(
            self.repX_np, self.repG_np, self.repF_np, self.mu, self.L
        )
        
        # Compare (both should have same ordering)
        np.testing.assert_allclose(np.array(A_jax), A_np, rtol=1e-10)
        np.testing.assert_allclose(np.array(b_jax), b_np, rtol=1e-10)
    
    def test_jit_compilation(self):
        """Test that JIT compilation works and produces same results."""
        # First call (triggers compilation)
        A1, b1 = smooth_strongly_convex_interp(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        
        # Second call (uses compiled version)
        A2, b2 = smooth_strongly_convex_interp(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        
        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(b1, b2)
    
    def test_differentiability(self):
        """Test that the function is differentiable w.r.t. inputs."""
        def loss_fn(repX, repG, repF):
            A_vals, b_vals = smooth_strongly_convex_interp(
                repX, repG, repF, self.mu, self.L, self.n_points
            )
            return jnp.sum(A_vals ** 2) + jnp.sum(b_vals ** 2)
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
        grads = grad_fn(self.repX, self.repG, self.repF)
        
        # Check gradients have correct shapes
        self.assertEqual(grads[0].shape, self.repX.shape)
        self.assertEqual(grads[1].shape, self.repG.shape)
        self.assertEqual(grads[2].shape, self.repF.shape)
        
        # Check gradients are not all zero (non-trivial)
        self.assertGreater(jnp.abs(grads[0]).sum(), 0)
        self.assertGreater(jnp.abs(grads[1]).sum(), 0)
        self.assertGreater(jnp.abs(grads[2]).sum(), 0)
    
    def test_symmetry_of_A_matrices(self):
        """Test that each A matrix is symmetric."""
        A_vals, _ = smooth_strongly_convex_interp(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        
        for i in range(A_vals.shape[0]):
            np.testing.assert_allclose(
                A_vals[i], A_vals[i].T, rtol=1e-10,
                err_msg=f"A_vals[{i}] is not symmetric"
            )
    
    def test_mu_equals_zero(self):
        """Test with mu=0 (smooth convex, not strongly convex)."""
        A_vals, b_vals = smooth_strongly_convex_interp(
            self.repX, self.repG, self.repF, 0.0, self.L, self.n_points
        )
        
        # Should still produce valid output
        expected_num_constraints = (self.n_points + 1) * self.n_points
        self.assertEqual(A_vals.shape[0], expected_num_constraints)
        
        # Compare with numpy reference
        A_np, b_np = numpy_smooth_strongly_convex(
            self.repX_np, self.repG_np, self.repF_np, 0.0, self.L
        )
        np.testing.assert_allclose(np.array(A_vals), A_np, rtol=1e-10)


class TestSmoothStronglyConvexInterpConsecutive(unittest.TestCase):
    """Tests for smooth_strongly_convex_interp_consecutive function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_points = 4
        self.dimG = 5
        self.dimF = 3
        self.mu = 0.2
        self.L = 2.0
        
        np.random.seed(123)
        self.repX = jnp.array(np.random.randn(self.n_points + 1, self.dimG))
        self.repG = jnp.array(np.random.randn(self.n_points + 1, self.dimG))
        self.repF = jnp.array(np.random.randn(self.n_points + 1, self.dimF))
    
    def test_output_shapes(self):
        """Test that output shapes are correct."""
        A_vals, b_vals = smooth_strongly_convex_interp_consecutive(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        
        # (n_points - 1) consecutive + n_points optimal
        expected_num_constraints = 2 * self.n_points - 1
        
        self.assertEqual(A_vals.shape, (expected_num_constraints, self.dimG, self.dimG))
        self.assertEqual(b_vals.shape, (expected_num_constraints, self.dimF))
    
    def test_fewer_constraints_than_full(self):
        """Test that consecutive version has fewer constraints than full."""
        A_full, _ = smooth_strongly_convex_interp(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        A_consec, _ = smooth_strongly_convex_interp_consecutive(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        
        self.assertLess(A_consec.shape[0], A_full.shape[0])
    
    def test_jit_compilation(self):
        """Test that JIT compilation works."""
        # Trigger compilation
        A1, b1 = smooth_strongly_convex_interp_consecutive(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        
        # Second call
        A2, b2 = smooth_strongly_convex_interp_consecutive(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        
        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(b1, b2)
    
    def test_differentiability(self):
        """Test differentiability of consecutive version."""
        def loss_fn(repX, repG, repF):
            A_vals, b_vals = smooth_strongly_convex_interp_consecutive(
                repX, repG, repF, self.mu, self.L, self.n_points
            )
            return jnp.sum(A_vals ** 2) + jnp.sum(b_vals ** 2)
        
        grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
        grads = grad_fn(self.repX, self.repG, self.repF)
        
        self.assertEqual(grads[0].shape, self.repX.shape)
        self.assertGreater(jnp.abs(grads[0]).sum(), 0)
    
    def test_symmetry_of_A_matrices(self):
        """Test that each A matrix is symmetric."""
        A_vals, _ = smooth_strongly_convex_interp_consecutive(
            self.repX, self.repG, self.repF, self.mu, self.L, self.n_points
        )
        
        for i in range(A_vals.shape[0]):
            np.testing.assert_allclose(
                A_vals[i], A_vals[i].T, rtol=1e-10,
                err_msg=f"A_vals[{i}] is not symmetric"
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_single_point(self):
        """Test with n_points=1 (minimal case)."""
        n_points = 1
        dimG = 3
        dimF = 2
        
        repX = jnp.ones((n_points + 1, dimG))
        repG = jnp.ones((n_points + 1, dimG))
        repF = jnp.ones((n_points + 1, dimF))
        
        A_vals, b_vals = smooth_strongly_convex_interp(
            repX, repG, repF, 0.1, 1.0, n_points
        )
        
        # 2 points, 2 constraints (i,j) pairs with i != j
        self.assertEqual(A_vals.shape[0], 2)
    
    def test_large_L_over_mu_ratio(self):
        """Test with large condition number L/mu."""
        n_points = 2
        dimG = 4
        dimF = 3
        
        repX = jnp.array(np.random.randn(n_points + 1, dimG))
        repG = jnp.array(np.random.randn(n_points + 1, dimG))
        repF = jnp.array(np.random.randn(n_points + 1, dimF))
        
        # Large condition number
        mu = 0.001
        L = 1000.0
        
        A_vals, b_vals = smooth_strongly_convex_interp(
            repX, repG, repF, mu, L, n_points
        )
        
        # Should not produce NaN or Inf
        self.assertFalse(jnp.any(jnp.isnan(A_vals)))
        self.assertFalse(jnp.any(jnp.isinf(A_vals)))
        self.assertFalse(jnp.any(jnp.isnan(b_vals)))
        self.assertFalse(jnp.any(jnp.isinf(b_vals)))


if __name__ == '__main__':
    unittest.main()
