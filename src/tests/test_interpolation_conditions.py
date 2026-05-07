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

from learning.pep_constructions import (
    smooth_strongly_convex_interp,
    smooth_strongly_convex_interp_consecutive
)


def numpy_smooth_strongly_convex(repX, repG, repF, mu, L):
    """
    Reference numpy implementation for comparison.

    Arrays contain only algorithm iterates (no stationary point row).
    Optimal point constraints are computed with explicit zeros.

    Returns A_list, b_list as numpy arrays.
    """
    n_points = len(repX)  # Number of algorithm points
    dimG = repX.shape[1]
    dimF = repF.shape[1]

    # Explicit zeros for stationary point
    xs = np.zeros(dimG)
    gs = np.zeros(dimG)
    fs = np.zeros(dimF)

    A_list, b_list = [], []
    coeff = 1.0 / (2.0 * (1.0 - mu / L))

    def compute_constraint(xi, xj, gi, gj, fi, fj):
        diff_x = xi - xj
        diff_g = gi - gj
        Ai = (1 / 2) * np.outer(gj, diff_x) + (1 / 2) * np.outer(diff_x, gj)
        Ai += coeff * (
            (1 / L) * np.outer(diff_g, diff_g)
            + mu * np.outer(diff_x, diff_x)
            - (mu / L) * np.outer(diff_g, diff_x)
            - (mu / L) * np.outer(diff_x, diff_g)
        )
        bi = fj - fi
        return Ai, bi

    # Part 1: Algorithm point pairs (i, j) with i != j
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                Ai, bi = compute_constraint(
                    repX[i, :], repX[j, :],
                    repG[i, :], repG[j, :],
                    repF[i, :], repF[j, :]
                )
                A_list.append(Ai)
                b_list.append(bi)

    # Part 2: (i, s) constraints for each algorithm point i
    for i in range(n_points):
        Ai, bi = compute_constraint(
            repX[i, :], xs,
            repG[i, :], gs,
            repF[i, :], fs
        )
        A_list.append(Ai)
        b_list.append(bi)

    # Part 3: (s, j) constraints for each algorithm point j
    for j in range(n_points):
        Ai, bi = compute_constraint(
            xs, repX[j, :],
            gs, repG[j, :],
            fs, repF[j, :]
        )
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

        # Create random test data (n_points rows, no stationary point)
        np.random.seed(42)
        self.repX_np = np.random.randn(self.n_points, self.dimG)
        self.repG_np = np.random.randn(self.n_points, self.dimG)
        self.repF_np = np.random.randn(self.n_points, self.dimF)

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

        # Arrays without stationary point
        np.random.seed(123)
        self.repX = jnp.array(np.random.randn(self.n_points, self.dimG))
        self.repG = jnp.array(np.random.randn(self.n_points, self.dimG))
        self.repF = jnp.array(np.random.randn(self.n_points, self.dimF))
    
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

        # Arrays without stationary point
        repX = jnp.ones((n_points, dimG))
        repG = jnp.ones((n_points, dimG))
        repF = jnp.ones((n_points, dimF))

        A_vals, b_vals = smooth_strongly_convex_interp(
            repX, repG, repF, 0.1, 1.0, n_points
        )

        # 1 algorithm point + stationary point = 2 constraints: (0, s) and (s, 0)
        # n_points*(n_points+1) = 1*2 = 2
        self.assertEqual(A_vals.shape[0], 2)

    def test_large_L_over_mu_ratio(self):
        """Test with large condition number L/mu."""
        n_points = 2
        dimG = 4
        dimF = 3

        # Arrays without stationary point
        repX = jnp.array(np.random.randn(n_points, dimG))
        repG = jnp.array(np.random.randn(n_points, dimG))
        repF = jnp.array(np.random.randn(n_points, dimF))

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


def numpy_convex(repX, repG, repF):
    """
    Reference numpy implementation for convex function interpolation.

    Arrays contain only algorithm iterates (no stationary point row).
    Optimal point constraints are computed with explicit zeros.

    Returns A_list, b_list as numpy arrays.
    """
    n_points = len(repX)  # Number of algorithm points
    dimG = repX.shape[1]
    dimF = repF.shape[1]

    # Explicit zeros for stationary point
    xs = np.zeros(dimG)
    gs = np.zeros(dimG)
    fs = np.zeros(dimF)

    A_list, b_list = [], []

    def compute_constraint(xi, xj, gj, fi, fj):
        diff_x = xi - xj
        Ai = (1 / 2) * np.outer(gj, diff_x) + (1 / 2) * np.outer(diff_x, gj)
        bi = fj - fi
        return Ai, bi

    # Part 1: Algorithm point pairs (i, j) with i != j
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                Ai, bi = compute_constraint(
                    repX[i, :], repX[j, :],
                    repG[j, :],
                    repF[i, :], repF[j, :]
                )
                A_list.append(Ai)
                b_list.append(bi)

    # Part 2: (i, s) constraints for each algorithm point i
    for i in range(n_points):
        Ai, bi = compute_constraint(
            repX[i, :], xs,
            gs,
            repF[i, :], fs
        )
        A_list.append(Ai)
        b_list.append(bi)

    # Part 3: (s, j) constraints for each algorithm point j
    for j in range(n_points):
        Ai, bi = compute_constraint(
            xs, repX[j, :],
            repG[j, :],
            fs, repF[j, :]
        )
        A_list.append(Ai)
        b_list.append(bi)

    return np.array(A_list), np.array(b_list)


class TestConvexInterp(unittest.TestCase):
    """Tests for convex_interp function."""

    def setUp(self):
        """Set up test fixtures."""
        from learning.pep_constructions import convex_interp
        self.convex_interp = convex_interp

        self.n_points = 3
        self.dimG = 5
        self.dimF = 4

        # Create random test data (n_points rows, no stationary point)
        np.random.seed(42)
        self.repX_np = np.random.randn(self.n_points, self.dimG)
        self.repG_np = np.random.randn(self.n_points, self.dimG)
        self.repF_np = np.random.randn(self.n_points, self.dimF)

        # JAX versions
        self.repX = jnp.array(self.repX_np)
        self.repG = jnp.array(self.repG_np)
        self.repF = jnp.array(self.repF_np)
    
    def test_output_shapes(self):
        """Test that output shapes are correct."""
        A_vals, b_vals = self.convex_interp(
            self.repX, self.repG, self.repF, self.n_points
        )
        
        expected_num_constraints = (self.n_points + 1) * self.n_points
        
        self.assertEqual(A_vals.shape, (expected_num_constraints, self.dimG, self.dimG))
        self.assertEqual(b_vals.shape, (expected_num_constraints, self.dimF))
    
    def test_matches_numpy_reference(self):
        """Test that JAX implementation matches numpy reference."""
        # Get JAX result
        A_jax, b_jax = self.convex_interp(
            self.repX, self.repG, self.repF, self.n_points
        )
        
        # Get numpy reference result
        A_np, b_np = numpy_convex(
            self.repX_np, self.repG_np, self.repF_np
        )
        
        # Compare
        np.testing.assert_allclose(np.array(A_jax), A_np, rtol=1e-10)
        np.testing.assert_allclose(np.array(b_jax), b_np, rtol=1e-10)
    
    def test_symmetry_of_A_matrices(self):
        """Test that each A matrix is symmetric."""
        A_vals, _ = self.convex_interp(
            self.repX, self.repG, self.repF, self.n_points
        )
        
        for i in range(A_vals.shape[0]):
            np.testing.assert_allclose(
                A_vals[i], A_vals[i].T, rtol=1e-10,
                err_msg=f"A_vals[{i}] is not symmetric"
            )
    
    def test_differentiability(self):
        """Test that the function is differentiable w.r.t. inputs."""
        def loss_fn(repX, repG, repF):
            A_vals, b_vals = self.convex_interp(
                repX, repG, repF, self.n_points
            )
            return jnp.sum(A_vals ** 2) + jnp.sum(b_vals ** 2)
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
        grads = grad_fn(self.repX, self.repG, self.repF)
        
        # Check gradients have correct shapes
        self.assertEqual(grads[0].shape, self.repX.shape)
        self.assertEqual(grads[1].shape, self.repG.shape)
        self.assertEqual(grads[2].shape, self.repF.shape)
        
        # Check gradients are not all zero
        self.assertGreater(jnp.abs(grads[0]).sum(), 0)
        self.assertGreater(jnp.abs(grads[1]).sum(), 0)


class TestProximalGradientInterp(unittest.TestCase):
    """Tests for smooth_strongly_convex_proximal_gradient_interp function."""

    def setUp(self):
        """Set up test fixtures."""
        from learning.pep_constructions import (
            smooth_strongly_convex_proximal_gradient_interp
        )
        self.prox_grad_interp = smooth_strongly_convex_proximal_gradient_interp

        self.K = 3  # Number of iterations
        self.dimG = 2 * self.K + 3  # Gram basis dimension: 1 + (K+1) + (K+1) = 2K+3
        self.dimF1 = self.K + 1  # f1 function value dimension
        self.dimF2 = self.K + 1  # f2 function value dimension (now includes x0)
        self.mu = 0.1
        self.L = 1.0

        # Create random test data for f1 (smooth strongly convex)
        # f1 has K+1 algorithm points: x_0, x_1, ..., x_K (no x_s in arrays)
        np.random.seed(42)
        self.repX_f1 = jnp.array(np.random.randn(self.K + 1, self.dimG))
        self.repG_f1 = jnp.array(np.random.randn(self.K + 1, self.dimG))
        self.repF_f1 = jnp.array(np.random.randn(self.K + 1, self.dimF1))

        # Create random test data for f2 (convex)
        # f2 has K+1 algorithm points: x_0, x_1, ..., x_K (no x_s in arrays)
        self.repX_f2 = jnp.array(np.random.randn(self.K + 1, self.dimG))
        self.repG_f2 = jnp.array(np.random.randn(self.K + 1, self.dimG))
        self.repF_f2 = jnp.array(np.random.randn(self.K + 1, self.dimF2))
    
    def test_output_shapes(self):
        """Test that output shapes are correct."""
        A_f1, b_f1, A_f2, b_f2 = self.prox_grad_interp(
            self.repX_f1, self.repG_f1, self.repF_f1,
            self.repX_f2, self.repG_f2, self.repF_f2,
            self.mu, self.L, self.K
        )

        # f1 has K+1 algorithm points, so n_points*(n_points+1) = (K+1)*(K+2) constraints
        n_points_f1 = self.K + 1
        expected_f1_constraints = n_points_f1 * (n_points_f1 + 1)

        # f2 has K+1 algorithm points, so n_points*(n_points+1) = (K+1)*(K+2) constraints
        n_points_f2 = self.K + 1
        expected_f2_constraints = n_points_f2 * (n_points_f2 + 1)

        self.assertEqual(A_f1.shape, (expected_f1_constraints, self.dimG, self.dimG))
        self.assertEqual(b_f1.shape, (expected_f1_constraints, self.dimF1))
        self.assertEqual(A_f2.shape, (expected_f2_constraints, self.dimG, self.dimG))
        self.assertEqual(b_f2.shape, (expected_f2_constraints, self.dimF2))
    
    def test_symmetry_of_A_matrices(self):
        """Test that each A matrix is symmetric."""
        A_f1, _, A_f2, _ = self.prox_grad_interp(
            self.repX_f1, self.repG_f1, self.repF_f1,
            self.repX_f2, self.repG_f2, self.repF_f2,
            self.mu, self.L, self.K
        )
        
        # Check f1 A matrices
        for i in range(A_f1.shape[0]):
            np.testing.assert_allclose(
                A_f1[i], A_f1[i].T, rtol=1e-10,
                err_msg=f"A_f1[{i}] is not symmetric"
            )
        
        # Check f2 A matrices
        for i in range(A_f2.shape[0]):
            np.testing.assert_allclose(
                A_f2[i], A_f2[i].T, rtol=1e-10,
                err_msg=f"A_f2[{i}] is not symmetric"
            )
    
    def test_differentiability(self):
        """Test that the function is differentiable."""
        def loss_fn(repX_f1, repG_f1, repF_f1, repX_f2, repG_f2, repF_f2):
            A_f1, b_f1, A_f2, b_f2 = self.prox_grad_interp(
                repX_f1, repG_f1, repF_f1,
                repX_f2, repG_f2, repF_f2,
                self.mu, self.L, self.K
            )
            return (jnp.sum(A_f1 ** 2) + jnp.sum(b_f1 ** 2) + 
                    jnp.sum(A_f2 ** 2) + jnp.sum(b_f2 ** 2))
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))
        grads = grad_fn(
            self.repX_f1, self.repG_f1, self.repF_f1,
            self.repX_f2, self.repG_f2, self.repF_f2
        )
        
        # Check gradients have correct shapes
        self.assertEqual(grads[0].shape, self.repX_f1.shape)
        self.assertEqual(grads[1].shape, self.repG_f1.shape)
        self.assertEqual(grads[2].shape, self.repF_f1.shape)
        self.assertEqual(grads[3].shape, self.repX_f2.shape)
        self.assertEqual(grads[4].shape, self.repG_f2.shape)
        self.assertEqual(grads[5].shape, self.repF_f2.shape)
    
    def test_mu_equals_zero(self):
        """Test with mu=0 (smooth convex, not strongly convex for f1)."""
        A_f1, b_f1, A_f2, b_f2 = self.prox_grad_interp(
            self.repX_f1, self.repG_f1, self.repF_f1,
            self.repX_f2, self.repG_f2, self.repF_f2,
            0.0, self.L, self.K
        )
        
        # Should produce valid output without NaN
        self.assertFalse(jnp.any(jnp.isnan(A_f1)))
        self.assertFalse(jnp.any(jnp.isnan(A_f2)))


if __name__ == '__main__':
    unittest.main()
