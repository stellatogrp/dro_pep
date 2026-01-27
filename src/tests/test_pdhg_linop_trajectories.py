"""
Unit tests verifying PDHG trajectories satisfy PEP constraints from linop construction.

Tests the trajectory mapping against the PEP interpolation inequalities from
pep_construction_chambolle_pock_linop.py.

The new API uses LP data arrays instead of functions for JAX JIT compatibility.

IMPORTANT: The trajectory-to-Gram mapping currently only works correctly for
pure saddle-point problems with c=0 and q=0:
    min_x max_y  <Kx, y> + indicator_{[l,u]}(x) - indicator_{y>=0}(y)

For general LPs with nonzero c and q, the PEP operator constraints coupled with
KKT conditions result in zero-valued adjusted subgradients, which is inconsistent
with nonzero function values. Extending to handle linear terms would require
modifying either the PEP construction or the trajectory mapping.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning.trajectories_pdhg import problem_data_to_pdhg_trajectories
from learning.pep_construction_chambolle_pock_linop import (
    construct_chambolle_pock_pep_data,
    chambolle_pock_pep_data_to_numpy,
)


class TestPDHGLinopTrajectories(unittest.TestCase):
    """Test PDHG trajectories against linop PEP construction."""

    def test_simple_lp(self):
        """
        Test with a simple LP where saddle point is at origin.

        For a valid PEP test, the saddle point must satisfy the optimality conditions:
        - K^T @ y_opt = -c (primal stationarity for interior point)
        - K @ x_opt = q (dual stationarity for interior point)

        We use c=0, q=0, x_opt=0, y_opt=0 which trivially satisfies these conditions.
        """
        np.random.seed(42)

        n = 3  # primal dimension
        m = 4  # dual dimension (all inequality constraints)
        m1 = 4  # number of inequality constraints
        K_max = 2

        # Random coupling matrix
        K_mat = jnp.array(np.random.randn(m, n) * 0.5)
        M = np.linalg.norm(K_mat, ord=2)

        # Zero cost/RHS -> saddle point at origin is valid
        c = jnp.zeros(n)
        q = jnp.zeros(m)

        # Box constraints (large to be non-binding)
        l = jnp.array([-10.0, -10.0, -10.0])
        u = jnp.array([10.0, 10.0, 10.0])

        # Optimal saddle point at origin (satisfies KKT conditions with c=0, q=0)
        x_opt = jnp.zeros(n)
        y_opt = jnp.zeros(m)

        # Step sizes
        tau = 0.5 / M
        sigma = 0.5 / M
        theta = 1.0

        # Initial point (ORIGINAL coordinates)
        x0 = jnp.array(np.random.randn(n) * 0.5)
        y0 = jnp.array(np.abs(np.random.randn(m) * 0.3))  # non-negative

        # Compute initial radius for P-norm
        dx0 = x0 - x_opt
        dy0 = y0 - y_opt
        K_dx0 = K_mat @ dx0
        R_sq = jnp.sum(dx0**2) / tau + jnp.sum(dy0**2) / sigma - 2 * jnp.dot(K_dx0, dy0)
        R = jnp.sqrt(jnp.maximum(R_sq, 1.0))

        # Get trajectory
        stepsizes = (jnp.array(tau), jnp.array(sigma), jnp.array(theta))
        G_traj, F_traj = problem_data_to_pdhg_trajectories(
            stepsizes,
            c, K_mat, q, l, u,
            x0, y0,
            x_opt, y_opt,
            K_max, m1,
            M=M
        )

        # Get PEP constraints
        pep_data = construct_chambolle_pock_pep_data(tau, sigma, theta, M, float(R), K_max)
        pep_data_np = chambolle_pock_pep_data_to_numpy(pep_data)

        A_obj, b_obj, A_vals, b_vals, c_vals, PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes = pep_data_np

        G_np = np.array(G_traj)
        F_np = np.array(F_traj)

        # Check dimensions
        self.assertEqual(G_np.shape[0], A_obj.shape[0],
            f"Gram dim mismatch: got {G_np.shape[0]}, expected {A_obj.shape[0]}")
        self.assertEqual(F_np.shape[0], b_obj.shape[0],
            f"F dim mismatch: got {F_np.shape[0]}, expected {b_obj.shape[0]}")

        # Check G is symmetric
        np.testing.assert_allclose(G_np, G_np.T, atol=1e-10,
            err_msg="Gram matrix should be symmetric")

        # Check interpolation constraints
        violations = []
        for i in range(A_vals.shape[0]):
            val = np.trace(A_vals[i] @ G_np) + np.dot(b_vals[i], F_np) + c_vals[i]
            violations.append(val)

        violations = np.array(violations)
        max_violation = np.max(violations)

        print(f"\nSimple LP test: max violation = {max_violation:.2e}")
        print(f"  Violations > 0: {np.sum(violations > 1e-6)} / {len(violations)}")

        # Assert all constraints satisfied (tolerance for numerical precision)
        self.assertLess(max_violation, 1e-6,
            f"Constraint violation too large: {max_violation:.2e}")

    def test_dimension_consistency(self):
        """Test that dimensions match between trajectory and PEP construction."""
        np.random.seed(42)
        n = 5
        m = 4
        m1 = 4  # all inequality constraints

        K_mat = jnp.array(np.random.randn(m, n))
        M = np.linalg.norm(K_mat, ord=2)

        c = jnp.zeros(n)
        q = jnp.zeros(m)
        l = -jnp.ones(n) * 10.0
        u = jnp.ones(n) * 10.0
        x_opt = jnp.zeros(n)
        y_opt = jnp.zeros(m)

        for K_max in [1, 2, 3, 5]:
            # Expected dimensions
            expected_dimG = 4 + 2*(K_max + 2) + 2*K_max + 3
            expected_dimF = 2*(K_max + 2)

            stepsizes = (0.5/M, 0.5/M, 1.0)
            x0 = jnp.zeros(n) + 0.1
            y0 = jnp.zeros(m) + 0.1

            G, F = problem_data_to_pdhg_trajectories(
                stepsizes,
                c, K_mat, q, l, u,
                x0, y0,
                x_opt, y_opt,
                K_max, m1,
                M=M
            )

            self.assertEqual(G.shape[0], expected_dimG,
                f"K={K_max}: G dim = {G.shape[0]}, expected {expected_dimG}")
            self.assertEqual(F.shape[0], expected_dimF,
                f"K={K_max}: F dim = {F.shape[0]}, expected {expected_dimF}")

            # Check PEP construction dimensions
            pep_data = construct_chambolle_pock_pep_data(0.5/M, 0.5/M, 1.0, M, 1.0, K_max)
            A_obj, b_obj, _, _, _, _, _, _, _ = pep_data

            self.assertEqual(A_obj.shape[0], expected_dimG)
            self.assertEqual(b_obj.shape[0], expected_dimF)

            print(f"K_max={K_max}: dimG={G.shape[0]}, dimF={F.shape[0]} OK")

    def test_gram_matrix_structure(self):
        """Test that Gram matrix has expected structure."""
        np.random.seed(42)

        n = 3
        m = 4
        m1 = 4
        K_max = 2

        K_mat = jnp.array(np.random.randn(m, n) * 0.5)
        M = np.linalg.norm(K_mat, ord=2)

        c = jnp.zeros(n)
        q = jnp.zeros(m)
        l = -jnp.ones(n) * 10.0
        u = jnp.ones(n) * 10.0
        x_opt = jnp.zeros(n)
        y_opt = jnp.zeros(m)

        x0 = jnp.array([0.5, -0.3, 0.2])
        y0 = jnp.array([0.1, 0.2, 0.3, 0.4])  # non-negative

        stepsizes = (0.5/M, 0.5/M, 1.0)
        G, F = problem_data_to_pdhg_trajectories(
            stepsizes,
            c, K_mat, q, l, u,
            x0, y0,
            x_opt, y_opt,
            K_max, m1,
            M=M
        )

        G_np = np.array(G)

        # Check symmetry
        np.testing.assert_allclose(G_np, G_np.T, atol=1e-12)

        # Check G[0,0] = ||dx0||^2 (primal-primal inner product)
        # dx0 = x0 - x_opt = x0 (since x_opt = 0)
        expected_dx0_norm_sq = np.sum(np.array(x0)**2)
        np.testing.assert_allclose(G_np[0, 0], expected_dx0_norm_sq, rtol=1e-6,
            err_msg="G[0,0] should be ||dx0||^2")

        # Check G[1,1] = ||dy0||^2 (dual-dual inner product)
        expected_dy0_norm_sq = np.sum(np.array(y0)**2)
        np.testing.assert_allclose(G_np[1, 1], expected_dy0_norm_sq, rtol=1e-6,
            err_msg="G[1,1] should be ||dy0||^2")

        # Check G[2,2] = ||x_s||^2 = 0 (saddle point at origin in shifted coords)
        self.assertAlmostEqual(G_np[2, 2], 0.0, places=10,
            msg="G[2,2] should be 0 (x_s = 0)")

        # Check G[3,3] = ||y_s||^2 = 0
        self.assertAlmostEqual(G_np[3, 3], 0.0, places=10,
            msg="G[3,3] should be 0 (y_s = 0)")

        # Check cross inner product G[0, 1] = <dx0, dy0>_W = <K dx0, dy0> / M
        dx0_np = np.array(x0) - np.array(x_opt)
        K_dx0 = np.array(K_mat) @ dx0_np
        dy0_np = np.array(y0) - np.array(y_opt)
        expected_cross = np.dot(K_dx0, dy0_np) / M
        np.testing.assert_allclose(G_np[0, 1], expected_cross, rtol=1e-6,
            err_msg="G[0,1] should be <K dx0, dy0> / M")

        print(f"\nGram structure test passed!")
        print(f"  G[0,0] = {G_np[0,0]:.6f} (expected {expected_dx0_norm_sq:.6f})")
        print(f"  G[1,1] = {G_np[1,1]:.6f} (expected {expected_dy0_norm_sq:.6f})")
        print(f"  G[0,1] = {G_np[0,1]:.6f} (expected {expected_cross:.6f})")

    def test_function_values_structure(self):
        """Test that function values F have correct structure."""
        np.random.seed(42)

        n = 3
        m = 4
        m1 = 4
        K_max = 2

        K_mat = jnp.array(np.random.randn(m, n) * 0.5)
        M = np.linalg.norm(K_mat, ord=2)

        # For LP: f1(x) = c^T x, h(y) = q^T y
        c = jnp.array([1.0, 0.5, -0.3])
        q = jnp.array([0.2, -0.1, 0.4, 0.1])
        l = -jnp.ones(n) * 10.0
        u = jnp.ones(n) * 10.0
        x_opt = jnp.zeros(n)
        y_opt = jnp.zeros(m)

        x0 = jnp.array([0.5, -0.3, 0.2])
        y0 = jnp.array([0.1, 0.2, 0.3, 0.4])

        stepsizes = (0.5/M, 0.5/M, 1.0)
        G, F = problem_data_to_pdhg_trajectories(
            stepsizes,
            c, K_mat, q, l, u,
            x0, y0,
            x_opt, y_opt,
            K_max, m1,
            M=M
        )

        F_np = np.array(F)
        dimF1 = K_max + 2

        F1 = F_np[:dimF1]
        F_h = F_np[dimF1:]

        # Last elements should be 0 (at optimal, which is at origin in shifted coords)
        self.assertAlmostEqual(F1[-1], 0.0, places=10)
        self.assertAlmostEqual(F_h[-1], 0.0, places=10)

        # F1[0] = f1(x0) - f1(x_opt) = c^T x0 - c^T x_opt = c^T x0 (since x_opt = 0)
        expected_f1_0 = np.dot(np.array(c), np.array(x0))
        np.testing.assert_allclose(F1[0], expected_f1_0, rtol=1e-6)

        # F_h[0] = h(y0) - h(y_opt) = q^T y0 - q^T y_opt = q^T y0 (since y_opt = 0)
        expected_h_0 = np.dot(np.array(q), np.array(y0))
        np.testing.assert_allclose(F_h[0], expected_h_0, rtol=1e-6)

        print(f"\nFunction values test passed!")
        print(f"  F1[0] = {F1[0]:.6f} (expected {expected_f1_0:.6f})")
        print(f"  F_h[0] = {F_h[0]:.6f} (expected {expected_h_0:.6f})")
        print(f"  F1[-1] = {F1[-1]:.6f} (expected 0)")
        print(f"  F_h[-1] = {F_h[-1]:.6f} (expected 0)")

    def test_lp_with_equality_constraints(self):
        """
        Test LP with both inequality and equality constraints.

        Uses c=0, q=0, x_opt=0, y_opt=0 to ensure valid saddle point conditions.
        """
        np.random.seed(123)

        n = 4  # primal dimension
        m1 = 2  # inequality constraints
        m2 = 2  # equality constraints
        m = m1 + m2  # total dual dimension
        K_max = 3

        # Stacked constraint matrix K = [G; A]
        G_ineq = jnp.array(np.random.randn(m1, n) * 0.5)
        A_eq = jnp.array(np.random.randn(m2, n) * 0.5)
        K_mat = jnp.vstack([G_ineq, A_eq])
        M = np.linalg.norm(K_mat, ord=2)

        # Zero cost/RHS for valid saddle point at origin
        c = jnp.zeros(n)
        q = jnp.zeros(m)
        l = -jnp.ones(n) * 10.0
        u = jnp.ones(n) * 10.0

        # Optimal point at origin
        x_opt = jnp.zeros(n)
        y_opt = jnp.zeros(m)

        tau = 0.5 / M
        sigma = 0.5 / M
        theta = 1.0

        x0 = jnp.array(np.random.randn(n) * 0.3)
        y0 = jnp.array([0.2, 0.3, 0.1, 0.05])  # first m1 non-negative

        # Compute actual initial radius for P-norm
        dx0 = x0 - x_opt
        dy0 = y0 - y_opt
        K_dx0 = K_mat @ dx0
        R_sq = jnp.sum(dx0**2) / tau + jnp.sum(dy0**2) / sigma - 2 * jnp.dot(K_dx0, dy0)
        R = float(jnp.sqrt(jnp.maximum(R_sq, 1.0)))

        stepsizes = (jnp.array(tau), jnp.array(sigma), jnp.array(theta))
        G_traj, F_traj = problem_data_to_pdhg_trajectories(
            stepsizes,
            c, K_mat, q, l, u,
            x0, y0,
            x_opt, y_opt,
            K_max, m1,
            M=M
        )

        G_np = np.array(G_traj)
        F_np = np.array(F_traj)

        # Expected dimensions
        expected_dimG = 4 + 2*(K_max + 2) + 2*K_max + 3
        expected_dimF = 2*(K_max + 2)

        self.assertEqual(G_np.shape[0], expected_dimG)
        self.assertEqual(F_np.shape[0], expected_dimF)

        # Check symmetry
        np.testing.assert_allclose(G_np, G_np.T, atol=1e-10)

        # Check interpolation constraints
        pep_data = construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K_max)
        pep_data_np = chambolle_pock_pep_data_to_numpy(pep_data)
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data_np

        violations = []
        for i in range(A_vals.shape[0]):
            val = np.trace(A_vals[i] @ G_np) + np.dot(b_vals[i], F_np) + c_vals[i]
            violations.append(val)
        max_violation = np.max(violations)

        print(f"\nLP with equality constraints test:")
        print(f"  n={n}, m1={m1}, m2={m2}, K_max={K_max}")
        print(f"  dimG={G_np.shape[0]}, dimF={F_np.shape[0]}")
        print(f"  max violation = {max_violation:.2e}")

        self.assertLess(max_violation, 1e-6,
            f"Constraint violation too large: {max_violation:.2e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
