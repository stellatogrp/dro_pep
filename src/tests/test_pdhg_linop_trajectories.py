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
from learning.pep_constructions import (
    construct_chambolle_pock_pep_data,
    chambolle_pock_pep_data_to_numpy,
)
from learning_experiment_classes.pdlp import (
    generate_facility_location_problem,
    extract_constraint_matrices,
    FacilityLocationDPP,
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

    def test_nonzero_optimal_with_linear_terms(self):
        """
        Test with nonzero c, q AND nonzero x_opt, y_opt.

        This is the critical test for the coordinate shifting fix - we need
        to verify that the algorithm works correctly when both the linear
        cost terms (c, q) and the optimal point (x_opt, y_opt) are nonzero.
        """
        np.random.seed(123)

        n = 4
        m1 = 3
        m2 = 2
        m = m1 + m2
        K_max = 3

        # Nonzero linear cost terms
        c = jnp.array([1.0, -0.5, 0.3, -0.2])
        q = jnp.array([0.5, -0.3, 0.2, -0.1, 0.4])

        # Box constraints
        l = jnp.array([-2.0, -2.0, -2.0, -2.0])
        u = jnp.array([3.0, 3.0, 3.0, 3.0])

        # Constraint matrix
        K_mat = jnp.array(np.random.randn(m, n) * 0.3)
        M = np.linalg.norm(K_mat, ord=2)

        # NONZERO optimal point (this is the key difference from other tests)
        x_opt = jnp.array([0.5, -0.3, 0.7, 0.2])
        y_opt = jnp.array([0.2, -0.1, 0.3, -0.15, 0.1])

        # Initial point (different from optimal)
        x0 = jnp.array([1.0, 0.5, -0.5, 0.8])
        y0 = jnp.array([0.3, 0.2, 0.1, 0.0, 0.2])

        stepsizes = (0.5/M, 0.5/M, 1.0)

        # This should NOT crash and should produce valid results
        G, F = problem_data_to_pdhg_trajectories(
            stepsizes,
            c, K_mat, q, l, u,
            x0, y0,
            x_opt, y_opt,
            K_max, m1,
            M=M
        )

        G_np = np.array(G)
        F_np = np.array(F)

        # Basic sanity checks
        dimG = 4 * K_max + 11
        dimF = 2 * (K_max + 2)

        self.assertEqual(G_np.shape, (dimG, dimG))
        self.assertEqual(F_np.shape, (dimF,))

        # G should be symmetric
        np.testing.assert_allclose(G_np, G_np.T, atol=1e-10)

        # G should be PSD
        eigvals = np.linalg.eigvalsh(G_np)
        min_eig = np.min(eigvals)
        self.assertGreaterEqual(min_eig, -1e-8,
            f"Gram matrix should be PSD, but has min eigenvalue {min_eig}")

        # Function values at optimal should be 0
        dimF1 = K_max + 2
        F1 = F_np[:dimF1]
        F_h = F_np[dimF1:]

        self.assertAlmostEqual(F1[-1], 0.0, places=10,
            msg="F1 at optimal should be 0")
        self.assertAlmostEqual(F_h[-1], 0.0, places=10,
            msg="F_h at optimal should be 0")

        # Check function values are computed correctly
        # F1[0] = f1(x0) - f1(x_opt) = c^T x0 - c^T x_opt = c^T (x0 - x_opt)
        expected_f1_0 = np.dot(np.array(c), np.array(x0 - x_opt))
        np.testing.assert_allclose(F1[0], expected_f1_0, rtol=1e-6,
            err_msg="F1[0] with nonzero x_opt should be c^T (x0 - x_opt)")

        # F_h[0] = h(y0) - h(y_opt) = q^T y0 - q^T y_opt = q^T (y0 - y_opt)
        expected_h_0 = np.dot(np.array(q), np.array(y0 - y_opt))
        np.testing.assert_allclose(F_h[0], expected_h_0, rtol=1e-6,
            err_msg="F_h[0] with nonzero y_opt should be q^T (y0 - y_opt)")

        print(f"\nNonzero optimal point test passed!")
        print(f"  x_opt = {x_opt}")
        print(f"  y_opt = {y_opt}")
        print(f"  c = {c}")
        print(f"  q = {q}")
        print(f"  F1[0] = {F1[0]:.6f} (expected {expected_f1_0:.6f})")
        print(f"  F_h[0] = {F_h[0]:.6f} (expected {expected_h_0:.6f})")

    def test_pdlp_with_nonzero_cost_full_validation(self):
        """
        Test PDLP with nonzero c and q from a REAL facility location problem.

        This is the critical test: check that trajectories from a realistic
        PDLP problem (with nonzero objective) satisfy both:
        1. Inequality constraints: Tr(A @ G) + b^T @ F + c <= 0
        2. PSD constraints: constructed H matrices are PSD
        """
        import jax.random as jrandom

        # Create config for facility location problem
        class Config:
            fixed_costs = type('obj', (object,), {'l': 1.0, 'u': 2.0})()
            demands = type('obj', (object,), {'l': 0.5, 'u': 1.5})()
            transportation_costs = type('obj', (object,), {'l': 0.1, 'u': 1.0})()
            base_capacity = type('obj', (object,), {'base': 2.0, 'scaling': 1.5})()

        cfg = Config()

        # Generate facility location problem (GUARANTEED FEASIBLE)
        n_facilities = 5
        n_customers = 10  # Keep small for faster testing
        key = jrandom.PRNGKey(42)
        K_max = 2

        problem = generate_facility_location_problem(
            cfg=cfg,
            n_facilities=n_facilities,
            key=key,
            n_customers=n_customers,
        )

        # Extract constraint matrices
        c, A_eq, b_eq, A_ineq, b_ineq, lb, ub = extract_constraint_matrices(
            problem["fixed_costs"],
            problem["capacities"],
            problem["demands"],
            problem["transportation_costs"],
            n_facilities=n_facilities,
            n_customers=n_customers,
        )

        # Problem dimensions
        n_vars = c.shape[0]  # m + m*n
        m1 = A_ineq.shape[0]  # inequality constraints
        m2 = A_eq.shape[0]    # equality constraints

        # Solve for optimal point using cvxpy
        solver = FacilityLocationDPP(n_vars, m1, m2)
        x_opt, y_opt = solver.solve(
            np.array(c),
            np.array(A_ineq),
            np.array(b_ineq),
            np.array(A_eq),
            np.array(b_eq),
            np.array(lb),
            np.array(ub),
        )

        # Convert to JAX arrays
        x_opt = jnp.array(x_opt)
        y_opt = jnp.array(y_opt)  # Now only [s; ν], no λ

        # Build constraint matrix K for PDHG formulation (WITHOUT box constraints)
        # CVXPy constraint: -A_ineq @ x >= -b_ineq (equivalent to A_ineq @ x <= b_ineq)
        # In saddle point: L(x,y) = c^T x + y^T (Kx - q) where K = [-A_ineq; A_eq]
        # Box constraints (l <= x <= u) are handled in f1's indicator function
        #
        # y = [s; ν] where s >= 0 (for inequality), ν free (for equality)
        # q = [-b_ineq; b_eq]
        K_mat = jnp.vstack([-A_ineq, A_eq])
        q = jnp.concatenate([-b_ineq, b_eq])

        M = np.linalg.norm(K_mat, ord=2)

        # Initial point (random feasible point)
        key2 = jrandom.split(key)[0]
        x0 = jrandom.uniform(key2, shape=(n_vars,)) * 0.5 + 0.1  # in (0.1, 0.6)
        x0 = jnp.clip(x0, lb, ub)  # ensure within bounds

        # Initial dual point [s; ν] (no λ for box constraints)
        # - s >= 0 (inequality constraints)
        # - ν free (equality constraints)
        key3 = jrandom.split(key2)[0]
        y0_ineq = jnp.abs(jrandom.normal(key3, shape=(m1,))) * 0.1
        y0_eq = jrandom.normal(key3, shape=(m2,)) * 0.1
        y0 = jnp.concatenate([y0_ineq, y0_eq])

        # Step sizes
        tau = 0.5 / M
        sigma = 0.5 / M
        theta = 1.0

        # Generate trajectories using K = [-A_ineq; A_eq] (box constraints in f1)
        # Use dual y = [s; ν] (no λ)
        stepsizes = (jnp.array(tau), jnp.array(sigma), jnp.array(theta))
        G_traj, F_traj = problem_data_to_pdhg_trajectories(
            stepsizes,
            c, K_mat, q, lb, ub,  # K = [-A_ineq; A_eq], box constraints via f1
            x0, y0,               # Dual [s; ν]
            x_opt, y_opt,         # Dual [s; ν] from CVX (negated)
            K_max, m1,
            M=M
        )

        print(f"\n{'='*70}")
        print(f"Testing PDLP with FACILITY LOCATION problem")
        print(f"  n_facilities={n_facilities}, n_customers={n_customers}")
        print(f"  n_vars={n_vars}, m1={m1}, m2={m2}, K_max={K_max}")
        print(f"  Nonzero objective: ||c||={np.linalg.norm(c):.3f}")
        print(f"  Nonzero RHS: ||q||={np.linalg.norm(q):.3f}")
        print(f"  b_ineq (first 10): {np.array(b_ineq)[:10]}")
        print(f"  b_eq (first 5): {np.array(b_eq)[:5]}")
        print(f"  q vector (first 10 elements): {np.array(q)[:10]}")
        print(f"  K matrix shape: {K_mat.shape}")
        print(f"  y_opt shape (dual [s;ν]): {y_opt.shape}")
        print(f"  M = ||K||: {M:.3f}")

        # Verify KKT conditions
        Kt_y = np.array(K_mat.T @ y_opt)
        kkt_primal = c + Kt_y
        K_x = np.array(K_mat @ x_opt)
        print(f"  KKT verification:")
        print(f"    ||c + K^T y_opt|| = {np.linalg.norm(kkt_primal):.6e}")
        print(f"    K @ x_opt (first 5 / last 5): {K_x[:5]} / {K_x[-5:]}")
        print(f"    q (first 5 / last 5): {np.array(q)[:5]} / {np.array(q)[-5:]}")
        print(f"    ||K @ x_opt - q||: {np.linalg.norm(K_x - np.array(q)):.6e}")
        print(f"{'='*70}")

        # Print actual dual iterates for reference [s; ν]
        print(f"\nDual iterate values (for diagnostic):")
        print(f"  y0 [s;ν] (first 5): {np.array(y0)[:5]}")
        print(f"  y_opt [s;ν] (first 5): {np.array(y_opt)[:5]}")
        print(f"  dy0 = y0 - y_opt (first 5): {np.array(y0 - y_opt)[:5]}")

        # Get PEP constraints
        dx0 = x0 - x_opt
        dy0 = y0 - y_opt
        K_dx0 = K_mat @ dx0
        R_sq = jnp.sum(dx0**2) / tau + jnp.sum(dy0**2) / sigma - 2 * jnp.dot(K_dx0, dy0)
        R = float(jnp.sqrt(jnp.maximum(R_sq, 1.0)))

        pep_data = construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K_max)
        pep_data_np = chambolle_pock_pep_data_to_numpy(pep_data)

        A_obj, b_obj, A_vals, b_vals, c_vals, PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes = pep_data_np

        G_np = np.array(G_traj)
        F_np = np.array(F_traj)

        # ========================================================================
        # DIAGNOSTIC: Check if function values are computed correctly
        # ========================================================================
        dimF1 = K_max + 2
        F1 = F_np[:dimF1]
        F_h = F_np[dimF1:]

        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC: Function Value Validation")
        print(f"{'='*70}")
        print(f"\nFor h(y) = q^T y, function values should be:")
        print(f"  F_h[i] = h(y_i) - h(y_opt) = q^T (y_i - y_opt)")

        # Expected function values (manually computed)
        expected_F_h_0 = np.dot(np.array(q), np.array(y0 - y_opt))
        expected_F_h_last = np.dot(np.array(q), np.array(y_opt - y_opt))  # Should be 0

        print(f"\n  Expected F_h[0] = q^T (y0 - y_opt) = {expected_F_h_0:.6e}")
        print(f"  Actual   F_h[0] = {F_h[0]:.6e}")
        print(f"  Difference: {np.abs(F_h[0] - expected_F_h_0):.6e}")

        print(f"\n  Expected F_h[{K_max+1}] (at y_opt) = 0")
        print(f"  Actual   F_h[{K_max+1}] = {F_h[K_max+1]:.6e}")

        print(f"\n  All F_h values: {F_h}")
        print(f"{'='*70}")

        # ========================================================================
        # STEP 1: Validate Inequality Constraints
        # ========================================================================

        inequality_violations = []
        for i in range(A_vals.shape[0]):
            val = np.trace(A_vals[i] @ G_np) + np.dot(b_vals[i], F_np) + c_vals[i]
            inequality_violations.append(val)

        inequality_violations = np.array(inequality_violations)
        max_ineq_violation = np.max(inequality_violations)
        num_violated = np.sum(inequality_violations > 1e-6)

        print(f"\nInequality Constraints:")
        print(f"  Total constraints: {len(inequality_violations)}")
        print(f"  Violated (> 1e-6): {num_violated}")
        print(f"  Max violation: {max_ineq_violation:.6e}")

        if num_violated > 0:
            violated_indices = np.where(inequality_violations > 1e-6)[0]
            print(f"  Violated constraint indices: {violated_indices[:10]}...")  # show first 10
            print(f"  Violation values: {inequality_violations[violated_indices[:10]]}")

            # Decode which interpolation constraints are violated
            # For K_max=2: n_algo_points=3, total_points=4
            # convex_interp generates (total_points)*(total_points-1) = 4*3 = 12 constraints per function
            n_algo_points = K_max + 1
            total_points = n_algo_points + 1
            num_interp_per_func = total_points * (total_points - 1)

            print(f"\n  Constraint structure (K_max={K_max}):")
            print(f"    Indices 0-{num_interp_per_func-1}: f1 (primal) interpolation")
            print(f"    Indices {num_interp_per_func}-{2*num_interp_per_func-1}: h (dual) interpolation")
            print(f"    Indices {2*num_interp_per_func}-{2*num_interp_per_func+3}: value pinning")
            print(f"    Index {2*num_interp_per_func+4}: solution bound")
            print(f"    Remaining: adjoint consistency + P-norm IC")

            print(f"\n  Decoding violated constraints:")
            for idx in violated_indices[:10]:
                if idx < num_interp_per_func:
                    # f1 interpolation
                    local_idx = idx
                    i = local_idx // (total_points - 1)
                    j_offset = local_idx % (total_points - 1)
                    j = j_offset if j_offset < i else j_offset + 1
                    print(f"    [{idx}] f1 interp ({i},{j}): f1(x{i}) >= f1(x{j}) + <gf1({j}), x{i}-x{j}>")
                    print(f"         Violation: {inequality_violations[idx]:.6e}")
                elif idx < 2*num_interp_per_func:
                    # h interpolation
                    local_idx = idx - num_interp_per_func
                    i = local_idx // (total_points - 1)
                    j_offset = local_idx % (total_points - 1)
                    j = j_offset if j_offset < i else j_offset + 1

                    # Map to actual iterate indices (0=y0, 1=y1, 2=y2, 3=ys)
                    point_names = ['y0', 'y1', 'y2', 'ys']
                    print(f"    [{idx}] h interp ({i},{j}): h({point_names[i]}) >= h({point_names[j]}) + <gh({point_names[j]}), {point_names[i]}-{point_names[j]}>")
                    print(f"         Violation: {inequality_violations[idx]:.6e}")

                    # For h(y) = q^T y, gh(y) = q, so this should be:
                    # q^T y_i >= q^T y_j + q^T (y_i - y_j) => 0 >= 0 (always satisfied!)
                    print(f"         NOTE: For h(y)=q^T y, this should ALWAYS be satisfied (violation = 0)!")

                    # Check actual constraint value breakdown
                    # Constraint: Tr(A @ G) + b^T @ F <= 0
                    # For convex interp (i,j): A_ij = 0.5*(gj ⊗ dx + dx ⊗ gj), b_ij = fj - fi
                    A_ij = A_vals[idx]
                    b_ij = b_vals[idx]
                    c_ij = c_vals[idx]

                    trace_term = np.trace(A_ij @ G_np)
                    linear_term = np.dot(b_ij, F_np)
                    total = trace_term + linear_term + c_ij

                    print(f"         Breakdown: Tr(A@G)={trace_term:.6e}, b^T@F={linear_term:.6e}, c={c_ij:.6e}")
                    print(f"         Total: {total:.6e}")

                    # Get function values for diagnostic
                    dimF1 = K_max + 2
                    F1 = F_np[:dimF1]
                    F_h = F_np[dimF1:]
                    print(f"         F_h[{i}]={F_h[i]:.6e}, F_h[{j}]={F_h[j]:.6e}")
                    print(f"         Expected: F_h[{i}] - F_h[{j}] = {F_h[i] - F_h[j]:.6e}")
                else:
                    print(f"    [{idx}] Other constraint (value pinning, bound, adjoint, or IC)")
                    print(f"         Violation: {inequality_violations[idx]:.6e}")

        # ========================================================================
        # STEP 2: Validate PSD Constraints
        # ========================================================================
        print(f"\nPSD Constraints:")

        if PSD_A_vals is None:
            print("  No PSD constraints present")
            psd_all_satisfied = True
        else:
            M_psd = len(PSD_A_vals)
            print(f"  Number of PSD constraint groups: {M_psd}")

            psd_violations = []
            min_eigenvalues = []

            for m_psd in range(M_psd):
                # Construct H matrix: H = PSD_A[m_psd] : G + PSD_b[m_psd] @ F + PSD_c[m_psd]
                # PSD_A_vals[m_psd] has shape (mat_dim, mat_dim, dimG, dimG)
                # We need H[r,c] = Tr(PSD_A[m_psd][r,c] @ G) + 0 + 0 (assuming b,c are zero or handled)

                mat_dim = PSD_shapes[m_psd]
                H = np.zeros((mat_dim, mat_dim))

                # Reconstruct H from the constraint data
                for r in range(mat_dim):
                    for c in range(mat_dim):
                        H[r, c] = np.trace(PSD_A_vals[m_psd][r, c] @ G_np)
                        if PSD_b_vals is not None and PSD_b_vals[m_psd] is not None:
                            H[r, c] += np.dot(PSD_b_vals[m_psd][r, c], F_np)
                        if PSD_c_vals is not None and PSD_c_vals[m_psd] is not None:
                            H[r, c] += PSD_c_vals[m_psd][r, c]

                # Check if H is PSD
                eigvals = np.linalg.eigvalsh(H)
                min_eig = np.min(eigvals)
                min_eigenvalues.append(min_eig)

                is_psd = min_eig >= -1e-8
                psd_violations.append(not is_psd)

                print(f"  PSD[{m_psd}]: shape={mat_dim}x{mat_dim}, min_eig={min_eig:.6e}, satisfied={is_psd}")

                if not is_psd:
                    print(f"    VIOLATION: H matrix is not PSD!")
                    print(f"    Eigenvalues: {eigvals}")

            psd_all_satisfied = not any(psd_violations)

        # ========================================================================
        # STEP 3: Summary and Assertions
        # ========================================================================
        print(f"\n{'='*70}")
        print(f"SUMMARY:")
        print(f"  Inequality constraints satisfied: {num_violated == 0}")
        print(f"  PSD constraints satisfied: {psd_all_satisfied}")
        print(f"{'='*70}\n")

        # Assertions
        if num_violated > 0 or not psd_all_satisfied:
            self.fail(
                f"PEP constraints violated with nonzero c, q!\n"
                f"  Inequality violations: {num_violated}/{len(inequality_violations)}\n"
                f"  PSD violations: {sum(psd_violations) if PSD_A_vals else 0}\n"
                f"This explains why the DRO dual is unbounded (primal PEP is infeasible)."
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
