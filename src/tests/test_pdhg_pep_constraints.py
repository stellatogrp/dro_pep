"""
Comprehensive unit tests verifying PDHG trajectories satisfy PEP constraints.

This test uses the actual PEP construction from pep_construction_chambolle_pock.py
(the same function used during training) to verify that trajectory Gram matrices
satisfy all interpolation inequalities.
"""

import unittest
import numpy as np
import cvxpy as cp

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning.trajectories_pdhg import (
    problem_data_to_pdhg_trajectories,
    problem_data_to_pdhg_trajectories_decoupled,
    compute_pdhg_stepsizes_from_K,
)
from learning.pep_construction_chambolle_pock import (
    construct_chambolle_pock_pep_data,
    construct_chambolle_pock_pep_data_decoupled,
    chambolle_pock_pep_data_to_numpy,
)


def generate_random_lp(n, m1, m2, seed=42):
    """
    Generate a random LP problem.

    min c^T x
    s.t. G x >= h  (m1 inequality constraints)
         A x = b   (m2 equality constraints)
         l <= x <= u

    Returns feasible problem with optimal solution.
    """
    np.random.seed(seed)

    c = np.random.randn(n)

    # Generate constraints that are likely to be feasible
    G = np.random.randn(m1, n) / np.sqrt(m1)
    A = np.random.randn(m2, n) / np.sqrt(m2)

    # Generate a feasible point and set RHS to make it feasible
    x_feas = np.random.rand(n)  # In [0, 1]^n
    h = G @ x_feas - np.abs(np.random.randn(m1)) * 0.5
    b = A @ x_feas

    l = np.zeros(n)
    u = np.ones(n)

    return c, G, h, A, b, l, u


def solve_lp_cvxpy(c, G, h, A, b, l, u):
    """
    Solve LP using CVXPY to get optimal primal and dual solutions.

    Returns:
        x_opt: Optimal primal solution
        y_opt: Optimal dual solution [lambda; -mu_cvxpy]
        f_opt: Optimal objective value
    """
    n = c.shape[0]
    m1 = G.shape[0]

    x = cp.Variable(n)

    constraints = [
        G @ x >= h,
        A @ x == b,
        x >= l,
        x <= u,
    ]

    objective = cp.Minimize(c @ x)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='CLARABEL', verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"LP solve failed with status: {prob.status}")

    x_opt = x.value
    lambda_opt = constraints[0].dual_value
    mu_opt_cvxpy = constraints[1].dual_value

    # Our Lagrangian uses (q - Kx) not (Kx - q), so sign flip for equality duals
    y_opt = np.concatenate([lambda_opt, -mu_opt_cvxpy])
    f_opt = prob.value

    return x_opt, y_opt, f_opt


class TestPDHGPEPConstraints(unittest.TestCase):
    """
    Tests verifying PDHG trajectories satisfy PEP interpolation constraints.

    Uses the actual PEP construction from pep_construction_chambolle_pock.py.
    """

    def setUp(self):
        """Generate random LP problem data."""
        np.random.seed(42)

        self.n = 5
        self.m1 = 3
        self.m2 = 2
        self.K_iter = 3

        # Generate LP
        self.c, G, h, A, b, self.l, self.u = generate_random_lp(
            self.n, self.m1, self.m2, seed=42
        )

        # Stack constraints: K = [G; A], q = [h; b]
        self.K_mat = np.vstack([G, A])
        self.q = np.concatenate([h, b])

        # Solve LP
        self.x_opt, self.y_opt, self.f_opt = solve_lp_cvxpy(
            self.c, G, h, A, b, self.l, self.u
        )

        # Compute M = ||K||_2 (used in PEP as scalar coupling)
        self.M = np.linalg.norm(self.K_mat, ord=2)

        # Step sizes satisfying tau * sigma * M^2 < 1
        self.tau = 0.9 / self.M
        self.sigma = 0.9 / self.M
        self.theta = 1.0

        # Initial point
        self.x0 = 0.5 * (self.l + self.u)
        self.y0 = np.zeros(self.m1 + self.m2)
        self.y0[:self.m1] = 0.1

        # Initial radius for PEP
        delta_x0 = self.x0 - self.x_opt
        delta_y0 = self.y0 - self.y_opt
        self.R = np.sqrt(np.dot(delta_x0, delta_x0) + np.dot(delta_y0, delta_y0))

    def test_trajectory_satisfies_pep_constraints(self):
        """
        Core test: Verify trajectory (G, F) satisfies all PEP constraints.

        This uses the actual PEP construction function to generate constraint
        matrices, then checks that trace(A @ G) + b @ F + c <= 0 for all constraints.
        """
        # Convert to JAX arrays
        c_jax = jnp.array(self.c)
        K_jax = jnp.array(self.K_mat)
        q_jax = jnp.array(self.q)
        l_jax = jnp.array(self.l)
        u_jax = jnp.array(self.u)
        x0_jax = jnp.array(self.x0)
        y0_jax = jnp.array(self.y0)
        x_opt_jax = jnp.array(self.x_opt)
        y_opt_jax = jnp.array(self.y_opt)

        stepsizes = (jnp.array(self.tau), jnp.array(self.sigma), jnp.array(self.theta))

        # Get trajectory Gram matrix and function values
        G_traj, F_traj = problem_data_to_pdhg_trajectories(
            stepsizes, c_jax, K_jax, q_jax, l_jax, u_jax,
            x0_jax, y0_jax, x_opt_jax, y_opt_jax, self.f_opt,
            self.K_iter, self.m1, return_Gram_representation=True,
            M=self.M
        )

        # Get PEP constraint matrices using the actual construction function
        pep_data = construct_chambolle_pock_pep_data(
            tau=self.tau,
            sigma=self.sigma,
            theta=self.theta,
            M=self.M,
            R=self.R,
            K_max=self.K_iter
        )
        pep_data_np = chambolle_pock_pep_data_to_numpy(pep_data)

        (A_obj, b_obj, A_vals, b_vals, c_vals,
         PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = pep_data_np

        G_np = np.array(G_traj)
        F_np = np.array(F_traj)

        # Check dimensions match
        self.assertEqual(G_np.shape[0], A_obj.shape[0],
            f"Gram dimension mismatch: G has {G_np.shape[0]}, PEP expects {A_obj.shape[0]}")
        self.assertEqual(F_np.shape[0], b_obj.shape[0],
            f"F dimension mismatch: F has {F_np.shape[0]}, PEP expects {b_obj.shape[0]}")

        # Check G is symmetric and PSD
        np.testing.assert_allclose(G_np, G_np.T, atol=1e-10,
            err_msg="Gram matrix should be symmetric")

        eigvals = np.linalg.eigvalsh(G_np)
        min_eig = np.min(eigvals)
        self.assertGreaterEqual(min_eig, -1e-8,
            f"Gram matrix should be PSD, but has min eigenvalue {min_eig}")

        # Check all interpolation constraints: trace(A @ G) + b @ F + c <= 0
        violations = []
        for i in range(A_vals.shape[0]):
            constraint_val = np.trace(A_vals[i] @ G_np) + np.dot(b_vals[i], F_np) + c_vals[i]
            violations.append(constraint_val)

        violations = np.array(violations)
        max_violation = np.max(violations)

        # Allow small numerical tolerance
        self.assertLessEqual(max_violation, 1e-6,
            f"PEP constraints violated! Max violation: {max_violation}\n"
            f"Violations > 0: {np.sum(violations > 1e-6)} out of {len(violations)}")

    def test_initial_condition_satisfied(self):
        """Verify the initial condition ||delta_x0||^2 + ||delta_y0||^2 <= R^2."""
        c_jax = jnp.array(self.c)
        K_jax = jnp.array(self.K_mat)
        q_jax = jnp.array(self.q)
        l_jax = jnp.array(self.l)
        u_jax = jnp.array(self.u)
        x0_jax = jnp.array(self.x0)
        y0_jax = jnp.array(self.y0)
        x_opt_jax = jnp.array(self.x_opt)
        y_opt_jax = jnp.array(self.y_opt)

        stepsizes = (jnp.array(self.tau), jnp.array(self.sigma), jnp.array(self.theta))

        G_traj, F_traj = problem_data_to_pdhg_trajectories(
            stepsizes, c_jax, K_jax, q_jax, l_jax, u_jax,
            x0_jax, y0_jax, x_opt_jax, y_opt_jax, self.f_opt,
            self.K_iter, self.m1, return_Gram_representation=True,
            M=self.M
        )

        G_np = np.array(G_traj)

        # G[0,0] = ||delta_x0||^2, G[1,1] = ||delta_y0||^2
        init_dist_sq = G_np[0, 0] + G_np[1, 1]

        self.assertLessEqual(init_dist_sq, self.R**2 + 1e-10,
            f"Initial condition violated: {init_dist_sq} > R^2 = {self.R**2}")

    def test_gram_structure(self):
        """Verify Gram matrix has expected structure."""
        c_jax = jnp.array(self.c)
        K_jax = jnp.array(self.K_mat)
        q_jax = jnp.array(self.q)
        l_jax = jnp.array(self.l)
        u_jax = jnp.array(self.u)
        x0_jax = jnp.array(self.x0)
        y0_jax = jnp.array(self.y0)
        x_opt_jax = jnp.array(self.x_opt)
        y_opt_jax = jnp.array(self.y_opt)

        stepsizes = (jnp.array(self.tau), jnp.array(self.sigma), jnp.array(self.theta))

        G_traj, F_traj = problem_data_to_pdhg_trajectories(
            stepsizes, c_jax, K_jax, q_jax, l_jax, u_jax,
            x0_jax, y0_jax, x_opt_jax, y_opt_jax, self.f_opt,
            self.K_iter, self.m1, return_Gram_representation=True,
            M=self.M
        )

        G_np = np.array(G_traj)

        # Expected dimensions
        expected_dimG = 2 * self.K_iter + 6
        expected_dimF = 2 * (self.K_iter + 2)

        self.assertEqual(G_np.shape, (expected_dimG, expected_dimG))
        self.assertEqual(F_traj.shape, (expected_dimF,))

        # Verify cross inner product formula: <x, y>_W = -y^T K x / M
        # G[0, 3] = <delta_x0, ys>_W should equal -(delta_x0)^T K^T y_opt / M
        delta_x0 = self.x0 - self.x_opt
        expected_cross = -np.dot(delta_x0, self.K_mat.T @ self.y_opt) / self.M

        np.testing.assert_allclose(G_np[0, 3], expected_cross, rtol=1e-6,
            err_msg=f"Cross inner product <delta_x0, ys> mismatch")

    def test_function_values_structure(self):
        """Verify function values F have correct structure."""
        c_jax = jnp.array(self.c)
        K_jax = jnp.array(self.K_mat)
        q_jax = jnp.array(self.q)
        l_jax = jnp.array(self.l)
        u_jax = jnp.array(self.u)
        x0_jax = jnp.array(self.x0)
        y0_jax = jnp.array(self.y0)
        x_opt_jax = jnp.array(self.x_opt)
        y_opt_jax = jnp.array(self.y_opt)

        stepsizes = (jnp.array(self.tau), jnp.array(self.sigma), jnp.array(self.theta))

        G_traj, F_traj = problem_data_to_pdhg_trajectories(
            stepsizes, c_jax, K_jax, q_jax, l_jax, u_jax,
            x0_jax, y0_jax, x_opt_jax, y_opt_jax, self.f_opt,
            self.K_iter, self.m1, return_Gram_representation=True,
            M=self.M
        )

        F_np = np.array(F_traj)
        dimF1 = self.K_iter + 2

        F1 = F_np[:dimF1]
        F_h = F_np[dimF1:]

        # Last element of F1 and F_h should be 0 (value at optimal)
        self.assertAlmostEqual(F1[-1], 0.0, places=10,
            msg="F1 at optimal should be 0")
        self.assertAlmostEqual(F_h[-1], 0.0, places=10,
            msg="F_h at optimal should be 0")

        # F1[0] = f1(x0) - f1(x_opt) = c^T x0 - c^T x_opt = c^T (x0 - x_opt)
        expected_f1_0 = np.dot(self.c, self.x0 - self.x_opt)
        np.testing.assert_allclose(F1[0], expected_f1_0, rtol=1e-6,
            err_msg="F1[0] = f1(x0) - f1_opt mismatch")

        # F_h[0] = h(y0) - h(y_opt) = -q^T y0 - (-q^T y_opt) = -q^T (y0 - y_opt)
        expected_h_0 = -np.dot(self.q, self.y0 - self.y_opt)
        np.testing.assert_allclose(F_h[0], expected_h_0, rtol=1e-6,
            err_msg="F_h[0] = h(y0) - h_opt mismatch")


class TestPDHGPEPMultipleProblems(unittest.TestCase):
    """Test PEP constraint satisfaction across multiple random LP instances."""

    def test_multiple_lps_satisfy_constraints(self):
        """Test that trajectories from multiple LPs satisfy PEP constraints."""
        num_problems = 5
        K_iter = 2

        for seed in range(num_problems):
            with self.subTest(seed=seed):
                np.random.seed(seed + 100)

                n = 4
                m1 = 2
                m2 = 1

                c, G, h, A, b, l, u = generate_random_lp(n, m1, m2, seed=seed + 100)
                K_mat = np.vstack([G, A])
                q = np.concatenate([h, b])

                try:
                    x_opt, y_opt, f_opt = solve_lp_cvxpy(c, G, h, A, b, l, u)
                except RuntimeError:
                    continue  # Skip infeasible problems

                M = np.linalg.norm(K_mat, ord=2)
                tau = 0.9 / M
                sigma = 0.9 / M
                theta = 1.0

                x0 = 0.5 * (l + u)
                y0 = np.zeros(m1 + m2)
                y0[:m1] = 0.1

                delta_x0 = x0 - x_opt
                delta_y0 = y0 - y_opt
                R = np.sqrt(np.dot(delta_x0, delta_x0) + np.dot(delta_y0, delta_y0))

                # Get trajectory
                stepsizes = (jnp.array(tau), jnp.array(sigma), jnp.array(theta))
                G_traj, F_traj = problem_data_to_pdhg_trajectories(
                    stepsizes,
                    jnp.array(c), jnp.array(K_mat), jnp.array(q),
                    jnp.array(l), jnp.array(u),
                    jnp.array(x0), jnp.array(y0),
                    jnp.array(x_opt), jnp.array(y_opt), f_opt,
                    K_iter, m1, return_Gram_representation=True,
                    M=M
                )

                # Get PEP constraints
                pep_data = construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K_iter)
                pep_data_np = chambolle_pock_pep_data_to_numpy(pep_data)
                A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data_np

                G_np = np.array(G_traj)
                F_np = np.array(F_traj)

                # Check constraints
                max_violation = -np.inf
                for i in range(A_vals.shape[0]):
                    val = np.trace(A_vals[i] @ G_np) + np.dot(b_vals[i], F_np) + c_vals[i]
                    max_violation = max(max_violation, val)

                self.assertLessEqual(max_violation, 1e-6,
                    f"Problem {seed}: PEP constraints violated with max violation {max_violation}")


class TestPDHGPEPVectorStepsizes(unittest.TestCase):
    """Test PEP constraint satisfaction with vector step sizes."""

    def test_vector_stepsizes(self):
        """Test with iteration-varying step sizes."""
        np.random.seed(42)

        n, m1, m2 = 4, 2, 1
        K_iter = 3

        c, G, h, A, b, l, u = generate_random_lp(n, m1, m2, seed=42)
        K_mat = np.vstack([G, A])
        q = np.concatenate([h, b])

        x_opt, y_opt, f_opt = solve_lp_cvxpy(c, G, h, A, b, l, u)

        M = np.linalg.norm(K_mat, ord=2)

        # Vector step sizes (slightly varying)
        tau_vec = np.array([0.8, 0.85, 0.9]) / M
        sigma_vec = np.array([0.9, 0.85, 0.8]) / M
        theta_vec = np.array([1.0, 1.0, 1.0])

        x0 = 0.5 * (l + u)
        y0 = np.zeros(m1 + m2)
        y0[:m1] = 0.1

        delta_x0 = x0 - x_opt
        delta_y0 = y0 - y_opt
        R = np.sqrt(np.dot(delta_x0, delta_x0) + np.dot(delta_y0, delta_y0))

        # Get trajectory with vector stepsizes
        stepsizes = (jnp.array(tau_vec), jnp.array(sigma_vec), jnp.array(theta_vec))
        G_traj, F_traj = problem_data_to_pdhg_trajectories(
            stepsizes,
            jnp.array(c), jnp.array(K_mat), jnp.array(q),
            jnp.array(l), jnp.array(u),
            jnp.array(x0), jnp.array(y0),
            jnp.array(x_opt), jnp.array(y_opt), f_opt,
            K_iter, m1, return_Gram_representation=True,
            M=M
        )

        # Get PEP constraints with vector stepsizes
        pep_data = construct_chambolle_pock_pep_data(
            jnp.array(tau_vec), jnp.array(sigma_vec), jnp.array(theta_vec),
            M, R, K_iter
        )
        pep_data_np = chambolle_pock_pep_data_to_numpy(pep_data)
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data_np

        G_np = np.array(G_traj)
        F_np = np.array(F_traj)

        # Check constraints
        violations = []
        for i in range(A_vals.shape[0]):
            val = np.trace(A_vals[i] @ G_np) + np.dot(b_vals[i], F_np) + c_vals[i]
            violations.append(val)

        max_violation = max(violations)
        self.assertLessEqual(max_violation, 1e-6,
            f"Vector stepsize PEP constraints violated with max violation {max_violation}")


class TestPDHGPEPConstraintsDecoupled(unittest.TestCase):
    """
    Tests verifying DECOUPLED PDHG trajectories satisfy PEP interpolation constraints.

    Uses the decoupled PEP construction where primal and dual blocks are separated,
    so interpolation constraints only use same-block inner products.

    Dimensions for decoupled version:
        - dimG = 4K + 8 (vs. 2K + 6 for coupled)
        - dimF = 2K + 4 (same as coupled)
    """

    def setUp(self):
        """Generate random LP problem data."""
        np.random.seed(42)

        self.n = 5
        self.m1 = 3
        self.m2 = 2
        self.K_iter = 3

        # Generate LP
        self.c, G, h, A, b, self.l, self.u = generate_random_lp(
            self.n, self.m1, self.m2, seed=42
        )

        # Stack constraints
        self.K_mat = np.vstack([G, A])
        self.q = np.concatenate([h, b])

        # Solve LP
        self.x_opt, self.y_opt, self.f_opt = solve_lp_cvxpy(
            self.c, G, h, A, b, self.l, self.u
        )

        # Compute M = ||K||_2
        self.M = np.linalg.norm(self.K_mat, ord=2)

        # Step sizes
        self.tau = 0.9 / self.M
        self.sigma = 0.9 / self.M
        self.theta = 1.0

        # Initial point
        self.x0 = 0.5 * (self.l + self.u)
        self.y0 = np.zeros(self.m1 + self.m2)
        self.y0[:self.m1] = 0.1

        # Initial radius
        delta_x0 = self.x0 - self.x_opt
        delta_y0 = self.y0 - self.y_opt
        self.R = np.sqrt(np.dot(delta_x0, delta_x0) + np.dot(delta_y0, delta_y0))

    def test_decoupled_dimensions(self):
        """Verify decoupled Gram matrix has correct dimensions."""
        stepsizes = (jnp.array(self.tau), jnp.array(self.sigma), jnp.array(self.theta))

        G_traj, F_traj = problem_data_to_pdhg_trajectories_decoupled(
            stepsizes,
            jnp.array(self.c), jnp.array(self.K_mat), jnp.array(self.q),
            jnp.array(self.l), jnp.array(self.u),
            jnp.array(self.x0), jnp.array(self.y0),
            jnp.array(self.x_opt), jnp.array(self.y_opt), self.f_opt,
            self.K_iter, self.m1, M=self.M
        )

        # Decoupled dimensions: dimG = 4K + 8, dimF = 2K + 4
        expected_dimG = 4 * self.K_iter + 8
        expected_dimF = 2 * (self.K_iter + 2)

        G_np = np.array(G_traj)
        F_np = np.array(F_traj)

        self.assertEqual(G_np.shape, (expected_dimG, expected_dimG),
            f"Expected dimG = {expected_dimG}, got {G_np.shape[0]}")
        self.assertEqual(F_np.shape, (expected_dimF,),
            f"Expected dimF = {expected_dimF}, got {F_np.shape[0]}")

    def test_decoupled_trajectory_satisfies_pep_constraints(self):
        """
        Core test: Verify decoupled trajectory (G, F) satisfies all PEP constraints.

        This is the key test that should PASS, unlike the coupled version.
        """
        stepsizes = (jnp.array(self.tau), jnp.array(self.sigma), jnp.array(self.theta))

        # Get decoupled trajectory
        G_traj, F_traj = problem_data_to_pdhg_trajectories_decoupled(
            stepsizes,
            jnp.array(self.c), jnp.array(self.K_mat), jnp.array(self.q),
            jnp.array(self.l), jnp.array(self.u),
            jnp.array(self.x0), jnp.array(self.y0),
            jnp.array(self.x_opt), jnp.array(self.y_opt), self.f_opt,
            self.K_iter, self.m1, M=self.M
        )

        # Get decoupled PEP constraints
        pep_data = construct_chambolle_pock_pep_data_decoupled(
            tau=self.tau,
            sigma=self.sigma,
            theta=self.theta,
            M=self.M,
            R=self.R,
            K_max=self.K_iter
        )
        pep_data_np = chambolle_pock_pep_data_to_numpy(pep_data)

        (A_obj, b_obj, A_vals, b_vals, c_vals,
         PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = pep_data_np

        G_np = np.array(G_traj)
        F_np = np.array(F_traj)

        # Check dimensions match
        self.assertEqual(G_np.shape[0], A_obj.shape[0],
            f"Gram dimension mismatch: G has {G_np.shape[0]}, PEP expects {A_obj.shape[0]}")
        self.assertEqual(F_np.shape[0], b_obj.shape[0],
            f"F dimension mismatch: F has {F_np.shape[0]}, PEP expects {b_obj.shape[0]}")

        # Check G is symmetric and PSD
        np.testing.assert_allclose(G_np, G_np.T, atol=1e-10,
            err_msg="Gram matrix should be symmetric")

        eigvals = np.linalg.eigvalsh(G_np)
        min_eig = np.min(eigvals)
        self.assertGreaterEqual(min_eig, -1e-8,
            f"Gram matrix should be PSD, but has min eigenvalue {min_eig}")

        # Check all interpolation constraints
        violations = []
        for i in range(A_vals.shape[0]):
            constraint_val = np.trace(A_vals[i] @ G_np) + np.dot(b_vals[i], F_np) + c_vals[i]
            violations.append(constraint_val)

        violations = np.array(violations)
        max_violation = np.max(violations)

        self.assertLessEqual(max_violation, 1e-6,
            f"Decoupled PEP constraints violated! Max violation: {max_violation}\n"
            f"Violations > 0: {np.sum(violations > 1e-6)} out of {len(violations)}")

    def test_decoupled_initial_condition(self):
        """Verify initial condition ||delta_x0||^2 + ||delta_y0||^2 <= R^2 in decoupled structure."""
        stepsizes = (jnp.array(self.tau), jnp.array(self.sigma), jnp.array(self.theta))

        G_traj, F_traj = problem_data_to_pdhg_trajectories_decoupled(
            stepsizes,
            jnp.array(self.c), jnp.array(self.K_mat), jnp.array(self.q),
            jnp.array(self.l), jnp.array(self.u),
            jnp.array(self.x0), jnp.array(self.y0),
            jnp.array(self.x_opt), jnp.array(self.y_opt), self.f_opt,
            self.K_iter, self.m1, M=self.M
        )

        G_np = np.array(G_traj)

        # In decoupled structure:
        # - delta_x0 is at index 0 in primal block
        # - delta_y0 is at index dim_primal in dual block
        dim_primal = 2 * self.K_iter + 4

        # G[0, 0] = ||delta_x0||^2
        # G[dim_primal, dim_primal] = ||delta_y0||^2
        init_dist_sq = G_np[0, 0] + G_np[dim_primal, dim_primal]

        self.assertLessEqual(init_dist_sq, self.R**2 + 1e-10,
            f"Initial condition violated: {init_dist_sq} > R^2 = {self.R**2}")

    def test_decoupled_primal_block_structure(self):
        """Verify the primal block has correct inner products."""
        stepsizes = (jnp.array(self.tau), jnp.array(self.sigma), jnp.array(self.theta))

        G_traj, F_traj = problem_data_to_pdhg_trajectories_decoupled(
            stepsizes,
            jnp.array(self.c), jnp.array(self.K_mat), jnp.array(self.q),
            jnp.array(self.l), jnp.array(self.u),
            jnp.array(self.x0), jnp.array(self.y0),
            jnp.array(self.x_opt), jnp.array(self.y_opt), self.f_opt,
            self.K_iter, self.m1, M=self.M
        )

        G_np = np.array(G_traj)
        K_iter = self.K_iter

        # delta_x0 is at index 0
        # ||delta_x0||^2 = (x0 - x_opt)^T (x0 - x_opt)
        delta_x0 = self.x0 - self.x_opt
        expected_norm_sq = np.dot(delta_x0, delta_x0)

        np.testing.assert_allclose(G_np[0, 0], expected_norm_sq, rtol=1e-6,
            err_msg="||delta_x0||^2 mismatch")

        # xs is at index K_iter + 1
        # ||xs||^2 = x_opt^T x_opt
        idx_xs = K_iter + 1
        expected_xs_norm_sq = np.dot(self.x_opt, self.x_opt)

        np.testing.assert_allclose(G_np[idx_xs, idx_xs], expected_xs_norm_sq, rtol=1e-6,
            err_msg="||xs||^2 mismatch")

    def test_decoupled_function_values(self):
        """Verify function values have correct structure."""
        stepsizes = (jnp.array(self.tau), jnp.array(self.sigma), jnp.array(self.theta))

        G_traj, F_traj = problem_data_to_pdhg_trajectories_decoupled(
            stepsizes,
            jnp.array(self.c), jnp.array(self.K_mat), jnp.array(self.q),
            jnp.array(self.l), jnp.array(self.u),
            jnp.array(self.x0), jnp.array(self.y0),
            jnp.array(self.x_opt), jnp.array(self.y_opt), self.f_opt,
            self.K_iter, self.m1, M=self.M
        )

        F_np = np.array(F_traj)
        dimF1 = self.K_iter + 2

        F1 = F_np[:dimF1]
        F_h = F_np[dimF1:]

        # Last elements should be 0 (at optimal)
        self.assertAlmostEqual(F1[-1], 0.0, places=10)
        self.assertAlmostEqual(F_h[-1], 0.0, places=10)

        # F1[0] = f1(x0) - f1(x_opt) = c^T (x0 - x_opt)
        expected_f1_0 = np.dot(self.c, self.x0 - self.x_opt)
        np.testing.assert_allclose(F1[0], expected_f1_0, rtol=1e-6)

        # F_h[0] = h(y0) - h(y_opt) = -q^T (y0 - y_opt)
        expected_h_0 = -np.dot(self.q, self.y0 - self.y_opt)
        np.testing.assert_allclose(F_h[0], expected_h_0, rtol=1e-6)


class TestPDHGPEPDecoupledMultipleProblems(unittest.TestCase):
    """Test decoupled PEP constraint satisfaction across multiple random LP instances."""

    def test_multiple_lps_decoupled(self):
        """Test that decoupled trajectories from multiple LPs satisfy PEP constraints."""
        num_problems = 5
        K_iter = 2

        for seed in range(num_problems):
            with self.subTest(seed=seed):
                np.random.seed(seed + 100)

                n = 4
                m1 = 2
                m2 = 1

                c, G, h, A, b, l, u = generate_random_lp(n, m1, m2, seed=seed + 100)
                K_mat = np.vstack([G, A])
                q = np.concatenate([h, b])

                try:
                    x_opt, y_opt, f_opt = solve_lp_cvxpy(c, G, h, A, b, l, u)
                except RuntimeError:
                    continue

                M = np.linalg.norm(K_mat, ord=2)
                tau = 0.9 / M
                sigma = 0.9 / M
                theta = 1.0

                x0 = 0.5 * (l + u)
                y0 = np.zeros(m1 + m2)
                y0[:m1] = 0.1

                delta_x0 = x0 - x_opt
                delta_y0 = y0 - y_opt
                R = np.sqrt(np.dot(delta_x0, delta_x0) + np.dot(delta_y0, delta_y0))

                stepsizes = (jnp.array(tau), jnp.array(sigma), jnp.array(theta))
                G_traj, F_traj = problem_data_to_pdhg_trajectories_decoupled(
                    stepsizes,
                    jnp.array(c), jnp.array(K_mat), jnp.array(q),
                    jnp.array(l), jnp.array(u),
                    jnp.array(x0), jnp.array(y0),
                    jnp.array(x_opt), jnp.array(y_opt), f_opt,
                    K_iter, m1, M=M
                )

                pep_data = construct_chambolle_pock_pep_data_decoupled(tau, sigma, theta, M, R, K_iter)
                pep_data_np = chambolle_pock_pep_data_to_numpy(pep_data)
                A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data_np

                G_np = np.array(G_traj)
                F_np = np.array(F_traj)

                max_violation = -np.inf
                for i in range(A_vals.shape[0]):
                    val = np.trace(A_vals[i] @ G_np) + np.dot(b_vals[i], F_np) + c_vals[i]
                    max_violation = max(max_violation, val)

                self.assertLessEqual(max_violation, 1e-6,
                    f"Problem {seed}: Decoupled PEP constraints violated with max violation {max_violation}")


if __name__ == '__main__':
    unittest.main()
