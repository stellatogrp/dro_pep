"""
Unit tests for PDHG with custom interpolation conditions.

This test verifies that actual PDHG trajectories on LP problems
satisfy the interpolation conditions in our custom representation.
"""

import unittest
import numpy as np
import cvxpy as cp

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning.interpolation_conditions import convex_interp
from learning.trajectories_pdhg import (
    problem_data_to_pdhg_trajectories,
    compute_pdhg_stepsizes_from_K,
    build_lp_matrices,
    proj_box,
    proj_nonneg_first_m1,
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

    # Generate problem data
    c = np.random.randn(n)

    # Generate constraints that are likely to be feasible
    G = np.random.randn(m1, n) / np.sqrt(m1)
    A = np.random.randn(m2, n) / np.sqrt(m2)

    # Generate a feasible point and set RHS to make it feasible
    x_feas = np.random.rand(n)  # In [0, 1]^n
    h = G @ x_feas - np.abs(np.random.randn(m1)) * 0.5  # Ensure Gx_feas >= h
    b = A @ x_feas  # Ensure Ax_feas = b

    l = np.zeros(n)
    u = np.ones(n)

    return c, G, h, A, b, l, u


def solve_lp_cvxpy(c, G, h, A, b, l, u):
    """
    Solve LP using CVXPY to get optimal primal and dual solutions.

    Our Lagrangian is: L(x, y) = c^T x + lambda^T (h - Gx) + mu^T (b - Ax)
    CVXPY uses:        L_cvxpy = c^T x + lambda^T (Gx - h) + mu_cvxpy^T (Ax - b)

    So our mu = -mu_cvxpy (sign flip for equality constraints).

    Returns:
        x_opt: Optimal primal solution
        y_opt: Optimal dual solution [lambda; -mu_cvxpy] for our Lagrangian
        f_opt: Optimal objective value
    """
    n = c.shape[0]
    m1 = G.shape[0]
    m2 = A.shape[0]

    x = cp.Variable(n)

    constraints = [
        G @ x >= h,  # Inequality constraints
        A @ x == b,  # Equality constraints
        x >= l,
        x <= u,
    ]

    objective = cp.Minimize(c @ x)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='CLARABEL', verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"LP solve failed with status: {prob.status}")

    x_opt = x.value

    # Get dual variables
    lambda_opt = constraints[0].dual_value  # For Gx >= h
    mu_opt_cvxpy = constraints[1].dual_value  # For Ax - b = 0 (CVXPY convention)

    # Combine dual variables: y = [lambda; -mu_cvxpy]
    # The sign flip on mu is because our Lagrangian uses (b - Ax) not (Ax - b)
    y_opt = np.concatenate([lambda_opt, -mu_opt_cvxpy])

    f_opt = prob.value

    return x_opt, y_opt, f_opt


def run_pdhg_numpy(c, K, q, l, u, x0, y0, tau, sigma, theta, K_iter, m1):
    """
    Run PDHG in NumPy for verification.

    Returns trajectories in original (unshifted) coordinates.
    """
    n = c.shape[0]
    m = K.shape[0]

    x_traj = [x0.copy()]
    y_traj = [y0.copy()]

    x_curr = x0.copy()
    y_curr = y0.copy()

    for k in range(K_iter):
        # Primal update
        v_x = x_curr - tau * (c - K.T @ y_curr)
        x_new = np.clip(v_x, l, u)

        # Extrapolation
        x_bar = x_new + theta * (x_new - x_curr)

        # Dual update
        v_y = y_curr + sigma * (q - K @ x_bar)
        y_new = v_y.copy()
        y_new[:m1] = np.maximum(0, v_y[:m1])  # Project inequality duals to non-negative

        x_traj.append(x_new.copy())
        y_traj.append(y_new.copy())

        x_curr = x_new
        y_curr = y_new

    return x_traj, y_traj


def build_pdhg_symbolic_reps(K_iter, tau, sigma, theta, M_coupling, m1):
    """
    Build symbolic representations for PDHG interpolation conditions.

    Note: For LP, the coupling is a matrix K, not scalar M.
    The interpolation conditions for convex functions don't depend on coupling,
    so we can still verify f1 and h interpolation separately.

    Gram basis (dimG = 2K + 6):
        [delta_x0, delta_y0, xs, ys, gf1_0, gh_0, gf1_1, gh_1, ..., gf1_K, gh_K]

    Function values:
        F1: (K+2,) for f1 at x_0, ..., x_K, x_s
        F_h: (K+2,) for h at y_0, ..., y_K, y_s
    """
    dimG = 2 * K_iter + 6
    dimF1 = K_iter + 2
    dimF_h = K_iter + 2

    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)
    eyeF_h = jnp.eye(dimF_h)

    # Index helpers
    idx_delta_x0 = 0
    idx_delta_y0 = 1
    idx_xs = 2
    idx_ys = 3

    def idx_gf1(k):
        return 4 + 2 * k

    def idx_gh(k):
        return 5 + 2 * k

    n_points = K_iter + 1  # x_0, ..., x_K and y_0, ..., y_K

    # Representations for f1 (primal function, K+2 points including optimal)
    repX_f1 = jnp.zeros((n_points + 1, dimG))
    repG_f1 = jnp.zeros((n_points + 1, dimG))
    repF_f1 = jnp.zeros((n_points + 1, dimF1))

    # Representations for h (dual function, K+2 points including optimal)
    repY_h = jnp.zeros((n_points + 1, dimG))
    repG_h = jnp.zeros((n_points + 1, dimG))
    repF_h = jnp.zeros((n_points + 1, dimF_h))

    # For f1 interpolation, we need x_k - x_s and subgradient at each point
    # x_0 - x_s = delta_x0 (index 0 in Gram basis)
    # x_s - x_s = 0 (zero vector)

    # We cannot symbolically compute x_k representations from PDHG dynamics
    # because the coupling involves matrix K, not scalar M.
    #
    # However, for testing actual trajectories, we compute G and F numerically
    # and verify they satisfy interpolation conditions.
    #
    # For symbolic reps, we just set up the structure that matches the numerical G, F.

    # x_0: represented by delta_x0
    repX_f1 = repX_f1.at[0].set(eyeG[idx_delta_x0, :])
    repG_f1 = repG_f1.at[0].set(eyeG[idx_gf1(0), :])
    repF_f1 = repF_f1.at[0].set(eyeF1[0, :])

    # x_s (optimal): x_s - x_s = 0
    repX_f1 = repX_f1.at[n_points].set(jnp.zeros(dimG))
    # gf1_s: For the shifted problem, at x_s = 0, the subgradient depends on optimality conditions
    # We don't have a simple symbolic form, so we leave it as a placeholder
    # The actual test will use the numerical values
    repG_f1 = repG_f1.at[n_points].set(jnp.zeros(dimG))  # Placeholder
    repF_f1 = repF_f1.at[n_points].set(jnp.zeros(dimF1))

    # y_0: represented by delta_y0
    repY_h = repY_h.at[0].set(eyeG[idx_delta_y0, :])
    repG_h = repG_h.at[0].set(eyeG[idx_gh(0), :])
    repF_h = repF_h.at[0].set(eyeF_h[0, :])

    # y_s (optimal): y_s - y_s = 0
    repY_h = repY_h.at[n_points].set(jnp.zeros(dimG))
    repG_h = repG_h.at[n_points].set(jnp.zeros(dimG))  # Placeholder
    repF_h = repF_h.at[n_points].set(jnp.zeros(dimF_h))

    # For k = 1, ..., K: we cannot compute symbolic x_k - x_s without knowing the dynamics
    # We'll build these from the numerical Gram matrix

    return repX_f1, repG_f1, repF_f1, repY_h, repG_h, repF_h


def extract_f1_rep_from_gram(G, F, K_iter, n, m):
    """
    Extract f1 interpolation representation from numerical Gram matrix.

    For f1, we need (x_i - x_s, gf1_i, f1_i) for each point.

    The Gram basis is:
        [delta_x0, delta_y0, xs, ys, gf1_0, gh_0, gf1_1, gh_1, ..., gf1_K, gh_K]

    For f1 interpolation, we use the primal subspace (first n components of embedded vectors).
    """
    dimG = G.shape[0]
    dimF1 = K_iter + 2
    dimF_h = K_iter + 2

    n_points = K_iter + 1  # x_0, ..., x_K

    # F1 is the first K+2 values of F
    F1 = F[:dimF1]

    # Build representations for f1 from the Gram structure
    # x_k - x_s is the k-th iterate minus optimal
    # gf1_k is at index 4 + 2*k

    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)

    repX_f1 = jnp.zeros((n_points + 1, dimG))
    repG_f1 = jnp.zeros((n_points + 1, dimG))
    repF_f1 = jnp.zeros((n_points + 1, dimF1))

    # For the actual trajectory, we need to extract the x representations
    # from the Gram matrix structure.
    #
    # The embedded vectors are:
    #   delta_x0 = [x_0 - x_opt; 0]  (column 0)
    #   delta_y0 = [0; y_0 - y_opt]  (column 1)
    #   xs = [0; 0]                   (column 2)
    #   ys = [0; 0]                   (column 3)
    #   gf1_k = [gf1_k; 0]           (column 4 + 2*k)
    #   gh_k = [0; gh_k]             (column 5 + 2*k)
    #
    # For f1 interpolation, we only need the primal parts.
    # The Gram matrix G captures inner products between all embedded vectors.
    #
    # repX_f1[k] should be the Gram-basis representation of (x_k - x_s).
    # For k = 0: this is delta_x0, so repX_f1[0] = eyeG[0, :]
    # For k = 1, ..., K: we need to express x_k - x_s in terms of Gram basis.
    #
    # The issue is that x_k for k >= 1 is computed through PDHG dynamics
    # and cannot be expressed as a simple linear combination of basis vectors
    # without knowing the numerical values.
    #
    # Solution: We compute the actual inner products from the trajectory
    # and build a smaller representation that only involves the points
    # needed for f1 interpolation.

    # For testing purposes, we'll verify interpolation using the actual
    # numerical Gram matrix and function values, not symbolic representations.

    return repX_f1, repG_f1, repF_f1, F1


def check_convex_interpolation_numerical(x_points, g_points, f_values, tol=1e-6):
    """
    Check convex interpolation conditions numerically.

    For convex function f, for all pairs (i, j) with i != j:
        f(x_i) - f(x_j) - <g_j, x_i - x_j> >= 0

    Args:
        x_points: List of points (n,) arrays
        g_points: List of subgradients (n,) arrays at each point
        f_values: List of function values at each point
        tol: Tolerance for constraint satisfaction

    Returns:
        satisfied: True if all conditions are satisfied
        max_violation: Maximum violation
        violations: Array of all constraint values
    """
    n_points = len(x_points)
    violations = []

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # f(x_i) - f(x_j) - <g_j, x_i - x_j> >= 0
                lhs = f_values[i] - f_values[j] - np.dot(g_points[j], x_points[i] - x_points[j])
                violations.append(-lhs)  # We want lhs >= 0, so violation is -lhs if positive

    violations = np.array(violations)
    max_violation = np.max(violations)
    satisfied = max_violation <= tol

    return satisfied, max_violation, violations


class TestPDHGInterpolation(unittest.TestCase):
    """Tests verifying PDHG trajectories satisfy interpolation conditions."""

    def setUp(self):
        """Generate random LP problem data."""
        np.random.seed(42)

        self.n = 5    # Primal dimension
        self.m1 = 3   # Number of inequality constraints
        self.m2 = 2   # Number of equality constraints
        self.K_iter = 3  # Number of PDHG iterations

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

        # Compute step sizes
        K_norm = np.linalg.norm(self.K_mat, ord=2)
        self.tau = 0.9 / K_norm
        self.sigma = 0.9 / K_norm
        self.theta = 1.0

        # Initial point (feasible)
        self.x0 = 0.5 * (self.l + self.u)  # Midpoint of box
        self.y0 = np.zeros(self.m1 + self.m2)
        self.y0[:self.m1] = 0.1  # Small positive values for inequality duals

    def test_f1_interpolation_numerical(self):
        """Test that f1 (primal function) satisfies convex interpolation."""
        # Run PDHG in original coordinates
        x_traj, y_traj = run_pdhg_numpy(
            self.c, self.K_mat, self.q, self.l, self.u,
            self.x0, self.y0, self.tau, self.sigma, self.theta,
            self.K_iter, self.m1
        )

        # Shift to coordinates where optimal is at origin
        x_traj_shifted = [x - self.x_opt for x in x_traj]

        # Compute subgradients of f1 at each point
        # f1(x) = c^T x + indicator_{[l,u]}(x)
        # In shifted coords: f1_shifted(x) = c^T (x + x_opt)
        # Subgradient: c + normal_cone
        #
        # For points in interior: gf1 = c
        # For points on boundary: gf1 = c + normal_cone_element

        gf1_points = []
        f1_values = []

        l_shifted = self.l - self.x_opt
        u_shifted = self.u - self.x_opt

        for k, x_shifted in enumerate(x_traj_shifted):
            # Compute subgradient from PDHG optimality
            if k == 0:
                # Initial point: use c (assuming interior or any valid subgrad)
                gf1 = self.c.copy()
            else:
                # From PDHG: gf1_{k} = (v_prox - x_k) / tau
                # where v_prox = x_{k-1} + tau * K^T y_{k-1} (input to prox before -tau*c)
                # Note: x_prev is in shifted coords, y_prev is in original coords
                x_prev_original = x_traj[k-1]  # Original coordinates
                y_prev = y_traj[k-1]  # Original coords
                v_prox = x_prev_original + self.tau * (self.K_mat.T @ y_prev)
                x_curr_original = x_traj[k]
                gf1 = (v_prox - x_curr_original) / self.tau

            gf1_points.append(gf1)

            # f1_shifted(x) = c^T (x + x_opt)
            f1_val = np.dot(self.c, x_shifted + self.x_opt)
            f1_values.append(f1_val)

        # Add optimal point (shifted: x_s = 0)
        x_s_shifted = np.zeros(self.n)
        x_traj_shifted.append(x_s_shifted)

        # Subgradient at optimal: from KKT conditions
        # Our Lagrangian: L = c^T x + y^T (q - K x) = c^T x - y^T K x + y^T q
        # Stationarity: c - K^T y_opt + n = 0 where n ∈ N_{[l,u]}(x_opt)
        # So gf1_s = c + n = K^T y_opt ∈ ∂f1(x_opt) = c + N_{[l,u]}(x_opt)
        gf1_s = self.K_mat.T @ self.y_opt

        gf1_points.append(gf1_s)
        f1_values.append(np.dot(self.c, self.x_opt))  # f1(x_opt)

        # Check interpolation
        satisfied, max_violation, _ = check_convex_interpolation_numerical(
            x_traj_shifted, gf1_points, f1_values
        )

        self.assertLessEqual(max_violation, 1e-4,
            f"f1 interpolation violated! Max violation: {max_violation}")

    def test_h_interpolation_numerical(self):
        """Test that h (dual function) satisfies convex interpolation."""
        # Run PDHG in original coordinates
        x_traj, y_traj = run_pdhg_numpy(
            self.c, self.K_mat, self.q, self.l, self.u,
            self.x0, self.y0, self.tau, self.sigma, self.theta,
            self.K_iter, self.m1
        )

        # Shift to coordinates where optimal is at origin
        y_traj_shifted = [y - self.y_opt for y in y_traj]

        # Compute subgradients of h at each point
        # h(y) = -q^T y + indicator_Y(y) where Y = {y: y[:m1] >= 0}
        # In shifted coords: h_shifted(y) = -q^T (y + y_opt) + indicator
        # Subgradient: -q + normal_cone

        gh_points = []
        h_values = []

        for k, y_shifted in enumerate(y_traj_shifted):
            y_actual = y_shifted + self.y_opt

            if k == 0:
                # Initial point: use -q (assuming interior)
                gh = -self.q.copy()
            else:
                # From PDHG: gh_k = -q + (v_y - y_k) / sigma
                y_prev = y_traj_shifted[k-1] + self.y_opt
                x_bar_prev = x_traj[k] + self.theta * (x_traj[k] - x_traj[k-1])  # Extrapolated
                v_y = y_prev + self.sigma * (self.q - self.K_mat @ x_bar_prev)

                # Project v_y to get y_actual
                y_proj = v_y.copy()
                y_proj[:self.m1] = np.maximum(0, v_y[:self.m1])

                # Normal cone contribution
                normal_cone = (v_y - y_proj) / self.sigma
                gh = -self.q + normal_cone

            gh_points.append(gh)

            # h_shifted(y) = -q^T (y + y_opt)
            h_val = -np.dot(self.q, y_shifted + self.y_opt)
            h_values.append(h_val)

        # Add optimal point (shifted: y_s = 0)
        y_s_shifted = np.zeros(self.m1 + self.m2)
        y_traj_shifted.append(y_s_shifted)

        # Subgradient at optimal: from optimality conditions
        # At saddle point: q - K x_opt + normal_cone = 0 (for dual)
        # So the normal cone element is K x_opt - q
        # gh_s = -q + normal_cone = -q + K x_opt - q = K x_opt - 2q? That's not right.
        #
        # Let me reconsider. The dual function in the Lagrangian context:
        # We're maximizing over y, so at optimum: grad_y L = q - K x_opt
        # For constrained optimization, this should equal the normal cone to Y at y_opt.
        #
        # For h(y) = -q^T y + indicator_Y(y):
        # At the optimal y_opt: 0 in -q + N_Y(y_opt)
        # So a valid subgradient is gh_s = -q + n where n in N_Y(y_opt)
        #
        # From KKT: q - K x_opt in N_Y(y_opt) for the constrained part

        # For simplicity, use -q for interior parts and compute normal cone for boundary
        gh_s = -self.q.copy()

        # Adjust for boundary conditions (complementary slackness)
        for i in range(self.m1):
            if self.y_opt[i] < 1e-8:  # y_opt[i] = 0 (binding)
                # Normal cone allows any value <= 0
                # From KKT: (q - K x_opt)[i] <= 0
                kkt_residual = (self.q - self.K_mat @ self.x_opt)[i]
                gh_s[i] = -self.q[i] + min(0, kkt_residual)

        gh_points.append(gh_s)
        h_values.append(-np.dot(self.q, self.y_opt))  # h(y_opt)

        # Check interpolation
        satisfied, max_violation, _ = check_convex_interpolation_numerical(
            y_traj_shifted, gh_points, h_values
        )

        self.assertLessEqual(max_violation, 1e-4,
            f"h interpolation violated! Max violation: {max_violation}")

    def test_jax_trajectories_match_numpy(self):
        """Verify JAX trajectory function matches NumPy implementation."""
        # Run PDHG in NumPy
        x_traj_np, y_traj_np = run_pdhg_numpy(
            self.c, self.K_mat, self.q, self.l, self.u,
            self.x0, self.y0, self.tau, self.sigma, self.theta,
            self.K_iter, self.m1
        )

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

        # Run JAX implementation
        result = problem_data_to_pdhg_trajectories(
            stepsizes, c_jax, K_jax, q_jax, l_jax, u_jax,
            x0_jax, y0_jax, x_opt_jax, y_opt_jax, self.f_opt,
            self.K_iter, self.m1, return_Gram_representation=False
        )
        x_iter_jax, y_iter_jax, gf1_iter_jax, gh_iter_jax, f1_iter_jax, fh_iter_jax = result

        # Shift NumPy trajectories for comparison
        x_traj_np_shifted = [x - self.x_opt for x in x_traj_np]
        y_traj_np_shifted = [y - self.y_opt for y in y_traj_np]

        # Compare trajectories
        for k in range(self.K_iter + 1):
            np.testing.assert_allclose(
                np.array(x_iter_jax[:, k]), x_traj_np_shifted[k],
                atol=1e-10, err_msg=f"x mismatch at iteration {k}"
            )
            np.testing.assert_allclose(
                np.array(y_iter_jax[:, k]), y_traj_np_shifted[k],
                atol=1e-10, err_msg=f"y mismatch at iteration {k}"
            )

    def test_gram_representation_structure(self):
        """Verify Gram matrix has correct structure."""
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

        G, F = problem_data_to_pdhg_trajectories(
            stepsizes, c_jax, K_jax, q_jax, l_jax, u_jax,
            x0_jax, y0_jax, x_opt_jax, y_opt_jax, self.f_opt,
            self.K_iter, self.m1, return_Gram_representation=True
        )

        # Check dimensions
        expected_dimG = 2 * self.K_iter + 6
        expected_dimF = 2 * (self.K_iter + 2)

        self.assertEqual(G.shape, (expected_dimG, expected_dimG),
            f"G shape mismatch: expected {(expected_dimG, expected_dimG)}, got {G.shape}")
        self.assertEqual(F.shape, (expected_dimF,),
            f"F shape mismatch: expected {(expected_dimF,)}, got {F.shape}")

        # Check G is symmetric
        np.testing.assert_allclose(np.array(G), np.array(G.T), atol=1e-12,
            err_msg="G should be symmetric")

        # Check G is positive semidefinite
        eigvals = np.linalg.eigvalsh(np.array(G))
        self.assertTrue(np.all(eigvals >= -1e-10),
            f"G should be PSD, but has eigenvalue {np.min(eigvals)}")

        # Check xs and ys columns are zero (columns 2 and 3)
        # In shifted coordinates, x_s = 0 and y_s = 0
        # So G[:, 2] and G[:, 3] should be close to zero
        # (within numerical precision of inner products)
        G_np = np.array(G)
        np.testing.assert_allclose(G_np[:, 2], 0, atol=1e-10,
            err_msg="xs column should be zero")
        np.testing.assert_allclose(G_np[:, 3], 0, atol=1e-10,
            err_msg="ys column should be zero")

    def test_f1_interpolation_via_gram(self):
        """Test f1 interpolation using Gram representation and convex_interp."""
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

        # Get raw trajectories
        result = problem_data_to_pdhg_trajectories(
            stepsizes, c_jax, K_jax, q_jax, l_jax, u_jax,
            x0_jax, y0_jax, x_opt_jax, y_opt_jax, self.f_opt,
            self.K_iter, self.m1, return_Gram_representation=False
        )
        x_iter, y_iter, gf1_iter, gh_iter, f1_iter, fh_iter = result

        # Build f1-specific Gram representation
        # For f1, we only use primal vectors (embedded as [v; 0])
        n_points = self.K_iter + 1  # x_0, ..., x_K
        n = self.n
        m = self.m1 + self.m2

        # Build G_half for f1 only (primal space)
        # Columns: [x_0 - x_s, x_1 - x_s, ..., x_K - x_s, gf1_0, gf1_1, ..., gf1_K, gf1_s]
        # x_s = 0 in shifted coords

        G_half_f1_cols = []
        for k in range(n_points):
            G_half_f1_cols.append(x_iter[:, k])
        G_half_f1_cols.append(jnp.zeros(n))  # x_s = 0

        for k in range(n_points):
            G_half_f1_cols.append(gf1_iter[:, k])

        # gf1_s: subgradient at optimal
        # From KKT: c - K^T y_opt + n_x = 0, so gf1_s = c + n_x = K^T y_opt
        gf1_s = K_jax.T @ y_opt_jax
        G_half_f1_cols.append(gf1_s)

        G_half_f1 = jnp.column_stack(G_half_f1_cols)
        G_f1 = G_half_f1.T @ G_half_f1

        # Build representations for convex_interp
        dimG_f1 = 2 * (n_points + 1)  # (K+2) x points + (K+2) g points
        eyeG_f1 = jnp.eye(dimG_f1)
        dimF1 = n_points + 1
        eyeF1 = jnp.eye(dimF1)

        repX_f1 = jnp.zeros((n_points + 1, dimG_f1))
        repG_f1 = jnp.zeros((n_points + 1, dimG_f1))
        repF_f1 = jnp.zeros((n_points + 1, dimF1))

        for k in range(n_points + 1):
            repX_f1 = repX_f1.at[k].set(eyeG_f1[k, :])
            repG_f1 = repG_f1.at[k].set(eyeG_f1[n_points + 1 + k, :])
            repF_f1 = repF_f1.at[k].set(eyeF1[k, :])

        # Build F1 values
        f1_s = jnp.dot(c_jax, x_opt_jax)
        F1 = jnp.concatenate([f1_iter - f1_s, jnp.array([0.0])])

        # Compute interpolation constraints
        A_vals_f1, b_vals_f1 = convex_interp(repX_f1, repG_f1, repF_f1, n_points)

        # Check constraints: trace(A @ G) + b @ F <= 0
        violations = []
        for i in range(A_vals_f1.shape[0]):
            val = jnp.trace(A_vals_f1[i] @ G_f1) + jnp.dot(b_vals_f1[i], F1)
            violations.append(float(val))

        max_violation = max(violations)
        self.assertLessEqual(max_violation, 1e-4,
            f"f1 interpolation via Gram violated! Max violation: {max_violation}")

    def test_h_interpolation_via_gram(self):
        """Test h interpolation using Gram representation and convex_interp."""
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

        # Get raw trajectories
        result = problem_data_to_pdhg_trajectories(
            stepsizes, c_jax, K_jax, q_jax, l_jax, u_jax,
            x0_jax, y0_jax, x_opt_jax, y_opt_jax, self.f_opt,
            self.K_iter, self.m1, return_Gram_representation=False
        )
        x_iter, y_iter, gf1_iter, gh_iter, f1_iter, fh_iter = result

        # Build h-specific Gram representation
        n_points = self.K_iter + 1  # y_0, ..., y_K
        m = self.m1 + self.m2

        G_half_h_cols = []
        for k in range(n_points):
            G_half_h_cols.append(y_iter[:, k])
        G_half_h_cols.append(jnp.zeros(m))  # y_s = 0

        for k in range(n_points):
            G_half_h_cols.append(gh_iter[:, k])

        # gh_s: subgradient at optimal
        # For h(y) = -q^T y + indicator, at y_opt: -q + normal_cone = 0
        # From KKT: q - K x_opt in normal_cone (for inequality parts)
        gh_s = -q_jax + (q_jax - K_jax @ x_opt_jax)  # = -K x_opt
        # Actually, let's be more careful about the normal cone
        gh_s = -q_jax.copy()

        G_half_h_cols.append(gh_s)

        G_half_h = jnp.column_stack(G_half_h_cols)
        G_h = G_half_h.T @ G_half_h

        # Build representations for convex_interp
        dimG_h = 2 * (n_points + 1)
        eyeG_h = jnp.eye(dimG_h)
        dimF_h = n_points + 1
        eyeF_h = jnp.eye(dimF_h)

        repY_h = jnp.zeros((n_points + 1, dimG_h))
        repG_h = jnp.zeros((n_points + 1, dimG_h))
        repF_h = jnp.zeros((n_points + 1, dimF_h))

        for k in range(n_points + 1):
            repY_h = repY_h.at[k].set(eyeG_h[k, :])
            repG_h = repG_h.at[k].set(eyeG_h[n_points + 1 + k, :])
            repF_h = repF_h.at[k].set(eyeF_h[k, :])

        # Build F_h values
        h_s = -jnp.dot(q_jax, y_opt_jax)
        F_h = jnp.concatenate([fh_iter - h_s, jnp.array([0.0])])

        # Compute interpolation constraints
        A_vals_h, b_vals_h = convex_interp(repY_h, repG_h, repF_h, n_points)

        # Check constraints
        violations = []
        for i in range(A_vals_h.shape[0]):
            val = jnp.trace(A_vals_h[i] @ G_h) + jnp.dot(b_vals_h[i], F_h)
            violations.append(float(val))

        max_violation = max(violations)
        self.assertLessEqual(max_violation, 1e-4,
            f"h interpolation via Gram violated! Max violation: {max_violation}")


class TestPDHGMultipleProblems(unittest.TestCase):
    """Test PDHG interpolation across multiple random LP instances."""

    def test_multiple_lps(self):
        """Test interpolation on multiple random LPs."""
        num_problems = 5
        K_iter = 2

        for seed in range(num_problems):
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
                # Skip infeasible problems
                continue

            # Step sizes
            K_norm = np.linalg.norm(K_mat, ord=2)
            tau = 0.9 / K_norm
            sigma = 0.9 / K_norm
            theta = 1.0

            # Initial point
            x0 = 0.5 * (l + u)
            y0 = np.zeros(m1 + m2)
            y0[:m1] = 0.1

            # Run PDHG
            x_traj, y_traj = run_pdhg_numpy(
                c, K_mat, q, l, u, x0, y0, tau, sigma, theta, K_iter, m1
            )

            # Check f1 interpolation
            x_traj_shifted = [x - x_opt for x in x_traj]

            gf1_points = []
            f1_values = []

            for k, x_shifted in enumerate(x_traj_shifted):
                if k == 0:
                    gf1 = c.copy()
                else:
                    # gf1_k = (v_prox - x_k) / tau where v_prox = x_{k-1} + tau * K^T y_{k-1}
                    x_prev_original = x_traj[k-1]
                    y_prev = y_traj[k-1]
                    v_prox = x_prev_original + tau * (K_mat.T @ y_prev)
                    x_curr_original = x_traj[k]
                    gf1 = (v_prox - x_curr_original) / tau

                gf1_points.append(gf1)
                f1_values.append(np.dot(c, x_shifted + x_opt))

            # Add optimal
            x_traj_shifted.append(np.zeros(n))
            # gf1_s = K^T y_opt (from KKT conditions)
            gf1_s = K_mat.T @ y_opt
            gf1_points.append(gf1_s)
            f1_values.append(np.dot(c, x_opt))

            satisfied, max_viol, _ = check_convex_interpolation_numerical(
                x_traj_shifted, gf1_points, f1_values
            )

            self.assertLessEqual(max_viol, 1e-4,
                f"Problem {seed}: f1 interpolation violated with max violation {max_viol}")


if __name__ == '__main__':
    unittest.main()
