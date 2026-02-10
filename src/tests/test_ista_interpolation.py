"""
Unit tests for ISTA with custom interpolation conditions.

This test verifies that actual ISTA trajectories on generated lasso problems
satisfy the interpolation conditions in our custom representation.
"""

import unittest
import numpy as np
import cvxpy as cp

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning.pep_constructions import (
    smooth_strongly_convex_interp,
    convex_interp,
)


def soft_threshold(v, delta):
    """Soft-thresholding operator."""
    return np.sign(v) * np.maximum(np.abs(v) - delta, 0)


def solve_lasso(A, b, lambd):
    """Solve lasso problem using cvxpy to get optimal point."""
    n = A.shape[1]
    x = cp.Variable(n)
    obj = 0.5 * cp.sum_squares(A @ x - b) + lambd * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver='CLARABEL')
    return x.value, prob.value


def run_ista(x0, A, b, lambd, gamma, K):
    """
    Run ISTA for K iterations.
    
    Returns:
        x_iterates: List of x_k for k = 0, 1, ..., K
        g_iterates: List of grad_f1(x_k) for k = 0, 1, ..., K
        h_iterates: List of subgrad_f2(x_k) for k = 0, 1, ..., K
        f1_iterates: List of f1(x_k) for k = 0, 1, ..., K
        f2_iterates: List of f2(x_k) for k = 0, 1, ..., K
    """
    def f1(x):
        return 0.5 * np.linalg.norm(A @ x - b) ** 2
    
    def f2(x):
        return lambd * np.linalg.norm(x, 1)
    
    def grad_f1(x):
        return A.T @ (A @ x - b)
    
    def subgrad_f2(x):
        """Valid subgradient of lambda * ||x||_1."""
        return lambd * np.sign(x)
    
    x_iterates = [x0]
    g_iterates = [grad_f1(x0)]
    h_iterates = [subgrad_f2(x0)]
    f1_iterates = [f1(x0)]
    f2_iterates = [f2(x0)]
    
    x_curr = x0
    for k in range(K):
        y_k = x_curr - gamma * grad_f1(x_curr)
        x_new = soft_threshold(y_k, gamma * lambd)
        h_new = (y_k - x_new) / gamma  # Reverse-calculated subgradient
        
        x_iterates.append(x_new)
        g_iterates.append(grad_f1(x_new))
        h_iterates.append(h_new)
        f1_iterates.append(f1(x_new))
        f2_iterates.append(f2(x_new))
        
        x_curr = x_new
    
    return x_iterates, g_iterates, h_iterates, f1_iterates, f2_iterates


def build_gram_representation(x_iterates, g_iterates, h_iterates, f1_iterates, f2_iterates, 
                               x_s, f1_s, f2_s, A, b):
    """
    Build the Gram representation for ISTA.
    
    Gram basis (dimG = 2K + 5):
        [x_0 - x_s, g_0, h_0, h_1, g_1, h_2, g_2, ..., h_K, g_K, g_s, h_s]
    
    Returns:
        G: Gram matrix
        F1: f1 function values relative to f1_s
        F2: f2 function values relative to f2_s
    """
    K = len(x_iterates) - 1
    
    # Compute gradient and subgradient at optimal
    g_s = A.T @ (A @ x_s - b)  # grad_f1(x_s)
    h_s = -g_s  # At stationarity: g_s + h_s = 0
    
    # Build G_half for Gram matrix
    G_half_columns = []
    
    G_half_columns.append(x_iterates[0] - x_s)  # x_0 - x_s
    G_half_columns.append(g_iterates[0])  # g_0
    G_half_columns.append(h_iterates[0])  # h_0
    
    for k in range(1, K + 1):
        G_half_columns.append(h_iterates[k])
        G_half_columns.append(g_iterates[k])
    
    G_half_columns.append(g_s)
    G_half_columns.append(h_s)
    
    G_half = np.column_stack(G_half_columns)
    G = G_half.T @ G_half
    
    F1 = np.array([f1 - f1_s for f1 in f1_iterates] + [0.0])
    F2 = np.array([f2 - f2_s for f2 in f2_iterates] + [0.0])
    
    # Concatenate F1 and F2 for consistency with DRO pipeline
    F = np.concatenate([F1, F2])
    
    return G, F1, F2, F


def check_interpolation_constraints(G, F, A_vals, b_vals, tol=1e-6):
    """Check if all interpolation constraints trace(A @ G) + b @ F <= 0 are satisfied."""
    num_constraints = A_vals.shape[0]
    violations = []
    
    for m in range(num_constraints):
        val = np.trace(A_vals[m] @ G) + b_vals[m] @ F
        violations.append(val)
    
    violations = np.array(violations)
    max_violation = np.max(violations)
    satisfied = max_violation <= tol
    
    return satisfied, max_violation, violations


def build_symbolic_reps(K, gamma):
    """
    Build symbolic representations for interpolation conditions.
    
    Returns repX, repG, repF for both f1 and f2.
    """
    # dimG = 2K + 5: [x_0-x_s, g_0, h_0, h_1, g_1, ..., h_K, g_K, g_s, h_s]
    dimG = 2 * K + 5
    dimF1 = K + 2  # K+1 algorithm points + x_s
    dimF2 = K + 2
    
    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)
    eyeF2 = jnp.eye(dimF2)
    
    def idx_g(k):
        """Index of g_k in Gram basis."""
        if k == 0:
            return 1
        return 2 * k + 2
    
    def idx_h(k):
        """Index of h_k in Gram basis."""
        if k == 0:
            return 2
        return 2 * k + 1
    
    idx_gs = 2 * K + 3  # Index of g_s
    idx_hs = 2 * K + 4  # Index of h_s
    
    # f1 representations
    repX_f1 = jnp.zeros((K + 2, dimG))
    repG_f1 = jnp.zeros((K + 2, dimG))
    repF_f1 = jnp.zeros((K + 2, dimF1))
    
    # f2 representations
    repX_f2 = jnp.zeros((K + 2, dimG))
    repG_f2 = jnp.zeros((K + 2, dimG))
    repF_f2 = jnp.zeros((K + 2, dimF2))
    
    # Initial point x_0
    x_rep = eyeG[0, :]
    repX_f1 = repX_f1.at[0].set(x_rep)
    repG_f1 = repG_f1.at[0].set(eyeG[idx_g(0), :])
    repF_f1 = repF_f1.at[0].set(eyeF1[0, :])
    
    repX_f2 = repX_f2.at[0].set(x_rep)
    repG_f2 = repG_f2.at[0].set(eyeG[idx_h(0), :])
    repF_f2 = repF_f2.at[0].set(eyeF2[0, :])
    
    # Build x_k representations through ISTA dynamics
    for k in range(K):
        x_rep = x_rep - gamma * eyeG[idx_g(k), :] - gamma * eyeG[idx_h(k+1), :]
        
        repX_f1 = repX_f1.at[k + 1].set(x_rep)
        repG_f1 = repG_f1.at[k + 1].set(eyeG[idx_g(k + 1), :])
        repF_f1 = repF_f1.at[k + 1].set(eyeF1[k + 1, :])
        
        repX_f2 = repX_f2.at[k + 1].set(x_rep)
        repG_f2 = repG_f2.at[k + 1].set(eyeG[idx_h(k + 1), :])
        repF_f2 = repF_f2.at[k + 1].set(eyeF2[k + 1, :])
    
    # Optimal point x_s (at index K+1)
    xs_rep = jnp.zeros(dimG)  # x_s - x_s = 0
    repX_f1 = repX_f1.at[K + 1].set(xs_rep)
    repG_f1 = repG_f1.at[K + 1].set(eyeG[idx_gs, :])  # g_s
    repF_f1 = repF_f1.at[K + 1].set(jnp.zeros(dimF1))
    
    repX_f2 = repX_f2.at[K + 1].set(xs_rep)
    repG_f2 = repG_f2.at[K + 1].set(eyeG[idx_hs, :])  # h_s
    repF_f2 = repF_f2.at[K + 1].set(jnp.zeros(dimF2))
    
    return repX_f1, repG_f1, repF_f1, repX_f2, repG_f2, repF_f2


class TestISTAInterpolation(unittest.TestCase):
    """Tests verifying ISTA trajectories satisfy interpolation conditions."""
    
    def setUp(self):
        """Generate random lasso problem data."""
        np.random.seed(42)
        
        self.m = 20
        self.n = 10
        self.K = 3
        self.lambd = 0.1
        
        self.A = np.random.randn(self.m, self.n) / np.sqrt(self.m)
        
        ATA = self.A.T @ self.A
        eigvals = np.linalg.eigvalsh(ATA)
        self.L = np.max(eigvals)
        self.mu = np.min(eigvals) if self.m >= self.n else 0.0
        
        self.b = np.random.randn(self.m)
        
        self.x_s, self.f_s = solve_lasso(self.A, self.b, self.lambd)
        
        self.f1_s = 0.5 * np.linalg.norm(self.A @ self.x_s - self.b) ** 2
        self.f2_s = self.lambd * np.linalg.norm(self.x_s, 1)
        
        self.gamma = 1.0 / self.L
        self.x0 = np.zeros(self.n)
        self.R = np.linalg.norm(self.x0 - self.x_s)
    
    def test_ista_satisfies_f1_interpolation(self):
        """Test that ISTA trajectory satisfies f1 interpolation conditions."""
        x_iters, g_iters, h_iters, f1_iters, f2_iters = run_ista(
            self.x0, self.A, self.b, self.lambd, self.gamma, self.K
        )
        
        G, F1, F2, F = build_gram_representation(
            x_iters, g_iters, h_iters, f1_iters, f2_iters,
            self.x_s, self.f1_s, self.f2_s, self.A, self.b
        )
        
        repX_f1, repG_f1, repF_f1, _, _, _ = build_symbolic_reps(self.K, self.gamma)
        
        n_points_f1 = self.K + 1
        A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
            repX_f1, repG_f1, repF_f1, self.mu, self.L, n_points_f1
        )
        
        satisfied, max_violation, violations = check_interpolation_constraints(
            G, F1, np.array(A_vals_f1), np.array(b_vals_f1)
        )
        
        self.assertLessEqual(max_violation, 1e-4, 
            f"f1 interpolation violated! Max violation: {max_violation}")
    
    def test_ista_satisfies_f2_interpolation(self):
        """Test that ISTA trajectory satisfies f2 (convex) interpolation conditions."""
        x_iters, g_iters, h_iters, f1_iters, f2_iters = run_ista(
            self.x0, self.A, self.b, self.lambd, self.gamma, self.K
        )
        
        G, F1, F2, F = build_gram_representation(
            x_iters, g_iters, h_iters, f1_iters, f2_iters,
            self.x_s, self.f1_s, self.f2_s, self.A, self.b
        )
        
        _, _, _, repX_f2, repG_f2, repF_f2 = build_symbolic_reps(self.K, self.gamma)
        
        n_points_f2 = self.K + 1
        A_vals_f2, b_vals_f2 = convex_interp(
            repX_f2, repG_f2, repF_f2, n_points_f2
        )
        
        satisfied, max_violation, violations = check_interpolation_constraints(
            G, F2, np.array(A_vals_f2), np.array(b_vals_f2)
        )
        
        self.assertLessEqual(max_violation, 1e-4,
            f"f2 interpolation violated! Max violation: {max_violation}")
    
    def test_initial_condition_satisfied(self):
        """Test that initial condition ||x0 - xs||^2 <= R^2 is satisfied."""
        dist_sq = np.linalg.norm(self.x0 - self.x_s) ** 2
        R_sq = self.R ** 2
        self.assertLessEqual(dist_sq, R_sq + 1e-10)
    
    def test_multiple_random_problems(self):
        """Test interpolation across multiple random problem instances."""
        num_problems = 5
        K = 2
        
        for seed in range(num_problems):
            np.random.seed(seed + 100)
            
            m, n = 15, 8
            A = np.random.randn(m, n) / np.sqrt(m)
            b = np.random.randn(m)
            lambd = 0.05
            
            ATA = A.T @ A
            L = np.max(np.linalg.eigvalsh(ATA))
            mu = np.min(np.linalg.eigvalsh(ATA)) if m >= n else 0.0
            gamma = 1.0 / L
            
            x_s, _ = solve_lasso(A, b, lambd)
            f1_s = 0.5 * np.linalg.norm(A @ x_s - b) ** 2
            f2_s = lambd * np.linalg.norm(x_s, 1)
            
            x0 = np.random.randn(n) * 0.1
            
            x_iters, g_iters, h_iters, f1_iters, f2_iters = run_ista(
                x0, A, b, lambd, gamma, K
            )
            
            G, F1, F2, F = build_gram_representation(
                x_iters, g_iters, h_iters, f1_iters, f2_iters,
                x_s, f1_s, f2_s, A, b
            )
            
            repX_f1, repG_f1, repF_f1, repX_f2, repG_f2, repF_f2 = build_symbolic_reps(K, gamma)
            
            # Check f1 interpolation
            A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
                repX_f1, repG_f1, repF_f1, mu, L, K + 1
            )
            _, max_viol_f1, _ = check_interpolation_constraints(
                G, F1, np.array(A_vals_f1), np.array(b_vals_f1)
            )
            
            # Check f2 interpolation
            A_vals_f2, b_vals_f2 = convex_interp(
                repX_f2, repG_f2, repF_f2, K + 1
            )
            _, max_viol_f2, _ = check_interpolation_constraints(
                G, F2, np.array(A_vals_f2), np.array(b_vals_f2)
            )
            
            self.assertLessEqual(max_viol_f1, 1e-4,
                f"Problem {seed}: f1 interpolation violated with max violation {max_viol_f1}")
            self.assertLessEqual(max_viol_f2, 1e-4,
                f"Problem {seed}: f2 interpolation violated with max violation {max_viol_f2}")


# ============================================================================
# FISTA Algorithm Functions
# ============================================================================

def run_fista(x0, A, b, lambd, gamma, K):
    """
    Run FISTA for K iterations.
    
    FISTA algorithm:
        y_0 = x_0
        For k = 0, ..., K-1:
            x_{k+1} = soft_threshold(y_k - gamma * grad_f1(y_k), gamma * lambda)
            beta_{k+1} = (1 + sqrt(1 + 4*beta_k^2)) / 2
            y_{k+1} = x_{k+1} + (beta_k - 1) / beta_{k+1} * (x_{k+1} - x_k)
    
    Key difference from ISTA: gradients are evaluated at y_k, not x_k.
    
    Returns:
        x_iterates: [x_0, x_1, ..., x_K]
        y_iterates: [y_0, y_1, ..., y_{K-1}] (K points where f1 gradients are evaluated)
        g_iterates: [g(y_0), g(y_1), ..., g(y_{K-1})] (f1 gradients at y points)
        h_iterates: [h_0, h_1, ..., h_K] (f2 subgradients at x points)
        f1_y_iters: [f1(y_0), ..., f1(y_{K-1})] (f1 at y points)
        f2_x_iters: [f2(x_0), f2(x_1), ..., f2(x_K)] (f2 at x points)
        betas: beta values used
    """
    def f1(x):
        return 0.5 * np.linalg.norm(A @ x - b) ** 2
    
    def f2(x):
        return lambd * np.linalg.norm(x, 1)
    
    def grad_f1(x):
        return A.T @ (A @ x - b)
    
    def subgrad_f2(x):
        return lambd * np.sign(x)
    
    x_iterates = [x0]
    y_iterates = [x0]  # y_0 = x_0
    g_iterates = [grad_f1(x0)]  # g(y_0)
    h_iterates = [subgrad_f2(x0)]  # h_0 at x_0
    f1_y_iters = [f1(x0)]  # f1(y_0)
    f2_x_iters = [f2(x0)]  # f2(x_0)
    betas = [1.0]
    
    x_curr = x0
    y_curr = x0
    beta_curr = 1.0
    
    for k in range(K):
        # Gradient step at y_k
        g_yk = grad_f1(y_curr)
        ytilde = y_curr - gamma * g_yk
        
        # Proximal step
        x_new = soft_threshold(ytilde, gamma * lambd)
        h_new = (ytilde - x_new) / gamma  # Reverse-calculated subgradient
        
        # Momentum update for beta
        beta_new = 0.5 * (1 + np.sqrt(1 + 4 * beta_curr ** 2))
        
        # Momentum update for y
        y_new = x_new + (beta_curr - 1) / beta_new * (x_new - x_curr)
        
        x_iterates.append(x_new)
        h_iterates.append(h_new)
        f2_x_iters.append(f2(x_new))
        betas.append(beta_new)
        
        if k < K - 1:  # Don't need y_K, g(y_K) for interpolation
            y_iterates.append(y_new)
            g_iterates.append(grad_f1(y_new))
            f1_y_iters.append(f1(y_new))
        
        x_curr = x_new
        y_curr = y_new
        beta_curr = beta_new
    
    return x_iterates, y_iterates, g_iterates, h_iterates, f1_y_iters, f2_x_iters, betas


def build_fista_gram_representation(x_iterates, y_iterates, g_iterates, h_iterates, 
                                     f1_y_iters, f2_x_iters, betas,
                                     x_s, f1_s, f2_s, A, b, gamma):
    """
    Build Gram representation for FISTA.
    
    For FISTA, f1 interpolation is at y points, f2 interpolation is at x points.
    y depends linearly on x: y_{k+1} = x_{k+1} + beta_k/beta_{k+1} * (x_{k+1} - x_k)
    
    Gram basis (dimG = 2K + 3):
        [x_0 - x_s, g(y_0), h_0, h_1, g(y_1), h_2, g(y_2), ..., h_K, g_s]
        At stationarity: h_s = -g_s
        
    Function values:
        F1: f1 at y_0, y_1, ..., y_{K-1}, y_s (K + 1 values)
        F2: f2 at x_0, x_1, ..., x_K, x_s (K + 2 values)
    """
    K = len(x_iterates) - 1
    
    # Compute gradient at optimal (h_s = -g_s by stationarity)
    g_s = A.T @ (A @ x_s - b)  # grad_f1(x_s)
    
    # Build G_half for Gram matrix
    # Structure: [x_0 - x_s, g(y_0), h_0, h_1, g(y_1), ..., h_{K-1}, g(y_{K-1}), h_K, g_s]
    G_half_columns = []
    
    G_half_columns.append(x_iterates[0] - x_s)  # x_0 - x_s
    G_half_columns.append(g_iterates[0])  # g(y_0)
    G_half_columns.append(h_iterates[0])  # h_0
    
    for k in range(1, K):
        G_half_columns.append(h_iterates[k])  # h_k
        G_half_columns.append(g_iterates[k])  # g(y_k)
    
    G_half_columns.append(h_iterates[K])  # h_K
    G_half_columns.append(g_s)  # g_s (h_s = -g_s by stationarity)
    
    G_half = np.column_stack(G_half_columns)
    G = G_half.T @ G_half
    
    # F1: f1 at y points relative to f1_s
    F1 = np.array([f1 - f1_s for f1 in f1_y_iters] + [0.0])
    
    # F2: f2 at x points relative to f2_s
    F2 = np.array([f2 - f2_s for f2 in f2_x_iters] + [0.0])
    
    # Concatenate F1 and F2 for consistency with DRO pipeline
    F = np.concatenate([F1, F2])
    
    return G, F1, F2, F


def build_fista_symbolic_reps(K, gamma, betas):
    """
    Build symbolic representations for FISTA interpolation conditions.
    
    For f1: interpolation at y points (where gradients are evaluated)
    For f2: interpolation at x points (where proximal is applied)
    
    Gram basis indices:
        0: x_0 - x_s
        1: g(y_0)
        2: h_0
        3: h_1
        4: g(y_1)
        5: h_2
        6: g(y_2)
        ...
        2k+1: h_k (for k >= 1)
        2k+2: g(y_k) (for k >= 1)
        ...
        2K+1: h_K
        2K+2: g_s (h_s = -g_s by stationarity)
        
    dimG = 2K + 3
    """
    dimG = 2 * K + 3
    dimF1 = K + 1  # f1 at y_0, ..., y_{K-1}, y_s
    dimF2 = K + 2  # f2 at x_0, ..., x_K, x_s
    
    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)
    eyeF2 = jnp.eye(dimF2)
    
    def idx_g_y(k):
        """Index of g(y_k) in Gram basis."""
        if k == 0:
            return 1
        return 2 * k + 2
    
    def idx_h(k):
        """Index of h_k in Gram basis."""
        if k == 0:
            return 2
        return 2 * k + 1
    
    idx_gs = 2 * K + 2  # h_s = -g_s by stationarity
    
    # f1 representations: interpolation at y_0, ..., y_{K-1}, y_s
    n_y_points = K  # y_0 to y_{K-1}
    repY_f1 = jnp.zeros((n_y_points + 1, dimG))  # +1 for y_s
    repG_f1 = jnp.zeros((n_y_points + 1, dimG))
    repF_f1 = jnp.zeros((n_y_points + 1, dimF1))
    
    # f2 representations: interpolation at x_0, ..., x_K, x_s
    repX_f2 = jnp.zeros((K + 2, dimG))
    repG_f2 = jnp.zeros((K + 2, dimG))
    repF_f2 = jnp.zeros((K + 2, dimF2))
    
    # Initial: y_0 = x_0
    x_rep = eyeG[0, :]  # x_0 - x_s
    y_rep = x_rep  # y_0 = x_0
    
    repY_f1 = repY_f1.at[0].set(y_rep)
    repG_f1 = repG_f1.at[0].set(eyeG[idx_g_y(0), :])
    repF_f1 = repF_f1.at[0].set(eyeF1[0, :])
    
    repX_f2 = repX_f2.at[0].set(x_rep)
    repG_f2 = repG_f2.at[0].set(eyeG[idx_h(0), :])
    repF_f2 = repF_f2.at[0].set(eyeF2[0, :])
    
    x_prev = x_rep
    
    # Build representations through FISTA dynamics
    for k in range(K):
        # x_{k+1} = y_k - gamma * g(y_k) - gamma * h_{k+1}
        x_new = y_rep - gamma * eyeG[idx_g_y(k), :] - gamma * eyeG[idx_h(k+1), :]
        
        # Store x_{k+1} for f2
        repX_f2 = repX_f2.at[k + 1].set(x_new)
        repG_f2 = repG_f2.at[k + 1].set(eyeG[idx_h(k + 1), :])
        repF_f2 = repF_f2.at[k + 1].set(eyeF2[k + 1, :])
        
        if k < K - 1:
            # y_{k+1} = x_{k+1} + (beta_k - 1) / beta_{k+1} * (x_{k+1} - x_k)
            mom_coef = (betas[k] - 1) / betas[k + 1]
            y_new = x_new + mom_coef * (x_new - x_prev)
            
            repY_f1 = repY_f1.at[k + 1].set(y_new)
            repG_f1 = repG_f1.at[k + 1].set(eyeG[idx_g_y(k + 1), :])
            repF_f1 = repF_f1.at[k + 1].set(eyeF1[k + 1, :])
            
            y_rep = y_new
        
        x_prev = x_new
    
    # Optimal point: y_s = x_s
    # g_s for f1, h_s = -g_s for f2
    ys_rep = jnp.zeros(dimG)
    repY_f1 = repY_f1.at[K].set(ys_rep)
    repG_f1 = repG_f1.at[K].set(eyeG[idx_gs, :])  # g_s
    repF_f1 = repF_f1.at[K].set(jnp.zeros(dimF1))
    
    xs_rep = jnp.zeros(dimG)
    repX_f2 = repX_f2.at[K + 1].set(xs_rep)
    repG_f2 = repG_f2.at[K + 1].set(-eyeG[idx_gs, :])  # h_s = -g_s
    repF_f2 = repF_f2.at[K + 1].set(jnp.zeros(dimF2))
    
    return repY_f1, repG_f1, repF_f1, repX_f2, repG_f2, repF_f2


class TestFISTAInterpolation(unittest.TestCase):
    """Tests verifying FISTA trajectories satisfy interpolation conditions."""
    
    def setUp(self):
        """Generate random lasso problem data."""
        np.random.seed(42)
        
        self.m = 20
        self.n = 10
        self.K = 3
        self.lambd = 0.1
        
        self.A = np.random.randn(self.m, self.n) / np.sqrt(self.m)
        
        ATA = self.A.T @ self.A
        eigvals = np.linalg.eigvalsh(ATA)
        self.L = np.max(eigvals)
        self.mu = np.min(eigvals) if self.m >= self.n else 0.0
        
        self.b = np.random.randn(self.m)
        
        self.x_s, self.f_s = solve_lasso(self.A, self.b, self.lambd)
        
        self.f1_s = 0.5 * np.linalg.norm(self.A @ self.x_s - self.b) ** 2
        self.f2_s = self.lambd * np.linalg.norm(self.x_s, 1)
        
        self.gamma = 1.0 / self.L
        self.x0 = np.zeros(self.n)
    
    def test_fista_satisfies_f1_interpolation(self):
        """Test that FISTA trajectory satisfies f1 interpolation at y points."""
        x_iters, y_iters, g_iters, h_iters, f1_y_iters, f2_x_iters, betas = run_fista(
            self.x0, self.A, self.b, self.lambd, self.gamma, self.K
        )
        
        G, F1, F2, F = build_fista_gram_representation(
            x_iters, y_iters, g_iters, h_iters, f1_y_iters, f2_x_iters, betas,
            self.x_s, self.f1_s, self.f2_s, self.A, self.b, self.gamma
        )
        
        repY_f1, repG_f1, repF_f1, _, _, _ = build_fista_symbolic_reps(
            self.K, self.gamma, betas
        )
        
        n_points_f1 = self.K  # y_0 to y_{K-1}
        A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
            repY_f1, repG_f1, repF_f1, self.mu, self.L, n_points_f1
        )
        
        satisfied, max_violation, violations = check_interpolation_constraints(
            G, F1, np.array(A_vals_f1), np.array(b_vals_f1)
        )
        
        self.assertLessEqual(max_violation, 1e-4, 
            f"f1 interpolation violated! Max violation: {max_violation}")
    
    def test_fista_satisfies_f2_interpolation(self):
        """Test that FISTA trajectory satisfies f2 interpolation at x points."""
        x_iters, y_iters, g_iters, h_iters, f1_y_iters, f2_x_iters, betas = run_fista(
            self.x0, self.A, self.b, self.lambd, self.gamma, self.K
        )
        
        G, F1, F2, F = build_fista_gram_representation(
            x_iters, y_iters, g_iters, h_iters, f1_y_iters, f2_x_iters, betas,
            self.x_s, self.f1_s, self.f2_s, self.A, self.b, self.gamma
        )
        
        _, _, _, repX_f2, repG_f2, repF_f2 = build_fista_symbolic_reps(
            self.K, self.gamma, betas
        )
        
        n_points_f2 = self.K + 1  # x_0 to x_K
        A_vals_f2, b_vals_f2 = convex_interp(
            repX_f2, repG_f2, repF_f2, n_points_f2
        )
        
        satisfied, max_violation, violations = check_interpolation_constraints(
            G, F2, np.array(A_vals_f2), np.array(b_vals_f2)
        )
        
        self.assertLessEqual(max_violation, 1e-4,
            f"f2 interpolation violated! Max violation: {max_violation}")
    
    def test_fista_multiple_random_problems(self):
        """Test FISTA interpolation across multiple random problem instances."""
        num_problems = 5
        K = 2
        
        for seed in range(num_problems):
            np.random.seed(seed + 200)
            
            m, n = 15, 8
            A = np.random.randn(m, n) / np.sqrt(m)
            b = np.random.randn(m)
            lambd = 0.05
            
            ATA = A.T @ A
            L = np.max(np.linalg.eigvalsh(ATA))
            mu = np.min(np.linalg.eigvalsh(ATA)) if m >= n else 0.0
            gamma = 1.0 / L
            
            x_s, _ = solve_lasso(A, b, lambd)
            f1_s = 0.5 * np.linalg.norm(A @ x_s - b) ** 2
            f2_s = lambd * np.linalg.norm(x_s, 1)
            
            x0 = np.random.randn(n) * 0.1
            
            x_iters, y_iters, g_iters, h_iters, f1_y_iters, f2_x_iters, betas = run_fista(
                x0, A, b, lambd, gamma, K
            )
            
            G, F1, F2, F = build_fista_gram_representation(
                x_iters, y_iters, g_iters, h_iters, f1_y_iters, f2_x_iters, betas,
                x_s, f1_s, f2_s, A, b, gamma
            )
            
            repY_f1, repG_f1, repF_f1, repX_f2, repG_f2, repF_f2 = build_fista_symbolic_reps(
                K, gamma, betas
            )
            
            # Check f1 interpolation
            A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
                repY_f1, repG_f1, repF_f1, mu, L, K
            )
            _, max_viol_f1, _ = check_interpolation_constraints(
                G, F1, np.array(A_vals_f1), np.array(b_vals_f1)
            )
            
            # Check f2 interpolation
            A_vals_f2, b_vals_f2 = convex_interp(
                repX_f2, repG_f2, repF_f2, K + 1
            )
            _, max_viol_f2, _ = check_interpolation_constraints(
                G, F2, np.array(A_vals_f2), np.array(b_vals_f2)
            )
            
            self.assertLessEqual(max_viol_f1, 1e-4,
                f"Problem {seed}: f1 interpolation violated with max violation {max_viol_f1}")
            self.assertLessEqual(max_viol_f2, 1e-4,
                f"Problem {seed}: f2 interpolation violated with max violation {max_viol_f2}")


# ============================================================================
# Shifted Problem Tests (x_opt = 0, f_opt = 0)
# ============================================================================

def run_shifted_ista(x0, A, b, lambd, gamma, K, x_opt, f_opt):
    """
    Run ISTA on a shifted problem where x_opt = 0 and f1_opt = 0.
    
    The shifted functions are:
        f1_shifted(x) = 0.5 * ||A @ (x + x_opt) - b||^2 - f1_opt
        f2_shifted(x) = lambd * ||x + x_opt||_1
        
    With optimal at x = 0.
    
    Returns:
        x_iterates, g_iterates, h_iterates, f1_iterates, f2_iterates, problem_info
    """
    # f_opt is the full composite objective f1(x_opt) + f2(x_opt)
    # We shift f1 by the full f_opt so that f1_shifted(0) + f2_shifted(0) = 0
    
    def f1_shifted(x):
        # f1(x + x_opt) - f_opt
        return 0.5 * np.linalg.norm(A @ (x + x_opt) - b) ** 2 - f_opt
    
    def f2_shifted(x):
        # f2(x + x_opt)
        return lambd * np.linalg.norm(x + x_opt, 1)
    
    def grad_f1_shifted(x):
        # grad_f1(x + x_opt)
        return A.T @ (A @ (x + x_opt) - b)
    
    def subgrad_f2_shifted(x):
        # subgrad_f2(x + x_opt) = lambd * sign(x + x_opt)
        return lambd * np.sign(x + x_opt)
    
    # Shifted initial point: x0_shifted = x0 - x_opt
    x0_shifted = x0 - x_opt
    
    x_iterates = [x0_shifted]
    g_iterates = [grad_f1_shifted(x0_shifted)]
    h_iterates = [subgrad_f2_shifted(x0_shifted)]  # = lambd * sign(x0)
    f1_iterates = [f1_shifted(x0_shifted)]
    f2_iterates = [f2_shifted(x0_shifted)]
    
    x_curr = x0_shifted
    
    # Run ISTA
    for k in range(K):
        y_k = x_curr - gamma * grad_f1_shifted(x_curr)
        # Proximal step on f2_shifted
        # prox_{gamma * f2_shifted}(y) = arg min_z { 0.5*||z-y||^2 + gamma*lambd*||z+x_opt||_1 }
        # = soft_threshold(y + x_opt, gamma*lambd) - x_opt
        x_new_plus_xopt = soft_threshold(y_k + x_opt, gamma * lambd)
        x_new = x_new_plus_xopt - x_opt
        
        # Subgradient: h_{k+1} = (y_k - x_new) / gamma
        h_kp1 = (y_k - x_new) / gamma
        
        # Verify: h_kp1 should equal subgrad_f2_shifted(x_new) = lambd * sign(x_new + x_opt)
        expected_h = lambd * np.sign(x_new + x_opt)
        nonzero_mask = np.abs(x_new + x_opt) > 1e-8
        if np.any(nonzero_mask):
            assert np.allclose(h_kp1[nonzero_mask], expected_h[nonzero_mask], atol=1e-8), \
                f"Subgradient mismatch at k={k}: computed h differs from expected"
        
        x_iterates.append(x_new)
        g_iterates.append(grad_f1_shifted(x_new))
        h_iterates.append(h_kp1)
        f1_iterates.append(f1_shifted(x_new))
        f2_iterates.append(f2_shifted(x_new))
        
        x_curr = x_new
    
    # Return problem info needed for Gram representation
    problem_info = {'A': A, 'x_opt': x_opt, 'lambd': lambd, 'b': b}
    
    return x_iterates, g_iterates, h_iterates, f1_iterates, f2_iterates, problem_info


def build_shifted_gram_representation(x_iterates, g_iterates, h_iterates, f1_iterates, f2_iterates, problem_info):
    """
    Build Gram representation for shifted problem where x_s = 0, f1_s = 0.
    """
    K = len(x_iterates) - 1
    A = problem_info['A']
    x_opt = problem_info['x_opt']
    lambd = problem_info['lambd']
    b = problem_info['b']
    
    # At the shifted optimal x_s = 0:
    # g_s = grad_f1_shifted(0) = A^T(A @ x_opt - b)
    # h_s must satisfy g_s + h_s = 0 (stationarity)
    g_s = A.T @ (A @ x_opt - b)
    h_s = -g_s
    
    # Build G_half for Gram matrix
    G_half_columns = []
    
    x_s = np.zeros(x_iterates[0].shape)
    
    G_half_columns.append(x_iterates[0] - x_s)  # x_0 - x_s = x_0
    G_half_columns.append(g_iterates[0])  # g_0
    G_half_columns.append(h_iterates[0])  # h_0
    
    for k in range(1, K + 1):
        G_half_columns.append(h_iterates[k])
        G_half_columns.append(g_iterates[k])
    
    G_half_columns.append(g_s)
    G_half_columns.append(h_s)
    
    G_half = np.column_stack(G_half_columns)
    G = G_half.T @ G_half
    
    # At x_s = 0 for the shifted problem:
    # f1_shifted(0) = f1(x_opt) - f_opt = f1(x_opt) - (f1(x_opt) + f2(x_opt)) = -f2(x_opt)
    # f2_shifted(0) = f2(x_opt) = lambd * ||x_opt||_1
    # Total: f1_shifted(0) + f2_shifted(0) = 0  (optimal composite objective)
    f2_x_opt = lambd * np.linalg.norm(x_opt, 1)
    f1_s = -f2_x_opt  # f1_shifted at optimal
    f2_s = f2_x_opt   # f2_shifted at optimal
    
    F1 = np.array([f1 - f1_s for f1 in f1_iterates] + [0.0])
    F2 = np.array([f2 - f2_s for f2 in f2_iterates] + [0.0])
    
    # Concatenate F1 and F2 for consistency with DRO pipeline
    F = np.concatenate([F1, F2])
    
    return G, F1, F2, F


class TestShiftedISTAInterpolation(unittest.TestCase):
    """Test ISTA interpolation with shifted problem (x_opt = 0, f_opt = 0)."""
    
    def test_shifted_ista_interpolation(self):
        """Test that shifted ISTA still satisfies interpolation conditions."""
        np.random.seed(42)
        
        m, n = 20, 10
        A = np.random.randn(m, n) / np.sqrt(m)
        b = np.random.randn(m)
        lambd = 0.1
        
        ATA = A.T @ A
        L = np.max(np.linalg.eigvalsh(ATA))
        mu = np.min(np.linalg.eigvalsh(ATA)) if m >= n else 0.0
        gamma = 1.0 / L
        
        # Solve original problem
        x_opt, f_opt = solve_lasso(A, b, lambd)
        
        K = 3
        x0 = np.random.randn(n) * 0.5
        
        # Run shifted ISTA
        x_iters, g_iters, h_iters, f1_iters, f2_iters, problem_info = run_shifted_ista(
            x0, A, b, lambd, gamma, K, x_opt, f_opt
        )
        
        G, F1, F2, F = build_shifted_gram_representation(
            x_iters, g_iters, h_iters, f1_iters, f2_iters, problem_info
        )
        
        # Build symbolic representations
        repX_f1, repG_f1, repF_f1, repX_f2, repG_f2, repF_f2 = build_symbolic_reps(K, gamma)
        
        # Check f1 interpolation
        A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
            repX_f1, repG_f1, repF_f1, mu, L, K + 1
        )
        _, max_viol_f1, _ = check_interpolation_constraints(
            G, F1, np.array(A_vals_f1), np.array(b_vals_f1)
        )
        
        # Check f2 interpolation
        A_vals_f2, b_vals_f2 = convex_interp(
            repX_f2, repG_f2, repF_f2, K + 1
        )
        _, max_viol_f2, _ = check_interpolation_constraints(
            G, F2, np.array(A_vals_f2), np.array(b_vals_f2)
        )
        
        self.assertLessEqual(max_viol_f1, 1e-4,
            f"Shifted f1 interpolation violated! Max violation: {max_viol_f1}")
        self.assertLessEqual(max_viol_f2, 1e-4,
            f"Shifted f2 interpolation violated! Max violation: {max_viol_f2}")
    
    def test_multiple_shifted_problems(self):
        """Test shifted ISTA interpolation across multiple random problems."""
        num_problems = 5
        K = 2
        
        for seed in range(num_problems):
            np.random.seed(seed + 200)
            
            m, n = 15, 8
            A = np.random.randn(m, n) / np.sqrt(m)
            b = np.random.randn(m)
            lambd = 0.05
            
            ATA = A.T @ A
            L = np.max(np.linalg.eigvalsh(ATA))
            mu = np.min(np.linalg.eigvalsh(ATA)) if m >= n else 0.0
            gamma = 1.0 / L
            
            x_opt, f_opt = solve_lasso(A, b, lambd)
            x0 = np.random.randn(n) * 0.1
            
            x_iters, g_iters, h_iters, f1_iters, f2_iters, problem_info = run_shifted_ista(
                x0, A, b, lambd, gamma, K, x_opt, f_opt
            )
            
            G, F1, F2, F = build_shifted_gram_representation(
                x_iters, g_iters, h_iters, f1_iters, f2_iters, problem_info
            )
            
            repX_f1, repG_f1, repF_f1, repX_f2, repG_f2, repF_f2 = build_symbolic_reps(K, gamma)
            
            A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
                repX_f1, repG_f1, repF_f1, mu, L, K + 1
            )
            _, max_viol_f1, _ = check_interpolation_constraints(
                G, F1, np.array(A_vals_f1), np.array(b_vals_f1)
            )
            
            A_vals_f2, b_vals_f2 = convex_interp(
                repX_f2, repG_f2, repF_f2, K + 1
            )
            _, max_viol_f2, _ = check_interpolation_constraints(
                G, F2, np.array(A_vals_f2), np.array(b_vals_f2)
            )
            
            self.assertLessEqual(max_viol_f1, 1e-4,
                f"Problem {seed}: shifted f1 interpolation violated with max violation {max_viol_f1}")
            self.assertLessEqual(max_viol_f2, 1e-4,
                f"Problem {seed}: shifted f2 interpolation violated with max violation {max_viol_f2}")


class TestShiftedFISTAInterpolation(unittest.TestCase):
    """Test FISTA interpolation with shifted problem (x_opt = 0, f_opt = 0)."""
    
    def test_shifted_fista_interpolation(self):
        """Test that shifted FISTA still satisfies interpolation conditions."""
        np.random.seed(42)
        
        m, n = 20, 10
        A = np.random.randn(m, n) / np.sqrt(m)
        b = np.random.randn(m)
        lambd = 0.1
        
        ATA = A.T @ A
        L = np.max(np.linalg.eigvalsh(ATA))
        mu = np.min(np.linalg.eigvalsh(ATA)) if m >= n else 0.0
        gamma = 1.0 / L
        
        x_opt, f_opt = solve_lasso(A, b, lambd)
        
        K = 3
        x0 = np.random.randn(n) * 0.5
        x0_shifted = x0 - x_opt  # Shift so optimal is at 0
        
        # Define shifted functions for FISTA
        # f1 is shifted by full f_opt so composite optimal is 0
        def f1_shifted(x):
            return 0.5 * np.linalg.norm(A @ (x + x_opt) - b) ** 2 - f_opt
        
        def f2_shifted(x):
            return lambd * np.linalg.norm(x + x_opt, 1)
        
        def grad_f1_shifted(x):
            return A.T @ (A @ (x + x_opt) - b)
        
        def subgrad_f2_shifted(x):
            return lambd * np.sign(x + x_opt)
        
        # Run FISTA with shifted functions
        x_iterates = [x0_shifted]
        y_iterates = [x0_shifted]
        g_iterates = [grad_f1_shifted(x0_shifted)]
        h_iterates = [subgrad_f2_shifted(x0_shifted)]  # = lambd * sign(x0)
        f1_y_iters = [f1_shifted(x0_shifted)]
        f2_x_iters = [f2_shifted(x0_shifted)]
        betas = [1.0]
        
        x_curr = x0_shifted
        y_curr = x0_shifted
        beta_curr = 1.0
        
        for k in range(K):
            g_yk = grad_f1_shifted(y_curr)
            ytilde = y_curr - gamma * g_yk
            
            # Proximal step
            x_new_plus_xopt = soft_threshold(ytilde + x_opt, gamma * lambd)
            x_new = x_new_plus_xopt - x_opt
            h_new = (ytilde - x_new) / gamma
            
            # Verify subgradient
            expected_h = lambd * np.sign(x_new + x_opt)
            nonzero_mask = np.abs(x_new + x_opt) > 1e-8
            if np.any(nonzero_mask):
                assert np.allclose(h_new[nonzero_mask], expected_h[nonzero_mask], atol=1e-8), \
                    f"FISTA subgradient mismatch at k={k}"
            
            beta_new = 0.5 * (1 + np.sqrt(1 + 4 * beta_curr ** 2))
            y_new = x_new + (beta_curr - 1) / beta_new * (x_new - x_curr)
            
            x_iterates.append(x_new)
            h_iterates.append(h_new)
            f2_x_iters.append(f2_shifted(x_new))
            betas.append(beta_new)
            
            if k < K - 1:
                y_iterates.append(y_new)
                g_iterates.append(grad_f1_shifted(y_new))
                f1_y_iters.append(f1_shifted(y_new))
            
            x_curr = x_new
            y_curr = y_new
            beta_curr = beta_new
        
        # Build Gram representation for shifted FISTA with x_s = 0
        # For shifted problem at x_s = 0:
        # g_s = grad_f1_shifted(0) = A^T @ (A @ x_opt - b)  (NOT A^T @ (A @ 0 - b))
        # h_s = -g_s by stationarity
        # f1_shifted(0) = f1(x_opt) - f_opt = -f2(x_opt)
        # f2_shifted(0) = f2(x_opt)
        
        K = len(x_iterates) - 1
        
        g_s = A.T @ (A @ x_opt - b)  # Correct: gradient at x_opt, not at 0
        
        # Build G_half for Gram matrix
        G_half_columns = []
        G_half_columns.append(x_iterates[0])  # x_0 - x_s = x_0 (since x_s = 0)
        G_half_columns.append(g_iterates[0])  # g(y_0)
        G_half_columns.append(h_iterates[0])  # h_0
        
        for k in range(1, K):
            G_half_columns.append(h_iterates[k])  # h_k
            G_half_columns.append(g_iterates[k])  # g(y_k)
        
        G_half_columns.append(h_iterates[K])  # h_K
        G_half_columns.append(g_s)  # g_s
        
        G_half = np.column_stack(G_half_columns)
        G = G_half.T @ G_half
        
        # Function values at shifted optimal
        f2_x_opt = lambd * np.linalg.norm(x_opt, 1)
        f1_s = -f2_x_opt  # f1_shifted(0) = -f2(x_opt)
        f2_s = f2_x_opt   # f2_shifted(0) = f2(x_opt)
        
        F1 = np.array([f1 - f1_s for f1 in f1_y_iters] + [0.0])
        F2 = np.array([f2 - f2_s for f2 in f2_x_iters] + [0.0])
        
        # Build symbolic representations
        repY_f1, repG_f1, repF_f1, repX_f2, repG_f2, repF_f2 = build_fista_symbolic_reps(
            K, gamma, betas
        )
        
        # Check interpolation
        A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
            repY_f1, repG_f1, repF_f1, mu, L, K
        )
        _, max_viol_f1, _ = check_interpolation_constraints(
            G, F1, np.array(A_vals_f1), np.array(b_vals_f1)
        )
        
        A_vals_f2, b_vals_f2 = convex_interp(
            repX_f2, repG_f2, repF_f2, K + 1
        )
        _, max_viol_f2, _ = check_interpolation_constraints(
            G, F2, np.array(A_vals_f2), np.array(b_vals_f2)
        )
        
        self.assertLessEqual(max_viol_f1, 1e-4,
            f"Shifted FISTA f1 interpolation violated! Max violation: {max_viol_f1}")
        self.assertLessEqual(max_viol_f2, 1e-4,
            f"Shifted FISTA f2 interpolation violated! Max violation: {max_viol_f2}")


class TestJAXTrajectoryFunctions(unittest.TestCase):
    """Test that JAX trajectory functions match the test implementations exactly."""
    
    def test_jax_ista_matches_test_implementation(self):
        """Verify problem_data_to_ista_trajectories matches run_shifted_ista."""
        import jax.numpy as jnp
        from learning.trajectories_ista_fista import problem_data_to_ista_trajectories
        
        np.random.seed(42)
        m, n = 20, 10
        A_np = np.random.randn(m, n) / np.sqrt(m)
        b_np = np.random.randn(m)
        lambd = 0.1
        K = 3
        
        # Solve Lasso
        x_opt_np, f_opt_np = solve_lasso(A_np, b_np, lambd)
        
        ATA = A_np.T @ A_np
        L = np.max(np.linalg.eigvalsh(ATA))
        gamma_float = 1.0 / L
        x0_np = np.random.randn(n) * 0.5
        
        # Run JAX implementation
        A_jax = jnp.array(A_np)
        b_jax = jnp.array(b_np)
        x_opt_jax = jnp.array(x_opt_np)
        x0_jax = jnp.array(x0_np)
        gamma_jax = jnp.array([gamma_float] * K)
        
        G_jax, F_jax = problem_data_to_ista_trajectories(
            gamma_jax, A_jax, b_jax, x0_jax, x_opt_jax, float(f_opt_np), lambd, K_max=K
        )
        
        # Run test implementation
        x_iterates, g_iterates, h_iterates, f1_iterates, f2_iterates, problem_info = run_shifted_ista(
            x0_np, A_np, b_np, lambd, gamma_float, K, x_opt_np, f_opt_np
        )
        G_test, F1_test, F2_test, F_test = build_shifted_gram_representation(
            x_iterates, g_iterates, h_iterates, f1_iterates, f2_iterates, problem_info
        )
        
        # Compare
        np.testing.assert_allclose(np.array(G_jax), G_test, atol=1e-10,
            err_msg="ISTA JAX Gram matrix doesn't match test implementation")
        np.testing.assert_allclose(np.array(F_jax), F_test, atol=1e-10,
            err_msg="ISTA JAX F vector doesn't match test implementation")
    
    def test_jax_fista_matches_test_implementation(self):
        """Verify problem_data_to_fista_trajectories matches the shifted FISTA test."""
        import jax.numpy as jnp
        from learning.trajectories_ista_fista import problem_data_to_fista_trajectories
        
        np.random.seed(42)
        m, n = 20, 10
        A_np = np.random.randn(m, n) / np.sqrt(m)
        b_np = np.random.randn(m)
        lambd = 0.1
        K = 3
        
        # Solve Lasso
        x_opt_np, f_opt_np = solve_lasso(A_np, b_np, lambd)
        
        ATA = A_np.T @ A_np
        L = np.max(np.linalg.eigvalsh(ATA))
        gamma_float = 1.0 / L
        x0_np = np.random.randn(n) * 0.5
        
        # Compute betas like FISTA test
        betas_t = [1.0]
        for k in range(K):
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * betas_t[-1]**2))
            betas_t.append(t_new)
        effective_betas = [(betas_t[k] - 1) / betas_t[k+1] for k in range(K)]
        
        # Run JAX implementation
        A_jax = jnp.array(A_np)
        b_jax = jnp.array(b_np)
        x_opt_jax = jnp.array(x_opt_np)
        x0_jax = jnp.array(x0_np)
        gamma_jax = jnp.array([gamma_float] * K)
        betas_jax = jnp.array(effective_betas)
        
        G_jax, F_jax = problem_data_to_fista_trajectories(
            (gamma_jax, betas_jax), A_jax, b_jax, x0_jax, x_opt_jax, float(f_opt_np), lambd, K_max=K
        )
        
        # Run inline FISTA test implementation (matching test_shifted_fista_interpolation)
        x0_shifted = x0_np - x_opt_np
        
        def f1_shifted(x):
            return 0.5 * np.linalg.norm(A_np @ (x + x_opt_np) - b_np) ** 2 - f_opt_np
        
        def f2_shifted(x):
            return lambd * np.linalg.norm(x + x_opt_np, 1)
        
        def grad_f1_shifted(x):
            return A_np.T @ (A_np @ (x + x_opt_np) - b_np)
        
        def subgrad_f2_shifted(x):
            return lambd * np.sign(x + x_opt_np)
        
        x_iterates = [x0_shifted]
        y_iterates = [x0_shifted]
        g_iterates = [grad_f1_shifted(x0_shifted)]
        h_iterates = [subgrad_f2_shifted(x0_shifted)]
        f1_y_iters = [f1_shifted(x0_shifted)]
        f2_x_iters = [f2_shifted(x0_shifted)]
        
        x_curr = x0_shifted
        y_curr = x0_shifted
        beta_curr = 1.0
        
        for k in range(K):
            g_yk = grad_f1_shifted(y_curr)
            ytilde = y_curr - gamma_float * g_yk
            
            x_new_plus_xopt = soft_threshold(ytilde + x_opt_np, gamma_float * lambd)
            x_new = x_new_plus_xopt - x_opt_np
            h_new = (ytilde - x_new) / gamma_float
            
            beta_new = 0.5 * (1 + np.sqrt(1 + 4 * beta_curr ** 2))
            y_new = x_new + (beta_curr - 1) / beta_new * (x_new - x_curr)
            
            x_iterates.append(x_new)
            h_iterates.append(h_new)
            f2_x_iters.append(f2_shifted(x_new))
            
            if k < K - 1:
                y_iterates.append(y_new)
                g_iterates.append(grad_f1_shifted(y_new))
                f1_y_iters.append(f1_shifted(y_new))
            
            x_curr = x_new
            y_curr = y_new
            beta_curr = beta_new
        
        # Build Gram representation
        g_s = A_np.T @ (A_np @ x_opt_np - b_np)
        
        G_half_columns = []
        G_half_columns.append(x_iterates[0])
        G_half_columns.append(g_iterates[0])
        G_half_columns.append(h_iterates[0])
        
        for k in range(1, K):
            G_half_columns.append(h_iterates[k])
            G_half_columns.append(g_iterates[k])
        
        G_half_columns.append(h_iterates[K])
        G_half_columns.append(g_s)
        
        G_half = np.column_stack(G_half_columns)
        G_test = G_half.T @ G_half
        
        f2_x_opt = lambd * np.linalg.norm(x_opt_np, 1)
        f1_s = -f2_x_opt
        f2_s = f2_x_opt
        
        F1_test = np.array([f1 - f1_s for f1 in f1_y_iters] + [0.0])
        F2_test = np.array([f2 - f2_s for f2 in f2_x_iters] + [0.0])
        F_test = np.concatenate([F1_test, F2_test])
        
        # Compare
        np.testing.assert_allclose(np.array(G_jax), G_test, atol=1e-10,
            err_msg="FISTA JAX Gram matrix doesn't match test implementation")
        np.testing.assert_allclose(np.array(F_jax), F_test, atol=1e-10,
            err_msg="FISTA JAX F vector doesn't match test implementation")


if __name__ == '__main__':
    unittest.main()

