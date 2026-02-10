"""
Comparison tests between custom PEP construction and PEPit for proximal gradient descent.

These tests verify that our JAX-compatible PEP construction produces the same
SDP objective values as PEPit's proximal_gradient example.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction, ConvexFunction
from PEPit.primitive_steps import proximal_step

from learning.pep_constructions import (
    smooth_strongly_convex_interp,
    convex_interp,
)


def run_pepit_proximal_gradient(L, mu, gamma, n):
    """
    Run PEPit's proximal gradient example and return the optimal value.
    
    Args:
        L: Smoothness parameter for f1
        mu: Strong convexity parameter for f1
        gamma: Step size
        n: Number of iterations
    
    Returns:
        pepit_tau: Worst-case value from PEPit
    """
    # Instantiate PEP
    problem = PEP()
    
    # Declare a strongly convex smooth function and a closed convex proper function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    f2 = problem.declare_function(ConvexFunction)
    func = f1 + f2
    
    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()
    
    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()
    
    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    
    # Run the proximal gradient method starting from x0
    x = x0
    for _ in range(n):
        y = x - gamma * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, gamma)
    
    # Set the performance metric to the distance between x and xs
    problem.set_performance_metric((x - xs) ** 2)
    
    # Solve the PEP
    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL', verbose=0)
    
    return pepit_tau


def construct_proximal_gradient_pep_data(gamma, mu, L, R, K, pep_obj='opt_dist_sq_norm'):
    """
    Construct PEP constraint matrices for proximal gradient descent.
    
    This constructs the symbolic representation of PGD iterates and
    computes interpolation constraint matrices for both f1 and f2.
    
    Args:
        gamma: Step size
        mu: Strong convexity parameter for f1
        L: Lipschitz constant for gradient of f1
        R: Initial radius bound
        K: Number of PGD iterations
        pep_obj: Performance metric ('opt_dist_sq_norm', 'obj_val', 'grad_sq_norm')
    
    Returns:
        PEP data dict
    """
    # Dimensions for Gram representation
    # dimG = 2K + 3: [x0-xs, g0, h0, h1, g1, h2, g2, ..., hK, gK]
    #   - 1 column for x0 - xs
    #   - K+1 columns for f1 gradients: g0, g1, ..., gK
    #   - K+1 columns for f2 subgradients: h0, h1, h2, ..., hK
    #   Total = 1 + (K+1) + (K+1) = 2K + 3
    dimG = 2 * K + 3
    
    # Function value dimensions
    dimF1 = K + 2  # [f1(x0), f1(x1), ..., f1(xK), f1(xs)] - fs
    dimF2 = K + 2  # [f2(x0), f2(x1), ..., f2(xK), f2(xs)] - f2s
    
    # Identity matrices for symbolic representation
    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)
    eyeF2 = jnp.eye(dimF2)
    
    # Gram basis column indices:
    # 0: x0 - xs
    # 1: g0 (grad f1 at x0)
    # 2: h0 (subgrad f2 at x0) - free variable, not used in dynamics
    # 3: h1 (subgrad f2 at x1)
    # 4: g1 (grad f1 at x1)
    # 5: h2 (subgrad f2 at x2)
    # 6: g2 (grad f1 at x2)
    # ...
    # For k >= 1: h_k at index 2k+1, g_k at index 2k+2
    
    def idx_g(k):
        """Index of g_k (f1 gradient at x_k) in the Gram basis."""
        if k == 0:
            return 1
        return 2 * k + 2
    
    def idx_h(k):
        """Index of h_k (f2 subgradient at x_k) in the Gram basis."""
        if k == 0:
            return 2  # h0 is at index 2
        return 2 * k + 1
    
    # Build representations for f1 (smooth strongly convex)
    # Points: x_0, x_1, ..., x_K, x_s (K+2 points)
    n_points_f1 = K + 1  # Algorithm points (x_s excluded in count for interp function)
    
    repX_f1 = jnp.zeros((K + 2, dimG))  # x_0, ..., x_K, x_s
    repG_f1 = jnp.zeros((K + 2, dimG))  # g_0, ..., g_K, g_s
    repF_f1 = jnp.zeros((K + 2, dimF1)) # f1_0, ..., f1_K, f1_s (relative to f1_s)
    
    # Initial point x_0: basis vector 0 is x_0 - x_s
    x0 = eyeG[0, :]
    g0 = eyeG[idx_g(0), :]
    f1_0 = eyeF1[0, :]  # f1(x0) - f1(xs)
    
    repX_f1 = repX_f1.at[0].set(x0)
    repG_f1 = repG_f1.at[0].set(g0)
    repF_f1 = repF_f1.at[0].set(f1_0)
    
    # Build representations for f2 (convex)
    # Points: x_0, x_1, x_2, ..., x_K, x_s (K+2 points)
    n_points_f2 = K + 1  # Algorithm points for f2
    
    repX_f2 = jnp.zeros((K + 2, dimG))  # x_0, x_1, ..., x_K, x_s
    repG_f2 = jnp.zeros((K + 2, dimG))  # h_0, h_1, ..., h_K, h_s
    repF_f2 = jnp.zeros((K + 2, dimF2)) # f2_0, f2_1, ..., f2_K, f2_s
    
    # Set x_0 for f2
    h0 = eyeG[idx_h(0), :]  # h_0 is a free subgradient
    f2_0 = eyeF2[0, :]  # f2(x0) - f2(xs)
    
    repX_f2 = repX_f2.at[0].set(x0)
    repG_f2 = repG_f2.at[0].set(h0)
    repF_f2 = repF_f2.at[0].set(f2_0)
    
    # Proximal gradient iterations
    # y_k = x_k - gamma * g_k
    # x_{k+1} = y_k - gamma * h_{k+1} = x_k - gamma * (g_k + h_{k+1})
    
    x_curr = x0
    for k in range(K):
        g_k = eyeG[idx_g(k), :]
        h_kp1 = eyeG[idx_h(k + 1), :]
        
        # y_k = x_k - gamma * g_k
        y_k = x_curr - gamma * g_k
        
        # x_{k+1} = y_k - gamma * h_{k+1}
        x_kp1 = y_k - gamma * h_kp1
        
        # Store x_{k+1} representation for f1
        repX_f1 = repX_f1.at[k + 1].set(x_kp1)
        repG_f1 = repG_f1.at[k + 1].set(eyeG[idx_g(k + 1), :])
        repF_f1 = repF_f1.at[k + 1].set(eyeF1[k + 1, :])
        
        # Store for f2 (x_{k+1} is at index k+1 in repX_f2)
        repX_f2 = repX_f2.at[k + 1].set(x_kp1)
        repG_f2 = repG_f2.at[k + 1].set(h_kp1)
        repF_f2 = repF_f2.at[k + 1].set(eyeF2[k + 1, :])
        
        x_curr = x_kp1
    
    # Optimal point x_s: all zeros in relative representation
    xs = jnp.zeros(dimG)  # x_s - x_s = 0
    gs = jnp.zeros(dimG)  # g(x_s) = 0 for both f1 and f2
    f1_s = jnp.zeros(dimF1)  # f1(x_s) - f1(x_s) = 0
    f2_s = jnp.zeros(dimF2)  # f2(x_s) - f2(x_s) = 0
    
    repX_f1 = repX_f1.at[K + 1].set(xs)
    repG_f1 = repG_f1.at[K + 1].set(gs)
    repF_f1 = repF_f1.at[K + 1].set(f1_s)
    
    repX_f2 = repX_f2.at[K + 1].set(xs)
    repG_f2 = repG_f2.at[K + 1].set(gs)
    repF_f2 = repF_f2.at[K + 1].set(f2_s)
    
    # Compute interpolation conditions for f1 (smooth strongly convex)
    A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
        repX_f1, repG_f1, repF_f1, mu, L, n_points_f1
    )
    
    # Compute interpolation conditions for f2 (convex)
    A_vals_f2, b_vals_f2 = convex_interp(
        repX_f2, repG_f2, repF_f2, n_points_f2
    )
    
    # Initial condition: ||x0 - xs||^2 <= R^2
    A_init = jnp.outer(repX_f1[0], repX_f1[0])
    b_init_f1 = jnp.zeros(dimF1)
    b_init_f2 = jnp.zeros(dimF2)
    c_init = -R ** 2
    
    # Objective representation
    xK = repX_f1[K]  # Final iterate
    gK_f1 = repG_f1[K]
    
    if pep_obj == 'opt_dist_sq_norm':
        A_obj = jnp.outer(xK, xK)
        b_obj_f1 = jnp.zeros(dimF1)
        b_obj_f2 = jnp.zeros(dimF2)
    elif pep_obj == 'obj_val':
        A_obj = jnp.zeros((dimG, dimG))
        b_obj_f1 = repF_f1[K]  # f1(xK) - f1(xs)
        b_obj_f2 = repF_f2[K]  # f2(xK) - f2(xs)
    elif pep_obj == 'grad_sq_norm':
        # ||g_f1(xK) + h_f2(xK)||^2
        gK_total = gK_f1 + repG_f2[K]  # g_K + h_K
        A_obj = jnp.outer(gK_total, gK_total)
        b_obj_f1 = jnp.zeros(dimF1)
        b_obj_f2 = jnp.zeros(dimF2)
    else:
        raise ValueError(f"Unknown pep_obj: {pep_obj}")
    
    return {
        'A_vals_f1': A_vals_f1,
        'b_vals_f1': b_vals_f1,
        'A_vals_f2': A_vals_f2,
        'b_vals_f2': b_vals_f2,
        'A_init': A_init,
        'b_init_f1': b_init_f1,
        'b_init_f2': b_init_f2,
        'c_init': c_init,
        'A_obj': A_obj,
        'b_obj_f1': b_obj_f1,
        'b_obj_f2': b_obj_f2,
        'dimG': dimG,
        'dimF1': dimF1,
        'dimF2': dimF2,
    }


def solve_proximal_gradient_pep(gamma, mu, L, R, K, pep_obj='opt_dist_sq_norm'):
    """
    Solve the PEP for proximal gradient descent using custom construction.
    
    Returns:
        optimal_value: The optimal objective value
    """
    import cvxpy as cp
    
    pep_data = construct_proximal_gradient_pep_data(gamma, mu, L, R, K, pep_obj)
    
    dimG = pep_data['dimG']
    dimF1 = pep_data['dimF1']
    dimF2 = pep_data['dimF2']
    
    # Variables
    G = cp.Variable((dimG, dimG), symmetric=True)
    F1 = cp.Variable(dimF1)
    F2 = cp.Variable(dimF2)
    
    constraints = [G >> 0]
    
    # f1 interpolation constraints
    A_vals_f1 = np.array(pep_data['A_vals_f1'])
    b_vals_f1 = np.array(pep_data['b_vals_f1'])
    for i in range(A_vals_f1.shape[0]):
        constraints.append(cp.trace(A_vals_f1[i] @ G) + b_vals_f1[i] @ F1 <= 0)
    
    # f2 interpolation constraints
    A_vals_f2 = np.array(pep_data['A_vals_f2'])
    b_vals_f2 = np.array(pep_data['b_vals_f2'])
    for i in range(A_vals_f2.shape[0]):
        constraints.append(cp.trace(A_vals_f2[i] @ G) + b_vals_f2[i] @ F2 <= 0)
    
    # Initial condition
    A_init = np.array(pep_data['A_init'])
    c_init = float(pep_data['c_init'])
    constraints.append(cp.trace(A_init @ G) + c_init <= 0)
    
    # Objective
    A_obj = np.array(pep_data['A_obj'])
    b_obj_f1 = np.array(pep_data['b_obj_f1'])
    b_obj_f2 = np.array(pep_data['b_obj_f2'])
    
    objective = cp.trace(A_obj @ G) + b_obj_f1 @ F1 + b_obj_f2 @ F2
    
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver='CLARABEL')
    
    return prob.value


class TestPEPitComparison(unittest.TestCase):
    """Tests comparing custom PEP construction with PEPit proximal gradient."""
    
    def test_basic_proximal_gradient_n1(self):
        """Test proximal gradient with n=1 iteration."""
        L = 1.0
        mu = 0.1
        gamma = 1.0 / L
        n = 1
        R = 1.0
        
        # Run PEPit
        pepit_tau = run_pepit_proximal_gradient(L, mu, gamma, n)
        
        # Run our implementation
        custom_tau = solve_proximal_gradient_pep(gamma, mu, L, R, n, 'opt_dist_sq_norm')
        
        # They should match within tolerance
        np.testing.assert_allclose(custom_tau, pepit_tau, rtol=1e-4,
                                   err_msg=f"Mismatch: custom={custom_tau}, pepit={pepit_tau}")
    
    def test_proximal_gradient_n2(self):
        """Test proximal gradient with n=2 iterations."""
        L = 1.0
        mu = 0.1
        gamma = 1.0
        n = 2
        R = 1.0
        
        # Run PEPit
        pepit_tau = run_pepit_proximal_gradient(L, mu, gamma, n)
        
        # Run our implementation
        custom_tau = solve_proximal_gradient_pep(gamma, mu, L, R, n, 'opt_dist_sq_norm')
        
        # They should match within tolerance
        np.testing.assert_allclose(custom_tau, pepit_tau, rtol=1e-4,
                                   err_msg=f"Mismatch: custom={custom_tau}, pepit={pepit_tau}")
    
    def test_proximal_gradient_n3(self):
        """Test proximal gradient with n=3 iterations."""
        L = 1.0
        mu = 0.0  # Just smooth convex
        gamma = 1.0 / L
        n = 3
        R = 1.0
        
        # Run PEPit
        pepit_tau = run_pepit_proximal_gradient(L, mu, gamma, n)
        
        # Run our implementation
        custom_tau = solve_proximal_gradient_pep(gamma, mu, L, R, n, 'opt_dist_sq_norm')
        
        # They should match within tolerance
        np.testing.assert_allclose(custom_tau, pepit_tau, rtol=1e-4,
                                   err_msg=f"Mismatch: custom={custom_tau}, pepit={pepit_tau}")
    
    def test_theoretical_bound(self):
        """Test that the result matches the theoretical bound."""
        L = 1.0
        mu = 0.1
        gamma = 1.0
        n = 2
        
        # Theoretical bound: max{(1-mu*gamma)^2, (1-L*gamma)^2}^n
        theoretical_tau = max((1 - mu * gamma) ** 2, (1 - L * gamma) ** 2) ** n
        
        # Run PEPit
        pepit_tau = run_pepit_proximal_gradient(L, mu, gamma, n)
        
        # PEPit should match theoretical
        np.testing.assert_allclose(pepit_tau, theoretical_tau, rtol=1e-4,
                                   err_msg=f"PEPit={pepit_tau}, theoretical={theoretical_tau}")


# ============================================================================
# Accelerated Proximal Gradient (FISTA) Tests
# ============================================================================

def run_pepit_accelerated_proximal_gradient(L, mu, n):
    """
    Run PEPit's accelerated proximal gradient (FISTA) example.
    
    Algorithm (from PEPit):
        λ_1 = 1, y_1 = x_0
        For t = 1, ..., n:
            λ_{t+1} = (1 + sqrt(4*λ_t^2 + 1)) / 2
            x_t = prox_{h/L}(y_t - (1/L) * ∇f(y_t))
            y_{t+1} = x_t + (λ_t - 1) / λ_{t+1} * (x_t - x_{t-1})
    
    Performance metric: F(x_n) - F(x_s)
    
    Returns:
        pepit_tau: Worst-case value from PEPit
    """
    from math import sqrt
    
    problem = PEP()
    
    f = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    h = problem.declare_function(ConvexFunction)
    F = f + h
    
    xs = F.stationary_point()
    Fs = F(xs)
    
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    
    # FISTA iterations
    x_new = x0
    y = x0
    lam = 1.0
    
    for i in range(n):
        lam_old = lam
        lam = (1 + sqrt(4 * lam_old ** 2 + 1)) / 2
        x_old = x_new
        x_new, _, hx_new = proximal_step(y - 1/L * f.gradient(y), h, 1/L)
        y = x_new + (lam_old - 1) / lam * (x_new - x_old)
    
    # Objective: function value
    problem.set_performance_metric((f(x_new) + hx_new) - Fs)
    
    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL', verbose=0)
    
    return pepit_tau


def construct_accelerated_proximal_gradient_pep_data(L, mu, K, R=1.0, pep_obj='obj_val'):
    """
    Construct PEP constraint matrices for accelerated proximal gradient (FISTA).
    
    Algorithm structure:
        - f1 gradients evaluated at y points: y_0, y_1, ..., y_{K-1}
        - f2 subgradients at x points: x_0, x_1, ..., x_K
        - y depends linearly on x: y_{k+1} = x_{k+1} + (λ_k - 1)/λ_{k+1} * (x_{k+1} - x_k)
    
    Gram basis (dimG = 2K + 3):
        [x_0 - x_s, g(y_0), h_0, h_1, g(y_1), h_2, g(y_2), ..., h_K, g_s]
        - 1 column for x_0 - x_s
        - K columns for g(y_0), ..., g(y_{K-1})
        - K+1 columns for h_0, ..., h_K
        - 1 column for g_s (at optimal, h_s = -g_s by stationarity)
        Total = 1 + K + (K+1) + 1 = 2K + 3
    
    Function values:
        F1: f1 at y_0, ..., y_{K-1}, y_s (size K + 1)
        F2: f2 at x_0, ..., x_K, x_s (size K + 2)
    """
    from math import sqrt
    
    gamma = 1.0 / L  # Step size
    
    # Compute the lambda sequence
    lams = [1.0]
    for k in range(K):
        lam_new = (1 + sqrt(4 * lams[-1] ** 2 + 1)) / 2
        lams.append(lam_new)
    
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
    
    idx_gs = 2 * K + 2  # Index of g_s (combined gradient at optimal)
    # h_s = -g_s at stationarity, so h_s representation is -eyeG[idx_gs, :]
    
    # f1 representations: y_0, ..., y_{K-1}, y_s
    n_y_points = K
    repY_f1 = jnp.zeros((K + 1, dimG))
    repG_f1 = jnp.zeros((K + 1, dimG))
    repF_f1 = jnp.zeros((K + 1, dimF1))
    
    # f2 representations: x_0, ..., x_K, x_s
    n_x_points = K + 1
    repX_f2 = jnp.zeros((K + 2, dimG))
    repG_f2 = jnp.zeros((K + 2, dimG))
    repF_f2 = jnp.zeros((K + 2, dimF2))
    
    # Initial: y_0 = x_0
    x0 = eyeG[0, :]
    y_rep = x0
    
    repY_f1 = repY_f1.at[0].set(y_rep)
    repG_f1 = repG_f1.at[0].set(eyeG[idx_g_y(0), :])
    repF_f1 = repF_f1.at[0].set(eyeF1[0, :])
    
    repX_f2 = repX_f2.at[0].set(x0)
    repG_f2 = repG_f2.at[0].set(eyeG[idx_h(0), :])
    repF_f2 = repF_f2.at[0].set(eyeF2[0, :])
    
    x_curr = x0
    
    # FISTA iterations
    for k in range(K):
        # x_{k+1} = y_k - gamma * g(y_k) - gamma * h_{k+1}
        x_new = y_rep - gamma * eyeG[idx_g_y(k), :] - gamma * eyeG[idx_h(k+1), :]
        
        # Store x_{k+1} for f2
        repX_f2 = repX_f2.at[k + 1].set(x_new)
        repG_f2 = repG_f2.at[k + 1].set(eyeG[idx_h(k + 1), :])
        repF_f2 = repF_f2.at[k + 1].set(eyeF2[k + 1, :])
        
        if k < K - 1:
            # y_{k+1} = x_{k+1} + (λ_k - 1) / λ_{k+1} * (x_{k+1} - x_k)
            mom_coef = (lams[k] - 1) / lams[k + 1]
            y_new = x_new + mom_coef * (x_new - x_curr)
            
            repY_f1 = repY_f1.at[k + 1].set(y_new)
            repG_f1 = repG_f1.at[k + 1].set(eyeG[idx_g_y(k + 1), :])
            repF_f1 = repF_f1.at[k + 1].set(eyeF1[k + 1, :])
            
            y_rep = y_new
        
        x_curr = x_new
    
    # Optimal point: y_s = x_s = 0 in relative representation
    # For f1: g_s at y_s = g_s
    # For f2: h_s at x_s = -g_s (stationarity)
    ys_rep = jnp.zeros(dimG)
    repY_f1 = repY_f1.at[K].set(ys_rep)
    repG_f1 = repG_f1.at[K].set(eyeG[idx_gs, :])  # g_s
    repF_f1 = repF_f1.at[K].set(jnp.zeros(dimF1))
    
    xs_rep = jnp.zeros(dimG)
    repX_f2 = repX_f2.at[K + 1].set(xs_rep)
    repG_f2 = repG_f2.at[K + 1].set(-eyeG[idx_gs, :])  # h_s = -g_s
    repF_f2 = repF_f2.at[K + 1].set(jnp.zeros(dimF2))
    
    # Compute interpolation conditions
    A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
        repY_f1, repG_f1, repF_f1, mu, L, n_y_points
    )
    
    A_vals_f2, b_vals_f2 = convex_interp(
        repX_f2, repG_f2, repF_f2, n_x_points
    )
    
    # Initial condition: ||x0 - xs||^2 <= R^2
    A_init = jnp.outer(x0, x0)
    c_init = -R ** 2
    
    # Objective
    xK = repX_f2[K]  # Final x iterate
    
    if pep_obj == 'opt_dist_sq_norm':
        A_obj = jnp.outer(xK, xK)
        b_obj_f1 = jnp.zeros(dimF1)
        b_obj_f2 = jnp.zeros(dimF2)
    elif pep_obj == 'obj_val':
        raise NotImplementedError("obj_val requires f1(x_K) which is not in y-rep")
    else:
        raise ValueError(f"Unknown pep_obj: {pep_obj}")
    
    return {
        'A_vals_f1': A_vals_f1,
        'b_vals_f1': b_vals_f1,
        'A_vals_f2': A_vals_f2,
        'b_vals_f2': b_vals_f2,
        'A_init': A_init,
        'c_init': c_init,
        'A_obj': A_obj,
        'b_obj_f1': b_obj_f1,
        'b_obj_f2': b_obj_f2,
        'dimG': dimG,
        'dimF1': dimF1,
        'dimF2': dimF2,
    }


def solve_accelerated_proximal_gradient_pep(L, mu, K, R=1.0, pep_obj='opt_dist_sq_norm'):
    """Solve the PEP for accelerated proximal gradient using custom construction."""
    import cvxpy as cp
    
    pep_data = construct_accelerated_proximal_gradient_pep_data(L, mu, K, R, pep_obj)
    
    dimG = pep_data['dimG']
    dimF1 = pep_data['dimF1']
    dimF2 = pep_data['dimF2']
    
    G = cp.Variable((dimG, dimG), symmetric=True)
    F1 = cp.Variable(dimF1)
    F2 = cp.Variable(dimF2)
    
    constraints = [G >> 0]
    
    # f1 interpolation constraints
    A_vals_f1 = np.array(pep_data['A_vals_f1'])
    b_vals_f1 = np.array(pep_data['b_vals_f1'])
    for i in range(A_vals_f1.shape[0]):
        constraints.append(cp.trace(A_vals_f1[i] @ G) + b_vals_f1[i] @ F1 <= 0)
    
    # f2 interpolation constraints
    A_vals_f2 = np.array(pep_data['A_vals_f2'])
    b_vals_f2 = np.array(pep_data['b_vals_f2'])
    for i in range(A_vals_f2.shape[0]):
        constraints.append(cp.trace(A_vals_f2[i] @ G) + b_vals_f2[i] @ F2 <= 0)
    
    # Initial condition
    A_init = np.array(pep_data['A_init'])
    c_init = float(pep_data['c_init'])
    constraints.append(cp.trace(A_init @ G) + c_init <= 0)
    
    # Objective
    A_obj = np.array(pep_data['A_obj'])
    b_obj_f1 = np.array(pep_data['b_obj_f1'])
    b_obj_f2 = np.array(pep_data['b_obj_f2'])
    
    objective = cp.trace(A_obj @ G) + b_obj_f1 @ F1 + b_obj_f2 @ F2
    
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver='CLARABEL')
    
    return prob.value


def run_pepit_accelerated_proximal_gradient_distance(L, mu, n):
    """
    Run PEPit's accelerated proximal gradient with distance metric.
    
    Performance metric: ||x_n - x_s||^2
    """
    from math import sqrt
    
    problem = PEP()
    
    f = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    h = problem.declare_function(ConvexFunction)
    F = f + h
    
    xs = F.stationary_point()
    
    x0 = problem.set_initial_point()
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)
    
    x_new = x0
    y = x0
    lam = 1.0
    
    for i in range(n):
        lam_old = lam
        lam = (1 + sqrt(4 * lam_old ** 2 + 1)) / 2
        x_old = x_new
        x_new, _, _ = proximal_step(y - 1/L * f.gradient(y), h, 1/L)
        y = x_new + (lam_old - 1) / lam * (x_new - x_old)
    
    # Distance metric
    problem.set_performance_metric((x_new - xs) ** 2)
    
    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL', verbose=0)
    
    return pepit_tau


class TestAcceleratedProximalGradientComparison(unittest.TestCase):
    """Tests comparing custom PEP construction with PEPit accelerated proximal gradient."""
    
    def test_accelerated_proximal_gradient_n1(self):
        """Test accelerated proximal gradient with n=1 iteration."""
        L = 1.0
        mu = 0.0
        n = 1
        R = 1.0
        
        pepit_tau = run_pepit_accelerated_proximal_gradient_distance(L, mu, n)
        custom_tau = solve_accelerated_proximal_gradient_pep(L, mu, n, R, 'opt_dist_sq_norm')
        
        np.testing.assert_allclose(custom_tau, pepit_tau, rtol=1e-4,
                                   err_msg=f"Mismatch: custom={custom_tau}, pepit={pepit_tau}")
    
    def test_accelerated_proximal_gradient_n2(self):
        """Test accelerated proximal gradient with n=2 iterations."""
        L = 1.0
        mu = 0.0
        n = 2
        R = 1.0
        
        pepit_tau = run_pepit_accelerated_proximal_gradient_distance(L, mu, n)
        custom_tau = solve_accelerated_proximal_gradient_pep(L, mu, n, R, 'opt_dist_sq_norm')
        
        np.testing.assert_allclose(custom_tau, pepit_tau, rtol=1e-4,
                                   err_msg=f"Mismatch: custom={custom_tau}, pepit={pepit_tau}")
    
    def test_accelerated_proximal_gradient_n3(self):
        """Test accelerated proximal gradient with n=3 iterations."""
        L = 1.0
        mu = 0.0
        n = 3
        R = 1.0
        
        pepit_tau = run_pepit_accelerated_proximal_gradient_distance(L, mu, n)
        custom_tau = solve_accelerated_proximal_gradient_pep(L, mu, n, R, 'opt_dist_sq_norm')
        
        np.testing.assert_allclose(custom_tau, pepit_tau, rtol=1e-4,
                                   err_msg=f"Mismatch: custom={custom_tau}, pepit={pepit_tau}")
    
    def test_accelerated_proximal_gradient_strongly_convex(self):
        """Test accelerated proximal gradient with strongly convex f."""
        L = 1.0
        mu = 0.1
        n = 2
        R = 1.0
        
        pepit_tau = run_pepit_accelerated_proximal_gradient_distance(L, mu, n)
        custom_tau = solve_accelerated_proximal_gradient_pep(L, mu, n, R, 'opt_dist_sq_norm')
        
        np.testing.assert_allclose(custom_tau, pepit_tau, rtol=1e-4,
                                   err_msg=f"Mismatch: custom={custom_tau}, pepit={pepit_tau}")


if __name__ == '__main__':
    unittest.main()

