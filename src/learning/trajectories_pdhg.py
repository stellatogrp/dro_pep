"""
PDHG (Chambolle-Pock) trajectory computation for DRO-PEP optimization on Linear Programs.

This module contains trajectory computation functions for the Primal-Dual Hybrid Gradient
(PDHG) algorithm on LP problems, with shifted functions so x_opt = 0, y_opt = 0.

The LP problem is:
    min_x  c^T x
    s.t.   G x >= h      (m1 inequality constraints)
           A x = b       (m2 equality constraints)
           l <= x <= u   (box constraints)

Lagrangian: L(x, y) = c^T x + y^T (q - K x)  where K = [G; A], q = [h; b], y = [λ; μ]

Saddle point formulation:
    min_x max_y  f1(x) + <-K^T y, x> + q^T y  = f1(x) - y^T K x + q^T y

where:
    f1(x) = c^T x + indicator_{[l,u]}(x)   (convex, primal function)
    h(y) = -q^T y + indicator_{Y}(y)       (convex, dual function where Y = {y: y[:m1] >= 0})

PDHG Algorithm:
    For k = 0, ..., K-1:
        x_{k+1} = proj_{[l,u]}(x_k - τ (c - K^T y_k))
        x_bar_{k+1} = x_{k+1} + θ (x_{k+1} - x_k)
        y_{k+1} = proj_Y(y_k + σ (q - K x_bar_{k+1}))

For the DRO-PEP framework, we shift so the optimal saddle point is at origin:
    x_shifted = x - x_opt
    y_shifted = y - y_opt
"""

import jax
import jax.numpy as jnp
import logging
from functools import partial

log = logging.getLogger(__name__)


def proj_box(v, l, u):
    """Project v onto box [l, u]."""
    return jnp.minimum(u, jnp.maximum(v, l))


def proj_nonneg_first_m1(v, m1):
    """Project first m1 components to non-negative, leave rest unchanged."""
    v_ineq = jax.nn.relu(v[:m1])
    v_eq = v[m1:]
    return jnp.concatenate([v_ineq, v_eq])


@partial(jax.jit, static_argnames=['K_max', 'm1', 'return_Gram_representation'])
def problem_data_to_pdhg_trajectories(stepsizes, c, K, q, l, u, x0, y0, x_opt, y_opt, f_opt,
                                       K_max, m1, return_Gram_representation=True):
    """
    Compute PDHG trajectories on shifted LP problem and return Gram representation.

    The shifted problem has optimal saddle point at (x, y) = (0, 0):
        f1_shifted(x) = c^T (x + x_opt) + indicator_{[l-x_opt, u-x_opt]}(x)
        h_shifted(y) = -q^T (y + y_opt) + indicator constraints

    The LP matrices K, q, c are NOT shifted - only the variables.

    The Gram representation for PDHG with K iterations has:
        Gram basis (dimG = 2K + 6):
            [delta_x0, delta_y0, xs, ys, gf1_0, gh_0, gf1_1, gh_1, ..., gf1_K, gh_K]
        where:
            delta_x0 = x_0 - x_s (in shifted coords, x_s = 0)
            delta_y0 = y_0 - y_s (in shifted coords, y_s = 0)
            xs, ys = 0 (saddle point in shifted coordinates)
            gf1_k = subgradient of f1 at x_k
            gh_k = subgradient of h at y_k

        Function values:
            F1: (K+2,) [f1(x_0) - f1_s, ..., f1(x_K) - f1_s, 0]
            F_h: (K+2,) [h(y_0) - h_s, ..., h(y_K) - h_s, 0]

    The primal and dual vectors are embedded into R^{n+m}:
        - Primal vectors: [v; 0_m]
        - Dual vectors: [0_n; v]

    Args:
        stepsizes: Tuple (tau, sigma, theta) where:
            - tau: Primal step size (scalar or array of K_max)
            - sigma: Dual step size (scalar or array of K_max)
            - theta: Extrapolation parameter (scalar or array of K_max)
        c: (n,) cost vector
        K: (m, n) constraint matrix [G; A] stacked
        q: (m,) RHS vector [h; b] stacked
        l: (n,) lower bounds
        u: (n,) upper bounds
        x0: (n,) initial primal point in ORIGINAL coordinates
        y0: (m,) initial dual point in ORIGINAL coordinates
        x_opt: (n,) optimal primal point
        y_opt: (m,) optimal dual point
        f_opt: Optimal Lagrangian value (c^T x_opt + q^T y_opt - y_opt^T K x_opt)
        K_max: Number of PDHG iterations
        m1: Number of inequality constraints (y[:m1] >= 0)
        return_Gram_representation: If True, return (G, F) Gram representation.
            Else: raw trajectories

    Returns:
        If return_Gram_representation:
            G: (dimG, dimG) Gram matrix where dimG = 2K + 6
            F: (2K+4,) concatenated function values [F1, F_h]
        Else:
            x_iter, y_iter, gf1_iter, gh_iter, f1_iter, fh_iter
    """
    tau_raw, sigma_raw, theta_raw = stepsizes
    n = c.shape[0]
    m = K.shape[0]
    m2 = m - m1  # Number of equality constraints

    # Broadcast step sizes to vectors if scalar
    tau = jnp.broadcast_to(tau_raw, (K_max,))
    sigma = jnp.broadcast_to(sigma_raw, (K_max,))
    theta = jnp.broadcast_to(theta_raw, (K_max,))

    # Shifted box constraints: [l - x_opt, u - x_opt]
    l_shifted = l - x_opt
    u_shifted = u - x_opt

    # Shifted initial points
    x0_shifted = x0 - x_opt
    y0_shifted = y0 - y_opt

    # Shifted functions
    # f1_shifted(x) = c^T (x + x_opt) + indicator_{[l_shifted, u_shifted]}(x)
    #               = c^T x + c^T x_opt + indicator
    # At x = 0 (optimal): f1_shifted(0) = c^T x_opt + indicator(0 in [l_s, u_s])
    #
    # h_shifted(y) = -q^T (y + y_opt) + indicator_Y(y + y_opt)
    #              = -q^T y - q^T y_opt + indicator
    # At y = 0: h_shifted(0) = -q^T y_opt + indicator(y_opt in Y)

    def f1_shifted(x):
        """f1(x + x_opt) in shifted coordinates."""
        return jnp.dot(c, x + x_opt)

    def h_shifted(y):
        """h(y + y_opt) in shifted coordinates."""
        return -jnp.dot(q, y + y_opt)

    # Storage for K+1 iterates
    x_iter = jnp.zeros((n, K_max + 1))
    y_iter = jnp.zeros((m, K_max + 1))
    gf1_iter = jnp.zeros((n, K_max + 1))  # subgradients of f1 at x_k
    gh_iter = jnp.zeros((m, K_max + 1))   # subgradients of h at y_k
    f1_iter = jnp.zeros(K_max + 1)
    fh_iter = jnp.zeros(K_max + 1)

    # Initial point
    x_iter = x_iter.at[:, 0].set(x0_shifted)
    y_iter = y_iter.at[:, 0].set(y0_shifted)
    f1_iter = f1_iter.at[0].set(f1_shifted(x0_shifted))
    fh_iter = fh_iter.at[0].set(h_shifted(y0_shifted))

    # Initial subgradients
    # gf1_0: subgradient of f1 at x_0 (in shifted coords, x_0 + x_opt is the actual point)
    # For f1(x) = c^T x + indicator_{[l,u]}(x), a valid subgradient in interior is c
    # On boundary, we add normal cone. For initial point, we just use c.
    gf1_iter = gf1_iter.at[:, 0].set(c)

    # gh_0: subgradient of h at y_0
    # For h(y) = -q^T y + indicator_Y(y), a valid subgradient is -q (in interior)
    gh_iter = gh_iter.at[:, 0].set(-q)

    x_curr = x0_shifted
    y_curr = y0_shifted

    def pdhg_step(k, val):
        x_iter, y_iter, gf1_iter, gh_iter, f1_iter, fh_iter, x_curr, y_curr = val

        tau_k = tau[k]
        sigma_k = sigma[k]
        theta_k = theta[k]

        # Primal update: x_{k+1} = proj_{[l_s, u_s]}(x_k - tau * (c - K^T y_k))
        # In shifted coords with shifted y: y_actual = y_shifted + y_opt
        # The PDHG update for f1(x) = c^T x + indicator_{[l,u]}(x):
        #   prox_{τf1}(v) = proj_{[l,u]}(v - τc)
        # So: x_{k+1} = proj(x_k + τ K^T y_k - τc)  [v = x_k + τ K^T y_k]
        #
        # Subgradient formula: gf1 = (v - x_new) / τ where v is input to prox (before -τc)
        y_actual = y_curr + y_opt
        v_prox = x_curr + tau_k * (K.T @ y_actual)  # Input to prox (before -τc)
        v_proj = v_prox - tau_k * c  # Input to projection
        x_new = proj_box(v_proj, l_shifted, u_shifted)

        # Subgradient of f1 at x_{k+1} from prox optimality
        # For f1(x) = c^T x + indicator, ∂f1(x) = c + N_{[l,u]}(x)
        # gf1_{k+1} = (v_prox - x_new) / tau_k ∈ ∂f1(x_new)
        gf1_new = (v_prox - x_new) / tau_k

        # Extrapolation
        x_bar = x_new + theta_k * (x_new - x_curr)

        # Dual update: y_{k+1} = proj_Y(y_k + sigma * (q - K x_bar))
        # In shifted coords: x_bar_actual = x_bar + x_opt
        v_y = y_curr + sigma_k * (q - K @ (x_bar + x_opt))
        # Project: y_shifted + y_opt should satisfy constraints
        # So we project v_y + y_opt to Y, then subtract y_opt
        y_new_actual = proj_nonneg_first_m1(v_y + y_opt, m1)
        y_new = y_new_actual - y_opt

        # Subgradient of h at y_{k+1} from projection optimality
        # gh_{k+1} = -q + normal_cone_element
        # The normal cone element is (v_y - y_new) / sigma_k for the constrained components
        normal_cone = (v_y - y_new) / sigma_k
        gh_new = -q + normal_cone

        # Store
        x_iter = x_iter.at[:, k + 1].set(x_new)
        y_iter = y_iter.at[:, k + 1].set(y_new)
        gf1_iter = gf1_iter.at[:, k + 1].set(gf1_new)
        gh_iter = gh_iter.at[:, k + 1].set(gh_new)
        f1_iter = f1_iter.at[k + 1].set(f1_shifted(x_new))
        fh_iter = fh_iter.at[k + 1].set(h_shifted(y_new))

        return (x_iter, y_iter, gf1_iter, gh_iter, f1_iter, fh_iter, x_new, y_new)

    result = jax.lax.fori_loop(
        0, K_max, pdhg_step,
        (x_iter, y_iter, gf1_iter, gh_iter, f1_iter, fh_iter, x_curr, y_curr)
    )
    x_iter, y_iter, gf1_iter, gh_iter, f1_iter, fh_iter, _, _ = result

    if not return_Gram_representation:
        return x_iter, y_iter, gf1_iter, gh_iter, f1_iter, fh_iter

    # Build Gram representation
    # Gram basis (dimG = 2K + 6):
    # [delta_x0, delta_y0, xs, ys, gf1_0, gh_0, gf1_1, gh_1, ..., gf1_K, gh_K]
    #
    # Embedding: primal vectors as [v; 0_m], dual vectors as [0_n; v]

    embedded_dim = n + m
    dimG = 2 * K_max + 6

    G_half = jnp.zeros((embedded_dim, dimG))

    # delta_x0 = x_0 - x_s = x_0 (since x_s = 0 in shifted coords)
    G_half = G_half.at[:n, 0].set(x_iter[:, 0])

    # delta_y0 = y_0 - y_s = y_0
    G_half = G_half.at[n:, 1].set(y_iter[:, 0])

    # xs = 0 (saddle point in shifted coords) - column 2 is zeros
    # ys = 0 - column 3 is zeros

    # gf1_k and gh_k for k = 0, 1, ..., K
    for k in range(K_max + 1):
        # gf1_k at index 4 + 2*k
        G_half = G_half.at[:n, 4 + 2 * k].set(gf1_iter[:, k])
        # gh_k at index 5 + 2*k
        G_half = G_half.at[n:, 5 + 2 * k].set(gh_iter[:, k])

    G = G_half.T @ G_half

    # Function values
    # f1_s = f1_shifted(0) = c^T x_opt
    # h_s = h_shifted(0) = -q^T y_opt
    f1_s = jnp.dot(c, x_opt)
    h_s = -jnp.dot(q, y_opt)

    # F1: [f1(x_0) - f1_s, ..., f1(x_K) - f1_s, 0] (K+2 values)
    F1 = jnp.concatenate([f1_iter - f1_s, jnp.array([0.0])])

    # F_h: [h(y_0) - h_s, ..., h(y_K) - h_s, 0] (K+2 values)
    F_h = jnp.concatenate([fh_iter - h_s, jnp.array([0.0])])

    # Combined F = [F1, F_h]
    F = jnp.concatenate([F1, F_h])

    return G, F


@partial(jax.jit, static_argnames=['K_max', 'm1'])
def problem_data_to_pdhg_trajectories_raw(stepsizes, c, K, q, l, u, x0, y0, x_opt, y_opt, f_opt,
                                           K_max, m1):
    """
    Compute raw PDHG trajectories without Gram representation.

    Convenience wrapper that always returns raw trajectories.
    """
    return problem_data_to_pdhg_trajectories(
        stepsizes, c, K, q, l, u, x0, y0, x_opt, y_opt, f_opt,
        K_max, m1, return_Gram_representation=False
    )


def compute_pdhg_stepsizes_from_K(K_matrix):
    """
    Compute default PDHG step sizes from constraint matrix K.

    For convergence, we need tau * sigma * ||K||^2 < 1.
    We use tau = sigma = 0.9 / ||K||_2.

    Args:
        K_matrix: (m, n) constraint matrix

    Returns:
        tau, sigma, theta: Default step sizes
    """
    K_norm = jnp.linalg.norm(K_matrix, ord=2)
    tau = 0.9 / K_norm
    sigma = 0.9 / K_norm
    theta = 1.0
    return tau, sigma, theta


def build_lp_matrices(G_ineq, h_ineq, A_eq, b_eq):
    """
    Stack inequality and equality constraint matrices for PDHG.

    Args:
        G_ineq: (m1, n) inequality constraint matrix (Gx >= h)
        h_ineq: (m1,) inequality RHS
        A_eq: (m2, n) equality constraint matrix (Ax = b)
        b_eq: (m2,) equality RHS

    Returns:
        K: (m1 + m2, n) stacked constraint matrix
        q: (m1 + m2,) stacked RHS
        m1: Number of inequality constraints
    """
    K = jnp.vstack([G_ineq, A_eq])
    q = jnp.concatenate([h_ineq, b_eq])
    m1 = G_ineq.shape[0]
    return K, q, m1
