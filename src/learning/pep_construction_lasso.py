"""
JAX-compatible PEP data construction for ISTA/FISTA (Lasso/proximal problems).

Constructs PEP constraint matrices (A_vals, b_vals, A_obj, b_obj) directly
using JAX, enabling autodifferentiation through step sizes.

ISTA/FISTA optimize composite problems: min f1(x) + f2(x)
- f1: smooth (L-smooth, μ-strongly convex)
- f2: convex (nonsmooth, e.g., L1 regularization)

Key difference from GD: requires TWO sets of interpolation conditions:
1. f1 interpolation: smooth strongly convex conditions (gradients g)
2. f2 interpolation: convex conditions (subgradients h)
"""

import jax
import jax.numpy as jnp
from functools import partial

from .interpolation_conditions import smooth_strongly_convex_interp, convex_interp


@partial(jax.jit, static_argnames=['K_max', 'pep_obj'])
def construct_ista_pep_data(t, mu, L, R, K_max, pep_obj):
    """
    Construct PEP constraint matrices for ISTA using step sizes t.
    
    ISTA dynamics: x_{k+1} = prox_{t_k * f2}(x_k - t_k * g_k)
                           = x_k - t_k * g_k - t_k * h_{k+1}
    where g_k = grad f1(x_k) and h_{k+1} is a subgradient of f2 at x_{k+1}.
    
    Args:
        t: Step sizes - scalar (same for all iterations) or vector of length K_max
        mu: Strong convexity parameter of f1
        L: Lipschitz constant of gradient of f1
        R: Initial radius bound (||x0 - xs|| <= R)
        K_max: Number of ISTA iterations
        pep_obj: Performance metric type:
            'obj_val': f(xK) - f(xs) (requires both F1 and F2)
            'grad_sq_norm': ||gK + hK||^2 (composite gradient)
            'opt_dist_sq_norm': ||xK - xs||^2
    
    Returns:
        pep_data: Tuple (A_obj, b_obj, A_vals, b_vals, c_vals,
                        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
        
    Representation structure (from test_ista_interpolation.py):
        - dimG = 2K + 5: [x_0-x_s, g_0, h_0, h_1, g_1, h_2, g_2, ..., h_K, g_K, g_s, h_s]
        - dimF1 = K + 2: f1 values at [x_0, x_1, ..., x_K, x_s]
        - dimF2 = K + 2: f2 values at [x_0, x_1, ..., x_K, x_s]
        - Points: x_0, x_1, ..., x_K, x_s (optimal point last)
        - Stationarity at x_s: g_s + h_s = 0
    """
    # Broadcast t to vector if scalar
    t_vec = jnp.broadcast_to(t, (K_max,))
    
    # Dimensions for Gram representation
    # dimG = 2K + 5: [x_0-x_s, g_0, h_0, h_1, g_1, ..., h_K, g_K, g_s, h_s]
    dimG = 2 * K_max + 5
    dimF1 = K_max + 2  # K+1 algorithm points + x_s
    dimF2 = K_max + 2  # K+1 algorithm points + x_s
    
    # Identity matrices for symbolic representation
    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)
    eyeF2 = jnp.eye(dimF2)
    
    # Precompute indices for Gram basis (avoids traced conditionals)
    # idx_g[k] = index of gradient g_k: g_0 at 1, g_k at 2k+2 for k >= 1
    # idx_h[k] = index of subgradient h_k: h_0 at 2, h_k at 2k+1 for k >= 1
    idx_g = jnp.array([1] + [2 * k + 2 for k in range(1, K_max + 1)])  # Shape (K_max+1,)
    idx_h = jnp.array([2] + [2 * k + 1 for k in range(1, K_max + 1)])  # Shape (K_max+1,)
    
    idx_gs = 2 * K_max + 3  # Index of g_s
    idx_hs = 2 * K_max + 4  # Index of h_s
    
    # Number of algorithm points: K+1 (x_0, x_1, ..., x_K)
    n_points = K_max + 1
    
    # Initialize representations
    repX_f1 = jnp.zeros((n_points + 1, dimG))  # +1 for x_s
    repG_f1 = jnp.zeros((n_points + 1, dimG))
    repF_f1 = jnp.zeros((n_points + 1, dimF1))
    
    repX_f2 = jnp.zeros((n_points + 1, dimG))
    repG_f2 = jnp.zeros((n_points + 1, dimG))  # Subgradients h
    repF_f2 = jnp.zeros((n_points + 1, dimF2))
    
    # Initial point x_0
    x_rep = eyeG[0, :]  # x_0 - x_s
    
    repX_f1 = repX_f1.at[0].set(x_rep)
    repG_f1 = repG_f1.at[0].set(eyeG[idx_g[0], :])  # g_0
    repF_f1 = repF_f1.at[0].set(eyeF1[0, :])
    
    repX_f2 = repX_f2.at[0].set(x_rep)
    repG_f2 = repG_f2.at[0].set(eyeG[idx_h[0], :])  # h_0
    repF_f2 = repF_f2.at[0].set(eyeF2[0, :])
    
    # ISTA dynamics: x_{k+1} = x_k - t_k * g_k - t_k * h_{k+1}
    def ista_step(k, carry):
        repX_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_prev = carry
        t_k = t_vec[k]
        
        # x_{k+1} = x_k - t_k * g_k - t_k * h_{k+1}
        g_k = eyeG[idx_g[k], :]
        h_kp1 = eyeG[idx_h[k + 1], :]
        
        x_new = x_prev - t_k * g_k - t_k * h_kp1
        
        # Store x_{k+1}
        repX_f1 = repX_f1.at[k + 1].set(x_new)
        repG_f1 = repG_f1.at[k + 1].set(eyeG[idx_g[k + 1], :])
        repF_f1 = repF_f1.at[k + 1].set(eyeF1[k + 1, :])
        
        repX_f2 = repX_f2.at[k + 1].set(x_new)
        repG_f2 = repG_f2.at[k + 1].set(eyeG[idx_h[k + 1], :])
        repF_f2 = repF_f2.at[k + 1].set(eyeF2[k + 1, :])
        
        return (repX_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_new)
    
    init_carry = (repX_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_rep)
    repX_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_final = jax.lax.fori_loop(
        0, K_max, ista_step, init_carry
    )
    
    # Optimal point x_s: all zeros in relative representation
    xs_rep = jnp.zeros(dimG)
    
    repX_f1 = repX_f1.at[n_points].set(xs_rep)
    repG_f1 = repG_f1.at[n_points].set(eyeG[idx_gs, :])  # g_s
    repF_f1 = repF_f1.at[n_points].set(jnp.zeros(dimF1))
    
    repX_f2 = repX_f2.at[n_points].set(xs_rep)
    repG_f2 = repG_f2.at[n_points].set(eyeG[idx_hs, :])  # h_s
    repF_f2 = repF_f2.at[n_points].set(jnp.zeros(dimF2))
    
    # Compute interpolation conditions
    A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
        repX_f1, repG_f1, repF_f1, mu, L, n_points
    )
    
    A_vals_f2, b_vals_f2 = convex_interp(
        repX_f2, repG_f2, repF_f2, n_points
    )
    
    # Combine constraints with F = [F1, F2]
    dimF = dimF1 + dimF2
    
    b_vals_f1_combined = jnp.concatenate([b_vals_f1, jnp.zeros((b_vals_f1.shape[0], dimF2))], axis=1)
    b_vals_f2_combined = jnp.concatenate([jnp.zeros((b_vals_f2.shape[0], dimF1)), b_vals_f2], axis=1)
    
    A_vals = jnp.concatenate([A_vals_f1, A_vals_f2], axis=0)
    b_vals = jnp.concatenate([b_vals_f1_combined, b_vals_f2_combined], axis=0)
    
    num_constraints = A_vals.shape[0]
    c_vals = jnp.zeros(num_constraints)
    
    # Initial condition: ||x0 - xs||^2 <= R^2
    A_init = jnp.outer(repX_f1[0], repX_f1[0])
    b_init = jnp.zeros(dimF)
    c_init = -R ** 2
    
    A_vals = jnp.concatenate([A_vals, A_init[None, :, :]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None, :]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)
    
    # Stationarity constraint: ||g_s + h_s||^2 = 0
    gs_plus_hs = eyeG[idx_gs, :] + eyeG[idx_hs, :]
    A_stationarity = jnp.outer(gs_plus_hs, gs_plus_hs)
    b_stationarity = jnp.zeros(dimF)
    c_stationarity = 0.0
    
    A_vals = jnp.concatenate([A_vals, A_stationarity[None, :, :]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_stationarity[None, :]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_stationarity])], axis=0)
    
    # Objective
    xK = repX_f1[K_max]
    gK = repG_f1[K_max]
    hK = repG_f2[K_max]
    fK_f1 = repF_f1[K_max]
    fK_f2 = repF_f2[K_max]
    
    if pep_obj == 'obj_val':
        A_obj = jnp.zeros((dimG, dimG))
        b_obj = jnp.concatenate([fK_f1, fK_f2])
    elif pep_obj == 'grad_sq_norm':
        gK_plus_hK = gK + hK
        A_obj = jnp.outer(gK_plus_hK, gK_plus_hK)
        b_obj = jnp.zeros(dimF)
    elif pep_obj == 'opt_dist_sq_norm':
        A_obj = jnp.outer(xK, xK)
        b_obj = jnp.zeros(dimF)
    else:
        raise ValueError(f"Unknown pep_obj: {pep_obj}")
    
    PSD_A_vals = []
    PSD_b_vals = []
    PSD_c_vals = []
    PSD_shapes = []
    
    pep_data = (A_obj, b_obj, A_vals, b_vals, c_vals,
                PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
    
    return pep_data


@partial(jax.jit, static_argnames=['K_max', 'pep_obj'])
def construct_fista_pep_data(t, beta, mu, L, R, K_max, pep_obj):
    """
    Construct PEP constraint matrices for FISTA using step sizes t and momentum beta.
    
    FISTA dynamics:
        y_0 = x_0
        For k = 0, ..., K-1:
            x_{k+1} = prox_{t_k * f2}(y_k - t_k * grad_f1(y_k))
                    = y_k - t_k * g(y_k) - t_k * h_{k+1}
            y_{k+1} = x_{k+1} + beta_k * (x_{k+1} - x_k)
    
    Key: gradients g are evaluated at y points, subgradients h at x points.
    
    Representation structure (extending test_ista_interpolation.py for x_K in f1):
        - Gram basis: [x_0-x_s, g(y_0), h_0, h_1, g(y_1), ..., h_K, g(x_K), g_s]
        - dimG = 2K + 4 (position + K y-gradients + 1 x_K gradient + 1 g_s + K+1 subgradients)
        - Actually: [x_0-x_s, g(y_0), h_0, h_1, g(y_1), h_2, g(y_2), ..., h_{K-1}, g(y_{K-1}), h_K, g(x_K), g_s]
          = 1 + K (y-grads) + (K+1) (subgrads) + 1 (g(x_K)) + 1 (g_s) = 2K + 4
          Wait, let me count: position(1) + [g(y_0), h_0](at k=0) + [h_1, g(y_1)](k=1)... 
          Following the pattern from ISTA test: position, then alternating g/h pairs plus extras
          
        Simpler approach matching FGM in pep_construction.py:
        - Points for f1 interpolation: y_0, y_1, ..., y_{K-1}, x_K, x_s
        - Points for f2 interpolation: x_0, x_1, ..., x_K, x_s
        - dimG includes gradients at all these points
        
    Args:
        t: Step sizes - scalar or vector of length K_max
        beta: Momentum parameters - vector of length K_max
        mu: Strong convexity parameter of f1
        L: Lipschitz constant of gradient of f1
        R: Initial radius bound
        K_max: Number of FISTA iterations
        pep_obj: Performance metric type
    
    Returns:
        pep_data tuple
    """
    # Broadcast t to vector if scalar
    t_vec = jnp.broadcast_to(t, (K_max,))
    
    # Dimensions - similar to FGM in pep_construction.py
    # f1 is evaluated at: y_0, y_1, ..., y_{K-1}, x_K, x_s (K+2 points)
    # f2 is evaluated at: x_0, x_1, ..., x_K, x_s (K+2 points)
    # Gram basis: [y_0-y_s, g(y_0), g(y_1), ..., g(y_{K-1}), g(x_K), h_0, h_1, ..., h_K]
    # Actually, for FISTA we have both y and x sequences, so:
    # Position basis: x_0 - x_s (since y_0 = x_0)
    # Gradients: g(y_0), g(y_1), ..., g(y_{K-1}), g(x_K) = K+1 gradients
    # Subgradients: h_0, h_1, ..., h_K = K+1 subgradients
    # Total dimG = 1 + (K+1) + (K+1) = 2K + 3
    # But we also need g_s for stationarity, so dimG = 2K + 4
    # Actually matching the test structure more closely:
    # [x_0-x_s, g(y_0), h_0, h_1, g(y_1), h_2, g(y_2), ..., h_K, g(x_K), g_s]
    
    # Let's use a cleaner structure:
    # Basis columns: [x_0-x_s, g(y_0), g(y_1), ..., g(y_{K-1}), g(x_K), h_0, h_1, ..., h_K, g_s]
    # This gives: 1 + K + 1 + (K+1) + 1 = 2K + 4
    dimG = 2 * K_max + 4
    dimF1 = K_max + 2  # f1 at y_0, ..., y_{K-1}, x_K, x_s
    dimF2 = K_max + 2  # f2 at x_0, ..., x_K, x_s
    
    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)
    eyeF2 = jnp.eye(dimF2)
    
    # Index arrays for Gram basis
    # g(y_k) at index 1 + k for k = 0, ..., K-1
    # g(x_K) at index 1 + K = K + 1
    # h_k at index K + 2 + k for k = 0, ..., K
    # g_s at index K + 2 + K + 1 = 2K + 3
    idx_g_y = jnp.arange(1, K_max + 1)  # indices 1 to K for g(y_0) to g(y_{K-1})
    idx_g_xK = K_max + 1  # index for g(x_K)
    idx_h = jnp.arange(K_max + 2, 2 * K_max + 3)  # indices K+2 to 2K+2 for h_0 to h_K
    idx_gs = 2 * K_max + 3  # g_s (and h_s = -g_s)
    
    # Number of points
    n_f1_points = K_max + 1  # y_0, ..., y_{K-1}, x_K (not counting x_s)
    n_f2_points = K_max + 1  # x_0, ..., x_K (not counting x_s)
    
    # f1 representations: y_0, ..., y_{K-1}, x_K, x_s
    repY_f1 = jnp.zeros((n_f1_points + 1, dimG))
    repG_f1 = jnp.zeros((n_f1_points + 1, dimG))
    repF_f1 = jnp.zeros((n_f1_points + 1, dimF1))
    
    # f2 representations: x_0, ..., x_K, x_s
    repX_f2 = jnp.zeros((n_f2_points + 1, dimG))
    repG_f2 = jnp.zeros((n_f2_points + 1, dimG))
    repF_f2 = jnp.zeros((n_f2_points + 1, dimF2))
    
    # Initial: y_0 = x_0
    x_rep = eyeG[0, :]  # x_0 - x_s
    y_rep = x_rep  # y_0 = x_0
    
    # y_0 for f1
    repY_f1 = repY_f1.at[0].set(y_rep)
    repG_f1 = repG_f1.at[0].set(eyeG[idx_g_y[0], :])  # g(y_0)
    repF_f1 = repF_f1.at[0].set(eyeF1[0, :])
    
    # x_0 for f2
    repX_f2 = repX_f2.at[0].set(x_rep)
    repG_f2 = repG_f2.at[0].set(eyeG[idx_h[0], :])  # h_0
    repF_f2 = repF_f2.at[0].set(eyeF2[0, :])
    
    # FISTA dynamics
    def fista_step(k, carry):
        repY_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_prev, y_prev = carry
        t_k = t_vec[k]
        beta_k = beta[k]
        
        # x_{k+1} = y_k - t_k * g(y_k) - t_k * h_{k+1}
        g_yk = eyeG[idx_g_y[k], :]
        h_kp1 = eyeG[idx_h[k + 1], :]
        
        x_new = y_prev - t_k * g_yk - t_k * h_kp1
        
        # Store x_{k+1} for f2
        repX_f2 = repX_f2.at[k + 1].set(x_new)
        repG_f2 = repG_f2.at[k + 1].set(h_kp1)
        repF_f2 = repF_f2.at[k + 1].set(eyeF2[k + 1, :])
        
        # y_{k+1} = x_{k+1} + beta_k * (x_{k+1} - x_k)
        y_new = x_new + beta_k * (x_new - x_prev)
        
        # Store y_{k+1} for f1 (only for k < K-1, since y_K is not used)
        # For f1 interpolation, we store y_1, ..., y_{K-1} at indices 1, ..., K-1
        repY_f1 = jax.lax.cond(
            k < K_max - 1,
            lambda args: args[0].at[args[1] + 1].set(args[2]),
            lambda args: args[0],
            (repY_f1, k, y_new)
        )
        repG_f1 = jax.lax.cond(
            k < K_max - 1,
            lambda args: args[0].at[args[1] + 1].set(eyeG[idx_g_y[args[1] + 1], :]),
            lambda args: args[0],
            (repG_f1, k, None)
        )
        repF_f1 = jax.lax.cond(
            k < K_max - 1,
            lambda args: args[0].at[args[1] + 1].set(eyeF1[args[1] + 1, :]),
            lambda args: args[0],
            (repF_f1, k, None)
        )
        
        return (repY_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_new, y_new)
    
    init_carry = (repY_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_rep, y_rep)
    repY_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_final, y_final = jax.lax.fori_loop(
        0, K_max, fista_step, init_carry
    )
    
    # x_K is already stored in repX_f2[K_max]
    # Now add x_K to f1 interpolation (for full obj_val)
    # x_K is at position K in repY_f1 (index K_max for f1 points: y_0, ..., y_{K-1}, x_K)
    repY_f1 = repY_f1.at[K_max].set(x_final)  # x_K position
    repG_f1 = repG_f1.at[K_max].set(eyeG[idx_g_xK, :])  # g(x_K)
    repF_f1 = repF_f1.at[K_max].set(eyeF1[K_max, :])  # f1(x_K)
    
    # Optimal point x_s = y_s
    xs_rep = jnp.zeros(dimG)
    repY_f1 = repY_f1.at[n_f1_points].set(xs_rep)
    repG_f1 = repG_f1.at[n_f1_points].set(eyeG[idx_gs, :])  # g_s
    repF_f1 = repF_f1.at[n_f1_points].set(jnp.zeros(dimF1))
    
    repX_f2 = repX_f2.at[n_f2_points].set(xs_rep)
    repG_f2 = repG_f2.at[n_f2_points].set(-eyeG[idx_gs, :])  # h_s = -g_s
    repF_f2 = repF_f2.at[n_f2_points].set(jnp.zeros(dimF2))
    
    # Compute interpolation conditions
    A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
        repY_f1, repG_f1, repF_f1, mu, L, n_f1_points
    )
    
    A_vals_f2, b_vals_f2 = convex_interp(
        repX_f2, repG_f2, repF_f2, n_f2_points
    )
    
    # Combine constraints
    dimF = dimF1 + dimF2
    
    b_vals_f1_combined = jnp.concatenate([b_vals_f1, jnp.zeros((b_vals_f1.shape[0], dimF2))], axis=1)
    b_vals_f2_combined = jnp.concatenate([jnp.zeros((b_vals_f2.shape[0], dimF1)), b_vals_f2], axis=1)
    
    A_vals = jnp.concatenate([A_vals_f1, A_vals_f2], axis=0)
    b_vals = jnp.concatenate([b_vals_f1_combined, b_vals_f2_combined], axis=0)
    
    num_constraints = A_vals.shape[0]
    c_vals = jnp.zeros(num_constraints)
    
    # Initial condition
    A_init = jnp.outer(repX_f2[0], repX_f2[0])
    b_init = jnp.zeros(dimF)
    c_init = -R ** 2
    
    A_vals = jnp.concatenate([A_vals, A_init[None, :, :]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None, :]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)
    
    # Objective at x_K
    xK = repY_f1[K_max]  # x_K position (in f1 representation)
    gK = repG_f1[K_max]  # g(x_K)
    hK = repG_f2[K_max]  # h_K
    fK_f1 = repF_f1[K_max]  # f1(x_K) - f1(x_s)
    fK_f2 = repF_f2[K_max]  # f2(x_K) - f2(x_s)
    
    if pep_obj == 'obj_val':
        # Minimize f(xK) - f(xs) = f1(xK) + f2(xK) - (f1(xs) + f2(xs))
        A_obj = jnp.zeros((dimG, dimG))
        b_obj = jnp.concatenate([fK_f1, fK_f2])
    elif pep_obj == 'grad_sq_norm':
        # Minimize ||g(xK) + h_K||^2
        gK_plus_hK = gK + hK
        A_obj = jnp.outer(gK_plus_hK, gK_plus_hK)
        b_obj = jnp.zeros(dimF)
    elif pep_obj == 'opt_dist_sq_norm':
        # Minimize ||xK - xs||^2
        A_obj = jnp.outer(xK, xK)
        b_obj = jnp.zeros(dimF)
    else:
        raise ValueError(f"Unknown pep_obj: {pep_obj}")
    
    PSD_A_vals = []
    PSD_b_vals = []
    PSD_c_vals = []
    PSD_shapes = []
    
    pep_data = (A_obj, b_obj, A_vals, b_vals, c_vals,
                PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
    
    return pep_data


def ista_pep_data_to_numpy(pep_data):
    """
    Convert JAX arrays in ISTA/FISTA pep_data to numpy arrays.
    
    Args:
        pep_data: Tuple from construct_ista_pep_data or construct_fista_pep_data
    
    Returns:
        pep_data_np: Same tuple structure with numpy arrays
    """
    import numpy as np
    
    (A_obj, b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = pep_data
    
    return (
        np.asarray(A_obj),
        np.asarray(b_obj),
        np.asarray(A_vals),
        np.asarray(b_vals),
        np.asarray(c_vals),
        [np.asarray(a) for a in PSD_A_vals],
        [np.asarray(b) for b in PSD_b_vals],
        [np.asarray(c) for c in PSD_c_vals],
        PSD_shapes
    )
