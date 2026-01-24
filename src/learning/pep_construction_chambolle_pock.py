"""
JAX-compatible PEP data construction for Chambolle-Pock (PDHG) algorithm.

Constructs PEP constraint matrices (A_vals, b_vals, A_obj, b_obj) directly
using JAX, enabling autodifferentiation through step sizes.

Chambolle-Pock solves: min_x max_y L(x,y) = f1(x) + M<x,y> - h(y)
where both f1 and h are convex functions.

Algorithm:
    For k = 0, ..., K-1:
        x_{k+1} = prox_{τ·f1}(x_k - τ·M·y_k)
        x_bar_{k+1} = x_{k+1} + θ·(x_{k+1} - x_k)
        y_{k+1} = prox_{σ·h}(y_k + σ·M·x_bar_{k+1})

Optimality conditions at saddle point (xs, ys):
    -M·ys ∈ ∂f1(xs)  →  subgradient for f1 at xs is -M·ys
     M·xs ∈ ∂h(ys)   →  subgradient for h at ys is M·xs
"""

import jax
import jax.numpy as jnp
from functools import partial

from .interpolation_conditions import convex_interp


@partial(jax.jit, static_argnames=['K_max'])
def construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K_max):
    """
    Construct PEP constraint matrices for Chambolle-Pock with gap objective.
    
    Args:
        tau: Primal step size (scalar or vector of length K_max)
        sigma: Dual step size (scalar or vector of length K_max)
        theta: Extrapolation parameter (scalar or vector of length K_max)
        M: Coupling matrix norm (scalar, assumes M is identity scaled by M)
        R: Initial radius bound (||x0 - xs||^2 + ||y0 - ys||^2 <= R^2)
        K_max: Number of Chambolle-Pock iterations
    
    Returns:
        pep_data: Tuple (A_obj, b_obj, A_vals, b_vals, c_vals,
                        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
        
    Representation structure:
        Gram basis (dimG = 2K + 6):
            [delta_x0, delta_y0, xs, ys, gf1_0, gh_0, gf1_1, gh_1, ..., gf1_K, gh_K]
        where:
            delta_x0 = x_0 - xs, delta_y0 = y_0 - ys
            xs, ys = saddle point coordinates
            gf1_k = subgradient of f1 at x_k
            gh_k = subgradient of h at y_k
            
        Function values:
            dimF1 = K + 2: f1 values at [x_0, x_1, ..., x_K, xs]  
            dimF_h = K + 2: h values at [y_0, y_1, ..., y_K, ys]
        
    Gap objective:
        gap = [f1(x_K) - f1(xs)] + [h(y_K) - h(ys)] + M<x_K - xs, ys> - M<xs, y_K - ys>
    """
    # Broadcast step sizes to vectors if scalar
    tau_vec = jnp.broadcast_to(tau, (K_max,))
    sigma_vec = jnp.broadcast_to(sigma, (K_max,))
    theta_vec = jnp.broadcast_to(theta, (K_max,))
    
    # Dimensions
    # [delta_x0, delta_y0, xs, ys, gf1_0, gh_0, gf1_1, gh_1, ..., gf1_K, gh_K]
    dimG = 2 * K_max + 6
    dimF1 = K_max + 2
    dimF_h = K_max + 2
    
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
    
    n_points = K_max + 1  # x_0, ..., x_K and y_0, ..., y_K
    
    # Initialize representations for f1 (at x points) and h (at y points)
    repX_f1 = jnp.zeros((n_points + 1, dimG))
    repG_f1 = jnp.zeros((n_points + 1, dimG))
    repF_f1 = jnp.zeros((n_points + 1, dimF1))
    
    repY_h = jnp.zeros((n_points + 1, dimG))
    repG_h = jnp.zeros((n_points + 1, dimG))
    repF_h = jnp.zeros((n_points + 1, dimF_h))
    
    # Initial points (absolute): x_0 = xs + delta_x0, y_0 = ys + delta_y0
    x_rep = eyeG[idx_xs, :] + eyeG[idx_delta_x0, :]
    y_rep = eyeG[idx_ys, :] + eyeG[idx_delta_y0, :]
    
    # Store x_0 - xs for f1 interpolation
    repX_f1 = repX_f1.at[0].set(eyeG[idx_delta_x0, :])
    repG_f1 = repG_f1.at[0].set(eyeG[idx_gf1(0), :])
    repF_f1 = repF_f1.at[0].set(eyeF1[0, :])
    
    # Store y_0 - ys for h interpolation
    repY_h = repY_h.at[0].set(eyeG[idx_delta_y0, :])
    repG_h = repG_h.at[0].set(eyeG[idx_gh(0), :])
    repF_h = repF_h.at[0].set(eyeF_h[0, :])
    
    # Chambolle-Pock dynamics
    def cp_step(k, carry):
        repX_f1, repG_f1, repF_f1, repY_h, repG_h, repF_h, x_abs, y_abs, x_prev_abs = carry
        
        tau_k = tau_vec[k]
        sigma_k = sigma_vec[k]
        theta_k = theta_vec[k]
        
        gf1_kp1 = eyeG[idx_gf1(k + 1), :]
        gh_kp1 = eyeG[idx_gh(k + 1), :]
        
        # x_{k+1} = x_k - tau_k * M * y_k - tau_k * gf1_{k+1}
        x_new_abs = x_abs - tau_k * M * y_abs - tau_k * gf1_kp1
        
        # Store x_{k+1} - xs for interpolation
        delta_x_new = x_new_abs - eyeG[idx_xs, :]
        repX_f1 = repX_f1.at[k + 1].set(delta_x_new)
        repG_f1 = repG_f1.at[k + 1].set(gf1_kp1)
        repF_f1 = repF_f1.at[k + 1].set(eyeF1[k + 1, :])
        
        # x_bar_{k+1} = x_{k+1} + theta_k * (x_{k+1} - x_k)
        x_bar_abs = x_new_abs + theta_k * (x_new_abs - x_abs)
        
        # y_{k+1} = y_k + sigma_k * M * x_bar - sigma_k * gh_{k+1}
        y_new_abs = y_abs + sigma_k * M * x_bar_abs - sigma_k * gh_kp1
        
        # Store y_{k+1} - ys for interpolation
        delta_y_new = y_new_abs - eyeG[idx_ys, :]
        repY_h = repY_h.at[k + 1].set(delta_y_new)
        repG_h = repG_h.at[k + 1].set(gh_kp1)
        repF_h = repF_h.at[k + 1].set(eyeF_h[k + 1, :])
        
        return (repX_f1, repG_f1, repF_f1, repY_h, repG_h, repF_h, x_new_abs, y_new_abs, x_abs)
    
    init_carry = (repX_f1, repG_f1, repF_f1, repY_h, repG_h, repF_h, x_rep, y_rep, x_rep)
    result = jax.lax.fori_loop(0, K_max, cp_step, init_carry)
    repX_f1, repG_f1, repF_f1, repY_h, repG_h, repF_h, _, _, _ = result
    
    # Saddle point (xs, ys): xs - xs = 0, ys - ys = 0
    # Subgradients from optimality: gf1_s = -M*ys, gh_s = M*xs
    repX_f1 = repX_f1.at[n_points].set(jnp.zeros(dimG))
    repG_f1 = repG_f1.at[n_points].set(-M * eyeG[idx_ys, :])
    repF_f1 = repF_f1.at[n_points].set(jnp.zeros(dimF1))
    
    repY_h = repY_h.at[n_points].set(jnp.zeros(dimG))
    repG_h = repG_h.at[n_points].set(M * eyeG[idx_xs, :])
    repF_h = repF_h.at[n_points].set(jnp.zeros(dimF_h))
    
    # Compute interpolation conditions
    A_vals_f1, b_vals_f1 = convex_interp(repX_f1, repG_f1, repF_f1, n_points)
    A_vals_h, b_vals_h = convex_interp(repY_h, repG_h, repF_h, n_points)
    
    # Combine constraints with F = [F1, F_h]
    dimF = dimF1 + dimF_h
    
    b_vals_f1_combined = jnp.concatenate([b_vals_f1, jnp.zeros((b_vals_f1.shape[0], dimF_h))], axis=1)
    b_vals_h_combined = jnp.concatenate([jnp.zeros((b_vals_h.shape[0], dimF1)), b_vals_h], axis=1)
    
    A_vals = jnp.concatenate([A_vals_f1, A_vals_h], axis=0)
    b_vals = jnp.concatenate([b_vals_f1_combined, b_vals_h_combined], axis=0)
    c_vals = jnp.zeros(A_vals.shape[0])
    
    # Initial condition: ||x_0 - xs||^2 + ||y_0 - ys||^2 <= R^2
    delta_x0 = eyeG[idx_delta_x0, :]
    delta_y0 = eyeG[idx_delta_y0, :]
    A_init = jnp.outer(delta_x0, delta_x0) + jnp.outer(delta_y0, delta_y0)
    b_init = jnp.zeros(dimF)
    c_init = -R ** 2
    
    A_vals = jnp.concatenate([A_vals, A_init[None, :, :]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None, :]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)
    
    # Gap objective: gap = (f1(xK) - f1(xs)) + (h(yK) - h(ys)) + M<xK-xs, ys> - M<xs, yK-ys>
    delta_xK = repX_f1[K_max, :]
    delta_yK = repY_h[K_max, :]
    ys_vec = eyeG[idx_ys, :]
    xs_vec = eyeG[idx_xs, :]
    
    A_gap_1 = 0.5 * M * (jnp.outer(delta_xK, ys_vec) + jnp.outer(ys_vec, delta_xK))
    A_gap_2 = -0.5 * M * (jnp.outer(xs_vec, delta_yK) + jnp.outer(delta_yK, xs_vec))
    A_obj = A_gap_1 + A_gap_2
    
    fK_f1 = repF_f1[K_max, :]
    fK_h = repF_h[K_max, :]
    b_obj = jnp.concatenate([fK_f1, fK_h])
    
    pep_data = (A_obj, b_obj, A_vals, b_vals, c_vals, [], [], [], [])
    
    return pep_data


def chambolle_pock_pep_data_to_numpy(pep_data):
    """Convert JAX arrays in Chambolle-Pock pep_data to numpy arrays."""
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
