"""
JAX-compatible PEP data construction.

Constructs PEP constraint matrices (A_vals, b_vals, A_obj, b_obj) directly
using JAX, enabling autodifferentiation through step sizes.
"""

import jax
import jax.numpy as jnp
from functools import partial

from .interpolation_conditions import smooth_strongly_convex_interp


@partial(jax.jit, static_argnames=['K_max', 'pep_obj'])
def construct_gd_pep_data(t, mu, L, R, K_max, pep_obj):
    """
    Construct PEP constraint matrices for gradient descent using step sizes t.
    
    This function creates the symbolic representation of GD iterates and
    computes the interpolation constraint matrices, enabling autodiff through t.
    
    Args:
        t: Step sizes - scalar (same for all iterations) or vector of length K_max
        mu: Strong convexity parameter
        L: Lipschitz constant of gradient
        R: Initial radius bound (||x0 - xs|| <= R)
        K_max: Number of GD iterations
        pep_obj: Performance metric type:
            'obj_val': f(xK) - f(xs)
            'grad_sq_norm': ||gK||^2
            'opt_dist_sq_norm': ||xK - xs||^2
    
    Returns:
        pep_data: Tuple (A_obj, b_obj, A_vals, b_vals, c_vals,
                        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
        
    Representation structure:
        - dimG = K_max + 2: columns for [x0-xs, g0, g1, ..., gK]
        - dimF = K_max + 1: entries for [f0-fs, f1-fs, ..., fK-fs]
        - Points: x0, x1, ..., xK, xs (optimal point last, index K_max+1)
    """
    # Broadcast t to vector if scalar
    t_vec = jnp.broadcast_to(t, (K_max,))
    
    # Dimensions for Gram representation
    # G matrix: captures inner products of (x0-xs, g0, g1, ..., gK)
    dimG = K_max + 2  # [x0-xs, g0, g1, ..., gK]
    dimF = K_max + 1  # [f0-fs, f1-fs, ..., fK-fs]
    
    # Identity matrices for symbolic representation
    eyeG = jnp.eye(dimG)
    eyeF = jnp.eye(dimF)
    
    # Build symbolic representations of algorithm iterates
    # repX[i]: representation of x_i in the Gram basis
    # repG[i]: representation of g_i in the Gram basis  
    # repF[i]: representation of f_i in the function value basis
    
    # Number of points: K_max + 2 (x0, x1, ..., xK, xs)
    n_points = K_max + 1  # Algorithm points (excluding xs which is added after)
    
    # Initialize arrays for representations
    repX = jnp.zeros((n_points + 1, dimG))  # +1 for xs
    repG = jnp.zeros((n_points + 1, dimG))
    repF = jnp.zeros((n_points + 1, dimF))
    
    # x0 - xs is the first basis vector (index 0)
    # g0 is the second basis vector (index 1)
    # g1, g2, ..., gK are at indices 2, 3, ..., K_max+1
    
    # Initial point x0
    x0 = eyeG[0, :]  # x0 - xs (relative position)
    g0 = eyeG[1, :]  # g0
    f0 = eyeF[0, :]  # f0 - fs
    
    repX = repX.at[0].set(x0)
    repG = repG.at[0].set(g0)
    repF = repF.at[0].set(f0)
    
    # GD iterations: x_{k+1} = x_k - t_k * g_k
    def gd_step(k, carry):
        repX, x_prev = carry
        t_k = t_vec[k]
        g_k = eyeG[k + 1, :]  # g_k is at index k+1 in eyeG
        
        # Update: x_{k+1} = x_k - t_k * g_k
        x_new = x_prev - t_k * g_k
        
        # Store x_{k+1}
        repX = repX.at[k + 1].set(x_new)
        
        return (repX, x_new)
    
    repX, x_final = jax.lax.fori_loop(0, K_max, gd_step, (repX, x0))
    
    # Set gradient representations (each g_k is a basis vector)
    def set_gradients(k, repG):
        g_k = eyeG[k + 1, :]  # g_k at index k+1
        repG = repG.at[k].set(g_k)
        return repG
    
    repG = jax.lax.fori_loop(0, n_points, set_gradients, repG)
    
    # Set function value representations (each f_k is a basis vector)
    def set_fvals(k, repF):
        f_k = eyeF[k, :]  # f_k at index k
        repF = repF.at[k].set(f_k)
        return repF
    
    repF = jax.lax.fori_loop(0, n_points, set_fvals, repF)
    
    # Optimal point xs: all zeros in relative representation
    xs = jnp.zeros(dimG)  # xs - xs = 0
    gs = jnp.zeros(dimG)  # gs = 0
    fs = jnp.zeros(dimF)  # fs - fs = 0
    
    repX = repX.at[n_points].set(xs)
    repG = repG.at[n_points].set(gs)
    repF = repF.at[n_points].set(fs)
    
    # Compute interpolation conditions using the full set
    A_vals, b_vals = smooth_strongly_convex_interp(
        repX, repG, repF, mu, L, n_points
    )
    
    # c_vals: constant terms for each constraint (all zeros for standard interp)
    num_constraints = A_vals.shape[0]
    c_vals = jnp.zeros(num_constraints)
    
    # Initial condition: ||x0 - xs||^2 <= R^2
    # This becomes: <A_init, G> + b_init^T F + c_init <= 0
    # where A_init = outer(x0, x0), b_init = 0, c_init = -R^2
    A_init = jnp.outer(repX[0], repX[0])
    b_init = jnp.zeros(dimF)
    c_init = -R ** 2
    
    # Append initial condition
    A_vals = jnp.concatenate([A_vals, A_init[None, :, :]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None, :]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)
    
    # Objective: performance metric
    # Index of final iterate xK is K_max (index n_points - 1 = K_max)
    xK = repX[K_max]  # Final iterate
    gK = repG[K_max]  # Final gradient
    fK = repF[K_max]  # Final function value (f_K - fs)
    
    if pep_obj == 'obj_val':
        # Minimize f(xK) - f(xs) = fK (since fs = 0)
        A_obj = jnp.zeros((dimG, dimG))
        b_obj = fK  # This is already f_K - f_s
    elif pep_obj == 'grad_sq_norm':
        # Minimize ||gK||^2 = <gK, gK>
        A_obj = jnp.outer(gK, gK)
        b_obj = jnp.zeros(dimF)
    elif pep_obj == 'opt_dist_sq_norm':
        # Minimize ||xK - xs||^2 = <xK, xK> (since xs = 0 in relative coords)
        A_obj = jnp.outer(xK, xK)
        b_obj = jnp.zeros(dimF)
    else:
        raise ValueError(f"Unknown pep_obj: {pep_obj}")
    
    # PSD constraints: empty for standard gradient descent PEP
    PSD_A_vals = []
    PSD_b_vals = []
    PSD_c_vals = []
    PSD_shapes = []
    
    # Return pep_data tuple
    pep_data = (A_obj, b_obj, A_vals, b_vals, c_vals,
                PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
    
    return pep_data


def pep_data_to_numpy(pep_data):
    """
    Convert JAX arrays in pep_data to numpy arrays for use with canonicalizers.
    
    Args:
        pep_data: Tuple from construct_gd_pep_data (JAX arrays)
    
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


@partial(jax.jit, static_argnames=['K_max', 'pep_obj'])
def construct_fgm_pep_data(t, beta, mu, L, R, K_max, pep_obj):
    """
    Construct PEP constraint matrices for Nesterov FGM using step sizes t and momentum beta.
    
    Uses the algorithm ordering:
        x_prev = x
        x = y - t_k * g(y)
        y = x + beta_k * (x - x_prev)
    
    Uses repY representation where interpolation conditions are written for y iterates:
    - repY[k] = y_k (where gradients are evaluated)
    - repG[k] = g(y_k) (gradient at the same point)
    - repF[k] = f(y_k)
    
    x_K is added as an extra point for the objective, computed as:
        x_K = y_{K-1} - t_{K-1} * g(y_{K-1})
    
    Args:
        t: Step sizes - scalar (same for all iterations) or vector of length K_max
        beta: Momentum parameters - vector of length K_max
        mu: Strong convexity parameter
        L: Lipschitz constant of gradient
        R: Initial radius bound (||x0 - xs|| <= R)
        K_max: Number of FGM iterations
        pep_obj: Performance metric type:
            'obj_val': f(xK) - f(xs)
            'grad_sq_norm': ||g(xK)||^2
            'opt_dist_sq_norm': ||xK - xs||^2
    
    Returns:
        pep_data: Tuple (A_obj, b_obj, A_vals, b_vals, c_vals,
                        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
        
    Representation structure:
        - dimG = K_max + 2: [y0-ys, g(y0), g(y1), ..., g(y_{K-1}), g(x_K)]
        - dimF = K_max + 1: [f(y0)-fs, f(y1)-fs, ..., f(y_{K-1})-fs, f(x_K)-fs]
        - Points: y0, y1, ..., y_{K-1}, x_K, ys (n_points = K_max + 1, plus ys)
    """
    # Broadcast t to vector if scalar
    t_vec = jnp.broadcast_to(t, (K_max,))
    
    # Dimensions for Gram representation
    # Gram basis: [y0-ys, g(y0), g(y1), ..., g(y_{K-1}), g(x_K)] = K+2 columns
    # Function values: [f(y0)-fs, ..., f(y_{K-1})-fs, f(x_K)-fs] = K+1 entries
    dimG = K_max + 2
    dimF = K_max + 1
    
    # Identity matrices for symbolic representation
    eyeG = jnp.eye(dimG)
    eyeF = jnp.eye(dimF)
    
    # Number of algorithm points (y0, ..., y_{K-1}, x_K) = K + 1
    # Total points including ys: K + 2
    n_points = K_max + 1  # Algorithm points (excluding ys)
    
    # Initialize arrays for representations
    # repY[0..K-1] = y_0..y_{K-1}, repY[K] = x_K, repY[K+1] = ys
    repY = jnp.zeros((n_points + 1, dimG))  # +1 for ys
    repG = jnp.zeros((n_points + 1, dimG))
    repF = jnp.zeros((n_points + 1, dimF))
    
    # Basis vector indices:
    # Index 0: y0-ys
    # Index 1..K: g(y_0)..g(y_{K-1}) 
    # Index K+1: g(x_K)
    
    # Initial: y0 = x0, so y0 - ys is the first basis vector
    y0 = eyeG[0, :]  # y0 - ys (relative position)
    g_y0 = eyeG[1, :]  # g(y0)
    f_y0 = eyeF[0, :]  # f(y0) - fs
    
    repY = repY.at[0].set(y0)
    repG = repG.at[0].set(g_y0)
    repF = repF.at[0].set(f_y0)
    
    # FGM iterations: x = y - t * g(y), then y_new = x + beta * (x - x_prev)
    def fgm_step(k, carry):
        repY, x_prev, y_prev = carry
        t_k = t_vec[k]
        beta_k = beta[k]
        g_yk = eyeG[k + 1, :]  # g(y_k) is at index k+1 in eyeG
        
        # Update x: x_{k+1} = y_k - t_k * g(y_k)
        x_new = y_prev - t_k * g_yk
        
        # Update y: y_{k+1} = x_{k+1} + beta_k * (x_{k+1} - x_k)
        y_new = x_new + beta_k * (x_new - x_prev)
        
        # Store y_{k+1} (for k=0..K-2, we store y_1..y_{K-1})
        # Note: repY[0] = y_0, so repY[k+1] = y_{k+1}
        repY = repY.at[k + 1].set(y_new)
        
        return (repY, x_new, y_new)
    
    # Initial x0 = y0
    x0 = y0
    repY, x_final, y_final = jax.lax.fori_loop(0, K_max, fgm_step, (repY, x0, y0))
    
    # x_K is x_final from the last iteration
    repY = repY.at[K_max].set(x_final)
    
    # Set gradient representations for y points (y0, ..., y_{K-1})
    # repG[k] = g(y_k) for k = 0, ..., K-1
    def set_y_gradients(k, repG):
        g_yk = eyeG[k + 1, :]  # g(y_k) at index k+1
        repG = repG.at[k].set(g_yk)
        return repG
    
    repG = jax.lax.fori_loop(0, K_max, set_y_gradients, repG)
    
    # g(x_K) is a new basis vector at index K_max + 1
    g_xK = eyeG[K_max + 1, :]
    repG = repG.at[K_max].set(g_xK)
    
    # Set function value representations
    # repF[k] = f(y_k) - fs for k = 0, ..., K-1
    # repF[K] = f(x_K) - fs
    def set_fvals(k, repF):
        f_k = eyeF[k, :]  # f(y_k) at index k for k < K, or f(x_K) at index K
        repF = repF.at[k].set(f_k)
        return repF
    
    repF = jax.lax.fori_loop(0, n_points, set_fvals, repF)
    
    # Optimal point ys: all zeros in relative representation
    ys = jnp.zeros(dimG)  # ys - ys = 0
    gs = jnp.zeros(dimG)  # g(ys) = 0
    fs = jnp.zeros(dimF)  # f(ys) - fs = 0
    
    repY = repY.at[n_points].set(ys)
    repG = repG.at[n_points].set(gs)
    repF = repF.at[n_points].set(fs)
    
    # Compute interpolation conditions using repY, repG, repF
    # These are now valid (y_k, g(y_k), f(y_k)) triplets
    A_vals, b_vals = smooth_strongly_convex_interp(
        repY, repG, repF, mu, L, n_points
    )
    
    # c_vals: constant terms (all zeros for standard interp)
    num_constraints = A_vals.shape[0]
    c_vals = jnp.zeros(num_constraints)
    
    # Initial condition: ||x0 - xs||^2 = ||y0 - ys||^2 <= R^2
    A_init = jnp.outer(repY[0], repY[0])
    b_init = jnp.zeros(dimF)
    c_init = -R ** 2
    
    # Append initial condition
    A_vals = jnp.concatenate([A_vals, A_init[None, :, :]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None, :]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)
    
    # Objective: performance metric at final iterate x_K
    rep_xK = repY[K_max]  # x_K representation
    g_xK_rep = repG[K_max]  # g(x_K) representation
    f_xK = repF[K_max]  # f(x_K) - fs representation
    
    if pep_obj == 'obj_val':
        # Minimize f(xK) - f(xs) = fK
        A_obj = jnp.zeros((dimG, dimG))
        b_obj = f_xK
    elif pep_obj == 'grad_sq_norm':
        # Minimize ||g(xK)||^2 = <g(xK), g(xK)>
        A_obj = jnp.outer(g_xK_rep, g_xK_rep)
        b_obj = jnp.zeros(dimF)
    elif pep_obj == 'opt_dist_sq_norm':
        # Minimize ||xK - xs||^2 = <xK, xK> (since xs = 0 in relative coords)
        A_obj = jnp.outer(rep_xK, rep_xK)
        b_obj = jnp.zeros(dimF)
    else:
        raise ValueError(f"Unknown pep_obj: {pep_obj}")
    
    # PSD constraints: empty for FGM
    PSD_A_vals = []
    PSD_b_vals = []
    PSD_c_vals = []
    PSD_shapes = []
    
    pep_data = (A_obj, b_obj, A_vals, b_vals, c_vals,
                PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
    
    return pep_data

