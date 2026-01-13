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
    
    where x values are the actual iterates. Interpolation conditions apply to x values only.
    
    Args:
        t: Step sizes - scalar (same for all iterations) or vector of length K_max
        beta: Momentum parameters - vector of length K_max
        mu: Strong convexity parameter
        L: Lipschitz constant of gradient
        R: Initial radius bound (||x0 - xs|| <= R)
        K_max: Number of FGM iterations
        pep_obj: Performance metric type:
            'obj_val': f(xK) - f(xs)
            'grad_sq_norm': ||gK||^2
            'opt_dist_sq_norm': ||xK - xs||^2
    
    Returns:
        pep_data: Tuple (A_obj, b_obj, A_vals, b_vals, c_vals,
                        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)
        
    Representation structure (same as GD):
        - dimG = K_max + 2: columns for [x0-xs, g0, g1, ..., gK]
        - dimF = K_max + 1: entries for [f0-fs, f1-fs, ..., fK-fs]
        - Points: x0, x1, ..., xK, xs (optimal point last, index K_max+1)
        
    Note: g_k in the representation is the gradient g(y_{k-1}) used in iteration k.
    """
    # Broadcast t to vector if scalar
    t_vec = jnp.broadcast_to(t, (K_max,))
    
    # Dimensions for Gram representation (same as GD)
    dimG = K_max + 2  # [x0-xs, g0, g1, ..., gK]
    dimF = K_max + 1  # [f0-fs, f1-fs, ..., fK-fs]
    
    # Identity matrices for symbolic representation
    eyeG = jnp.eye(dimG)
    eyeF = jnp.eye(dimF)
    
    # Number of points: K_max + 2 (x0, x1, ..., xK, xs)
    n_points = K_max + 1  # Algorithm points (excluding xs)
    
    # Initialize arrays for representations
    repX = jnp.zeros((n_points + 1, dimG))  # +1 for xs
    repG = jnp.zeros((n_points + 1, dimG))
    repF = jnp.zeros((n_points + 1, dimF))
    
    # Basis vectors:
    # x0 - xs is at index 0
    # g0, g1, ..., gK are at indices 1, 2, ..., K_max+1
    
    # Initial point x0 = y0
    x0 = eyeG[0, :]  # x0 - xs (relative position)
    g0 = eyeG[1, :]  # g0 = g(y0) = g(x0)
    f0 = eyeF[0, :]  # f0 - fs
    
    repX = repX.at[0].set(x0)
    repG = repG.at[0].set(g0)
    repF = repF.at[0].set(f0)
    
    # FGM iterations with symbolic updates
    # x_prev = x
    # x = y - t_k * g(y)  -> g(y) is the k-th gradient basis vector
    # y = x + beta_k * (x - x_prev)
    def fgm_step(k, carry):
        repX, x_prev, y_prev = carry
        t_k = t_vec[k]
        beta_k = beta[k]
        g_k = eyeG[k + 1, :]  # g_k = g(y_{k-1}) at index k+1 in eyeG
        
        # Update x (main iterate): x = y - t_k * g(y)
        x_new = y_prev - t_k * g_k
        
        # Store x_{k+1}
        repX = repX.at[k + 1].set(x_new)
        
        # Update y (momentum): y = x + beta_k * (x - x_prev)
        y_new = x_new + beta_k * (x_new - x_prev)
        
        return (repX, x_new, y_new)
    
    # Initial y0 = x0
    y0 = x0
    repX, x_final, y_final = jax.lax.fori_loop(0, K_max, fgm_step, (repX, x0, y0))
    
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
    
    # Compute interpolation conditions (same as GD)
    A_vals, b_vals = smooth_strongly_convex_interp(
        repX, repG, repF, mu, L, n_points
    )
    
    # c_vals: constant terms (all zeros for standard interp)
    num_constraints = A_vals.shape[0]
    c_vals = jnp.zeros(num_constraints)
    
    # Initial condition: ||x0 - xs||^2 <= R^2
    A_init = jnp.outer(repX[0], repX[0])
    b_init = jnp.zeros(dimF)
    c_init = -R ** 2
    
    # Append initial condition
    A_vals = jnp.concatenate([A_vals, A_init[None, :, :]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None, :]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)
    
    # Objective: performance metric at final iterate xK
    xK = repX[K_max]
    gK = repG[K_max]
    fK = repF[K_max]
    
    if pep_obj == 'obj_val':
        A_obj = jnp.zeros((dimG, dimG))
        b_obj = fK
    elif pep_obj == 'grad_sq_norm':
        A_obj = jnp.outer(gK, gK)
        b_obj = jnp.zeros(dimF)
    elif pep_obj == 'opt_dist_sq_norm':
        A_obj = jnp.outer(xK, xK)
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
