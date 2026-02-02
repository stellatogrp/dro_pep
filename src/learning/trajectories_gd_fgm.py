"""
Autodiff setup functions for DRO-PEP optimization.

This module contains trajectory computation and preconditioning utilities
for gradient-based learning of step sizes.
"""
import jax
import jax.numpy as jnp
import numpy as np
import logging
from functools import partial

log = logging.getLogger(__name__)


def compute_preconditioner_from_samples(G_batch, F_batch, precond_type='average'):
    """Compute preconditioning factors from sample Gram matrices.
    
    Computes inverse preconditioning factors used to scale the DRO constraints
    based on sample statistics. This improves numerical conditioning.
    
    Args:
        G_batch: Batch of Gram matrices (N, dimG, dimG)
        F_batch: Batch of function value vectors (N, dimF)
        precond_type: Type of preconditioning:
            - 'average': Use average of sample diagonals (default)
            - 'max': Use maximum values
            - 'min': Use minimum values
            - 'identity': No preconditioning
    
    Returns:
        precond_inv: Tuple (precond_inv_G, precond_inv_F) of inverse preconditioning factors
            - precond_inv_G: (dimG,) array for Gram matrix scaling
            - precond_inv_F: (dimF,) array for function value scaling
    """
    if precond_type == 'identity':
        dimG = G_batch.shape[1]
        dimF = F_batch.shape[1]
        return (np.ones(dimG), np.ones(dimF))
    
    # Compute sqrt of diagonals of each G matrix: shape (N, dimG)
    G_diag_sqrt = np.sqrt(np.array([np.diag(G_batch[i]) for i in range(G_batch.shape[0])]))
    
    # Compute F values: shape (N, dimF)
    F_vals = np.array(F_batch)
    
    if precond_type == 'average':
        avg_G = np.mean(G_diag_sqrt, axis=0)  # (dimG,)
        avg_F = np.mean(F_vals, axis=0)       # (dimF,)
        precond_G = 1 / avg_G
        precond_F = 1 / np.sqrt(np.maximum(avg_F, 1e-10))  # Avoid sqrt of negative
    elif precond_type == 'max':
        max_G = np.max(G_diag_sqrt, axis=0)
        max_F = np.max(F_vals, axis=0)
        precond_G = 1 / max_G
        precond_F = 1 / np.sqrt(np.maximum(max_F, 1e-10))
    elif precond_type == 'min':
        min_G = np.min(G_diag_sqrt, axis=0)
        min_F = np.min(F_vals, axis=0)
        precond_G = 1 / min_G
        precond_F = 1 / np.sqrt(np.maximum(min_F, 1e-10))
    else:
        raise ValueError(f'{precond_type} is invalid precond_type')
    
    # Handle NaN/inf values and apply scaling (from original implementation)
    dimG = G_batch.shape[1]
    dimF = F_batch.shape[1]
    precond_G = np.nan_to_num(precond_G, nan=1.0, posinf=1.0, neginf=1.0) * dimG
    precond_F = np.nan_to_num(precond_F, nan=1.0, posinf=1.0, neginf=1.0) * np.sqrt(dimF)
    
    # Return inverse preconditioner
    precond_inv_G = 1 / precond_G
    precond_inv_F = 1 / precond_F
    
    return (precond_inv_G, precond_inv_F)


@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def problem_data_to_gd_trajectories(stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True):
    """
    Compute GD trajectories and return Gram representation for DRO-PEP.
    
    The Gram representation matches the structure in pep_construction.py:
    - G_half columns: [z0-zs, g0, g1, ..., gK]  (dimG = K_max + 2 columns)
    - G = G_half.T @ G_half
    - F = [f0-fs, f1-fs, ..., fK-fs]  (dimF = K_max + 1)
    
    Args:
        stepsizes: Tuple/list of K_max step sizes (t_0, t_1, ..., t_{K-1})
        Q: (d, d) Hessian matrix (Q = (1/2)(L+mu)I + (1/2)(L-mu)U for some U, ||U|| <= 1)
        z0: (d,) initial point
        zs: (d,) optimal point (Q @ zs = 0)
        fs: scalar optimal function value
        K_max: Number of GD steps
        return_Gram_representation: If True, return (G, F) Gram representation.
            Else: (z_stack, g_stack, f_stack) raw trajectories
    """
    d = Q.shape[0]
    
    def f(x):
        return 0.5 * x @ Q @ x + fs
    
    def g(x):
        return Q @ x
    
    # Initialize storage for K+1 points (z0, z1, ..., zK)
    # z_iter stores z_k for k = 0, 1, ..., K_max
    z_iter = jnp.zeros((d, K_max + 1))  # columns are z_k
    g_iter = jnp.zeros((d, K_max + 1))  # columns are g_k = grad f(z_k)
    f_iter = jnp.zeros(K_max + 1)        # f_k values
    
    # Initial point
    z_iter = z_iter.at[:, 0].set(z0)
    g_iter = g_iter.at[:, 0].set(g(z0))
    f_iter = f_iter.at[0].set(f(z0))
    
    t = stepsizes[0]
    
    # Initial values for fori_loop
    z_curr = z0
    def body_fun(k, val):
        z_iter, g_iter, f_iter, z_curr = val
        g_curr = g(z_curr)
        tk = t[k] if t.ndim > 0 else t
        z_next = z_curr - tk * g_curr
        
        z_iter = z_iter.at[:, k + 1].set(z_next)
        g_iter = g_iter.at[:, k + 1].set(g(z_next))
        f_iter = f_iter.at[k + 1].set(f(z_next))
        
        return (z_iter, g_iter, f_iter, z_next)
    
    z_iter, g_iter, f_iter, _ = jax.lax.fori_loop(
        0, K_max, body_fun, (z_iter, g_iter, f_iter, z_curr)
    )
    
    if return_Gram_representation:
        # Build Gram representation matching pep_construction.py structure
        # G_half columns: [z0-zs, g0, g1, ..., gK] - shape (d, K_max + 2)
        z0_minus_zs = (z_iter[:, 0] - zs).reshape(-1, 1)  # (d, 1)
        G_half = jnp.concatenate([z0_minus_zs, g_iter], axis=1)  # (d, K_max + 2)
        
        G = G_half.T @ G_half  # (K_max + 2, K_max + 2)
        F = f_iter - fs        # (K_max + 1,)
        
        return G, F
    else:
        # Return raw trajectories for debugging/analysis
        z_stack = z_iter
        g_stack = g_iter
        f_stack = f_iter - fs
        return z_stack, g_stack, f_stack


@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def problem_data_to_nesterov_fgm_trajectories(stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True):
    """
    Compute Nesterov FGM trajectories and return Gram representation for DRO-PEP.
    
    Uses the algorithm ordering:
        x_prev = x
        x = y - t_k * g(y)  
        y = x + beta_k * (x - x_prev)
    
    The Gram representation matches the structure in pep_construction.py:
    - G_half columns: [y0-ys, g(y_0), g(y_1), ..., g(y_{K-1}), g(x_K)]  (dimG = K_max + 2 columns)
    - G = G_half.T @ G_half
    - F = [f(y_0)-fs, f(y_1)-fs, ..., f(y_{K-1})-fs, f(x_K)-fs]  (dimF = K_max + 1)
    
    Args:
        stepsizes: Tuple (t, beta) where:
            - t: tuple of K_max step sizes (t_0, ..., t_{K-1})
            - beta: tuple of K_max + 1 momentum coefficients (beta_0, ..., beta_K)
        Q: (d, d) Hessian matrix
        z0: (d,) initial point (y_0 = x_0 = z0)
        zs: (d,) optimal point
        fs: scalar optimal function value
        K_max: Number of FGM steps
        return_Gram_representation: If True, return (G, F) Gram representation.
            Else: (y_iter, g_iter, f_iter, x_K) raw trajectories
    """
    t, beta = stepsizes
    d = Q.shape[0]
    
    def f(x):
        return 0.5 * x @ Q @ x + fs
    
    def g(x):
        return Q @ x
    
    # Storage for y points (y_0, y_1, ..., y_{K-1}) and corresponding gradients/function values
    # Plus final x_K for output
    y_iter = jnp.zeros((d, K_max))       # y_k for k = 0, ..., K-1
    g_iter = jnp.zeros((d, K_max))       # g(y_k) for k = 0, ..., K-1
    f_y_iter = jnp.zeros(K_max)          # f(y_k) for k = 0, ..., K-1
    
    # Initial point: y_0 = x_0 = z0
    y_iter = y_iter.at[:, 0].set(z0)
    g_iter = g_iter.at[:, 0].set(g(z0))
    f_y_iter = f_y_iter.at[0].set(f(z0))
    
    # Initialize: x_prev = x_0 = y_0 = z0, x_curr = z0
    x_prev = z0
    x_curr = z0
    
    def body_fun(k, val):
        y_iter, g_iter, f_y_iter, x_prev, x_curr = val
        
        # Current y is stored in y_iter[:, k]
        y_curr = y_iter[:, k]
        g_y = g(y_curr)
        
        # x update: x_new = y_curr - t_k * g(y_curr)
        tk = t[k] if t.ndim > 0 else t
        x_new = y_curr - tk * g_y
        
        # y update: y_new = x_new + beta_k * (x_new - x_curr)
        # Note: beta[k] is the momentum coefficient for step k
        # For k=0, beta[0]=1.0 by convention so y_1 = x_1 (no momentum on first step)
        y_new = x_new + beta[k] * (x_new - x_curr)
        
        def store_y(val):
            y_iter, g_iter, f_y_iter, x_prev, x_curr, x_new, y_new = val
            y_iter = y_iter.at[:, k + 1].set(y_new)
            g_iter = g_iter.at[:, k + 1].set(g(y_new))
            f_y_iter = f_y_iter.at[k + 1].set(f(y_new))
            return y_iter, g_iter, f_y_iter
        
        def no_store(val):
            y_iter, g_iter, f_y_iter, x_prev, x_curr, x_new, y_new = val
            return y_iter, g_iter, f_y_iter
        
        # Only store y_{k+1} if k < K-1 (i.e., we have more iterations)
        y_iter, g_iter, f_y_iter = jax.lax.cond(
            k < K_max - 1,
            store_y,
            no_store,
            (y_iter, g_iter, f_y_iter, x_prev, x_curr, x_new, y_new)
        )
        
        return (y_iter, g_iter, f_y_iter, x_curr, x_new)
    
    y_iter, g_iter, f_y_iter, _, x_K = jax.lax.fori_loop(
        0, K_max, body_fun, (y_iter, g_iter, f_y_iter, x_prev, x_curr)
    )
    
    # Compute gradient and function value at x_K
    g_xK = g(x_K)
    f_xK = f(x_K)
    
    if return_Gram_representation:
        # Build Gram representation matching pep_construction.py structure
        # G_half columns: [y0-ys, g(y_0), g(y_1), ..., g(y_{K-1}), g(x_K)]
        # Shape: (d, K_max + 2)
        y0_minus_ys = (y_iter[:, 0] - zs).reshape(-1, 1)  # (d, 1)
        G_half = jnp.concatenate([y0_minus_ys, g_iter, g_xK.reshape(-1, 1)], axis=1)
        
        G = G_half.T @ G_half  # (K_max + 2, K_max + 2)
        
        # F = [f(y_0)-fs, f(y_1)-fs, ..., f(y_{K-1})-fs, f(x_K)-fs]
        F = jnp.concatenate([f_y_iter - fs, jnp.array([f_xK - fs])])  # (K_max + 1,)
        
        return G, F
    else:
        # Return raw trajectories
        # Stack g_iter with g(x_K) as the last column
        g_stack = jnp.concatenate([g_iter, g_xK.reshape(-1, 1)], axis=1)
        # Stack f_y_iter with f(x_K) as the last entry
        f_stack = jnp.concatenate([f_y_iter, jnp.array([f_xK])])
        return y_iter, g_stack, f_stack


@jax.jit
def dro_pep_obj_jax(eps, lambd_star, s_star):
    N = s_star.shape[0]
    return lambd_star * eps + 1 / N * jnp.sum(s_star)


# if cfg.pep_obj == 'obj_val':
#         problem.set_performance_metric(f_stack[-1] - fs)
#     elif obj == 'grad_sq_norm':
#         problem.set_performance_metric((g_stack[-1]) ** 2)
#     elif obj == 'opt_dist_sq_norm':
#         problem.set_performance_metric((z_stack[-1] - z_stack[0]) ** 2)
#     else:
#         log.info('should be unreachable code')
#         exit(0)


@partial(jax.jit, static_argnames=['jax_traj_func', 'pep_obj', 'K_max'])
def problem_data_to_pep_obj(stepsizes, Q, z0, zs, fs, K_max, jax_traj_func, pep_obj):
    '''
        jax_traj_func needs to be a function like problem_data_to_gd_trajectories
    '''
    z_stack, g_stack, f_stack = jax_traj_func(
        stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=False
    )
    if pep_obj == 'obj_val':
        return f_stack[-1] - fs
    elif pep_obj == 'grad_sq_norm':
        return jnp.linalg.norm(g_stack[:, -1]) ** 2
    elif pep_obj == 'opt_dist_sq_norm':
        return jnp.linalg.norm(z_stack[:, -1] - zs) ** 2
    else:
        log.info('should be unreachable code')
        exit(0)
