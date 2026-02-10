"""
ISTA and FISTA trajectory computation for DRO-PEP optimization.

This module contains trajectory computation functions for proximal gradient methods
(ISTA, FISTA) on Lasso problems, with shifted functions so x_opt = 0 and f_opt = 0.

The Lasso problem is:
    min_x  0.5 * ||A @ x - b||^2 + lambd * ||x||_1
         = f1(x) + f2(x)

where:
    f1(x) = 0.5 * ||A @ x - b||^2  (smooth, strongly convex if A full rank)
    f2(x) = lambd * ||x||_1        (convex, non-smooth)

For the DRO-PEP framework, we need to shift functions so the optimal is at origin:
    f1_shifted(x) = 0.5 * ||A @ (x + x_opt) - b||^2 - f_opt
    f2_shifted(x) = lambd * ||x + x_opt||_1

This ensures f1_shifted(0) + f2_shifted(0) = 0.
"""
import jax
import jax.numpy as jnp
import logging
from functools import partial

log = logging.getLogger(__name__)


def soft_threshold_jax(v, delta):
    """JAX-compatible soft-thresholding operator."""
    return jnp.sign(v) * jnp.maximum(jnp.abs(v) - delta, 0)


@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def problem_data_to_ista_trajectories(stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max, 
                                       return_Gram_representation=True):
    """
    Compute ISTA trajectories on shifted Lasso problem and return Gram representation.
    
    The shifted problem has optimal at x = 0:
        f1_shifted(x) = 0.5 * ||A @ (x + x_opt) - b||^2 - f_opt
        f2_shifted(x) = lambd * ||x + x_opt||_1
    
    The Gram representation for ISTA with K iterations has:
        Gram basis (dimG = 2K + 5):
            [x_0 - x_s, g_0, h_0, h_1, g_1, h_2, g_2, ..., h_K, g_K, g_s, h_s]
        where x_s = 0, g_s = grad_f1_shifted(0), h_s = -g_s (stationarity)
        
        Function values (dimF1 = K + 2, dimF2 = K + 2):
            F1 = [f1_shifted(x_0) - f1_s, ..., f1_shifted(x_K) - f1_s, 0]
            F2 = [f2_shifted(x_0) - f2_s, ..., f2_shifted(x_K) - f2_s, 0]
        where f1_s = f1_shifted(0) = -f2(x_opt), f2_s = f2_shifted(0) = f2(x_opt)
    
    Args:
        stepsizes: Array of K_max step sizes (gamma_0, gamma_1, ..., gamma_{K-1})
        A: (m, n) measurement matrix
        b: (m,) observation vector (the sample)
        x0: (n,) initial point in ORIGINAL coordinates (will be shifted by x_opt)
        x_opt: (n,) optimal point of the original problem
        f_opt: Optimal objective value of the original problem
        lambd: Regularization parameter
        K_max: Number of ISTA iterations
        return_Gram_representation: If True, return (G, F1, F2) Gram representation.
            Else: (x_iter, g_iter, h_iter, f1_iter, f2_iter) raw trajectories
    
    Returns:
        If return_Gram_representation:
            G: (2K+5, 2K+5) Gram matrix
            F: (2K+4,) concatenated function values [F1, F2] where:
                - F1: (K+2,) f1 values relative to optimal
                - F2: (K+2,) f2 values relative to optimal
        Else:
            x_iter, g_iter, h_iter, f1_iter, f2_iter: raw trajectory data
    """
    n = A.shape[1]
    
    # Shifted functions
    def f1_shifted(x):
        # f1(x + x_opt) - f_opt
        residual = A @ (x + x_opt) - b
        return 0.5 * jnp.sum(residual ** 2) - f_opt
    
    def f2_shifted(x):
        # f2(x + x_opt) = lambd * ||x + x_opt||_1
        return lambd * jnp.sum(jnp.abs(x + x_opt))
    
    def grad_f1_shifted(x):
        # grad f1(x + x_opt) = A^T @ (A @ (x + x_opt) - b)
        return A.T @ (A @ (x + x_opt) - b)
    
    def subgrad_f2_shifted(x):
        # subgrad f2(x + x_opt) = lambd * sign(x + x_opt)
        return lambd * jnp.sign(x + x_opt)
    
    # Shifted initial point: x0_shifted = x0 - x_opt
    x0_shifted = x0 - x_opt
    
    # Storage for K+1 iterates (x_0, x_1, ..., x_K)
    x_iter = jnp.zeros((n, K_max + 1))
    g_iter = jnp.zeros((n, K_max + 1))  # gradients of f1 at x_k
    h_iter = jnp.zeros((n, K_max + 1))  # subgradients of f2 at x_k
    f1_iter = jnp.zeros(K_max + 1)
    f2_iter = jnp.zeros(K_max + 1)
    
    # Initial point
    x_iter = x_iter.at[:, 0].set(x0_shifted)
    g_iter = g_iter.at[:, 0].set(grad_f1_shifted(x0_shifted))
    h_iter = h_iter.at[:, 0].set(subgrad_f2_shifted(x0_shifted))
    f1_iter = f1_iter.at[0].set(f1_shifted(x0_shifted))
    f2_iter = f2_iter.at[0].set(f2_shifted(x0_shifted))
    
    x_curr = x0_shifted
    
    def ista_step(k, val):
        x_iter, g_iter, h_iter, f1_iter, f2_iter, x_curr = val
        gamma = stepsizes[k]
        
        # Gradient step
        y_k = x_curr - gamma * grad_f1_shifted(x_curr)
        
        # Proximal step on shifted f2:
        # prox_{gamma * f2_shifted}(y) = soft_threshold(y + x_opt, gamma * lambd) - x_opt
        x_new_plus_xopt = soft_threshold_jax(y_k + x_opt, gamma * lambd)
        x_new = x_new_plus_xopt - x_opt
        
        # Subgradient from proximal optimality: h_{k+1} = (y_k - x_new) / gamma
        h_new = (y_k - x_new) / gamma
        
        # Store iterates
        x_iter = x_iter.at[:, k + 1].set(x_new)
        g_iter = g_iter.at[:, k + 1].set(grad_f1_shifted(x_new))
        h_iter = h_iter.at[:, k + 1].set(h_new)
        f1_iter = f1_iter.at[k + 1].set(f1_shifted(x_new))
        f2_iter = f2_iter.at[k + 1].set(f2_shifted(x_new))
        
        return (x_iter, g_iter, h_iter, f1_iter, f2_iter, x_new)
    
    x_iter, g_iter, h_iter, f1_iter, f2_iter, _ = jax.lax.fori_loop(
        0, K_max, ista_step, (x_iter, g_iter, h_iter, f1_iter, f2_iter, x_curr)
    )
    
    if return_Gram_representation:
        # Compute g_s and h_s at optimal (x_s = 0)
        # g_s = grad_f1_shifted(0) = A^T @ (A @ x_opt - b)
        g_s = A.T @ (A @ x_opt - b)
        # h_s = -g_s by stationarity
        h_s = -g_s
        
        # Build G_half: columns are [x_0-x_s, g_0, h_0, h_1, g_1, ..., h_K, g_K, g_s, h_s]
        # x_s = 0, so x_0 - x_s = x_0
        G_half_cols = [x_iter[:, 0].reshape(-1, 1)]  # x_0 - x_s
        G_half_cols.append(g_iter[:, 0].reshape(-1, 1))  # g_0
        G_half_cols.append(h_iter[:, 0].reshape(-1, 1))  # h_0
        
        for k in range(1, K_max + 1):
            G_half_cols.append(h_iter[:, k].reshape(-1, 1))  # h_k
            G_half_cols.append(g_iter[:, k].reshape(-1, 1))  # g_k
        
        G_half_cols.append(g_s.reshape(-1, 1))  # g_s
        G_half_cols.append(h_s.reshape(-1, 1))  # h_s
        
        G_half = jnp.concatenate(G_half_cols, axis=1)  # (n, 2K+5)
        G = G_half.T @ G_half  # (2K+5, 2K+5)
        
        # Function values at optimal:
        # f1_s = f1_shifted(0) = f1(x_opt) - f_opt = -f2(x_opt)
        # f2_s = f2_shifted(0) = f2(x_opt)
        f2_x_opt = lambd * jnp.sum(jnp.abs(x_opt))
        f1_s = -f2_x_opt
        f2_s = f2_x_opt
        
        # F1 and F2: values relative to optimal, plus 0 at the end for x_s
        F1 = jnp.concatenate([f1_iter - f1_s, jnp.array([0.0])])  # (K+2,)
        F2 = jnp.concatenate([f2_iter - f2_s, jnp.array([0.0])])  # (K+2,)
        
        # Concatenate F1 and F2 for DRO pipeline compatibility
        F = jnp.concatenate([F1, F2])  # (2K+4,)
        
        return G, F
    else:
        return x_iter, g_iter, h_iter, f1_iter, f2_iter


@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def problem_data_to_fista_trajectories(stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max,
                                        return_Gram_representation=True):
    """
    Compute FISTA trajectories on shifted Lasso problem and return Gram representation.
    
    FISTA algorithm:
        y_0 = x_0
        For k = 0, ..., K-1:
            x_{k+1} = prox_{gamma_k * f2}(y_k - gamma_k * grad_f1(y_k))
            y_{k+1} = x_{k+1} + beta_k * (x_{k+1} - x_k)
    
    Key difference from ISTA: gradients are evaluated at y_k, not x_k.
    
    The Gram representation for FISTA with K iterations has:
        Gram basis (dimG = 2K + 4):
            [x_0 - x_s, g(y_0), g(y_1), ..., g(y_{K-1}), g(x_K), h_0, h_1, ..., h_K, g_s]
        where g_s = grad_f1_shifted(0), h_s = -g_s (stationarity)
        
        Function values (dimF = 2K + 4):
            F1 = [f1(y_0) - f1_s, ..., f1(y_{K-1}) - f1_s, f1(x_K) - f1_s, 0]  (K + 2 values)
            F2 = [f2(x_0) - f2_s, ..., f2(x_K) - f2_s, 0]                       (K + 2 values)
    
    Args:
        stepsizes: Tuple (gamma, betas) where:
            - gamma: Array of K_max step sizes
            - betas: Array of K_max + 1 raw t_k sequence values (Nesterov sequence)
        A: (m, n) measurement matrix
        b: (m,) observation vector (the sample)
        x0: (n,) initial point in ORIGINAL coordinates
        x_opt: (n,) optimal point of the original problem
        f_opt: Optimal objective value
        lambd: Regularization parameter
        K_max: Number of FISTA iterations
        return_Gram_representation: If True, return (G, F) Gram representation.
    
    Returns:
        If return_Gram_representation:
            G: (2K+4, 2K+4) Gram matrix
            F: (2K+4,) concatenated function values [F1, F2]
        Else:
            x_iter, y_iter, g_y_iter, g_xK, h_iter, f1_y_iter, f1_xK, f2_x_iter
    """
    gamma, betas = stepsizes
    n = A.shape[1]
    
    # Shifted functions
    def f1_shifted(x):
        residual = A @ (x + x_opt) - b
        return 0.5 * jnp.sum(residual ** 2) - f_opt
    
    def f2_shifted(x):
        return lambd * jnp.sum(jnp.abs(x + x_opt))
    
    def grad_f1_shifted(x):
        return A.T @ (A @ (x + x_opt) - b)
    
    # Shifted initial point
    x0_shifted = x0 - x_opt
    
    # Storage
    x_iter = jnp.zeros((n, K_max + 1))     # x_k for k = 0, ..., K
    y_iter = jnp.zeros((n, K_max))         # y_k for k = 0, ..., K-1
    g_y_iter = jnp.zeros((n, K_max))       # g(y_k) for k = 0, ..., K-1
    h_iter = jnp.zeros((n, K_max + 1))     # h_k for k = 0, ..., K
    f1_y_iter = jnp.zeros(K_max)           # f1(y_k) for k = 0, ..., K-1
    f2_x_iter = jnp.zeros(K_max + 1)       # f2(x_k) for k = 0, ..., K
    
    # Initial: y_0 = x_0 = x0_shifted
    x_iter = x_iter.at[:, 0].set(x0_shifted)
    y_iter = y_iter.at[:, 0].set(x0_shifted)
    g_y_iter = g_y_iter.at[:, 0].set(grad_f1_shifted(x0_shifted))
    h_iter = h_iter.at[:, 0].set(lambd * jnp.sign(x0_shifted + x_opt))  # subgrad at x_0
    f1_y_iter = f1_y_iter.at[0].set(f1_shifted(x0_shifted))
    f2_x_iter = f2_x_iter.at[0].set(f2_shifted(x0_shifted))
    
    x_curr = x0_shifted
    x_prev = x0_shifted
    
    def fista_step(k, val):
        x_iter, y_iter, g_y_iter, h_iter, f1_y_iter, f2_x_iter, x_prev, x_curr = val
        
        y_curr = y_iter[:, k]
        g_y = grad_f1_shifted(y_curr)
        
        # Proximal gradient step
        ytilde = y_curr - gamma[k] * g_y
        x_new_plus_xopt = soft_threshold_jax(ytilde + x_opt, gamma[k] * lambd)
        x_new = x_new_plus_xopt - x_opt
        
        # Subgradient from proximal optimality
        h_new = (ytilde - x_new) / gamma[k]
        
        # Momentum step: compute momentum coefficient from raw t_k sequence
        # mom_coef = (t_k - 1) / t_{k+1} = (betas[k] - 1) / betas[k+1]
        mom_coef = (betas[k] - 1.0) / betas[k + 1]
        y_new = x_new + mom_coef * (x_new - x_curr)
        
        # Store x_{k+1}
        x_iter = x_iter.at[:, k + 1].set(x_new)
        h_iter = h_iter.at[:, k + 1].set(h_new)
        f2_x_iter = f2_x_iter.at[k + 1].set(f2_shifted(x_new))
        
        # Store y_{k+1} if within bounds (k < K-1)
        def store_y(val):
            y_iter, g_y_iter, f1_y_iter, y_new = val
            y_iter = y_iter.at[:, k + 1].set(y_new)
            g_y_iter = g_y_iter.at[:, k + 1].set(grad_f1_shifted(y_new))
            f1_y_iter = f1_y_iter.at[k + 1].set(f1_shifted(y_new))
            return y_iter, g_y_iter, f1_y_iter
        
        def no_store(val):
            y_iter, g_y_iter, f1_y_iter, y_new = val
            return y_iter, g_y_iter, f1_y_iter
        
        y_iter, g_y_iter, f1_y_iter = jax.lax.cond(
            k < K_max - 1,
            store_y,
            no_store,
            (y_iter, g_y_iter, f1_y_iter, y_new)
        )
        
        return (x_iter, y_iter, g_y_iter, h_iter, f1_y_iter, f2_x_iter, x_curr, x_new)
    
    x_iter, y_iter, g_y_iter, h_iter, f1_y_iter, f2_x_iter, _, x_K = jax.lax.fori_loop(
        0, K_max, fista_step, 
        (x_iter, y_iter, g_y_iter, h_iter, f1_y_iter, f2_x_iter, x_prev, x_curr)
    )
    
    # Compute g(x_K) and f1(x_K) for the final point
    g_xK = grad_f1_shifted(x_K)
    f1_xK = f1_shifted(x_K)
    
    if return_Gram_representation:
        # Compute g_s at optimal (x_s = 0)
        g_s = A.T @ (A @ x_opt - b)
        
        # Build G_half with PEP construction structure:
        # [x_0-x_s, g(y_0), g(y_1), ..., g(y_{K-1}), g(x_K), h_0, h_1, ..., h_K, g_s]
        # Total: 1 + K + 1 + (K+1) + 1 = 2K + 4
        G_half_cols = [x_iter[:, 0].reshape(-1, 1)]  # x_0 - x_s = x_0
        
        # Add g(y_k) for k = 0, ..., K-1
        for k in range(K_max):
            G_half_cols.append(g_y_iter[:, k].reshape(-1, 1))
        
        # Add g(x_K)
        G_half_cols.append(g_xK.reshape(-1, 1))
        
        # Add h_k for k = 0, ..., K
        for k in range(K_max + 1):
            G_half_cols.append(h_iter[:, k].reshape(-1, 1))
        
        # Add g_s
        G_half_cols.append(g_s.reshape(-1, 1))
        
        G_half = jnp.concatenate(G_half_cols, axis=1)  # (n, 2K+4)
        G = G_half.T @ G_half  # (2K+4, 2K+4)
        
        # Function values at optimal
        f2_x_opt = lambd * jnp.sum(jnp.abs(x_opt))
        f1_s = -f2_x_opt
        f2_s = f2_x_opt
        
        # F1: f1 at y points (K values) + f1(x_K) + 0 for optimal = K+2 values
        F1 = jnp.concatenate([f1_y_iter - f1_s, jnp.array([f1_xK - f1_s]), jnp.array([0.0])])
        
        # F2: f2 at x points (K+1 values) + 0 for optimal = K+2 values
        F2 = jnp.concatenate([f2_x_iter - f2_s, jnp.array([0.0])])
        
        # Concatenate F1 and F2 for DRO pipeline compatibility
        F = jnp.concatenate([F1, F2])  # (2K+4,)
        
        return G, F
    else:
        return x_iter, y_iter, g_y_iter, g_xK, h_iter, f1_y_iter, f1_xK, f2_x_iter
