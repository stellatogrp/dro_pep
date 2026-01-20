"""
Trajectory computation functions for logistic regression optimization.

Computes GD and Nesterov FGM trajectories for logistic regression problems
where the objective and gradient depend on problem data (A, b).

All functions operate in shifted coordinates where the optimal point is at the origin.
"""
import jax
import jax.numpy as jnp
import logging
from functools import partial

log = logging.getLogger(__name__)


# =============================================================================
# Logistic Regression Objective and Gradient (shifted coordinates)
# =============================================================================

@jax.jit
def logreg_f_shifted(z, A, b, x_opt, f_opt, delta):
    """
    Compute logistic regression objective in shifted coordinates.
    
    f(z) = f(z + x_opt) - f_opt
    
    where f(x) = 1/m * sum(-y_i * (A @ x)_i + log(1 + exp((A @ x)_i))) + delta/2 * ||x||^2
    
    Args:
        z: (n,) point in shifted coordinates (z = x - x_opt)
        A: (m, n) data matrix
        b: (m,) binary labels
        x_opt: (n,) optimal solution
        f_opt: Optimal objective value
        delta: L2 regularization
        
    Returns:
        Shifted objective value (scalar)
    """
    m = A.shape[0]
    x = z + x_opt
    Ax = A @ x
    log_likeli = jnp.sum(b * Ax - jnp.logaddexp(0, Ax))
    return -1 / m * log_likeli + 0.5 * delta * jnp.dot(x, x) - f_opt


@jax.jit
def logreg_grad_shifted(z, A, b, x_opt, delta):
    """
    Compute gradient of logistic regression objective in shifted coordinates.
    
    grad f(z) = 1/m * A^T @ (sigmoid(A @ (z + x_opt)) - y) + delta * (z + x_opt)
    
    Args:
        z: (n,) point in shifted coordinates
        A: (m, n) data matrix
        b: (m,) binary labels
        x_opt: (n,) optimal solution
        delta: L2 regularization
        
    Returns:
        (n,) gradient vector
    """
    m = A.shape[0]
    x = z + x_opt
    Ax = A @ x
    sigmoid_Ax = jax.nn.sigmoid(Ax)
    return 1 / m * A.T @ (sigmoid_Ax - b) + delta * x


# =============================================================================
# Gradient Descent Trajectories for Logistic Regression
# =============================================================================

@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def logreg_gd_trajectories(stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, return_Gram_representation=True):
    """
    Compute GD trajectories for logistic regression and return Gram representation.
    
    Gram representation structure:
    - G_half columns: [z0, g0, g1, ..., gK] with shape (n, K_max + 2)
    - G = G_half.T @ G_half
    - F = [f0, f1, ..., fK] with shape (K_max + 1,)
    
    Args:
        stepsizes: Tuple (t,) where t is array of K_max step sizes
        A: (m, n) data matrix
        b: (m,) binary labels
        z0: (n,) initial point in shifted coordinates
        x_opt: (n,) optimal solution
        f_opt: Optimal function value
        delta: L2 regularization parameter
        K_max: Number of GD steps
        return_Gram_representation: If True, return (G, F). Otherwise return raw trajectories.
        
    Returns:
        If return_Gram_representation: (G, F) tuple
        Otherwise: (z_stack, g_stack, f_stack) tuple
    """
    t = stepsizes[0]
    n = A.shape[1]
    
    def f(z):
        return logreg_f_shifted(z, A, b, x_opt, f_opt, delta)
    
    def g(z):
        return logreg_grad_shifted(z, A, b, x_opt, delta)
    
    # Storage for K+1 points (z0, z1, ..., zK)
    z_iter = jnp.zeros((n, K_max + 1))
    g_iter = jnp.zeros((n, K_max + 1))
    f_iter = jnp.zeros(K_max + 1)
    
    z_iter = z_iter.at[:, 0].set(z0)
    g_iter = g_iter.at[:, 0].set(g(z0))
    f_iter = f_iter.at[0].set(f(z0))
    
    z_curr = z0
    
    def body_fun(k, val):
        z_iter, g_iter, f_iter, z_curr = val
        g_curr = g(z_curr)
        z_next = z_curr - t[k] * g_curr
        
        z_iter = z_iter.at[:, k + 1].set(z_next)
        g_iter = g_iter.at[:, k + 1].set(g(z_next))
        f_iter = f_iter.at[k + 1].set(f(z_next))
        
        return (z_iter, g_iter, f_iter, z_next)
    
    z_iter, g_iter, f_iter, _ = jax.lax.fori_loop(
        0, K_max, body_fun, (z_iter, g_iter, f_iter, z_curr)
    )
    
    if return_Gram_representation:
        z0_col = z_iter[:, 0].reshape(-1, 1)
        G_half = jnp.concatenate([z0_col, g_iter], axis=1)
        G = G_half.T @ G_half
        F = f_iter
        return G, F
    else:
        return z_iter, g_iter, f_iter


# =============================================================================
# Nesterov FGM Trajectories for Logistic Regression
# =============================================================================

@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def logreg_fgm_trajectories(stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, return_Gram_representation=True):
    """
    Compute Nesterov FGM trajectories for logistic regression and return Gram representation.
    
    Algorithm ordering:
        x_prev = x
        x = y - t_k * g(y)  
        y = x + beta_k * (x - x_prev)
    
    Gram representation structure:
    - G_half columns: [y0, g(y_0), ..., g(y_{K-1}), g(x_K)] with shape (n, K_max + 2)
    - G = G_half.T @ G_half
    - F = [f(y_0), ..., f(y_{K-1}), f(x_K)] with shape (K_max + 1,)
    
    Args:
        stepsizes: Tuple (t, beta) where t has K_max elements, beta has K_max elements
        A: (m, n) data matrix
        b: (m,) binary labels
        z0: (n,) initial point in shifted coordinates
        x_opt: (n,) optimal solution
        f_opt: Optimal function value
        delta: L2 regularization parameter
        K_max: Number of FGM steps
        return_Gram_representation: If True, return (G, F). Otherwise return raw trajectories.
        
    Returns:
        If return_Gram_representation: (G, F) tuple
        Otherwise: (y_iter, g_stack, f_stack) tuple
    """
    t, beta = stepsizes
    n = A.shape[1]
    
    def f(z):
        return logreg_f_shifted(z, A, b, x_opt, f_opt, delta)
    
    def g(z):
        return logreg_grad_shifted(z, A, b, x_opt, delta)
    
    # Storage for y points and gradients
    y_iter = jnp.zeros((n, K_max))
    g_iter = jnp.zeros((n, K_max))
    f_y_iter = jnp.zeros(K_max)
    
    y_iter = y_iter.at[:, 0].set(z0)
    g_iter = g_iter.at[:, 0].set(g(z0))
    f_y_iter = f_y_iter.at[0].set(f(z0))
    
    x_prev = z0
    x_curr = z0
    
    def body_fun(k, val):
        y_iter, g_iter, f_y_iter, x_prev, x_curr = val
        
        y_curr = y_iter[:, k]
        g_y = g(y_curr)
        
        x_new = y_curr - t[k] * g_y
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
    
    g_xK = g(x_K)
    f_xK = f(x_K)
    
    if return_Gram_representation:
        y0_col = y_iter[:, 0].reshape(-1, 1)
        G_half = jnp.concatenate([y0_col, g_iter, g_xK.reshape(-1, 1)], axis=1)
        G = G_half.T @ G_half
        F = jnp.concatenate([f_y_iter, jnp.array([f_xK])])
        return G, F
    else:
        g_stack = jnp.concatenate([g_iter, g_xK.reshape(-1, 1)], axis=1)
        f_stack = jnp.concatenate([f_y_iter, jnp.array([f_xK])])
        return y_iter, g_stack, f_stack


# =============================================================================
# PEP Objective Computation (for L2O pipeline)
# =============================================================================

@partial(jax.jit, static_argnames=['jax_traj_func', 'pep_obj', 'K_max'])
def logreg_pep_obj(stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, jax_traj_func, pep_obj):
    """
    Compute PEP objective directly from logistic regression trajectory.
    
    Args:
        stepsizes: Step size parameters (tuple)
        A: Data matrix
        b: Labels
        z0: Initial point in shifted coordinates
        x_opt: Optimal solution
        f_opt: Optimal value
        delta: L2 regularization
        K_max: Number of iterations
        jax_traj_func: Trajectory function (logreg_gd_trajectories or logreg_fgm_trajectories)
        pep_obj: Objective type ('obj_val', 'grad_sq_norm', 'opt_dist_sq_norm')
        
    Returns:
        Scalar PEP objective value
    """
    z_stack, g_stack, f_stack = jax_traj_func(
        stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, return_Gram_representation=False
    )
    
    if pep_obj == 'obj_val':
        return f_stack[-1]
    elif pep_obj == 'grad_sq_norm':
        return jnp.linalg.norm(g_stack[:, -1]) ** 2
    elif pep_obj == 'opt_dist_sq_norm':
        return jnp.linalg.norm(z_stack[:, -1]) ** 2
    else:
        return 0.0


# =============================================================================
# Factory Functions (create trajectory functions with delta baked in)
# =============================================================================

def create_logreg_traj_fn_gd(delta):
    """
    Create a GD trajectory function with delta baked in.
    
    Args:
        delta: L2 regularization parameter
        
    Returns:
        Trajectory function with signature:
        (stepsizes, A, b, z0, x_opt, f_opt, K_max, return_Gram_representation) -> (G, F)
    """
    @partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
    def traj_fn(stepsizes, A, b, z0, x_opt, f_opt, K_max, return_Gram_representation=True):
        return logreg_gd_trajectories(
            stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, return_Gram_representation
        )
    return traj_fn


def create_logreg_traj_fn_fgm(delta):
    """
    Create an FGM trajectory function with delta baked in.
    
    Args:
        delta: L2 regularization parameter
        
    Returns:
        Trajectory function with signature:
        (stepsizes, A, b, z0, x_opt, f_opt, K_max, return_Gram_representation) -> (G, F)
    """
    @partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
    def traj_fn(stepsizes, A, b, z0, x_opt, f_opt, K_max, return_Gram_representation=True):
        return logreg_fgm_trajectories(
            stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, return_Gram_representation
        )
    return traj_fn


# Aliases for import compatibility
problem_data_to_logreg_pep_obj = logreg_pep_obj
