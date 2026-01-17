import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import logging
from functools import partial

log = logging.getLogger(__name__)

# NOTE: We investigated adding ignore_dpp=True to speed up cvxpylayers compilation,
# but this is incompatible with how cvxpylayers handles parametric problems.
# With ignore_dpp=True, CVXPY tries to evaluate parameters immediately, breaking the layer.
# The DPP warning about "too many parameters" can cause OOM for very large problems.
# Alternative solutions:
#   1. Reduce N (number of samples) or M (constraint count)
#   2. Use a different backend (e.g., custom JAX solver)
#   3. Split the problem into smaller sub-problems

from cvxpylayers.jax import CvxpyLayer


class CvxpyLayerWithDefaults:
    """Wrapper for CvxpyLayer that properly merges default solver_args.
    
    Works around a bug in cvxpylayers where solver_args passed to __init__
    are not merged with solver_args passed to __call__.
    Also sets ignore_dpp=True by default to avoid slow DPP compilation with many parameters.
    """
    def __init__(self, problem, parameters, variables, solver_args=None):
        self._default_solver_args = solver_args or {}
        # Note: The DPP warning about "too many parameters" is purely about compilation speed,
        # not correctness. Gradients still work correctly without DPP optimization.
        self._layer = CvxpyLayer(problem, parameters=parameters, variables=variables)
    
    def __call__(self, *params, solver_args=None):
        merged = {**self._default_solver_args, **(solver_args or {})}
        return self._layer(*params, solver_args=merged)


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
    - F: [f0-fs, f1-fs, ..., fK-fs]  (dimF = K_max + 1 entries)
    
    Args:
        stepsizes: Tuple (t,) where t is scalar or vector of length K_max
        Q: Quadratic matrix
        z0: Initial point
        zs: Optimal point
        fs: Optimal function value (scalar)
        K_max: Number of GD iterations
        return_Gram_representation: If True, return (G, F) tuple
    
    Returns:
        If return_Gram_representation=True: (G, F) where G is Gram matrix, F is function values
        Else: (z_stack, g_stack, f_stack) raw trajectories
    """
    t = stepsizes[0]
    t_vec = jnp.broadcast_to(t, (K_max,))
    
    def f(x):
        return .5 * x.T @ Q @ x
    
    def g(x):
        return Q @ x

    dim = z0.shape[0]
    
    # Dimensions matching pep_construction.py
    # dimG = K_max + 2: [z0-zs, g0, g1, ..., gK]
    # dimF = K_max + 1: [f0-fs, f1-fs, ..., fK-fs]
    dimG = K_max + 2
    dimF = K_max + 1
    
    # Store positions and gradients
    # z_iter[k] = z_k for k = 0, 1, ..., K_max
    z_iter = jnp.zeros((dim, K_max + 1))
    g_iter = jnp.zeros((dim, K_max + 1))
    f_iter = jnp.zeros(K_max + 1)
    
    # Initial point
    z_iter = z_iter.at[:, 0].set(z0)
    g_iter = g_iter.at[:, 0].set(g(z0))
    f_iter = f_iter.at[0].set(f(z0))
    
    # GD iterations: z_{k+1} = z_k - t_k * g(z_k)
    def body_fun(k, val):
        z_iter, g_iter, f_iter, z_curr = val
        t_k = t_vec[k]
        z_new = z_curr - t_k * g(z_curr)
        z_iter = z_iter.at[:, k + 1].set(z_new)
        g_iter = g_iter.at[:, k + 1].set(g(z_new))
        f_iter = f_iter.at[k + 1].set(f(z_new))
        return (z_iter, g_iter, f_iter, z_new)
    
    init_val = (z_iter, g_iter, f_iter, z0)
    z_iter, g_iter, f_iter, _ = jax.lax.fori_loop(0, K_max, body_fun, init_val)
    
    if return_Gram_representation:
        # Build G_half to match pep_construction.py structure:
        # G_half columns = [z0-zs, g0, g1, ..., gK]
        # This gives dimG = K_max + 2 columns
        z0_minus_zs = z0 - zs  # First column: z0 - zs
        
        # G_half: shape (dim, dimG) = (dim, K_max + 2)
        # Column 0: z0 - zs
        # Columns 1 to K_max+1: g0, g1, ..., gK
        G_half = jnp.zeros((dim, dimG))
        G_half = G_half.at[:, 0].set(z0_minus_zs)
        G_half = G_half.at[:, 1:].set(g_iter)  # g_iter has K_max+1 columns (g0, ..., gK)
        
        # Gram matrix G = G_half.T @ G_half
        G = G_half.T @ G_half
        
        # Function values F = [f0-fs, f1-fs, ..., fK-fs]
        F = f_iter - fs
        
        return G, F
    else:
        return z_iter, g_iter, f_iter


@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def problem_data_to_nesterov_fgm_trajectories(stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True):
    """
    Compute Nesterov FGM trajectories and return Gram representation for DRO-PEP.
    
    Uses the algorithm ordering:
        x_prev = x
        x = y - t_k * g(y)  
        y = x + beta_k * (x - x_prev)
    
    Uses repY representation where gradients are evaluated at y points.
    x_K is added as an extra point for the objective.
    
    The Gram representation matches the structure in construct_fgm_pep_data:
    - G_half columns: [y0-ys, g(y0), g(y1), ..., g(y_{K-1}), g(x_K)]  (dimG = K_max + 3 columns)
    - F: [f(y0)-fs, f(y1)-fs, ..., f(y_{K-1})-fs, f(x_K)-fs]  (dimF = K_max + 2 entries)
    
    Args:
        stepsizes: Tuple (t, beta) where t is scalar or vector, beta is vector
        Q: Quadratic matrix
        z0: Initial point (x0 = y0 = z0)
        zs: Optimal point
        fs: Optimal function value (scalar)
        K_max: Number of FGM iterations
        return_Gram_representation: If True, return (G, F) tuple
    
    Returns:
        If return_Gram_representation=True: (G, F) where G is Gram matrix, F is function values
        Else: (y_iter, g_iter, f_iter, x_K) raw trajectories
    """
    t, beta = stepsizes[0], stepsizes[1]
    t_vec = jnp.broadcast_to(t, (K_max,))
    
    def f(x):
        return .5 * x.T @ Q @ x
    
    def g(x):
        return Q @ x

    dim = z0.shape[0]
    
    # Dimensions matching new PEP structure (same as GD!)
    # dimG = K_max + 2: [y0-ys, g(y0), g(y1), ..., g(y_{K-1}), g(x_K)]
    #   = 1 position column + K y-gradient columns + 1 x_K gradient = K+2
    # dimF = K_max + 1: [f(y0)-fs, f(y1)-fs, ..., f(y_{K-1})-fs, f(x_K)-fs]
    #   = K y-function values + 1 x_K function value = K+1
    dimG = K_max + 2
    dimF = K_max + 1
    
    # Store y iterates (y0, y1, ..., y_{K-1}) and their gradients
    y_iter = jnp.zeros((dim, K_max))  # y_0 to y_{K-1}
    g_iter = jnp.zeros((dim, K_max))  # g(y_0) to g(y_{K-1})
    f_y_iter = jnp.zeros(K_max)       # f(y_0) to f(y_{K-1})
    
    # Initial: y0 = x0 = z0
    y0 = z0
    x0 = z0
    
    y_iter = y_iter.at[:, 0].set(y0)
    g_iter = g_iter.at[:, 0].set(g(y0))  # g(y0)
    f_y_iter = f_y_iter.at[0].set(f(y0))
    
    # FGM iterations:
    # x = y - t * g(y)
    # y_new = x + beta * (x - x_prev)
    def body_fun(k, val):
        y_iter, g_iter, f_y_iter, x, y = val
        t_k = t_vec[k]
        beta_k = beta[k]
        
        # Compute gradient at current y
        g_y = g(y)
        
        # Update x (main iterate)
        x_prev = x
        x = y - t_k * g_y
        
        # Update y (momentum point)
        y = x + beta_k * (x - x_prev)
        
        # Store y_{k+1} and g(y_{k+1}) for k = 0, ..., K-2
        # Note: y_iter[:, k+1] = y_{k+1}, but we only have K-1 more slots after y_0
        # After iteration k, we have computed y_{k+1}
        # For k=0..K-2, store y_{k+1} at position k+1
        # For k=K-1, we compute x_K but y_K is not needed
        
        # Use jax.lax.cond to avoid storing past the end
        def store_y(val):
            y_iter, g_iter, f_y_iter = val
            y_iter = y_iter.at[:, k + 1].set(y)
            g_iter = g_iter.at[:, k + 1].set(g(y))
            f_y_iter = f_y_iter.at[k + 1].set(f(y))
            return y_iter, g_iter, f_y_iter
        
        def no_store(val):
            return val
        
        y_iter, g_iter, f_y_iter = jax.lax.cond(
            k < K_max - 1,
            store_y,
            no_store,
            (y_iter, g_iter, f_y_iter)
        )
        
        return (y_iter, g_iter, f_y_iter, x, y)
    
    init_val = (y_iter, g_iter, f_y_iter, x0, y0)
    y_iter, g_iter, f_y_iter, x_K, y_final = jax.lax.fori_loop(0, K_max, body_fun, init_val)
    
    # Compute x_K gradient and function value
    g_xK = g(x_K)
    f_xK = f(x_K)
    
    if return_Gram_representation:
        # Build G_half to match structure:
        # G_half columns = [y0-ys, g(y0), g(y1), ..., g(y_{K-1}), g(x_K)]
        # This gives dimG = K_max + 3 columns
        y0_minus_ys = y0 - zs  # First column: y0 - ys
        
        # G_half: shape (dim, dimG) = (dim, K_max + 3)
        # Column 0: y0 - ys
        # Columns 1 to K: g(y_0), g(y_1), ..., g(y_{K-1})
        # Column K+1: g(x_K)
        G_half = jnp.zeros((dim, dimG))
        G_half = G_half.at[:, 0].set(y0_minus_ys)
        G_half = G_half.at[:, 1:K_max+1].set(g_iter)  # g_iter has K columns (g(y_0), ..., g(y_{K-1}))
        G_half = G_half.at[:, K_max+1].set(g_xK)      # g(x_K) as last column
        
        # Gram matrix G = G_half.T @ G_half
        G = G_half.T @ G_half
        
        # Function values F = [f(y0)-fs, f(y1)-fs, ..., f(y_{K-1})-fs, f(x_K)-fs]
        # f_y_iter has K entries, then append f_xK
        F = jnp.concatenate([f_y_iter - fs, jnp.array([f_xK - fs])])
        
        return G, F
    else:
        # Return 3 values like GD: (positions, gradients, function values)
        # Include x_K as the final position for objective computation
        # Stack y_iter with x_K as the last column
        z_stack = jnp.concatenate([y_iter, x_K.reshape(-1, 1)], axis=1)
        # Stack g_iter with g(x_K) as the last column
        g_stack = jnp.concatenate([g_iter, g_xK.reshape(-1, 1)], axis=1)
        # Stack f_y_iter with f(x_K) as the last entry
        f_stack = jnp.concatenate([f_y_iter, jnp.array([f_xK])])
        return z_stack, g_stack, f_stack


def create_full_pep_layer(M, mat_shape, vec_shape):
    """Create vanilla PEP cvxpylayer (primal formulation) with constraint matrices as Parameters.
    
    This enables full gradient flow through step sizes by making A_vals, b_vals, 
    A_obj, b_obj into cvxpy Parameters.
    
    The PEP primal problem is:
        max  trace(A_obj @ G) + b_obj^T @ F
        s.t. G >> 0
             trace(A_m @ G) + b_m^T @ F + c_m <= 0  for all m
    
    Args:
        M: Number of interpolation constraints
        mat_shape: Shape of each A_m matrix (dimG, dimG)
        vec_shape: Shape of each b_m vector (dimF,)
    
    Returns:
        CvxpyLayer that takes parameters in order:
        [A_0, ..., A_{M-1}, b_0, ..., b_{M-1}, c_vals, A_obj, b_obj]
        
        And returns: [G, F] the optimal Gram matrix and function values
    """
    # Create parameters for constraint matrices (computed from t via JAX)
    A_params = [cp.Parameter(mat_shape) for _ in range(M)]  # M constraint matrices
    b_params = [cp.Parameter(vec_shape) for _ in range(M)]  # M constraint vectors
    A_obj_param = cp.Parameter(mat_shape)                    # Objective matrix
    b_obj_param = cp.Parameter(vec_shape)                    # Objective vector
    c_vals_param = cp.Parameter(M)                           # Objective coefficients
    
    G = cp.Variable(mat_shape)
    F = cp.Variable(vec_shape)

    constraints = [G >> 0]

    for m in range(M):
        Am = A_params[m]
        bm = b_params[m]
        cm = c_vals_param[m]
        constraints += [cp.trace(Am @ G) + bm.T @ F + cm <= 0]
    
    obj = cp.trace(A_obj_param @ G) + b_obj_param.T @ F

    prob = cp.Problem(cp.Maximize(obj), constraints)

    all_params = A_params + b_params + [c_vals_param, A_obj_param, b_obj_param]
    
    return CvxpyLayerWithDefaults(prob, parameters=all_params, variables=[G, F])


def create_full_dro_exp_layer(M, N, mat_shape, vec_shape, obj_mat_shape, obj_vec_shape, 
                               c_vals, precond_inv, eps):
    """Create expectation-based DRO cvxpylayer with constraint matrices as Parameters.
    
    This enables full gradient flow through step sizes by making A_vals, b_vals, 
    A_obj, b_obj into cvxpy Parameters (not constants).
    
    Args:
        M: Number of interpolation constraints
        N: Number of samples
        mat_shape: Shape of each A_m matrix (dimG, dimG)
        vec_shape: Shape of each b_m vector (dimF,)
        obj_mat_shape: Shape of A_obj (dimG, dimG)
        obj_vec_shape: Shape of b_obj (dimF,)
        c_vals: Constraint constants (fixed numpy array)
        precond_inv: Tuple (precond_inv_G, precond_inv_F) for preconditioning
        eps: Wasserstein ball radius
    
    Returns:
        CvxpyLayer that takes parameters in order:
        [A_0, ..., A_{M-1}, b_0, ..., b_{M-1}, A_obj, b_obj, G_0, ..., G_{N-1}, F_0, ..., F_{N-1}]
    """
    # Create parameters for constraint matrices (computed from t via JAX)
    A_params = [cp.Parameter(mat_shape) for _ in range(M)]  # M constraint matrices
    b_params = [cp.Parameter(vec_shape) for _ in range(M)]  # M constraint vectors
    A_obj_param = cp.Parameter(obj_mat_shape)               # Objective matrix
    b_obj_param = cp.Parameter(obj_vec_shape)               # Objective vector
    
    # Create parameters for sample Gram matrices (also computed via JAX)
    G_params = [cp.Parameter(obj_mat_shape) for _ in range(N)]  # N sample G matrices
    F_params = [cp.Parameter(obj_vec_shape) for _ in range(N)]  # N sample F vectors
    
    # Variables
    lambd = cp.Variable()
    s = cp.Variable(N)
    y = cp.Variable((N, M))
    Gz = [cp.Variable(obj_mat_shape, symmetric=True) for _ in range(N)]
    Fz = [cp.Variable(obj_vec_shape) for _ in range(N)]
    
    # Objective: minimize lambda * eps + (1/N) * sum(s)
    obj = lambd * eps + 1 / N * cp.sum(s)
    
    # Constraints
    constraints = [y >= 0]
    
    # Preconditioning
    G_preconditioner = np.diag(precond_inv[0])
    F_preconditioner = precond_inv[1]
    
    for i in range(N):
        G_sample, F_sample = G_params[i], F_params[i]
        
        # Epigraph constraint
        constraints += [- c_vals.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
        
        # SOC constraint for Wasserstein ball
        constraints += [cp.SOC(lambd, cp.hstack([
            cp.vec(G_preconditioner @ Gz[i] @ G_preconditioner, order='F'),
            cp.multiply(F_preconditioner**2, Fz[i])
        ]))]
        
        # L* constraint: sum_m (y[i,m] * A_m) - Gz[i] - A_obj >> 0
        LstarG = 0
        LstarF = 0
        for m in range(M):
            LstarG = LstarG + y[i, m] * A_params[m]
            LstarF = LstarF + y[i, m] * b_params[m]
        
        constraints += [LstarG - Gz[i] - A_obj_param >> 0]
        constraints += [LstarF - Fz[i] - b_obj_param == 0]
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    
    # Parameters ordered: A_params, b_params, A_obj, b_obj, G_params, F_params
    all_params = A_params + b_params + [A_obj_param, b_obj_param] + G_params + F_params
    
    return CvxpyLayerWithDefaults(prob, parameters=all_params, variables=[lambd, s], solver_args={'verbose': True})


def create_full_dro_cvar_layer(M, N, mat_shape, vec_shape, obj_mat_shape, obj_vec_shape,
                                c_vals, precond_inv, eps, alpha):
    """Create CVaR-based DRO cvxpylayer with constraint matrices as Parameters.
    
    This enables full gradient flow through step sizes by making A_vals, b_vals, 
    A_obj, b_obj into cvxpy Parameters (not constants).
    
    Args:
        M: Number of interpolation constraints
        N: Number of samples
        mat_shape: Shape of each A_m matrix (dimG, dimG)
        vec_shape: Shape of each b_m vector (dimF,)
        obj_mat_shape: Shape of A_obj (dimG, dimG)
        obj_vec_shape: Shape of b_obj (dimF,)
        c_vals: Constraint constants (fixed numpy array)
        precond_inv: Tuple (precond_inv_G, precond_inv_F) for preconditioning
        eps: Wasserstein ball radius
        alpha: CVaR confidence level
    
    Returns:
        CvxpyLayer that takes parameters in order:
        [A_0, ..., A_{M-1}, b_0, ..., b_{M-1}, A_obj, b_obj, G_0, ..., G_{N-1}, F_0, ..., F_{N-1}]
    """
    alpha_inv = 1 / alpha
    
    # Create parameters for constraint matrices (computed from t via JAX)
    A_params = [cp.Parameter(mat_shape) for _ in range(M)]
    b_params = [cp.Parameter(vec_shape) for _ in range(M)]
    A_obj_param = cp.Parameter(obj_mat_shape)
    b_obj_param = cp.Parameter(obj_vec_shape)
    
    # Create parameters for sample Gram matrices
    G_params = [cp.Parameter(obj_mat_shape) for _ in range(N)]
    F_params = [cp.Parameter(obj_vec_shape) for _ in range(N)]
    
    # Variables
    lambd = cp.Variable()
    s = cp.Variable(N)
    t_var = cp.Variable()
    y1 = cp.Variable((N, M))
    y2 = cp.Variable((N, M))
    
    Gz1 = [cp.Variable(obj_mat_shape, symmetric=True) for _ in range(N)]
    Fz1 = [cp.Variable(obj_vec_shape) for _ in range(N)]
    Gz2 = [cp.Variable(obj_mat_shape, symmetric=True) for _ in range(N)]
    Fz2 = [cp.Variable(obj_vec_shape) for _ in range(N)]
    
    # Objective
    obj = lambd * eps + 1 / N * cp.sum(s)
    
    # Constraints
    constraints = [y1 >= 0, y2 >= 0]
    
    # Preconditioning
    G_preconditioner = np.diag(precond_inv[0])
    F_preconditioner = precond_inv[1]
    
    for i in range(N):
        G_sample, F_sample = G_params[i], F_params[i]
        
        # CVaR epigraph constraints
        constraints += [t_var - c_vals.T @ y1[i] - cp.trace(G_sample @ Gz1[i]) - F_sample.T @ Fz1[i] <= s[i]]
        constraints += [-(alpha_inv - 1) * t_var - c_vals.T @ y2[i] - cp.trace(G_sample @ Gz2[i]) - F_sample.T @ Fz2[i] <= s[i]]
        
        # SOC constraints
        constraints += [cp.SOC(lambd, cp.hstack([
            cp.vec(G_preconditioner @ Gz1[i] @ G_preconditioner, order='F'),
            cp.multiply(F_preconditioner**2, Fz1[i])
        ]))]
        constraints += [cp.SOC(lambd, cp.hstack([
            cp.vec(G_preconditioner @ Gz2[i] @ G_preconditioner, order='F'),
            cp.multiply(F_preconditioner**2, Fz2[i])
        ]))]
        
        # L* constraints with parameterized A_vals, b_vals
        y1A_adj = 0
        y2A_adj = 0
        y1b_adj = 0
        y2b_adj = 0
        
        for m in range(M):
            y1A_adj = y1A_adj + y1[i, m] * A_params[m]
            y2A_adj = y2A_adj + y2[i, m] * A_params[m]
            y1b_adj = y1b_adj + y1[i, m] * b_params[m]
            y2b_adj = y2b_adj + y2[i, m] * b_params[m]
        
        constraints += [y1A_adj - Gz1[i] >> 0]
        constraints += [y1b_adj - Fz1[i] == 0]
        constraints += [y2A_adj - Gz2[i] - alpha_inv * A_obj_param >> 0]
        constraints += [y2b_adj - Fz2[i] - alpha_inv * b_obj_param == 0]
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    
    # Parameters ordered: A_params, b_params, A_obj, b_obj, G_params, F_params
    all_params = A_params + b_params + [A_obj_param, b_obj_param] + G_params + F_params
    
    return CvxpyLayerWithDefaults(prob, parameters=all_params, variables=[lambd, s])


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
