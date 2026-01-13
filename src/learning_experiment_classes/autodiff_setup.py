import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import logging
from cvxpylayers.jax import CvxpyLayer
from functools import partial

log = logging.getLogger(__name__)


class CvxpyLayerWithDefaults:
    """Wrapper for CvxpyLayer that properly merges default solver_args.
    
    Works around a bug in cvxpylayers where solver_args passed to __init__
    are not merged with solver_args passed to __call__.
    """
    def __init__(self, *args, solver_args=None, **kwargs):
        self._default_solver_args = solver_args or {}
        self._layer = CvxpyLayer(*args, solver_args=solver_args, **kwargs)
    
    def __call__(self, *params, solver_args=None):
        merged = {**self._default_solver_args, **(solver_args or {})}
        return self._layer(*params, solver_args=merged)


@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def problem_data_to_gd_trajectories(stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True):
    t = stepsizes[0]
    t_vec = jnp.broadcast_to(t, (K_max,))
    
    def f(x):
        return .5 * x.T @ Q @ x
    
    def g(x):
        return Q @ x

    dim = z0.shape[0]
    # zs = jnp.zeros(dim)
    z_stack = jnp.zeros((dim, K_max + 2))
    g_stack = jnp.zeros((dim, K_max + 2))
    f_stack = jnp.zeros(K_max + 2)

    z_stack = z_stack.at[:, 0].set(zs)
    z_stack = z_stack.at[:, 1].set(z0)
    g_stack = g_stack.at[:, 0].set(g(zs))
    g_stack = g_stack.at[:, 1].set(g(z0))
    f_stack = f_stack.at[0].set(fs)
    f_stack = f_stack.at[1].set(f(z0))

    def body_fun(i, val):
        z, z_stack, g_stack, f_stack = val
        t_i = t_vec[jnp.minimum(i-1, t_vec.shape[0]-1)]
        z = z - t_i * g(z)
        z_stack = z_stack.at[:, i+1].set(z)
        g_stack = g_stack.at[:, i+1].set(g(z))
        f_stack = f_stack.at[i+1].set(f(z))
        return (z, z_stack, g_stack, f_stack)

    init_val = (z0, z_stack, g_stack, f_stack)
    _, z_stack, g_stack, f_stack = \
        jax.lax.fori_loop(1, K_max + 1, body_fun, init_val)

    if return_Gram_representation:
        G_half = jnp.hstack([z_stack[:,:2], g_stack[:,1:]])
        return G_half.T@G_half, f_stack
    else:
        return z_stack, g_stack, f_stack


@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def problem_data_to_nesterov_fgm_trajectories(stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True):
    t, beta = stepsizes[0], stepsizes[1]
    t_vec = jnp.broadcast_to(t, (K_max,))
    
    def f(x):
        return .5 * x.T @ Q @ x
    
    def g(x):
        return Q @ x

    dim = z0.shape[0]
    z_stack = jnp.zeros((dim, K_max + 2))
    g_stack = jnp.zeros((dim, K_max + 2))
    f_stack = jnp.zeros(K_max + 2)

    z_stack = z_stack.at[:, 0].set(zs)
    z_stack = z_stack.at[:, 1].set(z0)
    g_stack = g_stack.at[:, 0].set(g(zs))
    g_stack = g_stack.at[:, 1].set(g(z0))
    f_stack = f_stack.at[0].set(fs)
    f_stack = f_stack.at[1].set(f(z0))

    def body_fun(k, val):
        z, y, z_stack, g_stack, f_stack = val
        t_k = t_vec[jnp.minimum(k-1, t_vec.shape[0]-1)]
        beta_k = beta[k-1]

        y_prev = y
        y = z - t_k * g(z)
        z = y + beta_k * (y - y_prev)

        z_stack = z_stack.at[:, k+1].set(z)
        g_stack = g_stack.at[:, k+1].set(g(z))
        f_stack = f_stack.at[k+1].set(f(z))
        return (z, y, z_stack, g_stack, f_stack)
    
    init_val = (z0, z0, z_stack, g_stack, f_stack)
    _, _, z_stack, g_stack, f_stack = \
        jax.lax.fori_loop(1, K_max + 1, body_fun, init_val)

    if return_Gram_representation:
        G_half = jnp.hstack([z_stack[:,:2], g_stack[:,1:]])
        return G_half.T@G_half, f_stack
    else:
        return z_stack, g_stack, f_stack


def create_exp_cp_layer(DR, eps, G_shape, F_shape):
    """Create expectation-based DRO cvxpylayer.
    
    Uses list of 2D parameters (one per sample) to avoid 3D tensor canonicalization issues.
    
    Args:
        DR: DROReformulator object
        eps: Wasserstein ball radius
        G_shape: Shape of G batch (N, mat_dim, mat_dim)
        F_shape: Shape of F batch (N, vec_dim)
    
    Returns:
        CvxpyLayer that takes 2*N parameters: [G_0, G_1, ..., G_{N-1}, F_0, F_1, ..., F_{N-1}]
    """
    N = G_shape[0]
    mat_shape = G_shape[1:]  # (mat_dim, mat_dim)
    vec_shape = F_shape[1:]  # (vec_dim,)

    # Create N separate 2D parameters for G and F
    G_params = [cp.Parameter(mat_shape) for _ in range(N)]
    F_params = [cp.Parameter(vec_shape) for _ in range(N)]

    A_vals = DR.canon.A_vals
    b_vals = DR.canon.b_vals
    c_vals = DR.canon.c_vals
    A_obj = DR.canon.A_obj
    b_obj = DR.canon.b_obj

    M = A_vals.shape[0]
    obj_mat_shape = A_obj.shape
    obj_vec_shape = b_obj.shape

    lambd = cp.Variable()
    s = cp.Variable(N)
    y = cp.Variable((N, M))
    Gz = [cp.Variable(obj_mat_shape, symmetric=True) for _ in range(N)]
    Fz = [cp.Variable(obj_vec_shape) for _ in range(N)]

    obj = lambd * eps + 1 / N * cp.sum(s)
    constraints = [y >= 0]

    G_preconditioner = np.diag(DR.canon.precond_inv[0])
    F_preconditioner = DR.canon.precond_inv[1]

    for i in range(N):
        G_sample, F_sample = G_params[i], F_params[i]
        constraints += [- c_vals.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
        constraints += [cp.SOC(lambd, cp.hstack([cp.vec( G_preconditioner@Gz[i]@G_preconditioner, order='F'), cp.multiply(F_preconditioner**2, Fz[i])]))]

        LstarG = 0
        LstarF = 0
        for m in range(M):
            Am = A_vals[m]
            bm = b_vals[m]
            LstarG = LstarG + y[i, m] * Am
            LstarF = LstarF + y[i, m] * bm
    
        constraints += [LstarG - Gz[i] - A_obj >> 0]
        constraints += [LstarF - Fz[i] - b_obj == 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    # Parameters list: [G_0, G_1, ..., G_{N-1}, F_0, F_1, ..., F_{N-1}]
    all_params = G_params + F_params
    return CvxpyLayerWithDefaults(prob, parameters=all_params, variables=[lambd, s])


def create_cvar_cp_layer(DR, eps, alpha, G_shape, F_shape):
    """Create CVaR-based DRO cvxpylayer.
    
    Uses list of 2D parameters (one per sample) to avoid 3D tensor canonicalization issues.
    
    Args:
        DR: DROReformulator object
        eps: Wasserstein ball radius
        alpha: CVaR confidence level
        G_shape: Shape of G batch (N, mat_dim, mat_dim)
        F_shape: Shape of F batch (N, vec_dim)
    
    Returns:
        CvxpyLayer that takes 2*N parameters: [G_0, G_1, ..., G_{N-1}, F_0, F_1, ..., F_{N-1}]
    """
    alpha_inv = 1 / alpha

    N = G_shape[0]
    mat_shape = G_shape[1:]  # (mat_dim, mat_dim)
    vec_shape = F_shape[1:]  # (vec_dim,)

    # Create N separate 2D parameters for G and F
    G_params = [cp.Parameter(mat_shape) for _ in range(N)]
    F_params = [cp.Parameter(vec_shape) for _ in range(N)]

    A_vals = DR.canon.A_vals
    b_vals = DR.canon.b_vals
    c_vals = DR.canon.c_vals
    A_obj = DR.canon.A_obj
    b_obj = DR.canon.b_obj

    M = A_vals.shape[0]
    obj_mat_shape = A_obj.shape
    obj_vec_shape = b_obj.shape

    lambd = cp.Variable()
    s = cp.Variable(N)
    y1 = cp.Variable((N, M))
    y2 = cp.Variable((N, M))
    t = cp.Variable()

    Gz1 = [cp.Variable(obj_mat_shape, symmetric=True) for _ in range(N)]
    Fz1 = [cp.Variable(obj_vec_shape) for _ in range(N)]
    Gz2 = [cp.Variable(obj_mat_shape, symmetric=True) for _ in range(N)]
    Fz2 = [cp.Variable(obj_vec_shape) for _ in range(N)]

    obj = lambd * eps + 1 / N * cp.sum(s)
    constraints = [y1 >= 0, y2 >= 0]

    G_preconditioner = np.diag(DR.canon.precond_inv[0])
    F_preconditioner = DR.canon.precond_inv[1]

    for i in range(N):
        G_sample, F_sample = G_params[i], F_params[i]
        constraints += [t - c_vals.T @ y1[i] - cp.trace(G_sample @ Gz1[i]) - F_sample.T @ Fz1[i] <= s[i]]
        constraints += [-(alpha_inv - 1) * t - c_vals.T @ y2[i] - cp.trace(G_sample @ Gz2[i]) - F_sample.T @ Fz2[i] <= s[i]]

        constraints += [cp.SOC(lambd, cp.hstack([cp.vec( G_preconditioner@Gz1[i]@G_preconditioner, order='F'), cp.multiply(F_preconditioner**2, Fz1[i])]))]
        constraints += [cp.SOC(lambd, cp.hstack([cp.vec( G_preconditioner@Gz2[i]@G_preconditioner, order='F'), cp.multiply(F_preconditioner**2, Fz2[i])]))]

        y1A_adj = 0
        y2A_adj = 0
        y1b_adj = 0
        y2b_adj = 0

        for m in range(M):
            Am = A_vals[m]
            bm = b_vals[m]

            y1A_adj = y1A_adj + y1[i, m] * Am
            y2A_adj = y2A_adj + y2[i, m] * Am

            y1b_adj = y1b_adj + y1[i, m] * bm
            y2b_adj = y2b_adj + y2[i, m] * bm

        constraints += [y1A_adj - Gz1[i] >> 0]
        constraints += [y1b_adj - Fz1[i] == 0]
        constraints += [y2A_adj - Gz2[i] - alpha_inv * A_obj >> 0]
        constraints += [y2b_adj - Fz2[i] - alpha_inv * b_obj == 0]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    
    # Parameters list: [G_0, G_1, ..., G_{N-1}, F_0, F_1, ..., F_{N-1}]
    all_params = G_params + F_params
    return CvxpyLayerWithDefaults(prob, parameters=all_params, variables=[lambd, s])


@jax.jit
def dro_pep_obj_jax(eps, lambd_star, s_star):
    N = s_star.shape[0]
    return lambd_star * eps + 1 / N * jnp.sum(s_star)


def create_lpep_cp_layer(DR):
    A_vals = DR.canon.A_vals
    b_vals = DR.canon.b_vals
    c_vals = DR.canon.c_vals
    A_obj = DR.canon.A_obj
    b_obj = DR.canon.b_obj

    M = A_vals.shape[0]
    obj_mat_shape = A_obj.shape
    obj_vec_shape = b_obj.shape

    lambd = cp.Variable()
    y = cp.Variable(M)

    obj = -c_vals.T @ y
    constraints = [y >= 0]

    LstarG = 0
    LstarF = 0
    for m in range(M):
        Am = A_vals[m]
        bm = b_vals[m]
        LstarG = LstarG + y[m] * Am
        LstarF = LstarF + y[m] * bm

    constraints += [LstarG - A_obj >> 0]
    constraints += [LstarF - b_obj == 0]


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
