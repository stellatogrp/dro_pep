import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import time
from cvxpylayers.jax import CvxpyLayer
from frozendict import frozendict
from functools import partial
from tqdm import trange

# from .utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, nesterov_fgm, generate_trajectories, sample_x0_centered_disk, generate_P_fixed_mu_L
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexQuadraticFunction, SmoothStronglyConvexFunction
from reformulator.dro_reformulator import DROReformulator

log = logging.getLogger(__name__)


def marchenko_pastur(key, d, mu, L, M):
    sigma = (jnp.sqrt(L) + jnp.sqrt(mu)) / 2
    X = jax.random.normal(key, shape=(d, M)) * sigma
    H = X.T @ X / d
    return H


def rejection_sample_single(key, dim, mu, L, M):
    def body_fun(state):
        key, _, _ = state
        key, subkey = jax.random.split(key)
        
        H = marchenko_pastur(subkey, dim, mu, L, M)
        
        eigvals = jnp.real(jnp.linalg.eigvals(H))
        is_valid = (mu <= jnp.min(eigvals)) & (L >= jnp.max(eigvals))
        return (key, H, is_valid)

    def cond_fun(state):
        return ~state[2]

    key, subkey = jax.random.split(key)
    init_H = marchenko_pastur(subkey, dim, mu, L, M)
    init_valid = jnp.array(False) 
    
    final_state = jax.lax.while_loop(cond_fun, body_fun, (key, init_H, init_valid))
    return final_state[1]


@partial(jax.jit, static_argnames=['dim', 'mu', 'L', 'M'])
def get_Q_samples(subkeys, dim, mu, L, M):
    sampler = partial(rejection_sample_single, dim=dim, mu=mu, L=L, M=M)
    return jax.vmap(sampler)(subkeys)


def sample_z0_single(key, d, R):
    k1, k2 = jax.random.split(key)
    
    z = jax.random.normal(k1, shape=(d,))
    z = z / jnp.linalg.norm(z)
    
    u = jax.random.uniform(k2)
    dist = u ** (1.0 / d)
    
    return R * dist * z


@partial(jax.jit, static_argnames=['d'])
def get_z0_samples(subkeys, d, R):
    # Fix d and R using partial, then vmap over keys
    sampler = partial(sample_z0_single, d=d, R=R)
    return jax.vmap(sampler)(subkeys)


@partial(jax.jit, static_argnames=['K_max'])
def problem_data_to_gd_trajectories(t, Q, z0, zs, K_max):
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
    f_stack = f_stack.at[0].set(f(zs))
    f_stack = f_stack.at[1].set(f(z0))

    def body_fun(i, val):
        z, z_stack, g_stack, f_stack = val
        z = z - t * g(z)
        z_stack = z_stack.at[:, i+1].set(z)
        g_stack = g_stack.at[:, i+1].set(g(z))
        f_stack = f_stack.at[i+1].set(f(z))
        return (z, z_stack, g_stack, f_stack)

    init_val = (z0, z_stack, g_stack, f_stack)
    _, z_stack, g_stack, f_stack = \
        jax.lax.fori_loop(1, K_max + 1, body_fun, init_val)

    G_half = jnp.hstack([z_stack[:,:2], g_stack[:,1:]])
    return G_half.T@G_half, f_stack


def create_exp_cp_layer(DR, eps, alpha, G_shape, F_shape):
    G_param = cp.Parameter(G_shape)
    F_param = cp.Parameter(F_shape)

    A_vals = DR.canon.A_vals
    b_vals = DR.canon.b_vals
    c_vals = DR.canon.c_vals
    A_obj = DR.canon.A_obj
    b_obj = DR.canon.b_obj

    N = G_shape[0]
    M = A_vals.shape[0]
    mat_shape = A_obj.shape
    vec_shape = b_obj.shape

    lambd = cp.Variable()
    s = cp.Variable(N)
    y = cp.Variable((N, M))
    Gz = [cp.Variable(mat_shape, symmetric=True) for _ in range(N)]
    Fz = [cp.Variable(vec_shape) for _ in range(N)]

    obj = lambd * eps + 1 / N * cp.sum(s)
    constraints = [y >= 0]

    G_preconditioner = np.diag(DR.canon.precond_inv[0])
    F_preconditioner = DR.canon.precond_inv[1]

    for i in range(N):
        G_sample, F_sample = G_param[i], F_param[i]
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

    return CvxpyLayer(prob, parameters=[G_param, F_param], variables=[lambd, s])


@jax.jit
def dro_pep_obj_jax(eps, lambd_star, s_star):
    N = s_star.shape[0]
    return lambd_star * eps + 1 / N * jnp.sum(s_star)


def SCS_pipeline(t, Q_batch, z0_batch, zs_batch, K_max, eps, alpha, DR, large_sdp_layer):
    batch_GF_func = jax.vmap(problem_data_to_gd_trajectories, in_axes=(None, 0, 0, 0, None))
    G_batch, F_batch = batch_GF_func(t, Q_batch, z0_batch, zs_batch, K_max)
    (lambd_star, s_star) = large_sdp_layer(G_batch, F_batch)
    loss = dro_pep_obj_jax(eps, lambd_star, s_star)
    return loss


# Import optimizers from separate module
from learning_experiment_classes.adam_optimizers import AdamWMinMax, AdamMinMax


def quad_run(cfg):
    """
    Run SGDA experiment for quadratic functions.
    
    Loops over K_max values, runs SGDA for each K, and saves per-K progress CSV.
    Algorithm and SGDA type are selected via config (Slurm job parallelization).
    """
    log.info(cfg)
    
    # Extract config values
    d_val = cfg.dim
    mu_val = cfg.mu
    L_val = cfg.L
    R_val = cfg.R
    N_val = cfg.N
    
    # Compute matrix width for Marchenko-Pastur
    r_val = (np.sqrt(L_val) - np.sqrt(mu_val))**2 / (np.sqrt(L_val) + np.sqrt(mu_val))**2
    M_val = int(np.round(r_val * d_val))
    log.info(f"Precomputed matrix width M: {M_val}")
    
    # Initialize t: 1/L if mu=0, else 2/(mu+L)
    if mu_val == 0:
        t_init = 1.0 / L_val
    else:
        t_init = 2.0 / (mu_val + L_val)
    log.info(f"Initial step size t: {t_init}")
    
    # SGDA parameters from config
    sgda_iters = cfg.sgda_iters
    eta_t = cfg.eta_t
    eta_Q = cfg.eta_Q
    eta_z0 = cfg.eta_z0
    eps = cfg.eps
    alpha = cfg.alpha
    
    # Algorithm and SGDA type (for future extensibility)
    alg = cfg.alg
    sgda_type = cfg.sgda_type
    log.info(f"Algorithm: {alg}, SGDA type: {sgda_type}")
    
    # Ensure output directory exists
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Random key for sampling
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    # Loop over K_max values
    for K in cfg.K_max:
        log.info(f"=== Running SGDA for K={K} ===")
        
        # Create output directory for this K
        K_output_dir = os.path.join(output_dir, f"K_{K}")
        os.makedirs(K_output_dir, exist_ok=True)
        csv_path = os.path.join(K_output_dir, "progress.csv")
        
        # Run SGDA for this K value (CSV saved inside the loop)
        run_sgda_for_K(
            cfg, K, key, M_val, t_init, 
            sgda_iters, eta_t, eta_Q, eta_z0,
            eps, alpha, alg, sgda_type,
            mu_val, L_val, R_val, N_val, d_val,
            csv_path
        )
    
    log.info("=== SGDA experiment complete ===")


def run_sgda_for_K(cfg, K_max, key, M_val, t_init, 
                   sgda_iters, eta_t, eta_Q, eta_z0,
                   eps, alpha, alg, sgda_type,
                   mu_val, L_val, R_val, N_val, d_val,
                   csv_path):
    """
    Run SGDA for a specific K_max value.
    
    Saves progress to csv_path after each iteration (overwrites to preserve intermediate progress).
    """
    from algorithm import gradient_descent
    
    # Projection functions
    @jax.jit
    def proj_z0(v):
        norm = jnp.linalg.norm(v)
        scale = R_val / jnp.maximum(norm, R_val)
        return v * scale

    @jax.jit
    def proj_Q(M):
        evals, evecs = jnp.linalg.eigh(M)
        evals_clipped = jnp.clip(evals, mu_val, L_val)
        return (evecs * evals_clipped) @ evecs.T
    
    # PEP subproblem setup function
    def quad_pep_subproblem(algo, mu, L, R, t, k, obj, return_problem=False):
        from PEPit import PEP
        from PEPit.functions import SmoothStronglyConvexFunction
        
        problem = PEP()
        func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        xs = func.stationary_point()
        fs = func(xs)
        x0 = problem.set_initial_point()
        
        params = {'t': float(t), 'K_max': k}
        problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)
        x_stack, g_stack, f_stack = algo(func, func.gradient, x0, xs, params)
        
        if obj == 'obj_val':
            problem.set_performance_metric(f_stack[-1] - fs)
        elif obj == 'grad_sq_norm':
            problem.set_performance_metric((g_stack[-1]) ** 2)
        elif obj == 'opt_dist_sq_norm':
            problem.set_performance_metric((x_stack[-1] - xs) ** 2)
        
        if return_problem:
            return problem
        
        pepit_tau = problem.solve(wrapper='cvxpy', solver='MOSEK')
        return pepit_tau
    
    # Get CP layer for given batch
    def get_cp_layer(G_batch, F_batch, t_curr):
        pep_problem = quad_pep_subproblem(
            gradient_descent, mu_val, L_val, R_val, t_curr, K_max, 
            cfg.dro_pep_obj, return_problem=True
        )
        mosek_params = {
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': cfg.mosek_tol_dfeas,
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': cfg.mosek_tol_pfeas,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': cfg.mosek_tol_rel_gap,
        }
        pep_problem.solve(
            wrapper='cvxpy',
            solver='MOSEK',
            mosek_params=mosek_params,
            verbose=0,
        )
        samples = [(np.array(G_batch[i]), np.array(F_batch[i])) for i in range(G_batch.shape[0])]
        
        DR = DROReformulator(
            pep_problem,
            samples,
            cfg.dro_obj,
            'cvxpy',
            precond=True,
            precond_type='average',
        )
        return DR, create_exp_cp_layer(DR, eps, alpha, G_batch.shape, F_batch.shape)
    
    # Sample functions
    def sample_batch(key):
        key, k1, k2 = jax.random.split(key, 3)
        Q_subkeys = jax.random.split(k1, N_val)
        z0_subkeys = jax.random.split(k2, N_val)
        
        Q_batch = get_Q_samples(Q_subkeys, d_val, mu_val, L_val, M_val)
        z0_batch = get_z0_samples(z0_subkeys, M_val, R_val)
        zs_batch = jnp.zeros(z0_batch.shape)
        
        return key, Q_batch, z0_batch, zs_batch
    
    # Batched trajectory and gradient functions
    batch_GF_func = jax.vmap(problem_data_to_gd_trajectories, in_axes=(None, 0, 0, 0, None))
    grad_fn = jax.grad(SCS_pipeline, argnums=(0, 1, 2))
    
    # Initialize SGDA variables
    t = t_init
    key, Q_batch, z0_batch, zs_batch = sample_batch(key)
    
    # Initialize Q and z0 for ascent (sample one each)
    key, k1, k2 = jax.random.split(key, 3)
    Q_single_keys = jax.random.split(k1, 1)
    z0_single_keys = jax.random.split(k2, 1)
    Q = get_Q_samples(Q_single_keys, d_val, mu_val, L_val, M_val)[0]
    z0 = get_z0_samples(z0_single_keys, M_val, R_val)[0]
    
    # Track t values for logging
    all_t_vals = [float(t)]
    
    # Initialize optimizer if needed (Adam or AdamW share same interface)
    optimizer = None
    if sgda_type == "adamw":
        optimizer = AdamWMinMax(
            x_params=[jnp.array(t)],
            y_params=[Q, z0],
            lr=eta_t,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0
        )
    elif sgda_type == "adam":
        optimizer = AdamMinMax(
            x_params=[jnp.array(t)],
            y_params=[Q, z0],
            lr=eta_t,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    # SGDA iterations
    for iter_num in range(sgda_iters):
        log.info(f'K={K_max}, iter={iter_num}, t={t:.6f}')
        
        # Sample new batch
        key, Q_batch, z0_batch, zs_batch = sample_batch(key)
        
        # Get CP layer for current t
        G_batch, F_batch = batch_GF_func(t, Q_batch, z0_batch, zs_batch, K_max)
        DR, large_sdp_layer = get_cp_layer(G_batch, F_batch, float(t))
        
        # Compute gradients
        dt, dQ, dz0 = grad_fn(t, Q_batch, z0_batch, zs_batch, K_max, eps, alpha, DR, large_sdp_layer)
        
        # Average gradients over batch
        dQ = jnp.mean(dQ, axis=0)
        dz0 = jnp.mean(dz0, axis=0)
        
        # SGDA step: descent in t, ascent in Q and z0 with projections
        if sgda_type == "vanilla_sgda":
            t = float(t - eta_t * dt)  # Keep t as Python float
            Q = proj_Q(Q + eta_Q * dQ)
            z0 = proj_z0(z0 + eta_z0 * dz0)
        elif sgda_type in ("adam", "adamw"):
            # Projection function for y params [Q, z0]
            def proj_y_params(y_params):
                Q_proj = proj_Q(y_params[0])
                z0_proj = proj_z0(y_params[1])
                return [Q_proj, z0_proj]
            
            # Use optimizer step (Adam or AdamW share same interface)
            x_new, y_new = optimizer.step(
                x_params=[jnp.array(t)],
                y_params=[Q, z0],
                grads_x=[dt],
                grads_y=[dQ, dz0],
                proj_y_fn=proj_y_params
            )
            t = float(x_new[0])
            Q = y_new[0]
            z0 = y_new[1]
        elif sgda_type == "sgda_wd":
            # SGD with weight decay
            # x update: x_{k+1} = x_k - eta_x * (g_x + lambda * x_k)
            # y update: y_{k+1} = Proj(y_k + eta_y * (g_y - lambda * y_k))
            weight_decay = cfg.get('weight_decay', 1e-2)
            t = float(t - eta_t * (dt + weight_decay * t))
            Q = proj_Q(Q + eta_Q * (dQ - weight_decay * Q))
            z0 = proj_z0(z0 + eta_z0 * (dz0 - weight_decay * z0))
        else:
            raise ValueError(f"Unknown sgda_type: {sgda_type}")
        
        all_t_vals.append(float(t))
        
        # Save progress to CSV after each iteration (overwrite to preserve intermediate progress)
        df = pd.DataFrame({
            'iteration': list(range(len(all_t_vals))),
            't': all_t_vals,
        })
        df.to_csv(csv_path, index=False)
    
    # Final save after loop completes
    df = pd.DataFrame({
        'iteration': list(range(len(all_t_vals))),
        't': all_t_vals,
    })
    df.to_csv(csv_path, index=False)
    log.info(f'K={K_max} complete. Final t={t:.6f}. Saved to {csv_path}')


# Required import for os in quad_run
import os
