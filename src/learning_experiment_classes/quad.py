import diffcp_patch  # Apply COO -> CSC fix for diffcp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import logging
import time
from functools import partial

# from .utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, nesterov_fgm, generate_trajectories, sample_x0_centered_disk, generate_P_fixed_mu_L
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexQuadraticFunction, SmoothStronglyConvexFunction
from reformulator.dro_reformulator import DROReformulator
from learning_experiment_classes.adam_optimizers import AdamWMinMax, AdamMinMax
from learning_experiment_classes.algorithms_for_pep import (
    gradient_descent,
    nesterov_fgm,
)
from learning_experiment_classes.autodiff_setup import (
    problem_data_to_gd_trajectories,
    problem_data_to_nesterov_fgm_trajectories,
    problem_data_to_pep_obj,
    create_exp_cp_layer,
    create_cvar_cp_layer,
    dro_pep_obj_jax,
)
from learning_experiment_classes.silver_stepsizes import get_strongly_convex_silver_stepsizes
from learning_experiment_classes.acceleration_stepsizes import (
    get_nesterov_fgm_beta_sequence,
    jax_get_nesterov_fgm_beta_sequence,
)
jax.config.update("jax_enable_x64", True)

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


def SCS_pipeline(stepsizes, Q_batch, z0_batch, zs_batch, fs_batch, K_max, eps, large_sdp_layer, traj_fn):
    batch_GF_func = jax.vmap(
        lambda Q, z0, zs, fs: traj_fn(stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True),
        in_axes=(0, 0, 0, 0)
    )
    G_batch, F_batch = batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)
    
    # Unpack batch into list of 2D arrays: [G_0, ..., G_{N-1}, F_0, ..., F_{N-1}]
    N = G_batch.shape[0]
    params_list = [G_batch[i] for i in range(N)] + [F_batch[i] for i in range(N)]
    
    (lambd_star, s_star) = large_sdp_layer(*params_list)
    loss = dro_pep_obj_jax(eps, lambd_star, s_star)
    return loss


def l2o_pipeline(stepsizes, Q_batch, z0_batch, zs_batch, fs_batch, K_max, traj_fn, pep_obj, risk_type, alpha=0.1):
    """Pipeline for learning-to-optimize without DRO SDP.
    
    Computes PEP objectives for each sample in the batch and returns a risk measure.
    
    Args:
        stepsizes: Step size parameters (tuple)
        Q_batch: Batch of Q matrices (N, dim, dim)
        z0_batch: Batch of initial points (N, dim)
        zs_batch: Batch of optimal points (N, dim)
        fs_batch: Batch of optimal function values (N,)
        K_max: Number of algorithm iterations
        traj_fn: Trajectory function (e.g., problem_data_to_gd_trajectories)
        pep_obj: PEP objective type ('obj_val', 'grad_sq_norm', 'opt_dist_sq_norm')
        risk_type: Risk measure type ('expectation' or 'cvar')
        alpha: CVaR confidence level (only used if risk_type='cvar')
    
    Returns:
        Scalar loss value (mean or CVaR of PEP objectives)
    """
    # vmap over the batch to compute PEP objectives for each sample
    batch_pep_obj_func = jax.vmap(
        lambda Q, z0, zs, fs: problem_data_to_pep_obj(
            stepsizes, Q, z0, zs, fs, K_max, traj_fn, pep_obj
        ),
        in_axes=(0, 0, 0, 0)
    )
    pep_objs = batch_pep_obj_func(Q_batch, z0_batch, zs_batch, fs_batch)
    
    if risk_type == 'expectation':
        return jnp.mean(pep_objs)
    elif risk_type == 'cvar':
        # CVaR: expectation of the top alpha fraction of values
        # Note: N and alpha should be static/known at trace time for JAX compatibility
        N = pep_objs.shape[0]
        k = max(int(np.ceil(alpha * N)), 1)  # Use numpy, not jax (static computation)
        sorted_objs = jnp.sort(pep_objs)[::-1]  # Sort descending
        return jnp.mean(sorted_objs[:k])
    else:
        raise ValueError(f"Unknown risk_type: {risk_type}")


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
        t_init_scalar = 1.0 / L_val
    else:
        t_init_scalar = 2.0 / (mu_val + L_val)
    log.info(f"Initial step size t: {t_init_scalar}")
    
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
        # Determine step size initialization based on config
        is_vector = cfg.stepsize_type == "vector"
        
        # Check for invalid combination: nesterov_fgm with silver stepsizes
        if alg == 'nesterov_fgm' and cfg.vector_init == "silver":
            log.error("Silver stepsizes are not compatible with nesterov_fgm. Use 'fixed' vector_init instead.")
            raise ValueError("Silver stepsizes are not compatible with nesterov_fgm algorithm.")
        
        # Determine t_init based on algorithm
        if alg == 'nesterov_fgm':
            # Nesterov FGM uses t = 1/L (scalar or vector of same value)
            t_init = jnp.full(K, 1 / L_val) if is_vector else 1 / L_val
        elif is_vector:
            if cfg.vector_init == "fixed":
                t_init = jnp.full(K, t_init_scalar)
            else:
                t_init = jnp.array(get_strongly_convex_silver_stepsizes(K, mu=mu_val, L=L_val))
        else:
            t_init = t_init_scalar
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


def build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_t, has_beta):
    """Build a DataFrame from stepsizes history for CSV saving."""
    data = {'iteration': list(range(len(all_stepsizes_vals)))}
    
    # Extract t values (first element of each stepsizes tuple)
    if is_vector_t:
        for k in range(K_max):
            data[f't{k}'] = [float(ss[0][k]) for ss in all_stepsizes_vals]
    else:
        data['t'] = [float(ss[0]) for ss in all_stepsizes_vals]
    
    # Extract beta values if present (second element of each stepsizes tuple)
    if has_beta:
        for k in range(K_max):
            data[f'beta{k}'] = [float(ss[1][k]) for ss in all_stepsizes_vals]
    
    return pd.DataFrame(data)

def run_sgda_for_K(cfg, K_max, key, M_val, t_init, 
                   sgda_iters, eta_t, eta_Q, eta_z0,
                   eps, alpha, alg, sgda_type,
                   mu_val, L_val, R_val, N_val, d_val,
                   csv_path):
    """
    Run SGDA for a specific K_max value.
    
    Saves progress to csv_path after each iteration (overwrites to preserve intermediate progress).
    """
    
    # Select algorithm based on alg parameter
    if alg == 'vanilla_gd':
        algo = gradient_descent
        traj_fn = problem_data_to_gd_trajectories
    elif alg == 'nesterov_fgm':
        algo = nesterov_fgm
        traj_fn = problem_data_to_nesterov_fgm_trajectories
    else:
        log.error(f"Algorithm '{alg}' is not implemented.")
        exit(0)
    
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
    def quad_pep_subproblem(mu, L, R, t, k, obj, return_problem=False):
        from PEPit import PEP
        from PEPit.functions import SmoothStronglyConvexFunction
        
        problem = PEP()
        func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        xs = func.stationary_point()
        fs = func(xs)
        x0 = problem.set_initial_point()
        
        # Build stepsizes tuple based on algorithm
        is_vec = jnp.ndim(t) > 0 if hasattr(t, 'ndim') else isinstance(t, (list, tuple))
        if alg == 'vanilla_gd':
            t_val = t if is_vec else float(t)
            params = {'stepsizes': (t_val,), 'K_max': k}
        elif alg == 'nesterov_fgm':
            # For nesterov_fgm, t can be scalar or vector, beta is always vector
            if is_vec:
                t_val = [float(ti) for ti in t]  # Convert vector to list
            else:
                t_val = float(t)
            beta = get_nesterov_fgm_beta_sequence(mu, L, k)
            params = {'stepsizes': (t_val, beta.tolist()), 'K_max': k}
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
            mu_val, L_val, R_val, t_curr, K_max, 
            cfg.pep_obj, return_problem=True
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

        if cfg.dro_obj == 'expectation':
            return DR, create_exp_cp_layer(DR, eps, G_batch.shape, F_batch.shape)
        elif cfg.dro_obj == 'cvar':
            return DR, create_cvar_cp_layer(DR, eps, alpha, G_batch.shape, F_batch.shape)
    
    # Sample functions
    def sample_batch(key):
        key, k1, k2 = jax.random.split(key, 3)
        Q_subkeys = jax.random.split(k1, N_val)
        z0_subkeys = jax.random.split(k2, N_val)
        
        Q_batch = get_Q_samples(Q_subkeys, d_val, mu_val, L_val, M_val)
        z0_batch = get_z0_samples(z0_subkeys, M_val, R_val)
        zs_batch = jnp.zeros(z0_batch.shape)
        fs_batch = jnp.zeros(N_val)
        
        return key, Q_batch, z0_batch, zs_batch, fs_batch
    
    # Batched trajectory and gradient functions (using stepsizes and traj_fn set above)
    def make_batch_GF_func(stepsizes):
        return jax.vmap(
            lambda Q, z0, zs, fs: traj_fn(stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True),
            in_axes=(0, 0, 0, 0)
        )
    
    # Define grad_fn based on learning framework
    learning_framework = cfg.learning_framework
    if learning_framework == 'ldro-pep':
        grad_fn = jax.grad(SCS_pipeline, argnums=(0, 1, 2))
    elif learning_framework == 'l2o':
        grad_fn = jax.grad(l2o_pipeline, argnums=(0, 1, 2))
    else:
        raise ValueError(f"Unknown learning_framework: {learning_framework}")
    
    # Initialize SGDA stepsizes based on algorithm
    if alg == 'vanilla_gd':
        stepsizes = (t_init,)
    elif alg == 'nesterov_fgm':
        beta_init = jax_get_nesterov_fgm_beta_sequence(mu_val, L_val, K_max)
        stepsizes = (t_init, beta_init)
    
    key, Q_batch, z0_batch, zs_batch, fs_batch = sample_batch(key)
    
    # Initialize Q and z0 for ascent (sample one each)
    key, k1, k2 = jax.random.split(key, 3)
    Q_single_keys = jax.random.split(k1, 1)
    z0_single_keys = jax.random.split(k2, 1)
    Q = get_Q_samples(Q_single_keys, d_val, mu_val, L_val, M_val)[0]
    z0 = get_z0_samples(z0_single_keys, M_val, R_val)[0]
    
    # Track all stepsizes values for logging/CSV
    t = stepsizes[0]  # For logging compatibility
    is_vector_t = jnp.ndim(t) > 0
    has_beta = len(stepsizes) > 1
    
    # Store all stepsizes at each iteration
    all_stepsizes_vals = [stepsizes]  # List of tuples
    
    # Initialize optimizer if needed (Adam or AdamW share same interface)
    optimizer = None
    if sgda_type == "adamw":
        optimizer = AdamWMinMax(
            x_params=[jnp.array(s) for s in stepsizes],
            y_params=[Q, z0],
            lr=eta_t,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
    elif sgda_type == "adam":
        optimizer = AdamMinMax(
            x_params=[jnp.array(s) for s in stepsizes],
            y_params=[Q, z0],
            lr=eta_t,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    # SGDA iterations
    for iter_num in range(sgda_iters):
        t = stepsizes[0]  # For logging
        t_log = f'{t:.5f}' if not is_vector_t else '[' + ', '.join(f'{x:.5f}' for x in t.tolist()) + ']'
        if has_beta:
            beta = stepsizes[1]
            beta_log = '[' + ', '.join(f'{x:.5f}' for x in beta.tolist()) + ']'
            log.info(f'K={K_max}, iter={iter_num}, t={t_log}, beta={beta_log}')
        else:
            log.info(f'K={K_max}, iter={iter_num}, t={t_log}')
        
        # Sample new batch
        key, Q_batch, z0_batch, zs_batch, fs_batch = sample_batch(key)
        
        # Get CP layer for current stepsizes (convert JAX array to Python list for PEPit)
        batch_GF_func = make_batch_GF_func(stepsizes)
        G_batch, F_batch = batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)
        # For PEP subproblem: both vanilla_gd and nesterov_fgm can use t (scalar or vector)
        if alg == 'vanilla_gd':
            t_for_pep = t.tolist() if is_vector_t else float(t)
        elif alg == 'nesterov_fgm':
            t_for_pep = t.tolist() if is_vector_t else float(t)
        
        # Compute gradients w.r.t stepsizes tuple, Q_batch, z0_batch
        if learning_framework == 'ldro-pep':
            _, large_sdp_layer = get_cp_layer(G_batch, F_batch, t_for_pep)
            d_stepsizes, dQ, dz0 = grad_fn(stepsizes, Q_batch, z0_batch, zs_batch, fs_batch, K_max, eps, large_sdp_layer, traj_fn)
        elif learning_framework == 'l2o':
            d_stepsizes, dQ, dz0 = grad_fn(stepsizes, Q_batch, z0_batch, zs_batch, fs_batch, K_max, traj_fn, cfg.pep_obj, cfg.dro_obj, alpha=alpha)
        
        # Average gradients over batch
        dQ = jnp.mean(dQ, axis=0)
        dz0 = jnp.mean(dz0, axis=0)
        
        # SGDA step: descent in stepsizes, ascent in Q and z0 with projections
        if sgda_type == "vanilla_sgda":
            # Update all stepsizes in tuple
            stepsizes = tuple(s - eta_t * ds for s, ds in zip(stepsizes, d_stepsizes))
            Q = proj_Q(Q + eta_Q * dQ)
            z0 = proj_z0(z0 + eta_z0 * dz0)
        elif sgda_type in ("adam", "adamw"):
            # Projection function for y params [Q, z0]
            def proj_y_params(y_params):
                Q_proj = proj_Q(y_params[0])
                z0_proj = proj_z0(y_params[1])
                return [Q_proj, z0_proj]
            
            # Use optimizer step (Adam or AdamW share same interface)
            # Convert stepsizes tuple to list of arrays for optimizer
            x_params = [jnp.array(s) for s in stepsizes]
            grads_x = list(d_stepsizes)
            x_new, y_new = optimizer.step(
                x_params=x_params,
                y_params=[Q, z0],
                grads_x=grads_x,
                grads_y=[dQ, dz0],
                proj_y_fn=proj_y_params
            )
            stepsizes = tuple(x_new)
            Q = y_new[0]
            z0 = y_new[1]
        elif sgda_type == "sgda_wd":
            # SGD with weight decay
            weight_decay = cfg.get('weight_decay', 1e-2)
            stepsizes = tuple(s - eta_t * (ds + weight_decay * s) for s, ds in zip(stepsizes, d_stepsizes))
            Q = proj_Q(Q + eta_Q * (dQ - weight_decay * Q))
            z0 = proj_z0(z0 + eta_z0 * (dz0 - weight_decay * z0))
        else:
            raise ValueError(f"Unknown sgda_type: {sgda_type}")
        
        t = stepsizes[0]  # For logging
        all_stepsizes_vals.append(stepsizes)
        
        # Save progress to CSV after each iteration (overwrite to preserve intermediate progress)
        df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_t, has_beta)
        df.to_csv(csv_path, index=False)
    
    # Final save after loop completes
    df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_t, has_beta)
    t = stepsizes[0]
    t_str = str(t) if is_vector_t else f'{float(t):.6f}'
    log.info(f'K={K_max} complete. Final t={t_str}. Saved to {csv_path}')
    df.to_csv(csv_path, index=False)
