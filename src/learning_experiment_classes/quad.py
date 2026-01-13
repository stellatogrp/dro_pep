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
from reformulator.dro_reformulator import DROReformulator
from learning_experiment_classes.pep_construction import (
    construct_gd_pep_data,
    construct_fgm_pep_data,
    pep_data_to_numpy,
)
from learning_experiment_classes.adam_optimizers import AdamWMinMax, AdamMinMax
from learning_experiment_classes.autodiff_setup import (
    problem_data_to_gd_trajectories,
    problem_data_to_nesterov_fgm_trajectories,
    problem_data_to_pep_obj,
    create_full_dro_exp_layer,
    create_full_dro_cvar_layer,
    compute_preconditioner_from_samples,
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


def construct_pep_data_for_gd(mu, L, R, t, K_max, pep_obj):
    """Construct PEP data tuple for gradient descent using JAX.
    
    This replaces the PEPit-based quad_pep_subproblem for gradient descent.
    
    Args:
        mu: Strong convexity parameter
        L: Lipschitz constant of gradient
        R: Initial radius bound
        t: Step size (scalar or vector)
        K_max: Number of iterations
        pep_obj: Objective type ('obj_val', 'grad_sq_norm', 'opt_dist_sq_norm')
        
    Returns:
        pep_data: Tuple for use with DROReformulator
    """
    # Convert t to JAX array if needed
    t_jax = jnp.asarray(t) if not isinstance(t, jnp.ndarray) else t
    
    # Construct PEP data using the JAX function
    pep_data_jax = construct_gd_pep_data(
        t_jax, float(mu), float(L), float(R), K_max, pep_obj
    )
    
    # Convert to numpy for use with canonicalizers
    pep_data_np = pep_data_to_numpy(pep_data_jax)
    
    return pep_data_np


def pep_data_fn_gd(stepsizes, mu, L, R, K_max, pep_obj):
    """PEP data construction function for gradient descent.
    
    Used as pep_data_fn argument to full_SCS_pipeline.
    """
    t = stepsizes[0]
    return construct_gd_pep_data(t, mu, L, R, K_max, pep_obj)


def pep_data_fn_fgm(stepsizes, mu, L, R, K_max, pep_obj):
    """PEP data construction function for Nesterov FGM.
    
    Used as pep_data_fn argument to full_SCS_pipeline.
    """
    t, beta = stepsizes[0], stepsizes[1]
    return construct_fgm_pep_data(t, beta, mu, L, R, K_max, pep_obj)


def full_SCS_pipeline(stepsizes, Q_batch, z0_batch, zs_batch, fs_batch, 
                      mu, L, R, K_max, eps, pep_obj, large_sdp_layer, traj_fn, pep_data_fn):
    """Full DRO pipeline with constraint matrices inside JAX trace.
    
    This function computes both:
    1. Sample trajectories (G_batch, F_batch) from stepsizes
    2. Constraint matrices (A_vals, b_vals, A_obj, b_obj) from stepsizes
    
    Both are passed to the cvxpylayer, enabling full gradient flow from 
    stepsizes through both paths.
    
    Algorithm-agnostic: works for both GD and FGM by accepting appropriate
    trajectory and PEP construction functions.
    
    Args:
        stepsizes: Tuple of step size arrays (e.g., (t,) for GD, (t, beta) for FGM)
        Q_batch: Batch of quadratic matrices (N, d, d)
        z0_batch: Batch of initial points (N, d)
        zs_batch: Batch of optimal points (N, d)
        fs_batch: Batch of optimal function values (N,)
        mu: Strong convexity parameter
        L: Lipschitz constant
        R: Initial radius bound
        K_max: Number of algorithm iterations
        eps: Wasserstein ball radius
        pep_obj: PEP objective type
        large_sdp_layer: CvxpyLayer created by create_full_dro_exp_layer
        traj_fn: Trajectory function (problem_data_to_gd_trajectories or _fgm_trajectories)
        pep_data_fn: PEP data construction function that takes stepsizes and returns pep_data
    
    Returns:
        Scalar loss value
    """
    # 1. Compute sample Gram matrices (G, F) from trajectories
    batch_GF_func = jax.vmap(
        lambda Q, z0, zs, fs: traj_fn(stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True),
        in_axes=(0, 0, 0, 0)
    )
    G_batch, F_batch = batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)
    
    # 2. Compute constraint matrices from stepsizes (JAX-traced, algorithm-specific)
    pep_data = pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj)
    A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
    
    # 3. Unpack into parameter list for cvxpylayer
    # Order: [A_0, ..., A_{M-1}, b_0, ..., b_{M-1}, A_obj, b_obj, G_0, ..., G_{N-1}, F_0, ..., F_{N-1}]
    N = G_batch.shape[0]
    M = A_vals.shape[0]
    
    params_list = (
        [A_vals[m] for m in range(M)] +           # A_params
        [b_vals[m] for m in range(M)] +           # b_params
        [A_obj, b_obj] +                           # A_obj_param, b_obj_param
        [G_batch[i] for i in range(N)] +          # G_params
        [F_batch[i] for i in range(N)]            # F_params
    )
    
    # 4. Call cvxpylayer and compute loss
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


def run_gd_for_K_lpep(cfg, K_max, t_init, gd_iters, eta_t, 
                      mu_val, L_val, R_val, alg, csv_path):
    """
    Run gradient descent for learning PEP (lpep) - no samples, no min-max.
    
    Optimizes step sizes to minimize the standard (worst-case) PEP objective.
    
    Args:
        cfg: Configuration object
        K_max: Number of algorithm iterations
        t_init: Initial step size (scalar or vector)
        gd_iters: Number of gradient descent iterations
        eta_t: Learning rate for step sizes
        mu_val, L_val, R_val: Problem parameters
        alg: Algorithm name ('vanilla_gd' or 'nesterov_fgm')
        csv_path: Path to save progress CSV
    """
    log.info(f"=== Running lpep GD for K={K_max} ===")
    
    # Select trajectory function based on algorithm
    if alg == 'vanilla_gd':
        traj_fn = problem_data_to_gd_trajectories
    elif alg == 'nesterov_fgm':
        traj_fn = problem_data_to_nesterov_fgm_trajectories
    else:
        log.error(f"Algorithm '{alg}' is not implemented.")
        raise ValueError(f"Unknown algorithm: {alg}")
    
    # Placeholder for lpep-specific CP layer creation
    def get_pep_cp_layer(t_curr):
        """Create cvxpylayer for standard PEP (placeholder).
        
        TODO: Implement this function to create a differentiable layer
        for the standard PEP SDP without samples/DRO.
        
        Args:
            t_curr: Current step size (scalar or vector)
            
        Returns:
            Callable cvxpylayer that takes step sizes and returns PEP objective
        """
        pep_problem = quad_pep_subproblem(
            mu_val, L_val, R_val, t_curr, K_max, 
            cfg.pep_obj, alg, algo, return_problem=True
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
        # TODO: Create and return the cvxpylayer for lpep
        # For now, raise NotImplementedError
        raise NotImplementedError("lpep cvxpylayer not yet implemented. Need to create lpep-specific layer.")
    
    # Initialize stepsizes based on algorithm
    if alg == 'vanilla_gd':
        stepsizes = (t_init,)
    elif alg == 'nesterov_fgm':
        beta_init = jax_get_nesterov_fgm_beta_sequence(mu_val, L_val, K_max)
        stepsizes = (t_init, beta_init)
    
    # Track step size values
    t = stepsizes[0]
    is_vector_t = jnp.ndim(t) > 0
    has_beta = len(stepsizes) > 1
    all_stepsizes_vals = [stepsizes]
    
    # GD iterations (descent only, no ascent)
    for iter_num in range(gd_iters):
        t = stepsizes[0]
        t_log = f'{t:.5f}' if not is_vector_t else '[' + ', '.join(f'{x:.5f}' for x in t.tolist()) + ']'
        if has_beta:
            beta = stepsizes[1]
            beta_log = '[' + ', '.join(f'{x:.5f}' for x in beta.tolist()) + ']'
            log.info(f'K={K_max}, iter={iter_num}, t={t_log}, beta={beta_log}')
        else:
            log.info(f'K={K_max}, iter={iter_num}, t={t_log}')
        
        # Get PEP layer for current stepsizes
        t_for_pep = t.tolist() if is_vector_t else float(t)
        pep_layer = get_pep_cp_layer(t_for_pep)
        
        # TODO: Compute gradients via autodiff of pep_layer
        # For now, this will raise NotImplementedError from get_pep_cp_layer
        
        # Placeholder gradient update (will be replaced with actual autodiff)
        # d_stepsizes = grad_fn(stepsizes, ...)
        # stepsizes = tuple(s - eta_t * ds for s, ds in zip(stepsizes, d_stepsizes))
        
        all_stepsizes_vals.append(stepsizes)
        
        # Save progress
        df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_t, has_beta)
        df.to_csv(csv_path, index=False)
    
    # Final save
    df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_t, has_beta)
    t = stepsizes[0]
    t_str = str(t) if is_vector_t else f'{float(t):.6f}'
    log.info(f'K={K_max} complete. Final t={t_str}. Saved to {csv_path}')
    df.to_csv(csv_path, index=False)


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
    
    # Select trajectory function based on algorithm
    if alg == 'vanilla_gd':
        traj_fn = problem_data_to_gd_trajectories
    elif alg == 'nesterov_fgm':
        traj_fn = problem_data_to_nesterov_fgm_trajectories
    else:
        log.error(f"Algorithm '{alg}' is not implemented.")
        raise ValueError(f"Unknown algorithm: {alg}")
    
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
        # For ldro-pep, we need to create the full_dro_layer once with correct dimensions
        # Dimensions: M constraints, N samples, dimG x dimG matrices, dimF vectors
        # dimG = K_max + 2, dimF = K_max + 1 (from pep_construction.py)
        dimG = K_max + 2
        dimF = K_max + 1
        mat_shape = (dimG, dimG)
        vec_shape = (dimF,)
        
        # M = number of interpolation constraints = (K_max+2) * (K_max+1) + 1 (initial condition)
        # From smooth_strongly_convex_interp: (n_points+1) * n_points where n_points = K_max+1
        n_points = K_max + 1
        M_interp = (n_points + 1) * n_points  # Interpolation constraints
        M = M_interp + 1  # Plus initial condition
        
        # Get c_vals (fixed, don't depend on t)
        # Initial c_vals are all zeros for interpolation + one -R^2 for initial condition
        c_vals_init = np.zeros(M)
        c_vals_init[-1] = -R_val ** 2
        
        # Set up algorithm-specific functions
        if alg == 'vanilla_gd':
            pep_data_fn = pep_data_fn_gd
            stepsizes_for_precond = (t_init,)
        elif alg == 'nesterov_fgm':
            pep_data_fn = pep_data_fn_fgm
            beta_init = jax_get_nesterov_fgm_beta_sequence(mu_val, L_val, K_max)
            stepsizes_for_precond = (t_init, beta_init)
        else:
            raise ValueError(f"Unknown algorithm: {alg}")
        
        # Compute preconditioner from first sample batch
        # Sample a batch and compute trajectories to get G, F statistics
        key, k1, k2 = jax.random.split(key, 3)
        Q_precond_keys = jax.random.split(k1, N_val)
        z0_precond_keys = jax.random.split(k2, N_val)
        Q_precond_batch = get_Q_samples(Q_precond_keys, d_val, mu_val, L_val, M_val)
        z0_precond_batch = get_z0_samples(z0_precond_keys, M_val, R_val)
        zs_precond_batch = jnp.zeros(z0_precond_batch.shape)
        fs_precond_batch = jnp.zeros(N_val)
        
        # Compute G, F for preconditioner calculation using correct stepsizes
        batch_GF_func = jax.vmap(
            lambda Q, z0, zs, fs: traj_fn(stepsizes_for_precond, Q, z0, zs, fs, K_max, return_Gram_representation=True),
            in_axes=(0, 0, 0, 0)
        )
        G_precond_batch, F_precond_batch = batch_GF_func(Q_precond_batch, z0_precond_batch, zs_precond_batch, fs_precond_batch)
        
        # Compute preconditioner based on sample statistics
        precond_type = cfg.get('precond_type', 'average')
        precond_inv = compute_preconditioner_from_samples(
            np.array(G_precond_batch), np.array(F_precond_batch), precond_type=precond_type
        )
        log.info(f'Computed preconditioner from {N_val} samples using type: {precond_type}')
        
        # Create the full DRO layer based on dro_obj type
        if cfg.dro_obj == 'expectation':
            full_dro_layer = create_full_dro_exp_layer(
                M, N_val, mat_shape, vec_shape, mat_shape, vec_shape,
                c_vals_init, precond_inv, eps
            )
        elif cfg.dro_obj == 'cvar':
            full_dro_layer = create_full_dro_cvar_layer(
                M, N_val, mat_shape, vec_shape, mat_shape, vec_shape,
                c_vals_init, precond_inv, eps, alpha
            )
        else:
            raise ValueError(f"Unknown dro_obj: {cfg.dro_obj}")
        
        # Gradient function uses full_SCS_pipeline with all args traced
        grad_fn = jax.grad(full_SCS_pipeline, argnums=(0, 1, 2))
    elif learning_framework == 'l2o':
        grad_fn = jax.grad(l2o_pipeline, argnums=(0, 1, 2))
        full_dro_layer = None  # Not used for l2o
        pep_data_fn = None  # Not used for l2o
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
            # Use full_SCS_pipeline which traces gradients through both G,F and A_vals,b_vals
            d_stepsizes, dQ, dz0 = grad_fn(
                stepsizes, Q_batch, z0_batch, zs_batch, fs_batch,
                mu_val, L_val, R_val, K_max, eps, cfg.pep_obj, full_dro_layer, traj_fn, pep_data_fn
            )
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
