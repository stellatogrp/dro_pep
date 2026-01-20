"""
Logistic Regression learning experiment pipeline.

Mirrors the quad.py pipeline but for gradient descent / Nesterov FGM on 
regularized logistic regression problems.
"""
import diffcp_patch  # Apply COO -> CSC fix for diffcp
import jax
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
import logging
import time
import cvxpy as cp
from functools import partial

from learning.pep_construction import (
    construct_gd_pep_data,
    construct_fgm_pep_data,
)
from learning.adam_optimizers import AdamWMin
from learning.trajectories_gd_fgm import (
    compute_preconditioner_from_samples,
    dro_pep_obj_jax,
)
from learning.trajectories_logreg_gd_fgm import (
    logreg_pep_obj,
    create_logreg_traj_fn_gd,
    create_logreg_traj_fn_fgm,
)
from learning.cvxpylayers_setup import (
    create_full_dro_exp_layer,
    create_full_dro_cvar_layer,
    create_full_pep_layer,
)
from learning.jax_scs_layer import dro_scs_solve
from learning.acceleration_stepsizes import (
    jax_get_nesterov_fgm_beta_sequence,
)

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=6, suppress=True)

log = logging.getLogger(__name__)


# =============================================================================
# A Matrix Generation (fixed per experiment)
# =============================================================================

def generate_A_logreg(seed, N_data, n, A_std):
    """
    Generate A matrix for logistic regression.
    
    A[:, :-1] ~ Normal(0, A_std), A[:, -1] = 1 (intercept column)
    
    Args:
        seed: Random seed for reproducibility
        N_data: Number of data points (rows)
        n: Dimension of beta (columns)
        A_std: Standard deviation for A entries
        
    Returns:
        A: (N_data, n) numpy array with last column = 1
    """
    np.random.seed(seed)
    A = np.random.normal(size=(N_data, n - 1), scale=A_std)
    A = np.hstack([A, np.ones((N_data, 1))])
    return A


# =============================================================================
# Label Generation (JAX - JIT compiled)
# =============================================================================

@partial(jax.jit, static_argnames=['n', 'p_beta_nonzero'])
def generate_true_beta_jax(key, n, p_beta_nonzero, beta_scale):
    """
    Generate sparse true beta vector.
    """
    key1, key2 = jax.random.split(key)
    beta = jax.random.uniform(key1, (n,), minval=-beta_scale, maxval=beta_scale)
    mask = jax.random.bernoulli(key2, p=p_beta_nonzero, shape=(n,)).astype(jnp.float64)
    return beta * mask


@jax.jit
def generate_labels_jax(key, A, beta_true, eps_std):
    """
    Generate binary labels from A @ beta + noise.
    y = 1 if A @ beta + eps > 0 else 0
    """
    N_data = A.shape[0]
    noise = eps_std * jax.random.normal(key, (N_data,))
    Abeta_noise = A @ beta_true + noise
    return jnp.where(Abeta_noise > 0, 1.0, 0.0)


@partial(jax.jit, static_argnames=['n', 'p_beta_nonzero'])
def generate_single_logreg_labels_jax(key, A, n, p_beta_nonzero, beta_scale, eps_std):
    """
    Generate a single logreg problem instance (labels for fixed A).
    """
    key1, key2 = jax.random.split(key)
    beta_true = generate_true_beta_jax(key1, n, p_beta_nonzero, beta_scale)
    b = generate_labels_jax(key2, A, beta_true, eps_std)
    return b


def generate_batch_logreg_labels_jax(key, A, N_batch, n, p_beta_nonzero, beta_scale, eps_std):
    """
    Generate a batch of logreg problem labels.
    """
    keys = jax.random.split(key, N_batch)
    generate_one = partial(
        generate_single_logreg_labels_jax,
        A=A, n=n, p_beta_nonzero=p_beta_nonzero,
        beta_scale=beta_scale, eps_std=eps_std
    )
    b_batch = jax.vmap(generate_one)(keys)
    return b_batch


# =============================================================================
# Logistic Regression Solution (using CVXPY - not JIT compatible)
# =============================================================================

def solve_logreg_cvxpy(A_np, b_np, delta):
    """
    Solve single logistic regression problem using CVXPY.
    
    min 1/m * sum(-y_i * (A @ beta)_i + log(1 + exp((A @ beta)_i))) + delta/2 * ||beta||^2
    """
    m, n = A_np.shape
    beta = cp.Variable(n)
    
    log_likelihood = cp.sum(
        cp.multiply(b_np, A_np @ beta) - cp.logistic(A_np @ beta)
    )
    obj = -1 / m * log_likelihood + 0.5 * delta * cp.sum_squares(beta)
    
    problem = cp.Problem(cp.Minimize(obj))
    problem.solve(solver='CLARABEL')
    
    x_opt = beta.value
    f_opt = problem.value
    
    return x_opt, f_opt


def solve_batch_logreg_cvxpy(A_np, b_batch_np, delta):
    """
    Solve batch of logistic regression problems.
    """
    N = b_batch_np.shape[0]
    n = A_np.shape[1]
    
    x_opt_batch = np.zeros((N, n))
    f_opt_batch = np.zeros(N)
    R_max = 0.0
    
    for i in range(N):
        x_opt, f_opt = solve_logreg_cvxpy(A_np, b_batch_np[i], delta)
        x_opt_batch[i] = x_opt
        f_opt_batch[i] = f_opt
        R = np.linalg.norm(x_opt)
        R_max = max(R_max, R)
    
    return x_opt_batch, f_opt_batch, R_max


# =============================================================================
# Compute Problem Parameters
# =============================================================================

def compute_logreg_L(A, delta):
    """
    Compute Lipschitz constant L for logistic regression.
    L = lambda_max(A^T A) / (4 * m) + delta
    """
    m, n = A.shape
    ATA = A.T @ A
    eigvals = jnp.linalg.eigvalsh(ATA)
    lambd_max = jnp.max(eigvals)
    L = lambd_max / (4 * m) + delta
    mu = delta
    return float(L), float(mu)


def compute_sample_radius_logreg(cfg, A_jax, A_np):
    """
    Compute maximum radius R by solving many logreg problems.
    """
    log.info(f"Computing R from {cfg.R_sample_size} samples...")
    
    key = jax.random.PRNGKey(cfg.seed)
    b_batch = generate_batch_logreg_labels_jax(
        key, A_jax, cfg.R_sample_size,
        cfg.n, cfg.p_beta_nonzero, cfg.beta_scale, cfg.eps_std
    )
    b_batch_np = np.array(b_batch)
    
    _, _, R_max = solve_batch_logreg_cvxpy(A_np, b_batch_np, cfg.delta)
    
    log.info(f"Computed R = {R_max:.6f}")
    return R_max


# =============================================================================
# Sample x0 from Disk (JAX - JIT compiled)
# =============================================================================

@partial(jax.jit, static_argnames=['n'])
def sample_x0_disk_jax(key, n, R):
    """
    Sample initial point uniformly from disk of radius R centered at 0.
    """
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, (n,))
    x = x / jnp.linalg.norm(x)
    dist = jax.random.uniform(key2) ** (1.0 / n)
    return R * dist * x


def sample_batch_x0_disk_jax(key, N, n, R):
    """
    Sample batch of initial points uniformly from disk.
    """
    keys = jax.random.split(key, N)
    sample_one = partial(sample_x0_disk_jax, n=n, R=R)
    return jax.vmap(sample_one)(keys)


# =============================================================================
# Setup Function
# =============================================================================

def setup_logreg_problem(cfg):
    """
    Set up the logistic regression problem: generate A, compute L/mu, optionally compute R.
    """
    log.info("Setting up logistic regression problem...")
    
    log.info(f"Generating A matrix with seed={cfg.A_seed}")
    A_np = generate_A_logreg(cfg.A_seed, cfg.N_data, cfg.n, cfg.A_std)
    A_jax = jnp.array(A_np)
    
    L_cfg = cfg.get('L', None)
    if L_cfg is None:
        L, mu = compute_logreg_L(A_jax, cfg.delta)
        log.info(f"Computed L = {L:.6f}, mu = {mu:.6f}")
    else:
        L = float(L_cfg)
        mu = float(cfg.delta)
        log.info(f"Using provided L = {L:.6f}, mu = {mu:.6f}")
    
    R_cfg = cfg.get('R', None)
    if R_cfg is None:
        R = compute_sample_radius_logreg(cfg, A_jax, A_np)
    else:
        R = float(R_cfg)
        log.info(f"Using provided R = {R:.6f}")
    
    problem_data = {
        'A_jax': A_jax,
        'A_np': A_np,
        'L': L,
        'mu': mu,
        'R': R,
        'delta': cfg.delta,
        'N_data': cfg.N_data,
    }
    
    return problem_data


# =============================================================================
# Sample Batch for SGD Iteration
# =============================================================================

def sample_logreg_batch(key, A_jax, A_np, N, n, p_beta_nonzero, beta_scale, eps_std, delta):
    """
    Sample a batch of logreg problems for one SGD iteration.
    
    Returns:
        b_batch: (N, m) JAX array of label vectors
        x_opt_batch: (N, n) JAX array of optimal solutions
        f_opt_batch: (N,) JAX array of optimal objectives
    """
    b_batch = generate_batch_logreg_labels_jax(
        key, A_jax, N, n, p_beta_nonzero, beta_scale, eps_std
    )
    
    b_batch_np = np.array(b_batch)
    x_opt_batch_np, f_opt_batch_np, _ = solve_batch_logreg_cvxpy(A_np, b_batch_np, delta)
    
    return b_batch, jnp.array(x_opt_batch_np), jnp.array(f_opt_batch_np)


# =============================================================================
# PEP Data Construction Functions
# =============================================================================

def pep_data_fn_gd(stepsizes, mu, L, R, K_max, pep_obj):
    """PEP data construction function for gradient descent."""
    t = stepsizes[0]
    return construct_gd_pep_data(t, mu, L, R, K_max, pep_obj)


def pep_data_fn_fgm(stepsizes, mu, L, R, K_max, pep_obj):
    """PEP data construction function for Nesterov FGM."""
    t, beta = stepsizes[0], stepsizes[1]
    return construct_fgm_pep_data(t, beta, mu, L, R, K_max, pep_obj)


# =============================================================================
# L2O Pipeline for Logistic Regression
# =============================================================================

def l2o_logreg_pipeline(stepsizes, A, b_batch, z0_batch, x_opt_batch, f_opt_batch, 
                        delta, K_max, traj_fn, pep_obj, risk_type, alpha=0.1):
    """
    L2O pipeline for logistic regression without DRO SDP.
    
    Computes PEP objectives for each sample in the batch and returns a risk measure.
    """
    batch_pep_obj_func = jax.vmap(
        lambda b, z0, x_opt, f_opt: logreg_pep_obj(
            stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, traj_fn, pep_obj
        ),
        in_axes=(0, 0, 0, 0)
    )
    pep_objs = batch_pep_obj_func(b_batch, z0_batch, x_opt_batch, f_opt_batch)
    
    if risk_type == 'expectation':
        return jnp.mean(pep_objs)
    elif risk_type == 'cvar':
        N = pep_objs.shape[0]
        k = max(int(np.ceil(alpha * N)), 1)
        sorted_objs = jnp.sort(pep_objs)[::-1]
        return jnp.mean(sorted_objs[:k])
    else:
        raise ValueError(f"Unknown risk_type: {risk_type}")


# =============================================================================
# DataFrame Building for CSV Logging
# =============================================================================

def build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_t, has_beta, all_losses=None, all_times=None):
    """Build a DataFrame from stepsizes history for CSV saving."""
    data = {'iteration': list(range(len(all_stepsizes_vals)))}
    
    if all_losses is not None:
        padded_losses = list(all_losses) + [None] * (len(all_stepsizes_vals) - len(all_losses))
        data['loss'] = [float(l) if l is not None else float('nan') for l in padded_losses]
    
    if all_times is not None:
        padded_times = list(all_times) + [None] * (len(all_stepsizes_vals) - len(all_times))
        data['iter_time'] = [float(t) if t is not None else float('nan') for t in padded_times]
    
    if is_vector_t:
        for k in range(K_max):
            data[f't{k}'] = [float(ss[0][k]) for ss in all_stepsizes_vals]
    else:
        data['t'] = [float(ss[0]) for ss in all_stepsizes_vals]
    
    if has_beta:
        for k in range(K_max):
            data[f'beta{k}'] = [float(ss[1][k]) for ss in all_stepsizes_vals]
    
    return pd.DataFrame(data)


# =============================================================================
# SGD Run Function for Single K
# =============================================================================

def run_sgd_for_K(cfg, K_max, key, problem_data, t_init, 
                   sgd_iters, eta_t,
                   eps, alpha, alg, optimizer_type,
                   csv_path):
    """
    Run SGD for a specific K_max value.
    
    Minimizes over stepsizes only. Problem data (b, x_opt, f_opt) is resampled each iteration.
    Saves progress to csv_path after each iteration.
    """
    log.info(f"=== Running SGD for K={K_max} ===")
    
    # Extract problem data
    A_jax = problem_data['A_jax']
    A_np = problem_data['A_np']
    L = problem_data['L']
    mu = problem_data['mu']
    R = problem_data['R']
    delta = problem_data['delta']
    n = cfg.n
    N_val = cfg.N
    
    # Select trajectory function based on algorithm
    if alg == 'vanilla_gd':
        traj_fn = create_logreg_traj_fn_gd(delta)
    elif alg == 'nesterov_fgm':
        traj_fn = create_logreg_traj_fn_fgm(delta)
    else:
        log.error(f"Algorithm '{alg}' is not implemented.")
        raise ValueError(f"Unknown algorithm: {alg}")
    
    # Projection function for stepsizes
    def proj_stepsizes(stepsizes):
        return [jax.nn.relu(s) for s in stepsizes]
    
    # Sample function
    def sample_batch(sample_key):
        k1, k2, next_key = jax.random.split(sample_key, 3)
        
        # Sample b and solve for optimal
        b_batch, x_opt_batch, f_opt_batch = sample_logreg_batch(
            k1, A_jax, A_np, N_val,
            cfg.n, cfg.p_beta_nonzero, cfg.beta_scale, cfg.eps_std, delta
        )
        
        # Sample z0 in shifted coordinates (within ball of radius R)
        z0_batch = sample_batch_x0_disk_jax(k2, N_val, n, R)
        
        return next_key, b_batch, z0_batch, x_opt_batch, f_opt_batch
    
    # Define grad_fn based on learning framework
    learning_framework = cfg.learning_framework
    
    if learning_framework == 'ldro-pep':
        # Dimensions
        dimG = K_max + 2
        dimF = K_max + 1
        n_points = K_max + 1
        
        mat_shape = (dimG, dimG)
        vec_shape = (dimF,)
        
        M_interp = (n_points + 1) * n_points
        M = M_interp + 1
        
        c_vals_init = np.zeros(M)
        c_vals_init[-1] = -R ** 2
        
        # Set up algorithm-specific functions
        if alg == 'vanilla_gd':
            pep_data_fn = pep_data_fn_gd
            stepsizes_for_precond = (t_init,)
        elif alg == 'nesterov_fgm':
            pep_data_fn = pep_data_fn_fgm
            beta_init = jax_get_nesterov_fgm_beta_sequence(mu, L, K_max)
            stepsizes_for_precond = (t_init, beta_init)
        else:
            raise ValueError(f"Unknown algorithm: {alg}")
        
        # Compute preconditioner from first sample batch
        key, b_precond, z0_precond, x_opt_precond, f_opt_precond = sample_batch(key)
        
        batch_GF_func = jax.vmap(
            lambda b, z0, x_opt, f_opt: traj_fn(stepsizes_for_precond, A_jax, b, z0, x_opt, f_opt, K_max, return_Gram_representation=True),
            in_axes=(0, 0, 0, 0)
        )
        G_precond_batch, F_precond_batch = batch_GF_func(b_precond, z0_precond, x_opt_precond, f_opt_precond)
        
        precond_type = cfg.get('precond_type', 'average')
        precond_inv = compute_preconditioner_from_samples(
            np.array(G_precond_batch), np.array(F_precond_batch), precond_type=precond_type
        )
        log.info(f'Computed preconditioner from {N_val} samples using type: {precond_type}')
        
        dro_canon_backend = cfg.get('dro_canon_backend', 'manual_jax')
        log.info(f'Using DRO canon backend: {dro_canon_backend}')
        
        if dro_canon_backend == 'manual_jax':
            precond_inv_jax = (jnp.array(precond_inv[0]), jnp.array(precond_inv[1]))
            risk_type = 'cvar' if cfg.dro_obj == 'cvar' else 'expectation'
            
            def manual_jax_pipeline(stepsizes, b_batch, z0_batch, x_opt_batch, f_opt_batch):
                """Full DRO pipeline using manual JAX canonicalization."""
                batch_GF_func = jax.vmap(
                    lambda b, z0, x_opt, f_opt: traj_fn(stepsizes, A_jax, b, z0, x_opt, f_opt, K_max, return_Gram_representation=True),
                    in_axes=(0, 0, 0, 0)
                )
                G_batch, F_batch = batch_GF_func(b_batch, z0_batch, x_opt_batch, f_opt_batch)
                
                pep_data = pep_data_fn(stepsizes, mu, L, R, K_max, cfg.pep_obj)
                A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]
                
                return dro_scs_solve(
                    A_obj, b_obj, A_vals, b_vals, c_vals,
                    G_batch, F_batch,
                    eps, precond_inv_jax,
                    risk_type=risk_type,
                    alpha=alpha,
                )
            
            value_and_grad_fn = jax.value_and_grad(manual_jax_pipeline, argnums=0)
            full_dro_layer = None
            log.info(f'Using manual JAX pipeline with risk_type={risk_type}')
            
        elif dro_canon_backend == 'cvxpylayers':
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
            
            def cvxpylayers_pipeline(stepsizes, b_batch, z0_batch, x_opt_batch, f_opt_batch):
                """Full DRO pipeline using cvxpylayers."""
                batch_GF_func = jax.vmap(
                    lambda b, z0, x_opt, f_opt: traj_fn(stepsizes, A_jax, b, z0, x_opt, f_opt, K_max, return_Gram_representation=True),
                    in_axes=(0, 0, 0, 0)
                )
                G_batch, F_batch = batch_GF_func(b_batch, z0_batch, x_opt_batch, f_opt_batch)
                
                pep_data = pep_data_fn(stepsizes, mu, L, R, K_max, cfg.pep_obj)
                A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
                
                N = G_batch.shape[0]
                M = A_vals.shape[0]
                
                params_list = (
                    [A_vals[m] for m in range(M)] +
                    [b_vals[m] for m in range(M)] +
                    [A_obj, b_obj] +
                    [G_batch[i] for i in range(N)] +
                    [F_batch[i] for i in range(N)]
                )
                
                (lambd_star, s_star) = full_dro_layer(*params_list)
                loss = dro_pep_obj_jax(eps, lambd_star, s_star)
                return loss
            
            value_and_grad_fn = jax.value_and_grad(cvxpylayers_pipeline, argnums=0)
            log.info(f'Using cvxpylayers pipeline')
        else:
            raise ValueError(f"Unknown dro_canon_backend: {dro_canon_backend}")
        
    elif learning_framework == 'l2o':
        risk_type = cfg.dro_obj  # 'expectation' or 'cvar'
        
        def l2o_wrapper(stepsizes, b_batch, z0_batch, x_opt_batch, f_opt_batch):
            return l2o_logreg_pipeline(
                stepsizes, A_jax, b_batch, z0_batch, x_opt_batch, f_opt_batch,
                delta, K_max, traj_fn, cfg.pep_obj, risk_type, alpha
            )
        
        value_and_grad_fn = jax.value_and_grad(l2o_wrapper, argnums=0)
        full_dro_layer = None
        pep_data_fn = None
    else:
        raise ValueError(f"Unknown learning_framework: {learning_framework}")
    
    # Initialize stepsizes based on algorithm
    if alg == 'vanilla_gd':
        stepsizes = (t_init,)
    elif alg == 'nesterov_fgm':
        beta_init = jax_get_nesterov_fgm_beta_sequence(mu, L, K_max)
        stepsizes = (t_init, beta_init)
    
    t = stepsizes[0]
    is_vector_t = jnp.ndim(t) > 0
    has_beta = len(stepsizes) > 1
    
    all_stepsizes_vals = [stepsizes]
    all_losses = []
    all_times = []
    
    # Determine update mask for learn_beta
    learn_beta = cfg.get('learn_beta', True)
    if has_beta and not learn_beta:
        update_mask = [True, False]
        log.info(f'learn_beta=False: beta will NOT be updated during optimization')
    else:
        update_mask = None
    
    # Initialize optimizer if needed
    optimizer = None
    if optimizer_type == "adamw":
        optimizer = AdamWMin(
            x_params=[jnp.array(s) for s in stepsizes],
            lr=eta_t,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            update_mask=update_mask,
        )
    
    # SGD iterations
    for iter_num in range(sgd_iters):
        t = stepsizes[0]
        t_log = f'{t:.5f}' if not is_vector_t else '[' + ', '.join(f'{x:.5f}' for x in t.tolist()) + ']'
        if has_beta:
            beta = stepsizes[1]
            beta_log = '[' + ', '.join(f'{x:.5f}' for x in beta.tolist()) + ']'
            log.info(f'K={K_max}, iter={iter_num}, t={t_log}, beta={beta_log}')
        else:
            log.info(f'K={K_max}, iter={iter_num}, t={t_log}')
        
        # Sample new batch
        key, b_batch, z0_batch, x_opt_batch, f_opt_batch = sample_batch(key)
        
        # Compute loss and gradients
        iter_start_time = time.time()
        loss, d_stepsizes = value_and_grad_fn(stepsizes, b_batch, z0_batch, x_opt_batch, f_opt_batch)
        iter_time = time.time() - iter_start_time
        
        log.info(f'  loss: {float(loss):.6f}, iter_time: {iter_time:.3f}s')
        
        all_losses.append(float(loss))
        all_times.append(iter_time)
        
        # SGD step
        if optimizer_type == "vanilla_sgd":
            if update_mask is None:
                stepsizes = tuple(jax.nn.relu(s - eta_t * ds) for s, ds in zip(stepsizes, d_stepsizes))
            else:
                stepsizes = tuple(
                    jax.nn.relu(s - eta_t * ds) if should_update else s 
                    for s, ds, should_update in zip(stepsizes, d_stepsizes, update_mask)
                )
        elif optimizer_type == "adamw":
            x_params = [jnp.array(s) for s in stepsizes]
            grads_x = list(d_stepsizes)
            x_new = optimizer.step(
                x_params=x_params,
                grads_x=grads_x,
                proj_x_fn=proj_stepsizes,
            )
            stepsizes = tuple(x_new)
        elif optimizer_type == "sgd_wd":
            weight_decay = cfg.get('weight_decay', 1e-2)
            if update_mask is None:
                stepsizes = tuple(jax.nn.relu(s - eta_t * (ds + weight_decay * s)) for s, ds in zip(stepsizes, d_stepsizes))
            else:
                stepsizes = tuple(
                    jax.nn.relu(s - eta_t * (ds + weight_decay * s)) if should_update else s 
                    for s, ds, should_update in zip(stepsizes, d_stepsizes, update_mask)
                )
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
        
        all_stepsizes_vals.append(stepsizes)
        
        # Save progress
        df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_t, has_beta, all_losses, all_times)
        df.to_csv(csv_path, index=False)
    
    # Final save
    df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_t, has_beta, all_losses, all_times)
    t = stepsizes[0]
    t_str = str(t) if is_vector_t else f'{float(t):.6f}'
    log.info(f'K={K_max} complete. Final t={t_str}. Saved to {csv_path}')
    df.to_csv(csv_path, index=False)


# =============================================================================
# GD Run Function for LPEP (no samples, deterministic PEP)
# =============================================================================

def run_gd_for_K_lpep(cfg, K_max, problem_data, t_init, gd_iters, eta_t, alg, csv_path):
    """
    Run gradient descent for learning PEP (lpep) - no samples, no min-max.
    
    Optimizes step sizes to minimize the standard (worst-case) PEP objective
    using SCS solver with cvxpylayers for differentiation.
    """
    log.info(f"=== Running lpep GD for K={K_max} ===")
    
    L = problem_data['L']
    mu = problem_data['mu']
    R = problem_data['R']
    
    sdp_backend = cfg.get('sdp_backend', 'scs')
    if sdp_backend != 'scs':
        raise ValueError(f"Only 'scs' backend is supported for lpep. Got: {sdp_backend}")
    
    # Select PEP data construction function based on algorithm
    if alg == 'vanilla_gd':
        pep_data_fn_local = lambda t, mu, L, R, K, pep_obj: construct_gd_pep_data(t, mu, L, R, K, pep_obj)
    elif alg == 'nesterov_fgm':
        pep_data_fn_local = lambda stepsizes, mu, L, R, K, pep_obj: construct_fgm_pep_data(
            stepsizes[0], stepsizes[1], mu, L, R, K, pep_obj
        )
    else:
        log.error(f"Algorithm '{alg}' is not implemented.")
        raise ValueError(f"Unknown algorithm: {alg}")
    
    # Initialize stepsizes
    is_vector_t = cfg.stepsize_type == 'vector'
    
    if alg == 'vanilla_gd':
        if is_vector_t:
            t = jnp.atleast_1d(jnp.array(t_init))
            if t.ndim == 0 or t.shape[0] == 1:
                t = jnp.full(K_max, float(t_init))
        else:
            t = jnp.array(float(t_init))
        stepsizes = [t]
        has_beta = False
    elif alg == 'nesterov_fgm':
        if is_vector_t:
            t = jnp.atleast_1d(jnp.array(t_init))
            if t.ndim == 0 or t.shape[0] == 1:
                t = jnp.full(K_max, float(t_init))
        else:
            t = jnp.array(float(t_init))
        beta_init = jax_get_nesterov_fgm_beta_sequence(mu, L, K_max)
        stepsizes = [t, beta_init]
        has_beta = True
    
    all_stepsizes_vals = [tuple(stepsizes)]
    all_losses = []
    
    # Create SCS-based PEP layer
    if alg == 'vanilla_gd':
        t = stepsizes[0]
        pep_data = construct_gd_pep_data(t, mu, L, R, K_max, cfg.pep_obj)
    else:
        t, beta = stepsizes[0], stepsizes[1]
        pep_data = construct_fgm_pep_data(t, beta, mu, L, R, K_max, cfg.pep_obj)
    
    A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
    M = A_vals.shape[0]
    mat_shape = (A_obj.shape[0], A_obj.shape[1])
    vec_shape = (b_obj.shape[0],)
    pep_layer = create_full_pep_layer(M, mat_shape, vec_shape)
    log.info(f"Using SCS backend for lpep (M={M}, mat_shape={mat_shape}, vec_shape={vec_shape})")
    
    # Define the PEP loss function
    def pep_loss_fn(stepsizes_list):
        if alg == 'vanilla_gd':
            t = stepsizes_list[0]
            pep_data = construct_gd_pep_data(t, mu, L, R, K_max, cfg.pep_obj)
        else:
            t, beta = stepsizes_list[0], stepsizes_list[1]
            pep_data = construct_fgm_pep_data(t, beta, mu, L, R, K_max, cfg.pep_obj)
        
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        
        M = A_vals.shape[0]
        params_list = (
            [A_vals[m] for m in range(M)] +
            [b_vals[m] for m in range(M)] +
            [c_vals, A_obj, b_obj]
        )
        (G_opt, F_opt) = pep_layer(*params_list)
        loss = jnp.trace(A_obj @ G_opt) + jnp.dot(b_obj, F_opt)
        return loss
    
    value_and_grad_fn = jax.value_and_grad(pep_loss_fn)
    
    # Determine update mask for learn_beta
    learn_beta = cfg.get('learn_beta', True)
    if has_beta and not learn_beta:
        update_mask = [True, False]
        log.info(f'learn_beta=False: beta will NOT be updated during optimization')
    else:
        update_mask = None
    
    # Initialize AdamWMin optimizer
    optimizer = AdamWMin(
        x_params=stepsizes,
        lr=eta_t,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=getattr(cfg, 'weight_decay', 0.0),
        update_mask=update_mask,
    )
    
    def proj_nonneg(params):
        return [jnp.maximum(p, 1e-6) for p in params]
    
    # GD iterations
    for iter_num in range(gd_iters):
        t = stepsizes[0]
        if is_vector_t:
            t_log = '[' + ', '.join(f'{x:.5f}' for x in t.tolist()) + ']'
        else:
            t_log = f'{float(t):.5f}'
        if has_beta:
            beta = stepsizes[1]
            beta_log = '[' + ', '.join(f'{x:.5f}' for x in beta.tolist()) + ']'
            log.info(f'K={K_max}, iter={iter_num}, t={t_log}, beta={beta_log}')
        else:
            log.info(f'K={K_max}, iter={iter_num}, t={t_log}')
        
        current_loss, grads = value_and_grad_fn(stepsizes)
        log.info(f'  PEP loss: {current_loss:.6f}')
        
        all_losses.append(float(current_loss))
        
        if any(jnp.any(jnp.isnan(g)) for g in grads):
            log.warning(f'NaN gradients at iter {iter_num}, skipping update')
            all_stepsizes_vals.append(tuple(stepsizes))
            continue
        
        stepsizes = optimizer.step(stepsizes, grads, proj_x_fn=proj_nonneg)
        
        all_stepsizes_vals.append(tuple(stepsizes))
        
        df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_t, has_beta, all_losses)
        df.to_csv(csv_path, index=False)
    
    # Final save
    df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_t, has_beta, all_losses)
    t = stepsizes[0]
    if is_vector_t:
        t_str = '[' + ', '.join(f'{x:.6f}' for x in t.tolist()) + ']'
    else:
        t_str = f'{float(t):.6f}'
    log.info(f'K={K_max} complete. Final t={t_str}. Saved to {csv_path}')
    df.to_csv(csv_path, index=False)


# =============================================================================
# Main Entry Point
# =============================================================================

def logreg_run(cfg):
    """
    Main entry point for logistic regression learning experiment.
    
    Loops over K_max values, runs SGD/GD for each K, and saves per-K progress CSV.
    Algorithm and optimizer type are selected via config.
    """
    log.info("=" * 60)
    log.info("Starting Logistic Regression learning experiment")
    log.info("=" * 60)
    log.info(cfg)
    
    # Setup problem
    problem_data = setup_logreg_problem(cfg)
    
    log.info(f"Problem setup complete:")
    log.info(f"  A shape: {problem_data['A_jax'].shape}")
    log.info(f"  L = {problem_data['L']:.6f}")
    log.info(f"  mu = {problem_data['mu']:.6f}")
    log.info(f"  R = {problem_data['R']:.6f}")
    log.info(f"  delta = {problem_data['delta']}")
    log.info(f"  N_data = {problem_data['N_data']}")
    
    # Extract config values
    L = problem_data['L']
    mu = problem_data['mu']
    
    # Initialize t: 1/L if mu=0, else 2/(mu+L)
    if mu == 0:
        t_init_scalar = 1.0 / L
    else:
        t_init_scalar = 2.0 / (mu + L)
    log.info(f"Initial step size t: {t_init_scalar}")
    
    # SGD parameters from config
    sgd_iters = cfg.sgd_iters
    eta_t = cfg.eta_t
    eps = cfg.eps
    alpha = cfg.alpha
    
    alg = cfg.alg
    optimizer_type = cfg.optimizer_type
    log.info(f"Algorithm: {alg}, Optimizer type: {optimizer_type}")
    
    # Ensure output directory exists
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Random key for sampling
    seed = cfg.seed
    key = jax.random.PRNGKey(seed)
    
    # Loop over K_max values
    for K in cfg.K_max:
        is_vector = cfg.stepsize_type == "vector"
        
        # Check for invalid combination: nesterov_fgm with silver stepsizes
        if alg == 'nesterov_fgm' and cfg.vector_init == "silver":
            log.error("Silver stepsizes are not compatible with nesterov_fgm. Use 'fixed' vector_init instead.")
            raise ValueError("Silver stepsizes are not compatible with nesterov_fgm algorithm.")
        
        # Determine t_init based on algorithm
        if alg == 'nesterov_fgm':
            t_init = jnp.full(K, 1 / L) if is_vector else 1 / L
        elif is_vector:
            t_init = jnp.full(K, t_init_scalar)
        else:
            t_init = t_init_scalar
        
        # Create output directory for this K
        K_output_dir = os.path.join(output_dir, f"K_{K}")
        os.makedirs(K_output_dir, exist_ok=True)
        csv_path = os.path.join(K_output_dir, "progress.csv")
        
        # Select run function based on learning framework
        learning_framework = cfg.learning_framework
        
        if learning_framework == 'lpep':
            run_gd_for_K_lpep(
                cfg, K, problem_data, t_init, sgd_iters, eta_t,
                alg, csv_path
            )
        else:
            run_sgd_for_K(
                cfg, K, key, problem_data, t_init,
                sgd_iters, eta_t,
                eps, alpha, alg, optimizer_type,
                csv_path
            )
            log.info("=== SGD experiment complete ===")
    
    log.info("=" * 60)
    log.info("Logistic Regression learning experiment complete!")
    log.info("=" * 60)
