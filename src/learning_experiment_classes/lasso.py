import diffcp_patch  # Apply COO -> CSC fix for diffcp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import logging
import time
import cvxpy as cp
from functools import partial
from tqdm import trange

from learning.pep_construction_lasso import (
    construct_ista_pep_data,
    construct_fista_pep_data,
    ista_pep_data_to_numpy,
)
from learning.adam_optimizers import AdamWMin
from learning.trajectories_gd_fgm import (
    dro_pep_obj_jax,
)
from learning.trajectories_ista_fista import (
    problem_data_to_ista_trajectories,
    problem_data_to_fista_trajectories,
)
from learning.jax_scs_layer import dro_scs_solve, compute_preconditioner_from_samples

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=6, suppress=True)

log = logging.getLogger(__name__)


def generate_A(seed, m, n):
    """
    Generate sparse A matrix using NumPy.
    
    Args:
        seed: Random seed for reproducibility
        m: Number of rows
        n: Number of columns
        
    Returns:
        A: (m, n) numpy array with columns normalized to unit norm
    """
    np.random.seed(seed)
    A = np.random.normal(scale=1/m, size=(m, n))
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-10)
    A = A / col_norms
    return A

def generate_single_b_jax(key, A, p_xsamp_nonzero):
    """
    Generate a single b vector: b = A @ x_samp + noise.
    
    Args:
        key: JAX random key
        A: (m, n) matrix
        p_xsamp_nonzero: Probability of non-zero entries in x_samp
        noise_eps: Noise level
        
    Returns:
        b: (m,) vector
    """
    m, n = A.shape
    key1, key2 = jax.random.split(key, 2)
    
    # Generate sparse x sample
    x_samp = jax.random.normal(key1, (n,))
    x_mask = jax.random.bernoulli(key2, p=p_xsamp_nonzero, shape=(n,)).astype(jnp.float64)
    x_samp = x_samp * x_mask
    
    return A @ x_samp


def compute_lasso_params(A):
    m, n = A.shape
    ATA = A.T @ A
    eigvals = jnp.linalg.eigvalsh(ATA)
    L = jnp.max(eigvals)
    mu = jnp.min(eigvals) if m >= n else 0.0
    return float(L), float(mu)


def generate_batch_b_jax(key, A, N, p_xsamp_nonzero):
    """
    Generate a batch of b vectors.
    
    Args:
        key: JAX random key
        A: (m, n) matrix
        N: Number of samples
        p_xsamp_nonzero: Probability of non-zero entries in x_samp
        
    Returns:
        b_batch: (N, m) array of b vectors
    """
    keys = jax.random.split(key, N)
    generate_one = partial(generate_single_b_jax, A=A, 
                           p_xsamp_nonzero=p_xsamp_nonzero)
    b_batch = jax.vmap(generate_one)(keys)
    return b_batch


# =============================================================================
# Lasso Solution (using CVXPY - not JIT compatible)
# =============================================================================

def solve_lasso_cvxpy(A_np, b_np, lambd):
    n = A_np.shape[1]
    x = cp.Variable(n)
    obj = 0.5 * cp.sum_squares(A_np @ x - b_np) + lambd * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver='CLARABEL')
    
    x_opt = x.value
    f_opt = prob.value
    
    return x_opt, f_opt


def solve_batch_lasso_cvxpy(A_np, b_batch_np, lambd):
    """
    Solve batch of Lasso problems.
    
    Args:
        A_np: (m, n) numpy array
        b_batch_np: (N, m) numpy array of b vectors
        lambd: L1 regularization parameter
        
    Returns:
        x_opt_batch: (N, n) array of optimal solutions
        f_opt_batch: (N,) array of optimal objective values
        R_max: Maximum radius across all samples
    """
    N = b_batch_np.shape[0]
    n = A_np.shape[1]
    
    x_opt_batch = np.zeros((N, n))
    f_opt_batch = np.zeros(N)
    R_max = 0.0
    
    for i in range(N):
        x_opt, f_opt = solve_lasso_cvxpy(A_np, b_batch_np[i], lambd)
        x_opt_batch[i] = x_opt
        f_opt_batch[i] = f_opt
        R = np.linalg.norm(x_opt)
        R_max = max(R_max, R)
    
    return x_opt_batch, f_opt_batch, R_max


def compute_sample_radius(cfg, A_np):
    log.info(f"Computing R from {cfg.R_sample_size} samples...")
    
    # Generate b samples using the b_seed
    key = jax.random.PRNGKey(cfg.R_seed)
    b_batch = generate_batch_b_jax(
        key, jnp.array(A_np), cfg.R_sample_size,
        cfg.p_xsamp_nonzero
    )
    b_batch_np = np.array(b_batch)
    
    # Solve all Lasso problems
    _, _, R_max = solve_batch_lasso_cvxpy(A_np, b_batch_np, cfg.lambd)
    
    log.info(f"Computed R = {R_max:.6f}")
    return R_max


def build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_gamma, has_beta, all_losses=None, all_times=None):
    """Build a DataFrame from stepsizes history for CSV saving."""
    data = {'iteration': list(range(len(all_stepsizes_vals)))}
    
    # Add losses if provided
    if all_losses is not None:
        # Pad losses with NaN for rows that don't have loss yet
        padded_losses = all_losses + [np.nan] * (len(all_stepsizes_vals) - len(all_losses))
        data['loss'] = padded_losses
    
    # Add times if provided
    if all_times is not None:
        padded_times = all_times + [np.nan] * (len(all_stepsizes_vals) - len(all_times))
        data['iter_time'] = padded_times
    
    # Add gamma values
    if is_vector_gamma:
        for k in range(K_max):
            data[f'gamma_{k}'] = [float(ss[0][k]) for ss in all_stepsizes_vals]
    else:
        # Handle both scalar and 1-element array cases
        def get_scalar_gamma(ss):
            g = ss[0]
            if hasattr(g, 'item') and g.size == 1:
                return float(g.item())
            else:
                return float(g)
        data['gamma'] = [get_scalar_gamma(ss) for ss in all_stepsizes_vals]
    
    # Add beta values if present
    if has_beta:
        for k in range(K_max + 1):  # Beta has K+1 values for FISTA
            if k < K_max + 1 and all(len(ss) > 1 and len(ss[1]) > k for ss in all_stepsizes_vals):
                data[f'beta_{k}'] = [float(ss[1][k]) for ss in all_stepsizes_vals]
    
    return pd.DataFrame(data)


def setup_lasso_problem(cfg):
    log.info("Setting up Lasso problem...")
    
    # Generate A matrix (using NumPy for cvxpy compatibility, then convert)
    log.info(f"Generating A matrix with seed={cfg.A_seed}")
    A_np = generate_A(cfg.A_seed, cfg.m, cfg.n)
    A_jax = jnp.array(A_np)

    log.info(A_np)
    
    # Compute L and mu
    L, mu = compute_lasso_params(A_jax)
    log.info(f"L = {L:.6f}, mu = {mu:.6f}")
    
    # Select R based on strong convexity
    MU_TOL = 1e-6
    is_strongly_convex = mu > MU_TOL
    
    if is_strongly_convex:
        log.info("Problem is strongly convex (mu > 0)")
        R_config = cfg.R_strongcvx
        if R_config is None:
            R = compute_sample_radius(cfg, A_np)
        else:
            R = float(R_config)
            log.info(f"Using provided R_strongcvx = {R}")
    else:
        log.info("Problem is non-strongly convex (mu ~ 0)")
        R_config = cfg.R_nonstrongcvx
        if R_config is None:
            R = compute_sample_radius(cfg, A_np)
        else:
            R = float(R_config)
            log.info(f"Using provided R_nonstrongcvx = {R}")
    
    problem_data = {
        'A_jax': A_jax,
        'A_np': A_np,
        'L': L,
        'mu': mu,
        'R': R,
        'lambd': cfg.lambd,
    }
    
    return problem_data


def sample_lasso_batch(key, A_jax, A_np, N, p_xsamp_nonzero, lambd):
    # Generate b vectors
    b_batch = generate_batch_b_jax(key, A_jax, N, p_xsamp_nonzero)
    
    # Solve Lasso to get x_opt, f_opt for each sample
    b_batch_np = np.array(b_batch)
    x_opt_batch_np, f_opt_batch_np, _ = solve_batch_lasso_cvxpy(A_np, b_batch_np, lambd)
    
    return b_batch, jnp.array(x_opt_batch_np), jnp.array(f_opt_batch_np)


# =============================================================================
# L2O Pipeline (Learning to Optimize without DRO)
# =============================================================================

@partial(jax.jit, static_argnames=['traj_fn', 'pep_obj', 'K_max'])
def lasso_pep_obj_from_trajectory(stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max, traj_fn, pep_obj):
    """
    Compute PEP objective directly from ISTA/FISTA trajectory without SDP.
    
    For Lasso, the composite objective is:
        f(x) = f1(x) + f2(x) = 0.5*||Ax - b||^2 + lambd*||x||_1
    
    Args:
        stepsizes: Step sizes (gamma,) for ISTA or (gamma, beta) for FISTA
        A, b: Problem data
        x0: Initial point (shifted, so x0 = 0)
        x_opt, f_opt: Optimal point and value
        lambd: L1 regularization
        K_max: Number of iterations
        traj_fn: Trajectory function (ISTA or FISTA)
        pep_obj: 'obj_val', 'opt_dist_sq_norm', or 'grad_sq_norm'
    
    Returns:
        Scalar PEP objective value
    """
    # Run trajectory (returns without Gram representation)
    traj_result = traj_fn(stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max, return_Gram_representation=False)
    
    # Extract final iterate (x_K is in shifted coordinates, so x_K + x_opt is actual)
    x_iter = traj_result[0]  # Shape (n, K+1), columns are x_0, ..., x_K in shifted coords
    x_K_shifted = x_iter[:, -1]  # Final iterate in shifted coordinates
    x_K = x_K_shifted + x_opt    # Actual final iterate
    
    if pep_obj == 'obj_val':
        # Compute actual objective at x_K
        f1_xK = 0.5 * jnp.sum((A @ x_K - b) ** 2)
        f2_xK = lambd * jnp.sum(jnp.abs(x_K))
        return f1_xK + f2_xK - f_opt
    elif pep_obj == 'opt_dist_sq_norm':
        return jnp.sum((x_K - x_opt) ** 2)
    elif pep_obj == 'grad_sq_norm':
        # Composite gradient: g_f1(x_K) + h_f2(x_K) where h is a subgradient
        g_f1 = A.T @ (A @ x_K - b)
        h_f2 = lambd * jnp.sign(x_K)  # Subgradient
        return jnp.sum((g_f1 + h_f2) ** 2)
    else:
        raise ValueError(f"Unknown pep_obj: {pep_obj}")


def l2o_lasso_pipeline(stepsizes, A, b_batch, x0_batch, x_opt_batch, f_opt_batch, 
                       lambd, K_max, traj_fn, pep_obj, risk_type, alpha=0.1):
    """
    L2O pipeline for Lasso: compute PEP objectives without DRO SDP.
    
    Computes PEP objectives for each sample in the batch and returns a risk measure.
    
    Args:
        stepsizes: Step size parameters (gamma,) or (gamma, beta)
        A: Fixed A matrix
        b_batch: Batch of b vectors (N, m)
        x0_batch: Batch of initial points (N, n)
        x_opt_batch: Batch of optimal points (N, n)
        f_opt_batch: Batch of optimal values (N,)
        lambd: L1 regularization
        K_max: Number of iterations
        traj_fn: Trajectory function
        pep_obj: PEP objective type
        risk_type: 'expectation' or 'cvar'
        alpha: CVaR confidence level
    
    Returns:
        Scalar loss value
    """
    # vmap over the batch
    batch_pep_obj_func = jax.vmap(
        lambda b, x0, x_opt, f_opt: lasso_pep_obj_from_trajectory(
            stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max, traj_fn, pep_obj
        ),
        in_axes=(0, 0, 0, 0)
    )
    pep_objs = batch_pep_obj_func(b_batch, x0_batch, x_opt_batch, f_opt_batch)
    
    if risk_type == 'expectation':
        return jnp.mean(pep_objs)
    elif risk_type == 'cvar':
        # CVaR: expectation of the top alpha fraction of values
        N = pep_objs.shape[0]
        k = max(int(np.ceil(alpha * N)), 1)
        sorted_objs = jnp.sort(pep_objs)[::-1]  # Sort descending
        return jnp.mean(sorted_objs[:k])
    else:
        raise ValueError(f"Unknown risk_type: {risk_type}")


def run_sgd_for_K_lasso(cfg, K_max, problem_data, key, gamma_init, sgd_iters, eta_t,
                         eps, alpha, alg, optimizer_type, N_val, csv_path, precond_inv):
    log.info(f"=== Running SGD for K={K_max}, alg={alg} ===")
    
    # Extract problem data
    A_jax = problem_data['A_jax']
    A_np = problem_data['A_np']
    L = problem_data['L']
    mu = problem_data['mu']
    R = problem_data['R']
    lambd = problem_data['lambd']
    
    learning_framework = cfg.learning_framework

    # TODO: add ALISTA
    if learning_framework == 'ldro-pep':
        dro_canon_backend = cfg.get('dro_canon_backend', 'manual_jax')
        if dro_canon_backend != 'manual_jax':
            raise ValueError(f"Only 'manual_jax' dro_canon_backend is supported. Got: {dro_canon_backend}")
    elif learning_framework == 'l2o':
        pass  # L2O doesn't use DRO backend
    else:
        raise ValueError(f"Only 'ldro-pep' or 'l2o' learning_framework is supported for Lasso SGD. Got: {learning_framework}")

    if alg == 'ista':
        traj_fn = problem_data_to_ista_trajectories
        pep_data_fn = construct_ista_pep_data
        has_beta = False
    elif alg == 'fista':
        traj_fn = problem_data_to_fista_trajectories
        pep_data_fn = construct_fista_pep_data
        has_beta = True
    else:
        raise ValueError(f"Unknown algorithm: {alg}. Must be 'ista' or 'fista'.")

    beta_init = None
    if has_beta:
        betas_t = [1.0]
        for k in range(K_max):
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * betas_t[-1]**2))
            betas_t.append(t_new)
        beta_init = jnp.array(betas_t)  # Raw t_k sequence of length K+1
    
    # Helper to sample a batch of Lasso problems
    def sample_batch(sample_key):
        batch_key, next_key = jax.random.split(sample_key)
        b_batch, x_opt_batch, f_opt_batch = sample_lasso_batch(
            batch_key, A_jax, A_np, N_val,
            cfg.p_xsamp_nonzero, lambd
        )
        return next_key, b_batch, x_opt_batch, f_opt_batch
    
    risk_type = 'cvar' if cfg.dro_obj == 'cvar' else 'expectation'
    if learning_framework == 'ldro-pep':
        # DRO pipeline: compute Gram representation, then solve DRO SDP
        def lasso_dro_pipeline(stepsizes_tuple, b_batch, x0_batch, x_opt_batch, f_opt_batch):
            """Full DRO pipeline for Lasso using manual JAX canonicalization."""
            # Compute trajectories for all samples
            # ISTA expects gamma array directly, FISTA expects (gamma, beta) tuple
            if has_beta:
                traj_stepsizes = stepsizes_tuple
            else:
                traj_stepsizes = stepsizes_tuple[0]
            
            batch_GF_fn = jax.vmap(
                lambda b, x0, x_opt, f_opt: traj_fn(traj_stepsizes, A_jax, b, x0, x_opt, f_opt, lambd, K_max, return_Gram_representation=True),
                in_axes=(0, 0, 0, 0)
            )
            G_batch, F_batch = batch_GF_fn(b_batch, x0_batch, x_opt_batch, f_opt_batch)
            
            # Compute PEP constraint matrices (these depend on stepsizes)
            if has_beta:
                pep_data = pep_data_fn(stepsizes_tuple[0], stepsizes_tuple[1], mu, L, R, K_max, cfg.pep_obj)
            else:
                pep_data = pep_data_fn(stepsizes_tuple[0], mu, L, R, K_max, cfg.pep_obj)
            A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]

            return dro_scs_solve(
                A_obj, b_obj, A_vals, b_vals, c_vals,
                G_batch, F_batch,
                eps, precond_inv,
                risk_type=risk_type,
                alpha=alpha,
            )
        
        value_and_grad_fn = jax.value_and_grad(lasso_dro_pipeline, argnums=0)

    elif learning_framework == 'l2o':
        # L2O pipeline: compute PEP objectives directly without DRO SDP
        def lasso_l2o_wrapper(stepsizes_tuple, b_batch, x0_batch, x_opt_batch, f_opt_batch):
            """L2O pipeline wrapper for Lasso."""
            # ISTA expects gamma array directly, FISTA expects (gamma, beta) tuple
            if has_beta:
                traj_stepsizes = stepsizes_tuple
            else:
                traj_stepsizes = stepsizes_tuple[0]
            
            return l2o_lasso_pipeline(
                traj_stepsizes, A_jax, b_batch, x0_batch, x_opt_batch, f_opt_batch,
                lambd, K_max, traj_fn, cfg.pep_obj, risk_type, alpha
            )
        
        value_and_grad_fn = jax.value_and_grad(lasso_l2o_wrapper, argnums=0)

    if has_beta:
        stepsizes = (gamma_init, beta_init)
    else:
        stepsizes = (gamma_init,)
    
    gamma = stepsizes[0]
    is_vector_gamma = jnp.ndim(gamma) > 0
    
    all_stepsizes_vals = [stepsizes]
    all_losses = []
    all_times = []
    
    # Determine update mask for learn_beta
    learn_beta = cfg.get('learn_beta', True)
    if has_beta and not learn_beta:
        update_mask = [True, False]  # Update gamma, keep beta fixed
        log.info(f'learn_beta=False: beta will NOT be updated during optimization')
    else:
        update_mask = None

    def proj_stepsizes(x):
        if isinstance(x, list):
            return [jax.nn.relu(jnp.array(xi)) for xi in x]
        return jax.nn.relu(x)

    optimizer = None
    weight_decay = cfg.get('weight_decay', 1e-2)
    if optimizer_type == "adamw":
        optimizer = AdamWMin(
            x_params=[jnp.array(s) for s in stepsizes],
            lr=eta_t,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
            update_mask=update_mask,
        )
    
    # SGD iterations
    for iter_num in range(sgd_iters):
        gamma = stepsizes[0]
        gamma_log = '[' + ', '.join(f'{x:.5f}' for x in gamma.tolist()) + ']' if is_vector_gamma else f'{float(gamma):.5f}'
        if has_beta:
            beta = stepsizes[1]
            beta_log = '[' + ', '.join(f'{x:.5f}' for x in beta.tolist()) + ']'
            log.info(f'K={K_max}, iter={iter_num}, gamma={gamma_log}, beta={beta_log}')
        else:
            log.info(f'K={K_max}, iter={iter_num}, gamma={gamma_log}')
        
        # Sample new batch
        iter_start_time = time.perf_counter()
        log.info('sampling...')
        key, b_batch, x_opt_batch, f_opt_batch = sample_batch(key)
        log.info('samples found')
        x0_batch = jnp.zeros((N_val, A_jax.shape[1]))  # Shifted problem: start at 0
        
        # Compute loss and gradients
        log.info('calling value_and_grad_fn...')
        loss, d_stepsizes = value_and_grad_fn(stepsizes, b_batch, x0_batch, x_opt_batch, f_opt_batch)
        log.info('value_and_grad_fn returned, materializing results...')
        # Force materialization of JAX arrays
        loss_val = float(loss)
        d_stepsizes_materialized = tuple(jnp.array(ds) for ds in d_stepsizes)
        log.info(f'results materialized, loss={loss_val:.6f}')
        log.info(f'd_stepsizes: {d_stepsizes_materialized}')
        
        iter_time = time.perf_counter() - iter_start_time
        log.info(f'  iter_time (finding optimal sols + solving SDP): {iter_time:.3f}s')
        
        all_losses.append(loss_val)
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
        df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_gamma, has_beta, all_losses, all_times)
        df.to_csv(csv_path, index=False)
    
    # Final save
    df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_gamma, has_beta, all_losses, all_times)
    gamma = stepsizes[0]
    gamma_str = '[' + ', '.join(f'{x:.5f}' for x in gamma.tolist()) + ']' if is_vector_gamma else f'{float(gamma):.6f}'
    log.info(f'K={K_max} complete. Final gamma={gamma_str}. Saved to {csv_path}')
    df.to_csv(csv_path, index=False)
    

def lasso_run(cfg):
    log.info("=" * 60)
    log.info("Starting Lasso learning experiment")
    log.info("=" * 60)
    log.info(cfg)
    
    # Setup problem
    problem_data = setup_lasso_problem(cfg)

    alg = cfg.alg
    optimizer_type = cfg.optimizer_type
    sgd_iters = cfg.sgd_iters
    eta_t = cfg.eta_t
    eps = cfg.eps
    alpha = cfg.alpha
    N_val = cfg.N
    L = problem_data['L']
    mu = problem_data['mu']
    R = problem_data['R']
    lambd = problem_data['lambd']
    A_jax = problem_data['A_jax']
    A_np = problem_data['A_np']
    
    log.info(f"Algorithm: {alg}, Optimizer: {optimizer_type}")

    # Ensure output directory exists
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Random key for sampling
    seed = cfg.sgd_seed
    key = jax.random.PRNGKey(seed)

    gamma_init_scalar = 1.5 / L
    log.info(f"Initial step size gamma 1.5 / L: {gamma_init_scalar}")

    for K in cfg.K_max:
        # Determine step size initialization
        is_vector = cfg.stepsize_type == "vector"
        if is_vector:
            gamma_init = jnp.full(K, gamma_init_scalar)
        else:
            gamma_init = jnp.array([gamma_init_scalar])  # Still array for consistency
        
        K_output_dir = os.path.join(output_dir, f"K_{K}")
        os.makedirs(K_output_dir, exist_ok=True)
        csv_path = os.path.join(K_output_dir, "progress.csv")
        
        # Select run function based on learning framework
        learning_framework = cfg.learning_framework
        
        if learning_framework == 'lpep':
            # LPEP: deterministic PEP minimization (no samples, no DRO)
            # run_gd_for_K_lpep_lasso(
            #     cfg, K, problem_data, gamma_init, sgd_iters, eta_t,
            #     alg, csv_path
            # )
            raise NotImplementedError
        elif learning_framework == 'l2o':
            # L2O doesn't need the preconditioner
            precond_inv = None
        elif learning_framework == 'ldro-pep':
            # LDRO-PEP: precompute preconditioner using a large sample batch
            precond_sample_size = cfg.get('precond_sample_size', 100)
            log.info(f"Precomputing preconditioner using {precond_sample_size} samples...")
            
            precond_key, key = jax.random.split(key)
            b_batch_precond, x_opt_batch_precond, f_opt_batch_precond = sample_lasso_batch(
                precond_key, A_jax, A_np, precond_sample_size,
                cfg.p_xsamp_nonzero, lambd
            )
            x0_batch_precond = jnp.zeros((precond_sample_size, A_jax.shape[1]))
            
            # Compute G_batch and F_batch using initial stepsizes
            if alg == 'ista':
                traj_fn = problem_data_to_ista_trajectories
                traj_stepsizes = gamma_init
            elif alg == 'fista':
                traj_fn = problem_data_to_fista_trajectories
                # Initialize beta for FISTA
                betas_t = [1.0]
                for k in range(K):
                    t_new = 0.5 * (1 + np.sqrt(1 + 4 * betas_t[-1]**2))
                    betas_t.append(t_new)
                beta_init = jnp.array(betas_t)
                traj_stepsizes = (gamma_init, beta_init)
            else:
                raise ValueError(f"Unknown algorithm: {alg}")
            
            batch_GF_fn = jax.vmap(
                lambda b, x0, x_opt, f_opt: traj_fn(
                    traj_stepsizes, A_jax, b, x0, x_opt, f_opt, lambd, K, 
                    return_Gram_representation=True
                ),
                in_axes=(0, 0, 0, 0)
            )
            G_batch_precond, F_batch_precond = batch_GF_fn(
                b_batch_precond, x0_batch_precond, x_opt_batch_precond, f_opt_batch_precond
            )
            
            # Compute preconditioner (fixed for all SGD iterations)
            precond_type = cfg.get('precond_type', 'average')
            precond_inv = compute_preconditioner_from_samples(
                G_batch_precond, F_batch_precond, precond_type=precond_type
            )
            log.info(f"Preconditioner computed (type={precond_type})")
        else:
            raise ValueError(f"Unknown learning_framework: {learning_framework}")
        
        # Run SGD for both L2O and ldro-pep
        run_sgd_for_K_lasso(
            cfg, K, problem_data, key,
            gamma_init, sgd_iters, eta_t,
            eps, alpha, alg, optimizer_type,
            N_val, csv_path, precond_inv
        )
    
    log.info("=== Lasso SGD experiment complete ===")

