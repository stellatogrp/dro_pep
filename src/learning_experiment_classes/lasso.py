"""
Lasso learning experiment pipeline.

Mirrors the quad.py pipeline but for ISTA/FISTA on Lasso problems.
"""
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
    compute_preconditioner_from_samples,
    dro_pep_obj_jax,
)
from learning.trajectories_ista_fista import (
    problem_data_to_ista_trajectories,
    problem_data_to_fista_trajectories,
)
from learning.jax_scs_layer import dro_scs_solve

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=6, suppress=True)

log = logging.getLogger(__name__)


# =============================================================================
# A Matrix Generation (fixed per experiment)
# =============================================================================

def generate_A(seed, m, n, p_A_nonzero):
    """
    Generate sparse A matrix using NumPy.
    
    Args:
        seed: Random seed for reproducibility
        m: Number of rows
        n: Number of columns
        p_A_nonzero: Probability of non-zero entries
        
    Returns:
        A: (m, n) numpy array with columns normalized to unit norm
    """
    np.random.seed(seed)
    A = np.random.normal(scale=1/m, size=(m, n))
    mask = np.random.binomial(1, p=p_A_nonzero, size=(m, n))
    A = A * mask
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-10)
    A = A / col_norms
    return A


# =============================================================================
# b Vector Generation (sampled per problem instance)
# =============================================================================

def generate_single_b_jax(key, A, p_xsamp_nonzero, noise_eps):
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
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Generate sparse x sample
    x_samp = jax.random.normal(key1, (n,))
    x_mask = jax.random.bernoulli(key2, p=p_xsamp_nonzero, shape=(n,)).astype(jnp.float64)
    x_samp = x_samp * x_mask
    
    # Generate b = A @ x_samp + noise
    noise = noise_eps * jax.random.normal(key3, (m,))
    b = A @ x_samp + noise
    
    return b


def generate_batch_b_jax(key, A, N, p_xsamp_nonzero, noise_eps):
    """
    Generate a batch of b vectors.
    
    Args:
        key: JAX random key
        A: (m, n) matrix
        N: Number of samples
        p_xsamp_nonzero: Probability of non-zero entries in x_samp
        noise_eps: Noise level
        
    Returns:
        b_batch: (N, m) array of b vectors
    """
    keys = jax.random.split(key, N)
    generate_one = partial(generate_single_b_jax, A=A, 
                           p_xsamp_nonzero=p_xsamp_nonzero, 
                           noise_eps=noise_eps)
    b_batch = jax.vmap(generate_one)(keys)
    return b_batch


# =============================================================================
# Lasso Solution (using CVXPY - not JIT compatible)
# =============================================================================

def solve_lasso_cvxpy(A_np, b_np, lambd):
    """
    Solve single Lasso problem using CVXPY.
    
    min 0.5 * ||A @ x - b||^2 + lambd * ||x||_1
    
    Args:
        A_np: (m, n) numpy array
        b_np: (m,) numpy array
        lambd: L1 regularization parameter
        
    Returns:
        x_opt: Optimal solution
        f_opt: Optimal objective value
    """
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


# =============================================================================
# Compute Problem Parameters (L, mu from A^T A)
# =============================================================================

def compute_lasso_params(A):
    """
    Compute Lipschitz constant L and strong convexity mu from A.
    
    For Lasso, f1(x) = 0.5 * ||Ax - b||^2 has:
        - L = max eigenvalue of A^T A
        - mu = min eigenvalue of A^T A (0 if m < n)
    
    Args:
        A: (m, n) matrix (JAX or numpy)
        
    Returns:
        L: Lipschitz constant
        mu: Strong convexity parameter
    """
    m, n = A.shape
    ATA = A.T @ A
    eigvals = jnp.linalg.eigvalsh(ATA)
    L = jnp.max(eigvals)
    mu = jnp.min(eigvals) if m >= n else 0.0
    return float(L), float(mu)


# =============================================================================
# Radius Computation (if R is None in config)
# =============================================================================

def compute_sample_radius(cfg, A_np):
    """
    Compute maximum radius R by solving many Lasso problems.
    
    Since problems are shifted to have x_opt at origin, R is the
    maximum ||x_opt|| across samples.
    
    Args:
        cfg: Hydra config
        A_np: (m, n) numpy array
        
    Returns:
        R: Maximum radius
    """
    log.info(f"Computing R from {cfg.R_sample_size} samples...")
    
    # Generate b samples using the b_seed
    key = jax.random.PRNGKey(cfg.b_seed)
    b_batch = generate_batch_b_jax(
        key, jnp.array(A_np), cfg.R_sample_size,
        cfg.p_xsamp_nonzero, cfg.noise_eps
    )
    b_batch_np = np.array(b_batch)
    
    # Solve all Lasso problems
    _, _, R_max = solve_batch_lasso_cvxpy(A_np, b_batch_np, cfg.lambd)
    
    log.info(f"Computed R = {R_max:.6f}")
    return R_max


# =============================================================================
# Sample Batch for SGD Iteration
# =============================================================================

def sample_lasso_batch(key, A_jax, A_np, N, p_xsamp_nonzero, noise_eps, lambd):
    """
    Sample a batch of Lasso problems for one SGD iteration.
    
    Args:
        key: JAX random key
        A_jax: (m, n) JAX array
        A_np: (m, n) NumPy array
        N: Batch size
        p_xsamp_nonzero: Sparsity parameter
        noise_eps: Noise level
        lambd: L1 regularization
        
    Returns:
        b_batch: (N, m) JAX array of b vectors
        x_opt_batch: (N, n) JAX array of optimal solutions
        f_opt_batch: (N,) JAX array of optimal objectives
    """
    # Generate b vectors
    b_batch = generate_batch_b_jax(key, A_jax, N, p_xsamp_nonzero, noise_eps)
    
    # Solve Lasso to get x_opt, f_opt for each sample
    b_batch_np = np.array(b_batch)
    x_opt_batch_np, f_opt_batch_np, _ = solve_batch_lasso_cvxpy(A_np, b_batch_np, lambd)
    
    return b_batch, jnp.array(x_opt_batch_np), jnp.array(f_opt_batch_np)


# =============================================================================
# Setup Function (called once at experiment start)
# =============================================================================

def setup_lasso_problem(cfg):
    """
    Set up the Lasso problem: generate A, compute L/mu, optionally compute R.
    
    Args:
        cfg: Hydra config
        
    Returns:
        problem_data: Dict containing:
            - A_jax: JAX array of A matrix
            - A_np: NumPy array of A matrix
            - L: Lipschitz constant
            - mu: Strong convexity parameter
            - R: Initial radius
            - lambd: L1 regularization
    """
    log.info("Setting up Lasso problem...")
    
    # Generate A matrix (using NumPy for cvxpy compatibility, then convert)
    log.info(f"Generating A matrix with seed={cfg.A_seed}")
    A_np = generate_A(cfg.A_seed, cfg.m, cfg.n, cfg.p_A_nonzero)
    A_jax = jnp.array(A_np)
    
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


# =============================================================================
# Build Stepsizes DataFrame for CSV Logging
# =============================================================================

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


# =============================================================================
# L2O Pipeline (Learning to Optimize without DRO)
# =============================================================================

@partial(jax.jit, static_argnames=['traj_fn', 'pep_obj', 'K_max', 'return_Gram'])
def lasso_pep_obj_from_trajectory(stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max, traj_fn, pep_obj, return_Gram=True):
    """
    Compute PEP objective directly from ISTA/FISTA trajectory without SDP.
    
    For Lasso, the composite objective is:
        f(x) = f1(x) + f2(x) = 0.5*||Ax - b||^2 + lambd*||x||_1
    
    At optimal x_opt, f(x_opt) = f_opt.
    
    Args:
        stepsizes: Step sizes (gamma,) for ISTA or (gamma, beta) for FISTA
        A, b: Problem data
        x0: Initial point
        x_opt, f_opt: Optimal point and value
        lambd: L1 regularization
        K_max: Number of iterations
        traj_fn: Trajectory function (ISTA or FISTA)
        pep_obj: 'obj_val', 'opt_dist_sq_norm', or 'grad_sq_norm'
        return_Gram: Whether traj_fn returns Gram representation
    
    Returns:
        Scalar PEP objective value
    """
    if return_Gram:
        # Get Gram representation (not needed for l2o, but reuse trajectory function)
        G, F = traj_fn(stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max, return_Gram_representation=True)
        
        # For obj_val: F contains [F1, F2] concatenated
        # F1 = f1(x_k) - f1_s for k=0..K and 0 for optimal
        # F2 = f2(x_k) - f2_s for k=0..K and 0 for optimal
        # The objective is (f1(x_K) + f2(x_K)) - (f1_s + f2_s)
        # For ISTA: dimF1 = K+2, dimF2 = K+2, so F1[-2] is f1(x_K)-f1_s, F2[-2] is f2(x_K)-f2_s
        if pep_obj == 'obj_val':
            # The F vector structure gives us f1(x_K) - f1_s + f2(x_K) - f2_s at specific indices
            # But simpler: just compute directly from trajectory
            pass  # Fall through to direct computation below
        elif pep_obj == 'opt_dist_sq_norm':
            # Need ||x_K - x_s||^2, but x_s = 0 in shifted coordinates
            # G[0,0] is ||x_0 - x_s||^2, but for x_K we need to extract from G structure
            pass
    
    # Direct computation is cleaner - run trajectory and compute objective
    traj_result = traj_fn(stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max, return_Gram_representation=False)
    
    # Extract final iterate (x_K is in shifted coordinates, so x_K + x_opt is actual)
    # Both ISTA and FISTA return x_iter as first element
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


# =============================================================================
# LPEP Run Function for Single K (Deterministic PEP Optimization)
# =============================================================================

def run_gd_for_K_lpep_lasso(cfg, K_max, problem_data, gamma_init, gd_iters, eta_t,
                            alg, csv_path):
    """
    Run gradient descent for learning PEP (lpep) for Lasso - no samples, no DRO.
    
    Optimizes step sizes to minimize the standard (worst-case) PEP objective
    using cvxpylayers with SCS solver for differentiation.
    
    Args:
        cfg: Configuration object
        K_max: Number of algorithm iterations
        problem_data: Problem data dict with A, L, mu, R, lambd
        gamma_init: Initial step sizes
        gd_iters: Number of gradient descent iterations
        eta_t: Learning rate for step sizes
        alg: Algorithm name ('ista' or 'fista')
        csv_path: Path to save progress CSV
    """
    from learning.cvxpylayers_setup import create_full_pep_layer
    
    log.info(f"=== Running lpep GD for K={K_max}, alg={alg} ===")
    
    # Extract problem data
    L = problem_data['L']
    mu = problem_data['mu']
    R = problem_data['R']
    
    # Select PEP data function based on algorithm
    if alg == 'ista':
        pep_data_fn = construct_ista_pep_data
        has_beta = False
    elif alg == 'fista':
        pep_data_fn = construct_fista_pep_data
        has_beta = True
    else:
        raise ValueError(f"Unknown algorithm: {alg}. Must be 'ista' or 'fista'.")
    
    # Initialize stepsizes
    is_vector_gamma = cfg.stepsize_type == 'vector'
    
    if has_beta:
        # FISTA: compute raw t_k sequence (Nesterov sequence of length K+1)
        betas_t = [1.0]
        for k in range(K_max):
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * betas_t[-1]**2))
            betas_t.append(t_new)
        beta_init = jnp.array(betas_t)
        stepsizes = [gamma_init, beta_init]
    else:
        stepsizes = [gamma_init]
    
    # Track step size values for logging
    all_stepsizes_vals = [tuple(stepsizes)]
    all_losses = []
    
    # Create cvxpylayers-based PEP layer
    if has_beta:
        pep_data = pep_data_fn(gamma_init, beta_init, mu, L, R, K_max, cfg.pep_obj)
    else:
        pep_data = pep_data_fn(gamma_init, mu, L, R, K_max, cfg.pep_obj)
    
    A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
    M = A_vals.shape[0]
    mat_shape = (A_obj.shape[0], A_obj.shape[1])
    vec_shape = (b_obj.shape[0],)
    pep_layer = create_full_pep_layer(M, mat_shape, vec_shape)
    log.info(f"Using SCS backend for lpep (M={M}, mat_shape={mat_shape}, vec_shape={vec_shape})")
    
    # Define the PEP loss function (differentiable w.r.t. stepsizes)
    def pep_loss_fn(stepsizes_list):
        """Compute PEP worst-case bound for given stepsizes."""
        if has_beta:
            gamma, beta = stepsizes_list[0], stepsizes_list[1]
            pep_data = pep_data_fn(gamma, beta, mu, L, R, K_max, cfg.pep_obj)
        else:
            gamma = stepsizes_list[0]
            pep_data = pep_data_fn(gamma, mu, L, R, K_max, cfg.pep_obj)
        
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        
        # Use cvxpylayers with SCS
        M = A_vals.shape[0]
        params_list = (
            [A_vals[m] for m in range(M)] +
            [b_vals[m] for m in range(M)] +
            [c_vals, A_obj, b_obj]
        )
        (G_opt, F_opt) = pep_layer(*params_list)
        # Compute objective: trace(A_obj @ G) + b_obj^T @ F
        loss = jnp.trace(A_obj @ G_opt) + jnp.dot(b_obj, F_opt)
        return loss
    
    value_and_grad_fn = jax.value_and_grad(pep_loss_fn, argnums=0)
    
    # Projection function
    def proj_stepsizes(x):
        return [jax.nn.relu(xi) for xi in x]
    
    # Determine update mask for learn_beta
    learn_beta = cfg.get('learn_beta', True)
    if has_beta and not learn_beta:
        update_mask = [True, False]
        log.info(f'learn_beta=False: beta will NOT be updated during optimization')
    else:
        update_mask = None
    
    # Initialize optimizer
    optimizer = AdamWMin(
        x_params=[jnp.array(s) for s in stepsizes],
        lr=eta_t,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        update_mask=update_mask,
    )
    
    # GD iterations
    for iter_num in range(gd_iters):
        gamma = stepsizes[0]
        # gamma is always an array, format appropriately
        if not is_vector_gamma and gamma.size == 1:
            gamma_log = f'{float(gamma.item()):.5f}'
        else:
            gamma_log = '[' + ', '.join(f'{float(x):.5f}' for x in gamma.tolist()) + ']'
        if has_beta:
            beta_log = '[' + ', '.join(f'{x:.5f}' for x in stepsizes[1].tolist()[:4]) + ', ...]'
            log.info(f"K={K_max}, iter={iter_num}, gamma={gamma_log}, beta={beta_log}")
        else:
            log.info(f"K={K_max}, iter={iter_num}, gamma={gamma_log}")
        
        start_time = time.time()
        
        # Compute loss and gradient
        try:
            loss_val, grads = value_and_grad_fn(stepsizes)
        except Exception as e:
            log.warning(f"PEP solve failed: {e}")
            loss_val = float('nan')
            grads = [jnp.zeros_like(s) for s in stepsizes]
        
        iter_time = time.time() - start_time
        log.info(f"  loss: {loss_val:.6f}, iter_time: {iter_time:.3f}s")
        
        all_losses.append(float(loss_val))
        
        # Update step sizes using AdamWMin
        stepsizes = optimizer.step(stepsizes, grads, proj_stepsizes)
        
        all_stepsizes_vals.append(tuple(stepsizes))
        
        # Save progress
        df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_gamma, has_beta, all_losses)
        df.to_csv(csv_path, index=False)
    
    log.info(f"=== lpep GD for K={K_max} complete ===")


# =============================================================================
# SGD Run Function for Single K
# =============================================================================

def run_sgd_for_K_lasso(cfg, K_max, problem_data, key, gamma_init, sgd_iters, eta_t,
                         eps, alpha, alg, optimizer_type, N_val, csv_path):
    """
    Run SGD for a single K value for Lasso learning.
    
    Args:
        cfg: Config object
        K_max: Number of algorithm iterations
        problem_data: Dict with A_jax, A_np, L, mu, R, lambd
        key: JAX random key
        gamma_init: Initial step sizes
        sgd_iters: Number of SGD iterations
        eta_t: Learning rate for stepsizes
        eps: DRO radius
        alpha: CVaR alpha (if using CVaR)
        alg: Algorithm ('ista' or 'fista')
        optimizer_type: Optimizer type ('vanilla_sgd', 'adamw', 'sgd_wd')
        N_val: Batch size
        csv_path: Path to save progress CSV
    """
    log.info(f"=== Running SGD for K={K_max}, alg={alg} ===")
    
    # Extract problem data
    A_jax = problem_data['A_jax']
    A_np = problem_data['A_np']
    L = problem_data['L']
    mu = problem_data['mu']
    R = problem_data['R']
    lambd = problem_data['lambd']
    
    learning_framework = cfg.learning_framework
    
    # Validate backend based on learning framework
    if learning_framework == 'ldro-pep':
        dro_canon_backend = cfg.get('dro_canon_backend', 'manual_jax')
        if dro_canon_backend != 'manual_jax':
            raise ValueError(f"Only 'manual_jax' dro_canon_backend is supported. Got: {dro_canon_backend}")
    elif learning_framework == 'l2o':
        pass  # L2O doesn't use DRO backend
    else:
        raise ValueError(f"Only 'ldro-pep' or 'l2o' learning_framework is supported for Lasso SGD. Got: {learning_framework}")
    
    # Select trajectory function and PEP data function based on algorithm
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
    
    # Compute initial beta for FISTA (raw t_k sequence)
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
            cfg.p_xsamp_nonzero, cfg.noise_eps, lambd
        )
        return next_key, b_batch, x_opt_batch, f_opt_batch
    
    # Compute preconditioner from initial sample batch (only needed for ldro-pep)
    if learning_framework == 'ldro-pep':
        log.info("Computing preconditioner from initial sample batch...")
        key, b_precond, x_opt_precond, f_opt_precond = sample_batch(key)
        
        # Initial x0 for preconditioner: zeros (since problem is shifted, optimal is at 0)
        x0_precond = jnp.zeros((N_val, A_jax.shape[1]))
        
        # Compute initial stepsizes for preconditioner
        if has_beta:
            stepsizes_for_precond = (gamma_init, beta_init)
        else:
            stepsizes_for_precond = (gamma_init,)
        
        # Compute G, F for preconditioner
        # ISTA expects gamma array directly, FISTA expects (gamma, beta) tuple
        def compute_GF_single(stepsizes_tuple, b, x0, x_opt, f_opt):
            if has_beta:
                # FISTA: pass (gamma, beta) tuple
                return traj_fn(stepsizes_tuple, A_jax, b, x0, x_opt, f_opt, lambd, K_max, return_Gram_representation=True)
            else:
                # ISTA: pass gamma array directly
                return traj_fn(stepsizes_tuple[0], A_jax, b, x0, x_opt, f_opt, lambd, K_max, return_Gram_representation=True)
        
        batch_GF_func = jax.vmap(
            lambda b, x0, x_opt, f_opt: compute_GF_single(stepsizes_for_precond, b, x0, x_opt, f_opt),
            in_axes=(0, 0, 0, 0)
        )
        G_precond_batch, F_precond_batch = batch_GF_func(b_precond, x0_precond, x_opt_precond, f_opt_precond)
        
        precond_type = cfg.get('precond_type', 'average')
        precond_inv = compute_preconditioner_from_samples(
            np.array(G_precond_batch), np.array(F_precond_batch), precond_type=precond_type
        )
        precond_inv_jax = (jnp.array(precond_inv[0]), jnp.array(precond_inv[1]))
        log.info(f'Computed preconditioner from {N_val} samples using type: {precond_type}')
    
    # Determine risk type
    risk_type = 'cvar' if cfg.dro_obj == 'cvar' else 'expectation'
    
    # Define the pipeline function based on learning framework
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
            
            # Call the manual JAX SCS solver
            return dro_scs_solve(
                A_obj, b_obj, A_vals, b_vals, c_vals,
                G_batch, F_batch,
                eps, precond_inv_jax,
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
    
    # Initialize stepsizes
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
    
    # Projection function for stepsizes (handles list from AdamWMin)
    def proj_stepsizes(x):
        if isinstance(x, list):
            return [jax.nn.relu(jnp.array(xi)) for xi in x]
        return jax.nn.relu(x)
    
    # Initialize optimizer
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
        gamma = stepsizes[0]
        gamma_log = '[' + ', '.join(f'{x:.5f}' for x in gamma.tolist()) + ']' if is_vector_gamma else f'{float(gamma):.5f}'
        if has_beta:
            beta = stepsizes[1]
            beta_log = '[' + ', '.join(f'{x:.5f}' for x in beta.tolist()) + ']'
            log.info(f'K={K_max}, iter={iter_num}, gamma={gamma_log}, beta={beta_log}')
        else:
            log.info(f'K={K_max}, iter={iter_num}, gamma={gamma_log}')
        
        # Sample new batch
        key, b_batch, x_opt_batch, f_opt_batch = sample_batch(key)
        x0_batch = jnp.zeros((N_val, A_jax.shape[1]))  # Shifted problem: start at 0
        
        # Compute loss and gradients
        iter_start_time = time.time()
        loss, d_stepsizes = value_and_grad_fn(stepsizes, b_batch, x0_batch, x_opt_batch, f_opt_batch)
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
        df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_gamma, has_beta, all_losses, all_times)
        df.to_csv(csv_path, index=False)
    
    # Final save
    df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_gamma, has_beta, all_losses, all_times)
    gamma = stepsizes[0]
    gamma_str = '[' + ', '.join(f'{x:.5f}' for x in gamma.tolist()) + ']' if is_vector_gamma else f'{float(gamma):.6f}'
    log.info(f'K={K_max} complete. Final gamma={gamma_str}. Saved to {csv_path}')
    df.to_csv(csv_path, index=False)


# =============================================================================
# Main Entry Point
# =============================================================================

def lasso_run(cfg):
    """
    Main entry point for Lasso learning experiment.
    
    Args:
        cfg: Hydra config
    """
    log.info("=" * 60)
    log.info("Starting Lasso learning experiment")
    log.info("=" * 60)
    log.info(cfg)
    
    # Setup problem
    problem_data = setup_lasso_problem(cfg)
    
    log.info(f"Problem setup complete:")
    log.info(f"  A shape: {problem_data['A_jax'].shape}")
    log.info(f"  L = {problem_data['L']:.6f}")
    log.info(f"  mu = {problem_data['mu']:.6f}")
    log.info(f"  R = {problem_data['R']:.6f}")
    log.info(f"  lambd = {problem_data['lambd']}")
    
    # Extract config values
    alg = cfg.alg
    optimizer_type = cfg.optimizer_type
    sgd_iters = cfg.sgd_iters
    eta_t = cfg.eta_t
    eps = cfg.eps
    alpha = cfg.alpha
    N_val = cfg.N
    L = problem_data['L']
    
    log.info(f"Algorithm: {alg}, Optimizer: {optimizer_type}")
    
    # Ensure output directory exists
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Random key for sampling
    seed = cfg.seed
    key = jax.random.PRNGKey(seed)
    
    # Initial step size: 1/L
    gamma_init_scalar = 1.0 / L
    log.info(f"Initial step size gamma: {gamma_init_scalar}")
    
    # Loop over K_max values
    for K in cfg.K_max:
        # Determine step size initialization
        is_vector = cfg.stepsize_type == "vector"
        if is_vector:
            gamma_init = jnp.full(K, gamma_init_scalar)
        else:
            gamma_init = jnp.array([gamma_init_scalar])  # Still array for consistency
        
        # Create output directory for this K
        K_output_dir = os.path.join(output_dir, f"K_{K}")
        os.makedirs(K_output_dir, exist_ok=True)
        csv_path = os.path.join(K_output_dir, "progress.csv")
        
        # Select run function based on learning framework
        learning_framework = cfg.learning_framework
        
        if learning_framework == 'lpep':
            # LPEP: deterministic PEP minimization (no samples, no DRO)
            run_gd_for_K_lpep_lasso(
                cfg, K, problem_data, gamma_init, sgd_iters, eta_t,
                alg, csv_path
            )
        else:
            # LDRO-PEP or L2O: stochastic optimization with samples
            run_sgd_for_K_lasso(
                cfg, K, problem_data, key,
                gamma_init, sgd_iters, eta_t,
                eps, alpha, alg, optimizer_type,
                N_val, csv_path
            )
    
    log.info("=== Lasso SGD experiment complete ===")