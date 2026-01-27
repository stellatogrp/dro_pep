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
from learning.jax_scs_layer import dro_scs_solve, wc_pep_scs_solve
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


def generate_single_A_logreg_jax(key, N_data, n, A_std):
    """Generate single A matrix using JAX for vmapping."""
    A_features = jax.random.normal(key, shape=(N_data, n-1)) * A_std
    A_intercept = jnp.ones((N_data, 1))
    return jnp.hstack([A_features, A_intercept])


@partial(jax.jit, static_argnames=['N_batch', 'N_data', 'n'])
def generate_batch_A_logreg_jax(key, N_batch, N_data, n, A_std):
    """Generate batch of A matrices (N_batch, N_data, n)."""
    keys = jax.random.split(key, N_batch)
    sample_one = partial(generate_single_A_logreg_jax,
                        N_data=N_data, n=n, A_std=A_std)
    return jax.vmap(sample_one)(keys)


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

def compute_logreg_L_single(A, delta):
    """Compute L for a single A matrix."""
    m, n = A.shape
    ATA = A.T @ A
    eigvals = jnp.linalg.eigvalsh(ATA)
    lambd_max = jnp.max(eigvals)
    return lambd_max / (4 * m) + delta


def compute_logreg_L_worst_case(cfg):
    """Compute worst-case L over many A samples."""
    log.info(f"Computing worst-case L from {cfg.L_sample_size} A matrices...")

    key = jax.random.PRNGKey(cfg.L_seed)

    # Generate many A matrices
    A_batch = generate_batch_A_logreg_jax(
        key, cfg.L_sample_size, cfg.N_data, cfg.n, cfg.A_std
    )

    # Compute L for each using vmap
    L_fn = lambda A: compute_logreg_L_single(A, cfg.delta)
    L_vals = jax.vmap(L_fn)(A_batch)

    L_max = float(jnp.max(L_vals))
    L_mean = float(jnp.mean(L_vals))
    L_std = float(jnp.std(L_vals))

    log.info(f"L distribution: mean={L_mean:.6f}, std={L_std:.6f}, max={L_max:.6f}")
    return L_max, cfg.delta


def compute_sample_radius_logreg(cfg):
    """Compute R by sampling many (A, b) pairs and solving."""
    log.info(f"Computing R from {cfg.R_sample_size} (A, b) samples...")

    key = jax.random.PRNGKey(cfg.R_seed)
    k1, k2 = jax.random.split(key)

    # Generate A matrices
    A_batch_jax = generate_batch_A_logreg_jax(
        k1, cfg.R_sample_size, cfg.N_data, cfg.n, cfg.A_std
    )
    A_batch_np = np.array(A_batch_jax)

    # Generate b vectors (one per A matrix)
    b_keys = jax.random.split(k2, cfg.R_sample_size)

    def generate_b_for_A(key, A):
        return generate_single_logreg_labels_jax(
            key, A, cfg.n, cfg.p_beta_nonzero, cfg.beta_scale, cfg.eps_std
        )

    b_batch_jax = jax.vmap(generate_b_for_A)(b_keys, A_batch_jax)
    b_batch_np = np.array(b_batch_jax)

    # Solve each logreg problem and find max R
    R_max = 0.0
    for i in range(cfg.R_sample_size):
        x_opt, f_opt = solve_logreg_cvxpy(A_batch_np[i], b_batch_np[i], cfg.delta)
        R = np.linalg.norm(x_opt)
        R_max = max(R_max, R)

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
    """Set up the logistic regression problem parameters."""
    log.info("Setting up logistic regression problem...")

    # Compute L and mu (worst-case over A distribution)
    L_cfg = cfg.get('L', None)
    if L_cfg is None:
        L, mu = compute_logreg_L_worst_case(cfg)
    else:
        L = float(L_cfg)
        mu = float(cfg.delta)
        log.info(f"Using provided L = {L:.6f}, mu = {mu:.6f}")

    # Compute R (max over many (A, b) samples)
    R_cfg = cfg.get('R', None)
    if R_cfg is None:
        R = compute_sample_radius_logreg(cfg)
    else:
        R = float(R_cfg)
        log.info(f"Using provided R = {R:.6f}")

    problem_data = {
        # No A_jax, A_np here - sampled per iteration
        'L': L,
        'mu': mu,
        'R': R,
        'delta': cfg.delta,
        'N_data': cfg.N_data,
        'n': cfg.n,
        'A_std': cfg.A_std,
    }

    return problem_data


# =============================================================================
# Sample Batch for SGD Iteration
# =============================================================================

def sample_logreg_batch(key, N, N_data, n, A_std, p_beta_nonzero,
                        beta_scale, eps_std, delta):
    """
    Sample a batch of logreg problems (both A and b) for one SGD iteration.

    Returns:
        A_batch: (N, N_data, n) JAX array of A matrices
        b_batch: (N, N_data) JAX array of label vectors
        x_opt_batch: (N, n) JAX array of optimal solutions
        f_opt_batch: (N,) JAX array of optimal objectives
    """
    k1, k2 = jax.random.split(key)

    # Generate N different A matrices
    A_batch = generate_batch_A_logreg_jax(k1, N, N_data, n, A_std)

    # Generate b vectors (one per A matrix)
    b_keys = jax.random.split(k2, N)

    def generate_b_for_A(key, A):
        return generate_single_logreg_labels_jax(
            key, A, n, p_beta_nonzero, beta_scale, eps_std
        )

    b_batch = jax.vmap(generate_b_for_A)(b_keys, A_batch)

    # Solve each problem using CVXPY (on numpy arrays)
    A_batch_np = np.array(A_batch)
    b_batch_np = np.array(b_batch)

    x_opt_batch_np = np.zeros((N, n))
    f_opt_batch_np = np.zeros(N)

    for i in range(N):
        x_opt, f_opt = solve_logreg_cvxpy(A_batch_np[i], b_batch_np[i], delta)
        x_opt_batch_np[i] = x_opt
        f_opt_batch_np[i] = f_opt

    return A_batch, b_batch, jnp.array(x_opt_batch_np), jnp.array(f_opt_batch_np)


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

@partial(jax.jit, static_argnames=['traj_fn', 'K_max', 'pep_obj', 'loss_type', 'decay_rate'])
def logreg_trajectory_loss(stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, traj_fn, pep_obj,
                            loss_type='final', decay_rate=0.9):
    """
    Compute loss from GD/FGM trajectory with different loss formulations for logreg.

    This function provides multiple loss formulations to address gradient
    issues in L2O for logistic regression problems.

    Args:
        stepsizes: Step sizes (t,) for GD or (t, beta) for FGM
        A, b: Problem data (A is design matrix, b is labels)
        z0: Initial point (shifted coordinates, so z0 = 0)
        x_opt, f_opt: Optimal point and value
        delta: L2 regularization parameter
        K_max: Number of iterations
        traj_fn: Trajectory function (GD or FGM)
        pep_obj: 'obj_val', 'opt_dist_sq_norm', or 'grad_sq_norm'
        loss_type: Loss formulation type:
            - 'final': Only final iterate loss (original, may cause uniform gradients)
            - 'cumulative': Sum of losses at all iterates (gives each t_k influence)
            - 'weighted': Exponentially weighted sum (emphasizes later iterations)
            - 'per_step': Loss improvement per step (directly ties t_k to step k)
            - 'distance_cumulative': Cumulative distance to optimal (often smoother)
        decay_rate: For 'weighted' loss, the exponential decay rate (0 < decay < 1)

    Returns:
        Scalar loss value
    """
    # Run trajectory
    traj_result = traj_fn(stepsizes, A, b, z0, x_opt, f_opt, delta, K_max,
                          return_Gram_representation=False)
    z_iter = traj_result[0]  # Shape (n, K+1), columns are z_0, ..., z_K in shifted coords

    def compute_metric_at_k(k):
        """Compute the metric (based on pep_obj) at iterate k."""
        z_k_shifted = z_iter[:, k]
        z_k = z_k_shifted + x_opt

        if pep_obj == 'obj_val':
            # Objective value: f(z_k) - f_opt
            # f(x) = -1/m * sum(b_i * (A @ x)_i - log(1 + exp((A @ x)_i))) + delta/2 * ||x||^2
            m = A.shape[0]
            Ax = A @ z_k
            log_likelihood = jnp.sum(jnp.multiply(b, Ax) - jnp.logaddexp(0.0, Ax))
            f_zk = -1.0 / m * log_likelihood + 0.5 * delta * jnp.sum(z_k ** 2)
            return f_zk - f_opt
        elif pep_obj == 'opt_dist_sq_norm':
            # Squared distance to optimum: ||z_k - x_opt||^2
            return jnp.sum(z_k_shifted ** 2)  # z_k_shifted = z_k - x_opt
        elif pep_obj == 'grad_sq_norm':
            # Squared gradient norm: ||∇f(z_k)||^2
            # ∇f(x) = A^T @ (sigmoid(A @ x) - b) / m + delta * x
            m = A.shape[0]
            Ax = A @ z_k
            sigmoid_Ax = jax.nn.sigmoid(Ax)
            grad = A.T @ (sigmoid_Ax - b) / m + delta * z_k
            return jnp.sum(grad ** 2)
        else:
            raise ValueError(f"Unknown pep_obj: {pep_obj}")

    if loss_type == 'final':
        # Original: only final iterate (may cause uniform gradients)
        return compute_metric_at_k(K_max)

    elif loss_type == 'cumulative':
        # Sum of losses at all iterates: gives each t_k direct influence
        # t_k affects z_{k+1}, z_{k+2}, ..., z_K
        # This creates a "cascade" effect where early t values affect more terms
        losses = jnp.array([compute_metric_at_k(k) for k in range(1, K_max + 1)])
        return jnp.mean(losses)

    elif loss_type == 'weighted':
        # Exponentially weighted: later iterations weighted more
        # w_k = decay^(K-k), so w_K = 1, w_{K-1} = decay, etc.
        losses = jnp.array([compute_metric_at_k(k) for k in range(1, K_max + 1)])
        weights = jnp.array([decay_rate ** (K_max - k) for k in range(1, K_max + 1)])
        weights = weights / jnp.sum(weights)  # Normalize
        return jnp.sum(weights * losses)

    elif loss_type == 'per_step':
        # Loss improvement per step: directly ties t_k to its effect
        # loss_k = (metric(z_k) - metric(z_{k+1})) measures how much step k improved
        # We minimize negative improvement (maximize improvement)
        improvements = []
        for k in range(K_max):
            loss_k = compute_metric_at_k(k)
            loss_kp1 = compute_metric_at_k(k + 1)
            # Improvement should be positive; we want to maximize it
            improvements.append(loss_k - loss_kp1)
        # Return negative mean improvement (so minimizing = maximizing improvement)
        # Plus final loss to ensure we reach optimum
        return compute_metric_at_k(K_max) - 0.1 * jnp.mean(jnp.array(improvements))

    elif loss_type == 'distance_cumulative':
        # Cumulative distance to optimal (always uses opt_dist_sq_norm regardless of pep_obj)
        dists = jnp.array([jnp.sum(z_iter[:, k] ** 2) for k in range(1, K_max + 1)])
        return jnp.mean(dists)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def l2o_logreg_pipeline(stepsizes, A, b_batch, z0_batch, x_opt_batch, f_opt_batch,
                        delta, K_max, traj_fn, pep_obj, risk_type, alpha=0.1):
    """
    L2O pipeline for logistic regression without DRO SDP.

    Computes PEP objectives for each sample in the batch and returns a risk measure.
    Uses the standard final-iterate loss f(z_K) - f_opt.
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


def l2o_logreg_pipeline_v2(stepsizes, A, b_batch, z0_batch, x_opt_batch, f_opt_batch,
                            delta, K_max, traj_fn, loss_type, risk_type, alpha=0.1, decay_rate=0.9):
    """
    Improved L2O pipeline with alternative loss formulations for better gradients.

    This version addresses potential uniform gradient problems by using loss
    formulations that give each step size t_k more direct influence on the loss.

    Args:
        stepsizes: Step size parameters (tuple)
        A: Design matrix (fixed for batch, but different A per sample in outer vmap)
        b_batch: Batch of label vectors (N, N_data)
        z0_batch: Batch of initial points (N, n)
        x_opt_batch: Batch of optimal points (N, n)
        f_opt_batch: Batch of optimal function values (N,)
        delta: L2 regularization parameter
        K_max: Number of algorithm iterations
        traj_fn: Trajectory function
        loss_type: 'final', 'cumulative', 'weighted', 'per_step', 'distance_cumulative'
        risk_type: 'expectation' or 'cvar'
        alpha: CVaR confidence level
        decay_rate: For 'weighted' loss type

    Returns:
        Scalar loss value
    """
    # vmap over the batch
    batch_loss_func = jax.vmap(
        lambda b, z0, x_opt, f_opt: logreg_trajectory_loss(
            stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, traj_fn, 'obj_val',
            loss_type=loss_type, decay_rate=decay_rate
        ),
        in_axes=(0, 0, 0, 0)
    )
    losses = batch_loss_func(b_batch, z0_batch, x_opt_batch, f_opt_batch)

    if risk_type == 'expectation':
        return jnp.mean(losses)
    elif risk_type == 'cvar':
        N = losses.shape[0]
        k = max(int(np.ceil(alpha * N)), 1)
        sorted_losses = jnp.sort(losses)[::-1]
        return jnp.mean(sorted_losses[:k])
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
    L = problem_data['L']
    mu = problem_data['mu']
    R = problem_data['R']
    delta = problem_data['delta']
    N_data = problem_data['N_data']
    n = problem_data['n']
    A_std = problem_data['A_std']
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

        # Sample both A and b (not just b)
        A_batch, b_batch, x_opt_batch, f_opt_batch = sample_logreg_batch(
            k1, N_val, N_data, n, A_std, cfg.p_beta_nonzero, cfg.beta_scale,
            cfg.eps_std, delta
        )

        # Sample z0 in shifted coordinates
        z0_batch = sample_batch_x0_disk_jax(k2, N_val, n, R)

        return next_key, A_batch, b_batch, z0_batch, x_opt_batch, f_opt_batch
    
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
        key, A_precond, b_precond, z0_precond, x_opt_precond, f_opt_precond = sample_batch(key)

        batch_GF_func = jax.vmap(
            lambda A, b, z0, x_opt, f_opt: traj_fn(
                stepsizes_for_precond, A, b, z0, x_opt, f_opt, K_max,
                return_Gram_representation=True
            ),
            in_axes=(0, 0, 0, 0, 0)  # A is axis 0 (batched)
        )
        G_precond_batch, F_precond_batch = batch_GF_func(
            A_precond, b_precond, z0_precond, x_opt_precond, f_opt_precond
        )
        
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
            
            def manual_jax_pipeline(stepsizes, A_batch, b_batch, z0_batch, x_opt_batch, f_opt_batch):
                """Full DRO pipeline using manual JAX canonicalization."""
                # Compute trajectories (A is now batched)
                batch_GF_func = jax.vmap(
                    lambda A, b, z0, x_opt, f_opt: traj_fn(
                        stepsizes, A, b, z0, x_opt, f_opt, K_max,
                        return_Gram_representation=True
                    ),
                    in_axes=(0, 0, 0, 0, 0)  # A is batched
                )
                G_batch, F_batch = batch_GF_func(A_batch, b_batch, z0_batch, x_opt_batch, f_opt_batch)
                
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

        else:
            raise ValueError(
                f"Unknown dro_canon_backend: {dro_canon_backend}. "
                f"Only 'manual_jax' is supported."
            )
        
    elif learning_framework == 'l2o':
        # L2O pipeline: compute PEP objectives directly without DRO SDP
        # Use improved loss formulation if specified
        l2o_loss_type = cfg.get('l2o_loss_type', 'final')
        decay_rate = cfg.get('l2o_decay_rate', 0.9)
        log.info(f"L2O loss type: {l2o_loss_type}")

        # Determine risk type string
        risk_type = cfg.dro_obj  # 'expectation' or 'cvar'

        if l2o_loss_type == 'final':
            # Original loss (only final iterate)
            def l2o_wrapper(stepsizes, A_batch, b_batch, z0_batch, x_opt_batch, f_opt_batch):
                """L2O pipeline wrapper for logistic regression."""
                # Vmap over batched A matrices
                batch_pep_obj_func = jax.vmap(
                    lambda A, b, z0, x_opt, f_opt: logreg_pep_obj(
                        stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, traj_fn, cfg.pep_obj
                    ),
                    in_axes=(0, 0, 0, 0, 0)  # A is batched
                )
                pep_objs = batch_pep_obj_func(A_batch, b_batch, z0_batch, x_opt_batch, f_opt_batch)

                # Risk computation
                if risk_type == 'expectation':
                    return jnp.mean(pep_objs)
                elif risk_type == 'cvar':
                    N = pep_objs.shape[0]
                    k = max(int(np.ceil(alpha * N)), 1)
                    sorted_objs = jnp.sort(pep_objs)[::-1]
                    return jnp.mean(sorted_objs[:k])
                else:
                    raise ValueError(f"Unknown risk_type: {risk_type}")
        else:
            # Use improved loss formulation (cumulative, weighted, per_step, etc.)
            def l2o_wrapper(stepsizes, A_batch, b_batch, z0_batch, x_opt_batch, f_opt_batch):
                """L2O pipeline wrapper with improved loss for better gradients."""
                # Vmap over batched A matrices
                batch_loss_func = jax.vmap(
                    lambda A, b, z0, x_opt, f_opt: logreg_trajectory_loss(
                        stepsizes, A, b, z0, x_opt, f_opt, delta, K_max, traj_fn, cfg.pep_obj,
                        loss_type=l2o_loss_type, decay_rate=decay_rate
                    ),
                    in_axes=(0, 0, 0, 0, 0)  # A is batched
                )
                losses = batch_loss_func(A_batch, b_batch, z0_batch, x_opt_batch, f_opt_batch)

                # Risk computation
                if risk_type == 'expectation':
                    return jnp.mean(losses)
                elif risk_type == 'cvar':
                    N = losses.shape[0]
                    k = max(int(np.ceil(alpha * N)), 1)
                    sorted_losses = jnp.sort(losses)[::-1]
                    return jnp.mean(sorted_losses[:k])
                else:
                    raise ValueError(f"Unknown risk_type: {risk_type}")
        
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
        
        # Sample new batch (includes A now)
        key, A_batch, b_batch, z0_batch, x_opt_batch, f_opt_batch = sample_batch(key)

        # Compute loss and gradients
        iter_start_time = time.time()
        loss, d_stepsizes = value_and_grad_fn(
            stepsizes, A_batch, b_batch, z0_batch, x_opt_batch, f_opt_batch
        )
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

def run_gd_for_K_lpep_logreg(cfg, K_max, problem_data, t_init, gd_iters, eta_t,
                              alg, csv_path):
    """
    Run gradient descent for learning PEP (lpep) for logistic regression.

    Optimizes step sizes to minimize worst-case PEP objective using SCS solver.
    No samples, no min-max - pure deterministic PEP optimization.
    """
    log.info(f"=== Running lpep GD for K={K_max}, alg={alg} ===")

    L = problem_data['L']
    mu = problem_data['mu']
    R = problem_data['R']

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
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

    all_stepsizes_vals = [tuple(stepsizes)]
    all_losses = []

    log.info(f"Using wc_pep_scs_solve for lpep optimization")

    # Define PEP loss function
    def pep_loss_fn(stepsizes_list):
        """Compute PEP worst-case bound using wc_pep_scs_solve."""
        if alg == 'vanilla_gd':
            t = stepsizes_list[0]
            pep_data = construct_gd_pep_data(t, mu, L, R, K_max, cfg.pep_obj)
        else:  # nesterov_fgm
            t, beta = stepsizes_list[0], stepsizes_list[1]
            pep_data = construct_fgm_pep_data(t, beta, mu, L, R, K_max, cfg.pep_obj)

        A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]
        loss = wc_pep_scs_solve(A_obj, b_obj, A_vals, b_vals, c_vals)
        return loss

    value_and_grad_fn = jax.value_and_grad(pep_loss_fn)

    # Determine update mask for learn_beta
    learn_beta = cfg.get('learn_beta', True)
    if has_beta and not learn_beta:
        update_mask = [True, False]  # Update t, keep beta fixed
        log.info(f'learn_beta=False: beta will NOT be updated')
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

    # Projection to keep stepsizes non-negative
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

        # Compute loss and gradients
        current_loss, grads = value_and_grad_fn(stepsizes)
        log.info(f'  PEP loss: {current_loss:.6f}')

        all_losses.append(float(current_loss))

        # Check for NaN gradients
        if any(jnp.any(jnp.isnan(g)) for g in grads):
            log.warning(f'NaN gradients at iter {iter_num}, skipping update')
            all_stepsizes_vals.append(tuple(stepsizes))
            continue

        # Update stepsizes
        stepsizes = optimizer.step(stepsizes, grads, proj_x_fn=proj_nonneg)
        all_stepsizes_vals.append(tuple(stepsizes))

        # Save progress
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
    log.info(f"  L = {problem_data['L']:.6f}")
    log.info(f"  mu = {problem_data['mu']:.6f}")
    log.info(f"  R = {problem_data['R']:.6f}")
    log.info(f"  delta = {problem_data['delta']}")
    log.info(f"  N_data = {problem_data['N_data']}")
    log.info(f"  n = {problem_data['n']}")
    log.info(f"  A_std = {problem_data['A_std']}")
    
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
            # LPEP: deterministic PEP minimization
            run_gd_for_K_lpep_logreg(
                cfg, K, problem_data, t_init, sgd_iters, eta_t,
                alg, csv_path
            )
        else:
            # LDRO-PEP or L2O: with samples
            run_sgd_for_K(
                cfg, K, key, problem_data, t_init,
                sgd_iters, eta_t,
                eps, alpha, alg, optimizer_type,
                csv_path
            )

    log.info("=" * 60)
    log.info("Logistic Regression learning experiment complete!")
    log.info("=" * 60)
