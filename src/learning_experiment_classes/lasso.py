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
from learning.jax_scs_layer import dro_scs_solve, wc_pep_scs_solve, compute_preconditioner_from_samples

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=6, suppress=True)

log = logging.getLogger(__name__)


def generate_A(seed, m, n, scaling=1.0):
    """
    Generate sparse A matrix using NumPy.

    Args:
        seed: Random seed for reproducibility
        m: Number of rows
        n: Number of columns
        scaling: Scale parameter for the normal distribution (default 1.0)
                 For out-of-distribution, use scaling=4.0

    Returns:
        A: (m, n) numpy array with columns normalized to unit norm
    """
    np.random.seed(seed)
    A = np.random.normal(scale=scaling/m, size=(m, n))
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-10)
    A = A / col_norms
    return A

def generate_single_b_jax(key, A, p_xsamp_nonzero, b_noise_std):
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
    noise = b_noise_std * jax.random.normal(key3, (m,))
    b = A @ x_samp + noise
    
    return b


def compute_lasso_params(A):
    m, n = A.shape
    ATA = A.T @ A
    eigvals = jnp.linalg.eigvalsh(ATA)
    L = jnp.max(eigvals)
    mu = jnp.min(eigvals) if m >= n else 0.0
    return float(L), float(mu)


def generate_batch_b_jax(key, A, N, p_xsamp_nonzero, b_noise_std):
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
                           p_xsamp_nonzero=p_xsamp_nonzero,
                           b_noise_std=b_noise_std)
    b_batch = jax.vmap(generate_one)(keys)
    return b_batch


# =============================================================================
# Lasso Solution (using CVXPY with DPP for fast re-solves)
# =============================================================================

class LassoProblemDPP:
    """
    DPP-parametrized Lasso problem for fast batch solving.

    Creates the problem structure once, then updates parameters and re-solves
    without rebuilding the problem each time.
    """

    def __init__(self, A_np, lambd):
        """
        Initialize the parametrized Lasso problem.

        Args:
            A_np: (m, n) numpy array - fixed design matrix
            lambd: L1 regularization parameter
        """
        m, n = A_np.shape
        self.A = A_np
        self.lambd = lambd

        # Create CVXPY parameter for b (will be updated for each solve)
        self.b_param = cp.Parameter(m)

        # Create variable
        self.x = cp.Variable(n)

        # Build objective: 0.5 * ||Ax - b||^2 + lambd * ||x||_1
        self.obj = 0.5 * cp.sum_squares(self.A @ self.x - self.b_param) + lambd * cp.norm(self.x, 1)

        # Create problem (done once)
        self.prob = cp.Problem(cp.Minimize(self.obj))

    def solve(self, b_np):
        """
        Solve Lasso for a given b vector.

        Args:
            b_np: (m,) numpy array

        Returns:
            x_opt: (n,) optimal solution
            f_opt: optimal objective value
        """
        # Update parameter value
        self.b_param.value = b_np

        # Solve (reuses problem structure)
        self.prob.solve(solver='CLARABEL')

        return self.x.value, self.prob.value

    def solve_batch(self, b_batch_np):
        """
        Solve batch of Lasso problems efficiently.

        Args:
            b_batch_np: (N, m) numpy array of b vectors

        Returns:
            x_opt_batch: (N, n) array of optimal solutions
            f_opt_batch: (N,) array of optimal objective values
            R_max: Maximum radius across all samples
        """
        N = b_batch_np.shape[0]
        n = self.A.shape[1]

        x_opt_batch = np.zeros((N, n))
        f_opt_batch = np.zeros(N)
        R_max = 0.0

        for i in range(N):
            x_opt, f_opt = self.solve(b_batch_np[i])
            x_opt_batch[i] = x_opt
            f_opt_batch[i] = f_opt
            R = np.linalg.norm(x_opt)
            R_max = max(R_max, R)

        return x_opt_batch, f_opt_batch, R_max


def solve_lasso_cvxpy(A_np, b_np, lambd):
    """Legacy function for single Lasso solve (creates new problem each time)."""
    n = A_np.shape[1]
    x = cp.Variable(n)
    obj = 0.5 * cp.sum_squares(A_np @ x - b_np) + lambd * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver='CLARABEL')

    x_opt = x.value
    f_opt = prob.value

    return x_opt, f_opt


def solve_batch_lasso_cvxpy(A_np, b_batch_np, lambd, lasso_dpp=None):
    """
    Solve batch of Lasso problems.

    Args:
        A_np: (m, n) numpy array
        b_batch_np: (N, m) numpy array of b vectors
        lambd: L1 regularization parameter
        lasso_dpp: Optional pre-created LassoProblemDPP instance for speed

    Returns:
        x_opt_batch: (N, n) array of optimal solutions
        f_opt_batch: (N,) array of optimal objective values
        R_max: Maximum radius across all samples
    """
    if lasso_dpp is not None:
        # Fast path: use pre-created DPP problem
        return lasso_dpp.solve_batch(b_batch_np)

    # Slow path: create new problem for each sample (legacy behavior)
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


def compute_sample_radius(cfg, A_np, lasso_dpp=None):
    log.info(f"Computing R from {cfg.R_sample_size} samples...")

    # Create DPP problem if not provided
    if lasso_dpp is None:
        lasso_dpp = LassoProblemDPP(A_np, cfg.lambd)

    # Generate b samples using the b_seed
    key = jax.random.PRNGKey(cfg.R_seed)
    b_batch = generate_batch_b_jax(
        key, jnp.array(A_np), cfg.R_sample_size,
        cfg.p_xsamp_nonzero
    )
    b_batch_np = np.array(b_batch)

    # Solve all Lasso problems using DPP
    _, _, R_max = solve_batch_lasso_cvxpy(A_np, b_batch_np, cfg.lambd, lasso_dpp=lasso_dpp)

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


def sample_lasso_batch(key, A_jax, A_np, N, p_xsamp_nonzero, b_noise_std, lambd, lasso_dpp=None):
    # Generate b vectors
    b_batch = generate_batch_b_jax(key, A_jax, N, p_xsamp_nonzero, b_noise_std)

    # Solve Lasso to get x_opt, f_opt for each sample
    b_batch_np = np.array(b_batch)
    x_opt_batch_np, f_opt_batch_np, _ = solve_batch_lasso_cvxpy(
        A_np, b_batch_np, lambd, lasso_dpp=lasso_dpp
    )

    return b_batch, jnp.array(x_opt_batch_np), jnp.array(f_opt_batch_np)


def compute_dead_zone_fraction(traj_fn, stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max):
    """
    Count fraction of components in soft-threshold dead zone per iteration.

    This diagnostic function helps identify if gradients are uniform due to
    too many trajectory components falling into the soft-threshold dead zone
    (where |v| < gamma * lambd and gradients are zero).

    Args:
        traj_fn: Trajectory function (ISTA or FISTA)
        stepsizes: Step sizes (gamma array for ISTA, or (gamma, beta) for FISTA)
        A, b: Problem data
        x0: Initial point (shifted coordinates)
        x_opt: Optimal point
        f_opt: Optimal value
        lambd: L1 regularization parameter
        K_max: Number of iterations

    Returns:
        List of dead zone fractions for each iteration k = 0, ..., K_max-1
    """
    # Get the gamma values
    if isinstance(stepsizes, tuple):
        gamma = stepsizes[0]  # FISTA case
    else:
        gamma = stepsizes  # ISTA case

    result = traj_fn(stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max, return_Gram_representation=False)
    x_iter = result[0]  # (n, K+1) in shifted coordinates

    dead_fractions = []
    for k in range(K_max):
        # Get x_k in original coordinates
        x_k = x_iter[:, k] + x_opt
        # Compute gradient at x_k
        g_k = A.T @ (A @ x_k - b)
        # Compute pre-proximal point y_k
        y_k = x_k - gamma[k] * g_k
        # Check dead zone: |y_k| < gamma * lambd
        threshold = gamma[k] * lambd
        in_dead_zone = jnp.abs(y_k) < threshold
        dead_fractions.append(float(jnp.mean(in_dead_zone)))

    return dead_fractions


# =============================================================================
# L2O Pipeline (Learning to Optimize without DRO)
# =============================================================================

@partial(jax.jit, static_argnames=['traj_fn', 'K_max', 'loss_type', 'decay_rate'])
def lasso_trajectory_loss(stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max, traj_fn,
                          loss_type='final', decay_rate=0.9):
    """
    Compute loss from ISTA/FISTA trajectory with different loss formulations.

    This function provides multiple loss formulations to address the uniform
    gradient problem in L2O for Lasso.

    Args:
        stepsizes: Step sizes (gamma array for ISTA, or (gamma, beta) for FISTA)
        A, b: Problem data
        x0: Initial point (shifted coordinates, so x0 = 0)
        x_opt, f_opt: Optimal point and value
        lambd: L1 regularization parameter
        K_max: Number of iterations
        traj_fn: Trajectory function (ISTA or FISTA)
        loss_type: Loss formulation type:
            - 'final': Only final iterate loss (original, causes uniform gradients)
            - 'cumulative': Sum of losses at all iterates (gives each gamma_k influence)
            - 'weighted': Exponentially weighted sum (emphasizes later iterations)
            - 'per_step': Loss improvement per step (directly ties gamma_k to step k)
        decay_rate: For 'weighted' loss, the exponential decay rate (0 < decay < 1)

    Returns:
        Scalar loss value
    """
    # Run trajectory
    traj_result = traj_fn(stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max,
                          return_Gram_representation=False)
    x_iter = traj_result[0]  # Shape (n, K+1), columns are x_0, ..., x_K in shifted coords

    def compute_obj_at_k(k):
        """Compute f(x_k) - f_opt for iterate k."""
        x_k_shifted = x_iter[:, k]
        x_k = x_k_shifted + x_opt
        f1_xk = 0.5 * jnp.sum((A @ x_k - b) ** 2)
        f2_xk = lambd * jnp.sum(jnp.abs(x_k))
        return f1_xk + f2_xk - f_opt

    def compute_dist_at_k(k):
        """Compute ||x_k - x_opt||^2 for iterate k."""
        x_k_shifted = x_iter[:, k]
        return jnp.sum(x_k_shifted ** 2)  # x_k_shifted = x_k - x_opt

    if loss_type == 'final':
        # Original: only final iterate (causes uniform gradients)
        return compute_obj_at_k(K_max)

    elif loss_type == 'cumulative':
        # Sum of losses at all iterates: gives each gamma_k direct influence
        # gamma_k affects x_{k+1}, x_{k+2}, ..., x_K
        # This creates a "cascade" effect where early gammas affect more terms
        losses = jnp.array([compute_obj_at_k(k) for k in range(1, K_max + 1)])
        return jnp.mean(losses)

    elif loss_type == 'weighted':
        # Exponentially weighted: later iterations weighted more
        # w_k = decay^(K-k), so w_K = 1, w_{K-1} = decay, etc.
        losses = jnp.array([compute_obj_at_k(k) for k in range(1, K_max + 1)])
        weights = jnp.array([decay_rate ** (K_max - k) for k in range(1, K_max + 1)])
        weights = weights / jnp.sum(weights)  # Normalize
        return jnp.sum(weights * losses)

    elif loss_type == 'per_step':
        # Loss improvement per step: directly ties gamma_k to its effect
        # loss_k = (f(x_k) - f(x_{k+1})) measures how much step k improved
        # We minimize negative improvement (maximize improvement)
        improvements = []
        for k in range(K_max):
            loss_k = compute_obj_at_k(k)
            loss_kp1 = compute_obj_at_k(k + 1)
            # Improvement should be positive; we want to maximize it
            improvements.append(loss_k - loss_kp1)
        # Return negative mean improvement (so minimizing = maximizing improvement)
        # Plus final loss to ensure we reach optimum
        return compute_obj_at_k(K_max) - 0.1 * jnp.mean(jnp.array(improvements))

    elif loss_type == 'distance_cumulative':
        # Cumulative distance to optimal (often smoother than objective)
        dists = jnp.array([compute_dist_at_k(k) for k in range(1, K_max + 1)])
        return jnp.mean(dists)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


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
    Uses the standard final-iterate loss f(x_K) - f_opt.

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


def l2o_lasso_pipeline_v2(stepsizes, A, b_batch, x0_batch, x_opt_batch, f_opt_batch,
                          lambd, K_max, traj_fn, loss_type, risk_type, alpha=0.1, decay_rate=0.9):
    """
    Improved L2O pipeline with alternative loss formulations for better gradients.

    This version addresses the uniform gradient problem by using loss formulations
    that give each step size gamma_k more direct influence on the loss.

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
        loss_type: 'final', 'cumulative', 'weighted', 'per_step', 'distance_cumulative'
        risk_type: 'expectation' or 'cvar'
        alpha: CVaR confidence level
        decay_rate: For 'weighted' loss type

    Returns:
        Scalar loss value
    """
    # vmap over the batch
    batch_loss_func = jax.vmap(
        lambda b, x0, x_opt, f_opt: lasso_trajectory_loss(
            stepsizes, A, b, x0, x_opt, f_opt, lambd, K_max, traj_fn,
            loss_type=loss_type, decay_rate=decay_rate
        ),
        in_axes=(0, 0, 0, 0)
    )
    losses = batch_loss_func(b_batch, x0_batch, x_opt_batch, f_opt_batch)

    if risk_type == 'expectation':
        return jnp.mean(losses)
    elif risk_type == 'cvar':
        N = losses.shape[0]
        k = max(int(np.ceil(alpha * N)), 1)
        sorted_losses = jnp.sort(losses)[::-1]
        return jnp.mean(sorted_losses[:k])
    else:
        raise ValueError(f"Unknown risk_type: {risk_type}")


def compute_silver_stepsizes(L, K):
    """
    Compute silver step sizes for gradient descent.

    The silver step sizes are known to be optimal for gradient descent on
    smooth convex functions and provide a good differentiated initialization.

    Based on: "The Silver Stepsize Schedule" (Altschuler & Parrilo, 2023)

    Args:
        L: Smoothness constant
        K: Number of iterations

    Returns:
        Array of K step sizes
    """
    # Silver ratio
    rho = 1 + np.sqrt(2)

    # Silver step sizes: gamma_k = (1/L) * (rho^k - rho^{-k}) / (rho^k + rho^{-k})
    # This creates a schedule that starts small and increases
    stepsizes = []
    for k in range(1, K + 1):
        rho_k = rho ** k
        rho_neg_k = rho ** (-k)
        gamma_k = (1.0 / L) * (rho_k - rho_neg_k) / (rho_k + rho_neg_k)
        stepsizes.append(gamma_k)

    return jnp.array(stepsizes)


def compute_geometric_stepsizes(L, K, start_factor=0.5, end_factor=1.5):
    """
    Compute geometrically spaced step sizes.

    Creates a differentiated step size schedule by geometric interpolation.

    Args:
        L: Smoothness constant
        K: Number of iterations
        start_factor: Multiplier for first step size (relative to 1/L)
        end_factor: Multiplier for last step size (relative to 1/L)

    Returns:
        Array of K step sizes
    """
    base = 1.0 / L
    factors = np.geomspace(start_factor, end_factor, K)
    return jnp.array(factors * base)


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

    # Create DPP-parametrized Lasso problem for fast batch solving
    # This avoids rebuilding the CVXPY problem at each SGD iteration
    lasso_dpp = LassoProblemDPP(A_np, lambd)
    log.info("Created DPP-parametrized Lasso problem for fast batch solving")

    # Helper to sample a batch of Lasso problems
    def sample_batch(sample_key):
        batch_key, next_key = jax.random.split(sample_key)
        b_batch, x_opt_batch, f_opt_batch = sample_lasso_batch(
            batch_key, A_jax, A_np, N_val,
            cfg.p_xsamp_nonzero, cfg.b_noise_std, lambd, lasso_dpp=lasso_dpp
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
        # Use improved loss formulation if specified
        l2o_loss_type = cfg.get('l2o_loss_type', 'final')
        decay_rate = cfg.get('l2o_decay_rate', 0.9)
        log.info(f"L2O loss type: {l2o_loss_type}")

        if l2o_loss_type == 'final':
            # Original loss (only final iterate)
            def lasso_l2o_wrapper(stepsizes_tuple, b_batch, x0_batch, x_opt_batch, f_opt_batch):
                """L2O pipeline wrapper for Lasso."""
                if has_beta:
                    traj_stepsizes = stepsizes_tuple
                else:
                    traj_stepsizes = stepsizes_tuple[0]

                return l2o_lasso_pipeline(
                    traj_stepsizes, A_jax, b_batch, x0_batch, x_opt_batch, f_opt_batch,
                    lambd, K_max, traj_fn, cfg.pep_obj, risk_type, alpha
                )
        else:
            # Use improved loss formulation (cumulative, weighted, per_step, etc.)
            def lasso_l2o_wrapper(stepsizes_tuple, b_batch, x0_batch, x_opt_batch, f_opt_batch):
                """L2O pipeline wrapper with improved loss for better gradients."""
                if has_beta:
                    traj_stepsizes = stepsizes_tuple
                else:
                    traj_stepsizes = stepsizes_tuple[0]

                return l2o_lasso_pipeline_v2(
                    traj_stepsizes, A_jax, b_batch, x0_batch, x_opt_batch, f_opt_batch,
                    lambd, K_max, traj_fn, l2o_loss_type, risk_type, alpha, decay_rate
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

        # Debug logging for gradient analysis
        if cfg.get('debug_gradients', False):
            grad_gamma = d_stepsizes[0]
            grad_magnitudes = jnp.abs(grad_gamma)
            max_grad = jnp.max(grad_magnitudes)
            min_grad = jnp.min(grad_magnitudes)
            ratio = max_grad / (min_grad + 1e-12)
            log.info(f'[DEBUG] Per-k gradient magnitudes: {[f"{float(g):.4e}" for g in grad_magnitudes]}')
            log.info(f'[DEBUG] Gradient max/min ratio: {float(ratio):.4f}')
            log.info(f'[DEBUG] Gradient std: {float(jnp.std(grad_magnitudes)):.6e}')
            log.info(f'[DEBUG] Gradient mean: {float(jnp.mean(grad_magnitudes)):.6e}')
            # Warn if gradients are too uniform
            if float(ratio) < 2.0:
                log.warning(f'[DEBUG] Gradients are too uniform (ratio={float(ratio):.2f} < 2.0). '
                           f'This may indicate soft-threshold dead zone issues.')

            # Compute and log dead zone fractions for first sample
            if has_beta:
                traj_stepsizes_debug = stepsizes
            else:
                traj_stepsizes_debug = stepsizes[0]
            dead_fracs = compute_dead_zone_fraction(
                traj_fn, traj_stepsizes_debug, A_jax, b_batch[0], x0_batch[0],
                x_opt_batch[0], f_opt_batch[0], lambd, K_max
            )
            avg_dead = sum(dead_fracs) / len(dead_fracs)
            log.info(f'[DEBUG] Dead zone fractions per k: {[f"{d:.3f}" for d in dead_fracs]}')
            log.info(f'[DEBUG] Average dead zone fraction: {avg_dead:.3f}')
            if avg_dead > 0.5:
                log.warning(f'[DEBUG] High dead zone fraction ({avg_dead:.1%}) - '
                           f'many components have zero gradient through soft-threshold.')

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
    

def run_gd_for_K_lpep_lasso(cfg, K_max, problem_data, gamma_init, gd_iters, eta_t,
                             alg, csv_path):
    """
    Run gradient descent for learning PEP (lpep) for Lasso - no samples, no min-max.

    Optimizes step sizes to minimize the standard (worst-case) PEP objective
    using SCS solver with wc_pep_scs_solve for differentiation.

    Args:
        cfg: Configuration object
        K_max: Number of algorithm iterations
        problem_data: Dictionary with Lasso problem data (A, L, mu, R, lambd)
        gamma_init: Initial step size (scalar or vector)
        gd_iters: Number of gradient descent iterations
        eta_t: Learning rate for step sizes
        alg: Algorithm name ('ista' or 'fista')
        csv_path: Path to save progress CSV
    """
    log.info(f"=== Running lpep GD for K={K_max}, alg={alg} ===")

    # Extract problem data
    L = problem_data['L']
    mu = problem_data['mu']
    R = problem_data['R']

    # Initialize stepsizes based on algorithm and stepsize_type
    is_vector_gamma = cfg.stepsize_type == 'vector'

    if alg == 'ista':
        # Just step size gamma
        if is_vector_gamma:
            gamma = jnp.atleast_1d(jnp.array(gamma_init))
            if gamma.ndim == 0 or gamma.shape[0] == 1:
                gamma = jnp.full(K_max, float(gamma_init))
        else:
            gamma = jnp.array(float(gamma_init))  # Keep as scalar
        stepsizes = [gamma]
        has_beta = False
    elif alg == 'fista':
        # Step size gamma and momentum beta
        if is_vector_gamma:
            gamma = jnp.atleast_1d(jnp.array(gamma_init))
            if gamma.ndim == 0 or gamma.shape[0] == 1:
                gamma = jnp.full(K_max, float(gamma_init))
        else:
            gamma = jnp.array(float(gamma_init))  # Keep as scalar
        # FISTA beta initialization
        betas_t = [1.0]
        for k in range(K_max):
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * betas_t[-1]**2))
            betas_t.append(t_new)
        beta_init = jnp.array(betas_t)  # Raw t_k sequence of length K+1
        stepsizes = [gamma, beta_init]
        has_beta = True
    else:
        log.error(f"Algorithm '{alg}' is not implemented.")
        raise ValueError(f"Unknown algorithm: {alg}")

    # Track step size values for logging
    all_stepsizes_vals = [tuple(stepsizes)]
    all_losses = []  # Will be filled as we go - loss[i] corresponds to stepsizes[i]

    log.info(f"Using wc_pep_scs_solve for lpep optimization")

    # Define the PEP loss function (differentiable w.r.t. stepsizes)
    def pep_loss_fn(stepsizes_list):
        """Compute PEP worst-case bound for given stepsizes using wc_pep_scs_solve."""
        if alg == 'ista':
            gamma = stepsizes_list[0]
            pep_data = construct_ista_pep_data(gamma, mu, L, R, K_max, cfg.pep_obj)
        else:  # fista
            gamma, beta = stepsizes_list[0], stepsizes_list[1]
            pep_data = construct_fista_pep_data(gamma, beta, mu, L, R, K_max, cfg.pep_obj)

        A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]

        # Call wc_pep_scs_solve with PEP data
        # JAX autodiff will automatically differentiate through this
        loss = wc_pep_scs_solve(A_obj, b_obj, A_vals, b_vals, c_vals)
        return loss

    # Create value and gradient function
    value_and_grad_fn = jax.value_and_grad(pep_loss_fn)

    # Determine update mask for learn_beta
    # If learn_beta=False and we have beta (fista), only update gamma
    learn_beta = cfg.get('learn_beta', True)
    if has_beta and not learn_beta:
        update_mask = [True, False]  # Update gamma, keep beta fixed
        log.info(f'learn_beta=False: beta will NOT be updated during optimization')
    else:
        update_mask = None  # Update all parameters

    # Initialize AdamWMin optimizer
    optimizer = AdamWMin(
        x_params=stepsizes,
        lr=eta_t,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=getattr(cfg, 'weight_decay', 0.0),
        update_mask=update_mask,
    )

    # Projection function to keep stepsizes non-negative
    def proj_nonneg(params):
        return [jnp.maximum(p, 1e-6) for p in params]

    # GD iterations (descent only, no ascent)
    for iter_num in range(gd_iters):
        gamma = stepsizes[0]
        if is_vector_gamma:
            gamma_log = '[' + ', '.join(f'{x:.5f}' for x in gamma.tolist()) + ']'
        else:
            gamma_log = f'{float(gamma):.5f}'
        if has_beta:
            beta = stepsizes[1]
            beta_log = '[' + ', '.join(f'{x:.5f}' for x in beta.tolist()) + ']'
            log.info(f'K={K_max}, iter={iter_num}, gamma={gamma_log}, beta={beta_log}')
        else:
            log.info(f'K={K_max}, iter={iter_num}, gamma={gamma_log}')

        # Compute loss and gradients (loss corresponds to CURRENT stepsizes before update)
        current_loss, grads = value_and_grad_fn(stepsizes)
        log.info(f'  PEP loss: {current_loss:.6f}')

        # Store loss for current stepsizes (iteration iter_num)
        all_losses.append(float(current_loss))

        # Check for NaN gradients
        if any(jnp.any(jnp.isnan(g)) for g in grads):
            log.warning(f'NaN gradients at iter {iter_num}, skipping update')
            all_stepsizes_vals.append(tuple(stepsizes))
            continue

        # Update stepsizes via AdamWMin
        stepsizes = optimizer.step(stepsizes, grads, proj_x_fn=proj_nonneg)

        # Store updated stepsizes for next iteration
        all_stepsizes_vals.append(tuple(stepsizes))

        # Save progress
        df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_gamma, has_beta, all_losses)
        df.to_csv(csv_path, index=False)

    # Final save
    df = build_stepsizes_df(all_stepsizes_vals, K_max, is_vector_gamma, has_beta, all_losses)
    gamma = stepsizes[0]
    if is_vector_gamma:
        gamma_str = '[' + ', '.join(f'{x:.6f}' for x in gamma.tolist()) + ']'
    else:
        gamma_str = f'{float(gamma):.6f}'
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
        vector_init = cfg.get('vector_init', 'fixed')

        if is_vector:
            if vector_init == 'fixed':
                # Uniform initialization (original)
                gamma_init = jnp.full(K, gamma_init_scalar)
                log.info(f"Using fixed uniform initialization: gamma_k = {gamma_init_scalar:.6f}")
            elif vector_init == 'silver':
                # Silver step sizes (known-good differentiated schedule)
                gamma_init = compute_silver_stepsizes(L, K)
                log.info(f"Using silver step size initialization: {[f'{g:.6f}' for g in gamma_init]}")
            elif vector_init == 'geometric':
                # Geometric interpolation
                start_factor = cfg.get('geometric_start', 0.5)
                end_factor = cfg.get('geometric_end', 1.5)
                gamma_init = compute_geometric_stepsizes(L, K, start_factor, end_factor)
                log.info(f"Using geometric initialization [{start_factor}-{end_factor}]: {[f'{g:.6f}' for g in gamma_init]}")
            elif vector_init == 'increasing':
                # Linearly increasing step sizes
                factors = jnp.linspace(0.5, 1.5, K)
                gamma_init = factors / L
                log.info(f"Using increasing initialization: {[f'{g:.6f}' for g in gamma_init]}")
            elif vector_init == 'decreasing':
                # Linearly decreasing step sizes
                factors = jnp.linspace(1.5, 0.5, K)
                gamma_init = factors / L
                log.info(f"Using decreasing initialization: {[f'{g:.6f}' for g in gamma_init]}")
            else:
                raise ValueError(f"Unknown vector_init: {vector_init}")
        else:
            gamma_init = jnp.array([gamma_init_scalar])  # Still array for consistency
        
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
            # LPEP is complete, continue to next K
            continue
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
                cfg.p_xsamp_nonzero, cfg.b_noise_std, lambd
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
            log.info(precond_inv)
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


def lasso_out_of_sample_run(cfg):
    """
    Generate and save out-of-sample test problems for Lasso.

    For in-distribution (validation and test):
        - Use the SAME A matrix as training (A_seed)
        - Generate out_of_sample_val_N different b vectors for validation
        - Generate out_of_sample_test_N different b vectors for test
        - Solve all Lasso problems to get optimal solutions and values
        - Save A once, save validation and test data separately

    For out-of-distribution:
        - Generate NEW A matrix with scale=4/m (A_out_of_dist_seed, scaling=4)
        - Generate out_of_dist_N different b vectors
        - Solve all Lasso problems to get optimal solutions and values
        - Save A once, save all data

    Output files (saved in Hydra run directory):
        Validation (in-distribution):
            - A_in_dist.npz: Contains 'A' array of shape (m, n) [shared by val and test]
            - b_val_samples.npz: Contains 'b' array of shape (out_of_sample_val_N, m)
            - x_opt_val_samples.npz: Contains 'x_opt' array of shape (out_of_sample_val_N, n)
            - f_opt_val_samples.npz: Contains 'f_opt' array of shape (out_of_sample_val_N,)

        Test (in-distribution):
            - b_test_samples.npz: Contains 'b' array of shape (out_of_sample_test_N, m)
            - x_opt_test_samples.npz: Contains 'x_opt' array of shape (out_of_sample_test_N, n)
            - f_opt_test_samples.npz: Contains 'f_opt' array of shape (out_of_sample_test_N,)

        Out-of-distribution:
            - A_out_of_dist.npz: Contains 'A' array of shape (m, n)
            - b_out_of_dist_samples.npz: Contains 'b' array of shape (out_of_dist_N, m)
            - x_opt_out_of_dist_samples.npz: Contains 'x_opt' array of shape (out_of_dist_N, n)
            - f_opt_out_of_dist_samples.npz: Contains 'f_opt' array of shape (out_of_dist_N,)
    """
    log.info("=" * 60)
    log.info("Generating Lasso out-of-sample problems")
    log.info("=" * 60)
    log.info(cfg)

    # Extract config values
    m = cfg.m
    n = cfg.n
    lambd = cfg.lambd
    A_seed = cfg.A_seed
    A_out_of_dist_seed = cfg.A_out_of_dist_seed
    out_of_sample_val_N = cfg.out_of_sample_val_N
    out_of_sample_test_N = cfg.out_of_sample_test_N
    out_of_sample_val_seed = cfg.out_of_sample_val_seed
    out_of_sample_test_seed = cfg.out_of_sample_test_seed
    out_of_dist_N = cfg.out_of_dist_N
    out_of_dist_seed = cfg.get('out_of_dist_seed', out_of_sample_val_seed + 1)
    p_xsamp_nonzero = cfg.p_xsamp_nonzero
    b_noise_std = cfg.b_noise_std

    # =========================================================================
    # Generate A matrix (SAME as training, shared by validation and test)
    # =========================================================================
    log.info(f"Generating A matrix for in-distribution sets (A_seed={A_seed})")
    A_in_dist_np = generate_A(A_seed, m, n, scaling=1.0)
    A_in_dist_jax = jnp.array(A_in_dist_np)

    # Save A matrix once (shared by validation and test)
    A_in_dist_path = "A_in_dist.npz"
    np.savez_compressed(A_in_dist_path, A=A_in_dist_np)
    log.info(f"Saved in-distribution A to {A_in_dist_path}, shape: {A_in_dist_np.shape}")

    # Create DPP-parametrized Lasso problem for fast batch solving
    lasso_dpp = LassoProblemDPP(A_in_dist_np, lambd)
    log.info("Created DPP-parametrized Lasso problem for in-distribution sets")

    # =========================================================================
    # Validation Set (in-distribution): SAME A as training
    # =========================================================================
    log.info(f"Generating {out_of_sample_val_N} validation problems (in-distribution)...")

    # Generate validation b vectors using out_of_sample_val_seed
    key_val = jax.random.PRNGKey(out_of_sample_val_seed)
    b_val_batch = generate_batch_b_jax(
        key_val, A_in_dist_jax, out_of_sample_val_N,
        p_xsamp_nonzero, b_noise_std
    )

    # Convert to numpy and save
    b_val_np = np.array(b_val_batch)
    b_val_path = "b_val_samples.npz"
    np.savez_compressed(b_val_path, b=b_val_np)
    log.info(f"Saved validation b samples to {b_val_path}, shape: {b_val_np.shape}")

    # Solve validation Lasso problems and save solutions
    log.info(f"Solving {out_of_sample_val_N} validation Lasso problems...")
    x_opt_val_np = np.zeros((out_of_sample_val_N, n))
    f_opt_val_np = np.zeros(out_of_sample_val_N)

    for i in trange(out_of_sample_val_N, desc="Solving validation problems"):
        x_opt, f_opt = lasso_dpp.solve(b_val_np[i])
        x_opt_val_np[i] = x_opt
        f_opt_val_np[i] = f_opt

    x_opt_val_path = "x_opt_val_samples.npz"
    f_opt_val_path = "f_opt_val_samples.npz"
    np.savez_compressed(x_opt_val_path, x_opt=x_opt_val_np)
    np.savez_compressed(f_opt_val_path, f_opt=f_opt_val_np)
    log.info(f"Saved validation optimal solutions to {x_opt_val_path}, shape: {x_opt_val_np.shape}")
    log.info(f"Saved validation optimal values to {f_opt_val_path}, shape: {f_opt_val_np.shape}")

    # =========================================================================
    # Test Set (in-distribution): SAME A as training
    # =========================================================================
    log.info(f"Generating {out_of_sample_test_N} test problems (in-distribution)...")

    # Generate test b vectors using out_of_sample_test_seed
    key_test = jax.random.PRNGKey(out_of_sample_test_seed)
    b_test_batch = generate_batch_b_jax(
        key_test, A_in_dist_jax, out_of_sample_test_N,
        p_xsamp_nonzero, b_noise_std
    )

    # Convert to numpy and save
    b_test_np = np.array(b_test_batch)
    b_test_path = "b_test_samples.npz"
    np.savez_compressed(b_test_path, b=b_test_np)
    log.info(f"Saved test b samples to {b_test_path}, shape: {b_test_np.shape}")

    # Solve test Lasso problems and save solutions
    log.info(f"Solving {out_of_sample_test_N} test Lasso problems...")
    x_opt_test_np = np.zeros((out_of_sample_test_N, n))
    f_opt_test_np = np.zeros(out_of_sample_test_N)

    for i in trange(out_of_sample_test_N, desc="Solving test problems"):
        x_opt, f_opt = lasso_dpp.solve(b_test_np[i])
        x_opt_test_np[i] = x_opt
        f_opt_test_np[i] = f_opt

    x_opt_test_path = "x_opt_test_samples.npz"
    f_opt_test_path = "f_opt_test_samples.npz"
    np.savez_compressed(x_opt_test_path, x_opt=x_opt_test_np)
    np.savez_compressed(f_opt_test_path, f_opt=f_opt_test_np)
    log.info(f"Saved test optimal solutions to {x_opt_test_path}, shape: {x_opt_test_np.shape}")
    log.info(f"Saved test optimal values to {f_opt_test_path}, shape: {f_opt_test_np.shape}")

    # =========================================================================
    # Out-of-Distribution Test Set: DIFFERENT A with scale=4/m
    # =========================================================================
    log.info(f"Generating {out_of_dist_N} out-of-distribution test problems...")
    log.info(f"Using DIFFERENT A matrix (A_out_of_dist_seed={A_out_of_dist_seed}, scaling=4)")

    # Generate A matrix with scale=4/m (out-of-distribution)
    A_ood_np = generate_A(A_out_of_dist_seed, m, n, scaling=4.0)
    A_ood_jax = jnp.array(A_ood_np)

    # Save A matrix
    A_ood_path = "A_out_of_dist.npz"
    np.savez_compressed(A_ood_path, A=A_ood_np)
    log.info(f"Saved out-of-dist A to {A_ood_path}, shape: {A_ood_np.shape}")

    # Generate b vectors using out_of_dist_seed
    key_ood = jax.random.PRNGKey(out_of_dist_seed)
    b_ood_batch = generate_batch_b_jax(
        key_ood, A_ood_jax, out_of_dist_N,
        p_xsamp_nonzero, b_noise_std
    )

    # Convert to numpy and save
    b_ood_np = np.array(b_ood_batch)
    b_ood_path = "b_out_of_dist_samples.npz"
    np.savez_compressed(b_ood_path, b=b_ood_np)
    log.info(f"Saved out-of-dist b samples to {b_ood_path}, shape: {b_ood_np.shape}")

    # Create DPP-parametrized Lasso problem for out-of-distribution
    lasso_dpp_ood = LassoProblemDPP(A_ood_np, lambd)
    log.info("Created DPP-parametrized Lasso problem for out-of-distribution set")

    # Solve out-of-distribution Lasso problems and save solutions
    log.info(f"Solving {out_of_dist_N} out-of-distribution Lasso problems...")
    x_opt_ood_np = np.zeros((out_of_dist_N, n))
    f_opt_ood_np = np.zeros(out_of_dist_N)

    for i in trange(out_of_dist_N, desc="Solving out-of-dist problems"):
        x_opt, f_opt = lasso_dpp_ood.solve(b_ood_np[i])
        x_opt_ood_np[i] = x_opt
        f_opt_ood_np[i] = f_opt

    x_opt_ood_path = "x_opt_out_of_dist_samples.npz"
    f_opt_ood_path = "f_opt_out_of_dist_samples.npz"
    np.savez_compressed(x_opt_ood_path, x_opt=x_opt_ood_np)
    np.savez_compressed(f_opt_ood_path, f_opt=f_opt_ood_np)
    log.info(f"Saved out-of-dist optimal solutions to {x_opt_ood_path}, shape: {x_opt_ood_np.shape}")
    log.info(f"Saved out-of-dist optimal values to {f_opt_ood_path}, shape: {f_opt_ood_np.shape}")

    # =========================================================================
    # Save metadata for reference
    # =========================================================================
    metadata = {
        'out_of_sample_val_N': out_of_sample_val_N,
        'out_of_sample_test_N': out_of_sample_test_N,
        'out_of_sample_val_seed': out_of_sample_val_seed,
        'out_of_sample_test_seed': out_of_sample_test_seed,
        'out_of_dist_N': out_of_dist_N,
        'out_of_dist_seed': out_of_dist_seed,
        'm': m,
        'n': n,
        'lambd': lambd,
        'A_seed': A_seed,
        'A_out_of_dist_seed': A_out_of_dist_seed,
        'p_xsamp_nonzero': p_xsamp_nonzero,
        'b_noise_std': b_noise_std,
        'A_in_dist_shape': A_in_dist_np.shape,
        'b_val_shape': b_val_np.shape,
        'x_opt_val_shape': x_opt_val_np.shape,
        'f_opt_val_shape': f_opt_val_np.shape,
        'b_test_shape': b_test_np.shape,
        'x_opt_test_shape': x_opt_test_np.shape,
        'f_opt_test_shape': f_opt_test_np.shape,
        'A_ood_shape': A_ood_np.shape,
        'b_ood_shape': b_ood_np.shape,
        'x_opt_ood_shape': x_opt_ood_np.shape,
        'f_opt_ood_shape': f_opt_ood_np.shape,
    }
    metadata_path = "out_of_sample_metadata.npz"
    np.savez_compressed(metadata_path, **metadata)
    log.info(f"Saved metadata to {metadata_path}")

    log.info("=== Lasso out-of-sample generation complete ===")
