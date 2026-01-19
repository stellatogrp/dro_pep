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
        R: Distance from origin to optimal (||x_opt||)
    """
    n = A_np.shape[1]
    x = cp.Variable(n)
    obj = 0.5 * cp.sum_squares(A_np @ x - b_np) + lambd * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver='CLARABEL')
    
    x_opt = x.value
    R = np.linalg.norm(x_opt)
    
    return x_opt, R


def solve_batch_lasso_cvxpy(A_np, b_batch_np, lambd):
    """
    Solve batch of Lasso problems.
    
    Args:
        A_np: (m, n) numpy array
        b_batch_np: (N, m) numpy array of b vectors
        lambd: L1 regularization parameter
        
    Returns:
        x_opt_batch: (N, n) array of optimal solutions
        R_max: Maximum radius across all samples
    """
    N = b_batch_np.shape[0]
    n = A_np.shape[1]
    
    x_opt_batch = np.zeros((N, n))
    R_max = 0.0
    
    for i in trange(N):
        x_opt, R = solve_lasso_cvxpy(A_np, b_batch_np[i], lambd)
        x_opt_batch[i] = x_opt
        R_max = max(R_max, R)
    
    return x_opt_batch, R_max


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
    _, R_max = solve_batch_lasso_cvxpy(A_np, b_batch_np, cfg.lambd)
    
    log.info(f"Computed R = {R_max:.6f}")
    return R_max


# =============================================================================
# Sample Batch for SGD Iteration
# =============================================================================

def sample_lasso_batch(key, A, N, p_xsamp_nonzero, noise_eps):
    """
    Sample a batch of Lasso problems for one SGD iteration.
    
    Args:
        key: JAX random key
        A: (m, n) JAX array
        N: Batch size
        p_xsamp_nonzero: Sparsity parameter
        noise_eps: Noise level
        
    Returns:
        b_batch: (N, m) array of b vectors
    """
    return generate_batch_b_jax(key, A, N, p_xsamp_nonzero, noise_eps)


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

    log.info(A_jax)
    
    # Compute L and mu
    L, mu = compute_lasso_params(A_jax)
    log.info(f"L = {L:.6f}, mu = {mu:.6f}")
    
    # Select R based on strong convexity
    # mu close to 0 means non-strongly convex (m < n case)
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
# Main Entry Point (placeholder for now)
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
    
    # Setup problem
    problem_data = setup_lasso_problem(cfg)
    
    log.info(f"Problem setup complete:")
    log.info(f"  A shape: {problem_data['A_jax'].shape}")
    log.info(f"  L = {problem_data['L']:.6f}")
    log.info(f"  mu = {problem_data['mu']:.6f}")
    log.info(f"  R = {problem_data['R']:.6f}")
    log.info(f"  lambd = {problem_data['lambd']}")
    
    # Test sampling a batch
    log.info(f"\nSampling test batch of N={cfg.N} problems...")
    key = jax.random.PRNGKey(cfg.seed)
    b_batch = sample_lasso_batch(
        key, problem_data['A_jax'], cfg.N,
        cfg.p_xsamp_nonzero, cfg.noise_eps
    )
    log.info(f"  b_batch shape: {b_batch.shape}")
    
    # TODO: Stepsize initialization and SGD loop will be added later
    log.info("\nSetup complete. SGD loop to be implemented.")
    
    return problem_data