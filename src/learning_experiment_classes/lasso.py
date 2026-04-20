"""Lasso problem module for unified learning framework.

Implements LassoProblemModule using the ProblemModule ABC and UnifiedTrainer,
eliminating code duplication from the old lasso.py implementation.

The Lasso problem is:
    min_x  0.5 * ||A @ x - b||^2 + lambd * ||x||_1
         = f1(x) + f2(x)

Key differences from Quad:
- A matrix is FIXED across all samples (bound via closure, not in training_data)
- Optimal solutions must be computed via CVXPY per sample
- ISTA/FISTA algorithms instead of GD/FGM
- Composite PEP structure (f1 smooth + f2 non-smooth)
"""

import diffcp_patch  # noqa: F401  # Apply COO -> CSC fix for diffcp
import jax
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
import logging
import cvxpy as cp
from functools import partial
from typing import Any, Callable, Dict, Tuple
from tqdm import trange

from learning.problem_module import ProblemModule, ProblemData, GroundTruth, Stepsizes, ParameterNames
from learning.unified_trainer import UnifiedTrainer
from learning.pep_constructions import construct_ista_pep_data, construct_fista_pep_data
from learning.trajectories import (
    problem_data_to_ista_trajectories,
    problem_data_to_fista_trajectories,
)

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)


# =============================================================================
# Module-level helper functions for sampling and solving
# =============================================================================

def generate_A(seed, m, n, scaling=1.0):
    """Generate sparse A matrix using NumPy.

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
    A = np.random.normal(scale=scaling / m, size=(m, n))
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, 1e-10)
    A = A / col_norms
    return A


def generate_single_b_jax(key, A, p_xsamp_nonzero, b_noise_std):
    """Generate a single b vector: b = A @ x_samp + noise.

    Args:
        key: JAX random key
        A: (m, n) matrix
        p_xsamp_nonzero: Probability of non-zero entries in x_samp
        b_noise_std: Noise standard deviation

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


def generate_batch_b_jax(key, A, N, p_xsamp_nonzero, b_noise_std):
    """Generate a batch of b vectors.

    Args:
        key: JAX random key
        A: (m, n) matrix
        N: Number of samples
        p_xsamp_nonzero: Probability of non-zero entries in x_samp
        b_noise_std: Noise standard deviation

    Returns:
        b_batch: (N, m) array of b vectors
    """
    keys = jax.random.split(key, N)
    generate_one = partial(generate_single_b_jax, A=A,
                           p_xsamp_nonzero=p_xsamp_nonzero,
                           b_noise_std=b_noise_std)
    b_batch = jax.vmap(generate_one)(keys)
    return b_batch


def compute_lasso_params(A):
    """Compute smoothness L and strong convexity mu from A.

    Args:
        A: (m, n) JAX array

    Returns:
        L: Smoothness constant (max eigenvalue of A^T A)
        mu: Strong convexity (min eigenvalue of A^T A, or 0 if m < n)
    """
    m, n = A.shape
    ATA = A.T @ A
    eigvals = jnp.linalg.eigvalsh(ATA)
    L = jnp.max(eigvals)
    mu = jnp.min(eigvals) if m >= n else 0.0
    return float(L), float(mu)


# =============================================================================
# Lasso Solution (using CVXPY with DPP for fast re-solves)
# =============================================================================

class LassoProblemDPP:
    """DPP-parametrized Lasso problem for fast batch solving.

    Creates the problem structure once, then updates parameters and re-solves
    without rebuilding the problem each time.
    """

    def __init__(self, A_np, lambd):
        """Initialize the parametrized Lasso problem.

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
        """Solve Lasso for a given b vector.

        Args:
            b_np: (m,) numpy array

        Returns:
            x_opt: (n,) optimal solution
            f_opt: optimal objective value
        """
        self.b_param.value = b_np
        self.prob.solve(solver='CLARABEL')
        return self.x.value, self.prob.value

    def solve_batch(self, b_batch_np):
        """Solve batch of Lasso problems efficiently.

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

        for i in trange(N):
            x_opt, f_opt = self.solve(b_batch_np[i])
            x_opt_batch[i] = x_opt
            f_opt_batch[i] = f_opt
            R = np.linalg.norm(x_opt)
            R_max = max(R_max, R)

        return x_opt_batch, f_opt_batch, R_max


def solve_batch_lasso_cvxpy(A_np, b_batch_np, lambd, lasso_dpp=None):
    """Solve batch of Lasso problems.

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
        return lasso_dpp.solve_batch(b_batch_np)

    N = b_batch_np.shape[0]
    n = A_np.shape[1]

    x_opt_batch = np.zeros((N, n))
    f_opt_batch = np.zeros(N)
    R_max = 0.0

    for i in trange(N):
        x = cp.Variable(n)
        obj = 0.5 * cp.sum_squares(A_np @ x - b_batch_np[i]) + lambd * cp.norm(x, 1)
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve(solver='CLARABEL')
        x_opt_batch[i] = x.value
        f_opt_batch[i] = prob.value
        R_max = max(R_max, np.linalg.norm(x.value))

    return x_opt_batch, f_opt_batch, R_max


def compute_sample_radius(cfg, A_np, lasso_dpp=None):
    """Compute radius R from samples when not specified in config.

    Args:
        cfg: Hydra config with R_seed, R_sample_size, p_xsamp_nonzero, b_noise_std, lambd
        A_np: (m, n) numpy array
        lasso_dpp: Optional pre-created LassoProblemDPP

    Returns:
        R_max: Maximum ||x_opt|| across samples
    """
    log.info(f"Computing R from {cfg.R_sample_size} samples...")

    if lasso_dpp is None:
        lasso_dpp = LassoProblemDPP(A_np, cfg.lambd)

    key = jax.random.PRNGKey(cfg.R_seed)
    b_batch = generate_batch_b_jax(
        key, jnp.array(A_np), cfg.R_sample_size,
        cfg.p_xsamp_nonzero, cfg.b_noise_std
    )
    b_batch_np = np.array(b_batch)

    _, _, R_max = solve_batch_lasso_cvxpy(A_np, b_batch_np, cfg.lambd, lasso_dpp=lasso_dpp)

    log.info(f"Computed R = {R_max:.6f}")
    return R_max


# =============================================================================
# Stepsize initialization helpers
# =============================================================================

def compute_silver_stepsizes(L, K):
    """Compute silver step sizes for gradient descent.

    Based on "The Silver Stepsize Schedule" (Altschuler & Parrilo, 2023).

    Args:
        L: Smoothness constant
        K: Number of iterations

    Returns:
        Array of K step sizes
    """
    rho = 1 + np.sqrt(2)
    stepsizes = []
    for k in range(1, K + 1):
        rho_k = rho ** k
        rho_neg_k = rho ** (-k)
        gamma_k = (1.0 / L) * (rho_k - rho_neg_k) / (rho_k + rho_neg_k)
        stepsizes.append(gamma_k)
    return jnp.array(stepsizes)


def compute_geometric_stepsizes(L, K, start_factor=0.5, end_factor=1.5):
    """Compute geometrically spaced step sizes.

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


# =============================================================================
# PEP data function wrappers
# =============================================================================

def pep_data_fn_ista(stepsizes, mu, L, R, K_max, pep_obj,
                     composition_type='final', decay_rate=0.9):
    """PEP data construction function for ISTA."""
    gamma = stepsizes[0]
    return construct_ista_pep_data(gamma, mu, L, R, K_max, pep_obj,
                                   composition_type=composition_type,
                                   decay_rate=decay_rate)


def pep_data_fn_fista(stepsizes, mu, L, R, K_max, pep_obj,
                      composition_type='final', decay_rate=0.9):
    """PEP data construction function for FISTA."""
    gamma, beta = stepsizes[0], stepsizes[1]
    return construct_fista_pep_data(gamma, beta, mu, L, R, K_max, pep_obj,
                                    composition_type=composition_type,
                                    decay_rate=decay_rate)


# =============================================================================
# Trajectory function factories
# =============================================================================

def _make_ista_traj_fn(A_jax, lambd):
    """Create ISTA trajectory wrapper with A, lambd bound in closure.

    The unified trainer calls:
        traj_fn(stepsizes, b=..., x_opt=..., f_opt=..., K_max=K, ...)

    stepsizes is a tuple (gamma,) from the trainer. Raw ISTA expects
    stepsizes=gamma (plain array). This wrapper extracts stepsizes[0].

    Args:
        A_jax: (m, n) JAX array - fixed measurement matrix
        lambd: L1 regularization parameter

    Returns:
        Wrapped trajectory function with signature:
            (stepsizes, b, x_opt, f_opt, K_max, return_Gram_representation) -> (G, F) or trajectories
    """
    def wrapped_traj_fn(stepsizes, b, x_opt, f_opt, K_max, return_Gram_representation=True):
        gamma = stepsizes[0]  # Extract from trainer's tuple format
        x0 = jnp.zeros_like(x_opt)  # Always zero in shifted coords
        return problem_data_to_ista_trajectories(
            gamma, A_jax, b, x0, x_opt, f_opt, lambd, K_max,
            return_Gram_representation=return_Gram_representation
        )
    return wrapped_traj_fn


def _make_fista_traj_fn(A_jax, lambd):
    """Create FISTA trajectory wrapper with A, lambd bound in closure.

    stepsizes is (gamma, beta) from the trainer. Raw FISTA expects the same
    tuple format, so we pass stepsizes directly.

    Args:
        A_jax: (m, n) JAX array - fixed measurement matrix
        lambd: L1 regularization parameter

    Returns:
        Wrapped trajectory function.
    """
    def wrapped_traj_fn(stepsizes, b, x_opt, f_opt, K_max, return_Gram_representation=True):
        x0 = jnp.zeros_like(x_opt)  # Always zero in shifted coords
        return problem_data_to_fista_trajectories(
            stepsizes, A_jax, b, x0, x_opt, f_opt, lambd, K_max,
            return_Gram_representation=return_Gram_representation
        )
    return wrapped_traj_fn


# =============================================================================
# LassoProblemModule class
# =============================================================================

class LassoProblemModule(ProblemModule):
    """Problem module for Lasso (L1-regularized least squares) optimization.

    Handles min_x 0.5*||Ax-b||^2 + lambd*||x||_1 problems where:
    - A is a fixed measurement matrix (shared across all samples)
    - b vectors are sampled per instance
    - Optimal solutions are computed via CVXPY

    A is bound in trajectory function closures rather than stored in
    training_data, because the unified trainer's _get_minibatch() slices
    all dict values by batch index.
    """

    def __init__(self, cfg: Any):
        """Initialize LassoProblemModule with configuration.

        Args:
            cfg: Hydra configuration object containing:
                - m, n: Problem dimensions
                - lambd: L1 regularization parameter
                - A_seed: Seed for A matrix generation
                - p_xsamp_nonzero: Sparsity of x samples
                - b_noise_std: Noise level in b generation
                - stepsize_type: 'scalar' or 'vector'
                - vector_init: 'fixed', 'silver', 'geometric', etc.
                - alg: 'ista' or 'fista'
        """
        super().__init__(cfg)

        # Generate A matrix (fixed across all samples)
        log.info(f"Generating A matrix with seed={cfg.A_seed}")
        self.A_np = generate_A(cfg.A_seed, cfg.m, cfg.n)
        self.A_jax = jnp.array(self.A_np)

        # Compute L, mu from A^T A eigenvalues
        self.L_val, self.mu_val = compute_lasso_params(self.A_jax)
        log.info(f"L = {self.L_val:.6f}, mu = {self.mu_val:.6f}")

        # Regularization parameter
        self.lambd = cfg.lambd

        # Determine R based on strong convexity
        MU_TOL = 1e-6
        is_strongly_convex = self.mu_val > MU_TOL

        if is_strongly_convex:
            log.info("Problem is strongly convex (mu > 0)")
            R_config = cfg.get('R_strongcvx', None)
        else:
            log.info("Problem is non-strongly convex (mu ~ 0)")
            R_config = cfg.get('R_nonstrongcvx', None)

        # Create DPP solver for efficient batch solving
        self.lasso_dpp = LassoProblemDPP(self.A_np, self.lambd)
        log.info("Created DPP-parametrized Lasso problem for fast batch solving")

        if R_config is not None:
            self.R_val = float(R_config)
            log.info(f"Using provided R = {self.R_val}")
        else:
            self.R_val = compute_sample_radius(cfg, self.A_np, self.lasso_dpp)

    def sample_training_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Generate N training Lasso problem instances.

        Generates b vectors, solves each Lasso problem via CVXPY to get
        optimal solutions and values. Does NOT include A in the returned
        dicts (A is bound in trajectory function closures).

        Args:
            key: JAX random key for reproducible sampling.
            N: Number of problem instances to generate.

        Returns:
            problem_data: {'b_batch': (N, m)}
            ground_truth: {'x_opt_batch': (N, n), 'f_opt_batch': (N,)}
        """
        # Generate b vectors
        b_batch = generate_batch_b_jax(
            key, self.A_jax, N, self.cfg.p_xsamp_nonzero, self.cfg.b_noise_std
        )

        # Solve Lasso to get x_opt, f_opt for each sample
        b_batch_np = np.array(b_batch)
        x_opt_batch_np, f_opt_batch_np, _ = solve_batch_lasso_cvxpy(
            self.A_np, b_batch_np, self.lambd, lasso_dpp=self.lasso_dpp
        )

        return (
            {'b_batch': b_batch},
            {'x_opt_batch': jnp.array(x_opt_batch_np),
             'f_opt_batch': jnp.array(f_opt_batch_np)}
        )

    def sample_validation_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Generate N validation Lasso problem instances.

        Same distribution as training batch.

        Args:
            key: JAX random key for reproducible sampling.
            N: Number of problem instances to generate.

        Returns:
            problem_data: {'b_batch': (N, m)}
            ground_truth: {'x_opt_batch': (N, n), 'f_opt_batch': (N,)}
        """
        return self.sample_training_batch(key, N)

    def get_trajectory_fn(self, alg: str) -> Callable:
        """Return trajectory function for the specified algorithm.

        Returns a wrapper that binds A, lambd, and x0 via closure.

        Args:
            alg: Algorithm name ('ista' or 'fista')

        Returns:
            Trajectory function with signature:
                (stepsizes, b, x_opt, f_opt, K_max, return_Gram_representation) -> (G, F) or trajectories
        """
        if alg == 'ista':
            return _make_ista_traj_fn(self.A_jax, self.lambd)
        elif alg == 'fista':
            return _make_fista_traj_fn(self.A_jax, self.lambd)
        else:
            raise ValueError(f"Unknown algorithm: {alg}")

    def get_pep_data_fn(self, alg: str) -> Callable:
        """Return PEP constraint construction function for the algorithm.

        Args:
            alg: Algorithm name.

        Returns:
            PEP data function with signature:
                (stepsizes, mu, L, R, K_max, pep_obj) -> pep_data_tuple
        """
        if alg == 'ista':
            return pep_data_fn_ista
        elif alg == 'fista':
            return pep_data_fn_fista
        else:
            raise ValueError(f"Unknown algorithm: {alg}")

    def compute_L_mu_R(self, samples: ProblemData | None = None) -> Tuple[float, float, float]:
        """Return problem parameters computed from A in __init__.

        Args:
            samples: Unused for Lasso (L, mu depend only on A).

        Returns:
            Tuple of (L, mu, R).
        """
        return (self.L_val, self.mu_val, self.R_val)

    def get_initial_stepsizes(self, alg: str, K: int, L: float, mu: float) -> Stepsizes:
        """Get algorithm-specific stepsize initialization.

        Args:
            alg: Algorithm name ('ista' or 'fista').
            K: Number of algorithm iterations (K_max).
            L: Lipschitz constant (smoothness).
            mu: Strong convexity parameter.

        Returns:
            Tuple of stepsize arrays:
                ISTA: (gamma,) where gamma is scalar or (K,) vector
                FISTA: (gamma, beta) where beta has K+1 elements
        """
        is_vector = self.cfg.stepsize_type == "vector"
        gamma_init_scalar = 1.5 / L

        if is_vector:
            vector_init = self.cfg.get('vector_init', 'fixed')
            if vector_init == 'fixed':
                gamma = jnp.full(K, gamma_init_scalar)
            elif vector_init == 'silver':
                gamma = compute_silver_stepsizes(L, K)
            elif vector_init == 'geometric':
                start_factor = self.cfg.get('geometric_start', 0.5)
                end_factor = self.cfg.get('geometric_end', 1.5)
                gamma = compute_geometric_stepsizes(L, K, start_factor, end_factor)
            elif vector_init == 'increasing':
                factors = jnp.linspace(0.5, 1.5, K)
                gamma = factors / L
            elif vector_init == 'decreasing':
                factors = jnp.linspace(1.5, 0.5, K)
                gamma = factors / L
            else:
                raise ValueError(f"Unknown vector_init: {vector_init}")
        else:
            gamma = jnp.array(gamma_init_scalar)

        if alg == 'ista':
            return (gamma,)
        elif alg == 'fista':
            # Raw t_k Nesterov sequence of length K+1
            betas_t = [1.0]
            for k in range(K):
                t_new = 0.5 * (1 + np.sqrt(1 + 4 * betas_t[-1] ** 2))
                betas_t.append(t_new)
            beta = jnp.array(betas_t)
            return (gamma, beta)
        else:
            raise ValueError(f"Unknown algorithm: {alg}")

    def build_stepsizes_dataframe(
        self,
        stepsizes_history: list[Stepsizes],
        K_max: int,
        alg: str,
        training_losses: list[float] | None = None,
        validation_losses: list[float] | None = None,
        times: list[float] | None = None,
    ) -> pd.DataFrame:
        """Build DataFrame from stepsizes history for CSV saving.

        Args:
            stepsizes_history: List of stepsizes tuples, one per SGD iteration.
            K_max: Number of algorithm iterations.
            alg: Algorithm name.
            training_losses: Optional list of training loss values per iteration.
            validation_losses: Optional list of validation loss values per iteration.
            times: Optional list of iteration times in seconds.

        Returns:
            DataFrame with columns for iteration, stepsizes, losses, and times.
        """
        gamma_sample = stepsizes_history[0][0]
        is_vector_gamma = jnp.ndim(gamma_sample) > 0
        has_beta = len(stepsizes_history[0]) > 1

        data = {'iteration': list(range(len(stepsizes_history)))}

        if training_losses is not None:
            data['training_loss'] = [float(l) for l in training_losses]

        if validation_losses is not None:
            data['validation_loss'] = [float(l) for l in validation_losses]

        if times is not None:
            data['iter_time'] = [float(t) for t in times]

        # Gamma columns
        if is_vector_gamma:
            for k in range(K_max):
                data[f'gamma_{k}'] = [float(ss[0][k]) for ss in stepsizes_history]
        else:
            data['gamma'] = [float(ss[0]) for ss in stepsizes_history]

        # Beta columns (FISTA has K+1 values)
        if has_beta:
            for k in range(K_max + 1):
                data[f'beta_{k}'] = [float(ss[1][k]) for ss in stepsizes_history]

        return pd.DataFrame(data)

    def get_batched_parameters(self) -> ParameterNames:
        """Return names of parameters that vary across the batch.

        For Lasso, b, x_opt, and f_opt vary per sample. A is handled
        via closure in the trajectory function.

        Returns:
            ('b', 'x_opt', 'f_opt')
        """
        return ('b', 'x_opt', 'f_opt')

    def get_fixed_parameters(self) -> ParameterNames:
        """Return names of parameters that are fixed across the batch.

        A is bound via closure in the trajectory function, NOT passed
        through training_data/minibatch (because _get_minibatch slices
        all values by batch index, which would incorrectly slice A).

        Returns:
            ()
        """
        return ()

    def get_ground_truth_keys(self) -> ParameterNames:
        """Return names of ground truth keys.

        Returns:
            ('x_opt', 'f_opt')
        """
        return ('x_opt', 'f_opt')

    def get_gram_dimensions(self, alg: str, K: int) -> Tuple[int, int]:
        """Return Gram matrix dimensions for the algorithm.

        ISTA has composite structure with gradients + subgradients:
            dimG = 2K+5, dimF = 2(K+2) = 2K+4
        FISTA has slightly smaller Gram basis:
            dimG = 2K+4, dimF = 2(K+2) = 2K+4

        Args:
            alg: Algorithm name.
            K: Number of iterations (K_max).

        Returns:
            Tuple of (dimG, dimF) for Gram matrix dimensions.
        """
        if alg == 'ista':
            return (2 * K + 5, 2 * K + 4)
        elif alg == 'fista':
            return (2 * K + 4, 2 * K + 4)
        else:
            raise ValueError(f"Unknown algorithm: {alg}")

    def create_metric_fn(
        self, trajectories: Any, problem_data: ProblemData, ground_truth: GroundTruth, pep_obj: str
    ) -> Callable[[int], float]:
        """Create a metric function for trajectory loss computation.

        Args:
            trajectories: Algorithm trajectory data. For both ISTA and FISTA,
                trajectories[0] = x_iter of shape (n, K+1) in shifted coords.
            problem_data: {'b': (m,)}
            ground_truth: {'x_opt': (n,), 'f_opt': scalar}
            pep_obj: Metric type ('obj_val', 'grad_sq_norm', 'opt_dist_sq_norm').

        Returns:
            Callable with signature metric_fn(k: int) -> scalar.
        """
        A = self.A_jax
        lambd = self.lambd
        x_opt = ground_truth['x_opt']
        f_opt = ground_truth['f_opt']
        b = problem_data['b']

        # Both ISTA and FISTA return x_iter as first element
        x_iter = trajectories[0]  # Shape (n, K+1), shifted coordinates

        if pep_obj == 'obj_val':
            def metric_fn(k):
                x_k_shifted = x_iter[:, k]
                x_k = x_k_shifted + x_opt
                f1_xk = 0.5 * jnp.sum((A @ x_k - b) ** 2)
                f2_xk = lambd * jnp.sum(jnp.abs(x_k))
                return f1_xk + f2_xk - f_opt
        elif pep_obj == 'opt_dist_sq_norm':
            def metric_fn(k):
                # x_iter stores shifted coordinates (x_k - x_opt)
                return jnp.sum(x_iter[:, k] ** 2)
        elif pep_obj == 'grad_sq_norm':
            def metric_fn(k):
                x_k = x_iter[:, k] + x_opt
                g_f1 = A.T @ (A @ x_k - b)
                h_f2 = lambd * jnp.sign(x_k)
                return jnp.sum((g_f1 + h_f2) ** 2)
        else:
            raise ValueError(f"Unknown pep_obj: {pep_obj}")

        return metric_fn

    def compute_batched_trajectories(
        self,
        stepsizes: Stepsizes,
        batched_data: Dict[str, jnp.ndarray],
        fixed_data: Dict[str, jnp.ndarray],
        traj_fn: Callable,
        K_max: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Gram representations for a batch of problems.

        Vmaps over b, x_opt, f_opt while keeping A, lambd, x0 from the
        trajectory function closure.

        Args:
            stepsizes: Algorithm stepsizes tuple.
            batched_data: Dict with 'b', 'x_opt', 'f_opt' arrays (with batch dim).
            fixed_data: Dict of fixed parameters (empty for Lasso).
            traj_fn: Trajectory function from get_trajectory_fn().
            K_max: Number of algorithm iterations.

        Returns:
            Tuple of (G_batch, F_batch) Gram representations.
        """
        batch_GF_func = jax.vmap(
            lambda b, x_opt, f_opt: traj_fn(
                stepsizes, b, x_opt, f_opt, K_max,
                return_Gram_representation=True
            ),
            in_axes=(0, 0, 0)
        )

        return batch_GF_func(
            batched_data['b'],
            batched_data['x_opt'],
            batched_data['f_opt']
        )

    def generate_out_of_sample_data(
        self, key: jax.Array
    ) -> Dict[str, Tuple[ProblemData, GroundTruth]]:
        """Generate validation, test, and out-of-distribution problem sets.

        In-distribution: same A matrix, different b vectors.
        Out-of-distribution: different A matrix with scaling=4.0.

        Args:
            key: JAX random key for reproducible sampling.

        Returns:
            Dict with keys 'validation', 'test', 'ood', each mapping to
            (problem_data, ground_truth) tuple.
        """
        N_val = self.cfg.out_of_sample_val_N
        N_test = self.cfg.out_of_sample_test_N
        N_ood = self.cfg.out_of_dist_N

        key, val_key, test_key, ood_key = jax.random.split(key, 4)

        # In-distribution validation and test sets (same A)
        val_data = self.sample_validation_batch(val_key, N_val)
        test_data = self.sample_validation_batch(test_key, N_test)

        # Out-of-distribution set (different A with scaling=4.0)
        ood_data = self._sample_ood_batch(ood_key, N_ood)

        return {
            'validation': val_data,
            'test': test_data,
            'ood': ood_data,
        }

    def _sample_ood_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Sample out-of-distribution problems with different A matrix.

        Uses a different A matrix generated with scaling=4.0 and a different seed.

        Args:
            key: JAX random key.
            N: Number of samples.

        Returns:
            (problem_data, ground_truth) tuple.
        """
        A_ood_seed = self.cfg.A_out_of_dist_seed
        A_ood_np = generate_A(A_ood_seed, self.cfg.m, self.cfg.n, scaling=4.0)
        A_ood_jax = jnp.array(A_ood_np)

        # Generate b vectors using OOD A matrix
        b_batch = generate_batch_b_jax(
            key, A_ood_jax, N, self.cfg.p_xsamp_nonzero, self.cfg.b_noise_std
        )

        # Solve Lasso with OOD A matrix
        lasso_dpp_ood = LassoProblemDPP(A_ood_np, self.lambd)
        b_batch_np = np.array(b_batch)
        x_opt_batch_np, f_opt_batch_np, _ = solve_batch_lasso_cvxpy(
            A_ood_np, b_batch_np, self.lambd, lasso_dpp=lasso_dpp_ood
        )

        return (
            {'b_batch': b_batch},
            {'x_opt_batch': jnp.array(x_opt_batch_np),
             'f_opt_batch': jnp.array(f_opt_batch_np)}
        )

    def get_supported_algorithms(self) -> list[str]:
        """Return list of algorithms supported by Lasso problems.

        Returns:
            ['ista', 'fista']
        """
        return ['ista', 'fista']

    def validate_config(self) -> None:
        """Validate configuration for Lasso problems.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.cfg.alg not in ['ista', 'fista']:
            raise ValueError(
                f"Unknown algorithm: {self.cfg.alg}. Must be 'ista' or 'fista'."
            )


# =============================================================================
# Entry point functions
# =============================================================================

def lasso_run(cfg):
    """Run learning experiment for Lasso problems.

    Loops over K_max values, runs training for each K, and saves per-K progress CSV.
    Algorithm and learning framework are selected via config.

    Args:
        cfg: Hydra configuration object.
    """
    log.info("=" * 60)
    log.info("Starting Lasso learning experiment")
    log.info("=" * 60)
    log.info(cfg)

    # Initialize random key
    key = jax.random.PRNGKey(cfg.sgd_seed)

    # Create problem module
    problem_module = LassoProblemModule(cfg)

    # Ensure output directory exists
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Loop over K_max values
    for K in cfg.K_max:
        log.info(f"=== Starting training for K={K} ===")

        # Validate config for this K
        problem_module.validate_config()

        # Create output directory for this K
        K_output_dir = os.path.join(output_dir, f"K_{K}")
        os.makedirs(K_output_dir, exist_ok=True)
        csv_path = os.path.join(K_output_dir, "progress.csv")

        # Create trainer and run
        key, train_key = jax.random.split(key)
        trainer = UnifiedTrainer(problem_module, cfg, train_key)
        result = trainer.train(K, csv_path, K_output_dir)

        # Log final stepsizes
        gamma = result.stepsizes[0]
        is_vector = jnp.ndim(gamma) > 0
        gamma_str = str(gamma.tolist()) if is_vector else f'{float(gamma):.6f}'
        log.info(f'K={K} complete. Final gamma={gamma_str}. Saved to {csv_path}')

    log.info("=== Experiment complete ===")


def lasso_out_of_sample_run(cfg):
    """Generate and save out-of-sample test problems for Lasso.

    For in-distribution (validation and test):
        - Use the SAME A matrix as training (A_seed)
        - Generate different b vectors
        - Solve all Lasso problems to get optimal solutions and values

    For out-of-distribution:
        - Generate NEW A matrix with scaling=4.0 (A_out_of_dist_seed)
        - Generate b vectors and solve

    Output files saved in Hydra run directory.

    Args:
        cfg: Hydra configuration object.
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
    # In-distribution A matrix (shared by validation and test)
    # =========================================================================
    log.info(f"Generating A matrix for in-distribution sets (A_seed={A_seed})")
    A_in_dist_np = generate_A(A_seed, m, n, scaling=1.0)
    A_in_dist_jax = jnp.array(A_in_dist_np)

    A_in_dist_path = "A_in_dist.npz"
    np.savez_compressed(A_in_dist_path, A=A_in_dist_np)
    log.info(f"Saved in-distribution A to {A_in_dist_path}, shape: {A_in_dist_np.shape}")

    lasso_dpp = LassoProblemDPP(A_in_dist_np, lambd)

    # =========================================================================
    # Validation Set (in-distribution)
    # =========================================================================
    log.info(f"Generating {out_of_sample_val_N} validation problems (in-distribution)...")

    key_val = jax.random.PRNGKey(out_of_sample_val_seed)
    b_val_batch = generate_batch_b_jax(
        key_val, A_in_dist_jax, out_of_sample_val_N,
        p_xsamp_nonzero, b_noise_std
    )

    b_val_np = np.array(b_val_batch)
    np.savez_compressed("b_val_samples.npz", b=b_val_np)

    log.info(f"Solving {out_of_sample_val_N} validation Lasso problems...")
    x_opt_val_np, f_opt_val_np, _ = solve_batch_lasso_cvxpy(
        A_in_dist_np, b_val_np, lambd, lasso_dpp=lasso_dpp
    )

    np.savez_compressed("x_opt_val_samples.npz", x_opt=x_opt_val_np)
    np.savez_compressed("f_opt_val_samples.npz", f_opt=f_opt_val_np)

    # =========================================================================
    # Test Set (in-distribution)
    # =========================================================================
    log.info(f"Generating {out_of_sample_test_N} test problems (in-distribution)...")

    key_test = jax.random.PRNGKey(out_of_sample_test_seed)
    b_test_batch = generate_batch_b_jax(
        key_test, A_in_dist_jax, out_of_sample_test_N,
        p_xsamp_nonzero, b_noise_std
    )

    b_test_np = np.array(b_test_batch)
    np.savez_compressed("b_test_samples.npz", b=b_test_np)

    log.info(f"Solving {out_of_sample_test_N} test Lasso problems...")
    x_opt_test_np, f_opt_test_np, _ = solve_batch_lasso_cvxpy(
        A_in_dist_np, b_test_np, lambd, lasso_dpp=lasso_dpp
    )

    np.savez_compressed("x_opt_test_samples.npz", x_opt=x_opt_test_np)
    np.savez_compressed("f_opt_test_samples.npz", f_opt=f_opt_test_np)

    # =========================================================================
    # Out-of-Distribution Test Set (different A with scaling=4.0)
    # =========================================================================
    log.info(f"Generating {out_of_dist_N} out-of-distribution test problems...")

    A_ood_np = generate_A(A_out_of_dist_seed, m, n, scaling=4.0)
    A_ood_jax = jnp.array(A_ood_np)

    np.savez_compressed("A_out_of_dist.npz", A=A_ood_np)

    key_ood = jax.random.PRNGKey(out_of_dist_seed)
    b_ood_batch = generate_batch_b_jax(
        key_ood, A_ood_jax, out_of_dist_N,
        p_xsamp_nonzero, b_noise_std
    )

    b_ood_np = np.array(b_ood_batch)
    np.savez_compressed("b_out_of_dist_samples.npz", b=b_ood_np)

    lasso_dpp_ood = LassoProblemDPP(A_ood_np, lambd)

    log.info(f"Solving {out_of_dist_N} out-of-distribution Lasso problems...")
    x_opt_ood_np, f_opt_ood_np, _ = solve_batch_lasso_cvxpy(
        A_ood_np, b_ood_np, lambd, lasso_dpp=lasso_dpp_ood
    )

    np.savez_compressed("x_opt_out_of_dist_samples.npz", x_opt=x_opt_ood_np)
    np.savez_compressed("f_opt_out_of_dist_samples.npz", f_opt=f_opt_ood_np)

    # =========================================================================
    # Save metadata
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
    }
    np.savez_compressed("out_of_sample_metadata.npz", **metadata)

    log.info("=== Lasso out-of-sample generation complete ===")
