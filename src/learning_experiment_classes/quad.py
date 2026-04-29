"""Quadratic problem module for unified learning framework.

Implements QuadProblemModule using the ProblemModule ABC and UnifiedTrainer,
eliminating code duplication from the old quad.py implementation.
"""

import diffcp_patch  # noqa: F401  # Apply COO -> CSC fix for diffcp
import jax
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
import logging
from functools import partial
from typing import Any, Callable, Dict, Tuple

from learning.problem_module import ProblemModule, ProblemData, GroundTruth, Stepsizes, ParameterNames
from learning.unified_trainer import UnifiedTrainer
from learning.pep_constructions import construct_gd_pep_data, construct_fgm_pep_data
from learning.trajectories import (
    problem_data_to_gd_trajectories,
    problem_data_to_nesterov_fgm_trajectories,
)
from learning.silver_stepsizes import get_strongly_convex_silver_stepsizes
from learning.acceleration_stepsizes import jax_get_nesterov_fgm_beta_sequence

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)


# =============================================================================
# data_source_dir loader (mirror of lasso._load_and_subsample)
# =============================================================================

def _load_and_subsample_quad(npz_path, N, seed, dataset_label):
    """Load a saved quad problem-instance bundle and subsample N aligned rows.

    Samples indices once with a seeded np.random.Generator and applies them
    uniformly across Q_batch, z0_batch, zs_batch, fs_batch so row alignment is
    preserved. If N exceeds the available rows, returns all rows and logs a
    warning.

    Returns (Q, z0, zs, fs) as jnp.ndarrays, or None if the file is missing
    (caller falls back to fresh sampling).
    """
    if not os.path.isfile(npz_path):
        return None
    d = np.load(npz_path)
    total = int(d['Q_batch'].shape[0])
    if N >= total:
        log.warning(
            f"{dataset_label}: requested N={N} but {npz_path} has only "
            f"{total} rows; using all {total}."
        )
        idx = np.arange(total)
    else:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(total, size=N, replace=False)
    log.info(f"{dataset_label}: loaded {len(idx)} problems from {npz_path}")
    return (
        jnp.asarray(d['Q_batch'][idx]),
        jnp.asarray(d['z0_batch'][idx]),
        jnp.asarray(d['zs_batch'][idx]),
        jnp.asarray(d['fs_batch'][idx]),
    )


# =============================================================================
# Module-level helper functions for sampling
# =============================================================================

def marchenko_pastur(key, d, mu, L, M):
    """Generate a random matrix with Marchenko-Pastur distributed eigenvalues.

    Args:
        key: JAX random key
        d: Dimension parameter (related to aspect ratio)
        mu: Lower bound for eigenvalues
        L: Upper bound for eigenvalues
        M: Matrix dimension

    Returns:
        H: (M, M) symmetric positive semi-definite matrix
    """
    sigma = (jnp.sqrt(L) + jnp.sqrt(mu)) / 2
    X = jax.random.normal(key, shape=(d, M)) * sigma
    H = X.T @ X / d
    return H


def rejection_sample_single(key, dim, mu, L, M):
    """Rejection sample a single Q matrix with eigenvalues in [mu, L].

    Uses while_loop for rejection sampling until eigenvalue bounds are satisfied.

    Args:
        key: JAX random key
        dim: Dimension parameter for Marchenko-Pastur
        mu: Lower bound for eigenvalues
        L: Upper bound for eigenvalues
        M: Matrix dimension

    Returns:
        H: (M, M) matrix with eigenvalues in [mu, L]
    """
    def body_fun(state):
        key, _, _ = state
        key, subkey = jax.random.split(key)

        H = marchenko_pastur(subkey, dim, mu, L, M)

        eigvals = jnp.real(jnp.linalg.eigvals(H))
        if mu == 0:
            is_valid = (L >= jnp.max(eigvals))
        else:
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
    """Sample batch of Q matrices using rejection sampling.

    Args:
        subkeys: JAX random keys for each sample (N,)
        dim: Dimension parameter for Marchenko-Pastur
        mu: Lower bound for eigenvalues
        L: Upper bound for eigenvalues
        M: Matrix dimension

    Returns:
        Q_batch: (N, M, M) batch of Q matrices
    """
    sampler = partial(rejection_sample_single, dim=dim, mu=mu, L=L, M=M)
    return jax.vmap(sampler)(subkeys)


@partial(jax.jit, static_argnames=['dim'])
def get_out_of_dist_Q_samples(subkeys, dim, mu, L):
    """Generate out-of-distribution Q samples with random eigenvalues and rotation.

    Each sample is generated as Q = U @ diag(eigvals) @ U.T where:
    - eigvals are drawn from Beta(0.5, 0.5) scaled to [mu, L]
    - U is a random orthogonal matrix

    Args:
        subkeys: JAX random keys for each sample (N,)
        dim: Matrix dimension
        mu: Lower bound for eigenvalues
        L: Upper bound for eigenvalues

    Returns:
        Q_batch: (N, dim, dim) batch of symmetric PSD matrices
    """
    def sample_single(key):
        k1, k2 = jax.random.split(key)

        # Generate random eigenvalues using Beta distribution
        random_eigvals = jax.random.beta(k1, 0.5, 0.5, shape=(dim,))
        random_eigvals = random_eigvals * (L - mu) + mu

        # Generate random orthogonal matrix
        U = jax.random.orthogonal(k2, dim)

        # Q = U @ diag(eigvals) @ U.T
        Q = U @ jnp.diag(random_eigvals) @ U.T

        return Q

    return jax.vmap(sample_single)(subkeys)


def sample_z0_single(key, d, R):
    """Sample a single initial point uniformly from a ball of radius R.

    Args:
        key: JAX random key
        d: Dimension
        R: Radius of ball

    Returns:
        z0: (d,) initial point
    """
    k1, k2 = jax.random.split(key)

    z = jax.random.normal(k1, shape=(d,))
    z = z / jnp.linalg.norm(z)

    u = jax.random.uniform(k2)
    dist = u ** (1.0 / d)

    return R * dist * z


@partial(jax.jit, static_argnames=['d'])
def get_z0_samples(subkeys, d, R):
    """Sample batch of initial points uniformly from ball.

    Args:
        subkeys: JAX random keys for each sample (N,)
        d: Dimension
        R: Radius of ball

    Returns:
        z0_batch: (N, d) batch of initial points
    """
    sampler = partial(sample_z0_single, d=d, R=R)
    return jax.vmap(sampler)(subkeys)


# =============================================================================
# PEP data function wrappers
# =============================================================================

def pep_data_fn_gd(stepsizes, mu, L, R, K_max, pep_obj,
                   composition_type='final', decay_rate=0.9):
    """PEP data construction function for gradient descent."""
    t = stepsizes[0]
    return construct_gd_pep_data(t, mu, L, R, K_max, pep_obj,
                                 composition_type=composition_type,
                                 decay_rate=decay_rate)


def pep_data_fn_fgm(stepsizes, mu, L, R, K_max, pep_obj,
                    composition_type='final', decay_rate=0.9):
    """PEP data construction function for Nesterov FGM."""
    t, beta = stepsizes[0], stepsizes[1]
    return construct_fgm_pep_data(t, beta, mu, L, R, K_max, pep_obj,
                                  composition_type=composition_type,
                                  decay_rate=decay_rate)


# =============================================================================
# QuadProblemModule class
# =============================================================================

class QuadProblemModule(ProblemModule):
    """Problem module for quadratic function optimization.

    Handles Q @ z / 2 minimization problems with Marchenko-Pastur distributed
    Q matrices and uniformly sampled initial points.
    """

    def __init__(self, cfg: Any):
        """Initialize QuadProblemModule with configuration.

        Args:
            cfg: Hydra configuration object containing:
                - dim: Problem dimension
                - mu: Strong convexity parameter
                - L: Smoothness constant
                - R: Initial radius bound
                - stepsize_type: 'scalar' or 'vector'
                - vector_init: 'fixed' or 'silver'
                - alg: 'vanilla_gd' or 'nesterov_fgm'
        """
        super().__init__(cfg)

        # Extract problem parameters
        self.d_val = cfg.dim
        self.mu_val = cfg.mu
        self.L_val = cfg.L
        self.R_val = cfg.R

        # Compute matrix width for Marchenko-Pastur
        r_val = (np.sqrt(self.L_val) - np.sqrt(self.mu_val)) ** 2 / \
                (np.sqrt(self.L_val) + np.sqrt(self.mu_val)) ** 2
        self.M_val = int(np.round(r_val * self.d_val))
        log.info(f"Precomputed matrix width M: {self.M_val}")

    def sample_training_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Generate N training quadratic problem instances.

        When `data_source_dir` is configured and `training_set.npz` is present
        under it, load N rows from there (seeded by `training_seed`). Otherwise
        sample fresh via Marchenko-Pastur.

        Args:
            key: JAX random key for reproducible sampling (unused on load path).
            N: Number of problem instances to generate.

        Returns:
            problem_data: {'Q_batch': (N, M, M), 'z0_batch': (N, M)}
            ground_truth: {'zs_batch': (N, M), 'fs_batch': (N,)}
        """
        data_source_dir = self.cfg.get('data_source_dir', None)
        if data_source_dir is not None:
            loaded = _load_and_subsample_quad(
                os.path.join(data_source_dir, 'training_set.npz'),
                N, self.cfg.training_seed, 'training',
            )
            if loaded is not None:
                Q_batch, z0_batch, zs_batch, fs_batch = loaded
                return (
                    {'Q_batch': Q_batch, 'z0_batch': z0_batch},
                    {'zs_batch': zs_batch, 'fs_batch': fs_batch},
                )

        # Fresh-sampling path
        key, k1, k2 = jax.random.split(key, 3)
        Q_subkeys = jax.random.split(k1, N)
        z0_subkeys = jax.random.split(k2, N)

        Q_batch = get_Q_samples(Q_subkeys, self.d_val, self.mu_val, self.L_val, self.M_val)
        z0_batch = get_z0_samples(z0_subkeys, self.M_val, self.R_val)

        # Ground truth: optimal point is origin, optimal value is 0
        zs_batch = jnp.zeros(z0_batch.shape)
        fs_batch = jnp.zeros(N)

        return (
            {'Q_batch': Q_batch, 'z0_batch': z0_batch},
            {'zs_batch': zs_batch, 'fs_batch': fs_batch}
        )

    def sample_validation_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Generate N validation quadratic problem instances.

        When `data_source_dir` is configured and `validation_set.npz` is present
        under it, load N rows from there (seeded by `out_of_sample_val_seed`).
        Otherwise fall back to fresh sampling from the training distribution.

        Args:
            key: JAX random key for reproducible sampling (unused on load path).
            N: Number of problem instances to generate.

        Returns:
            problem_data: {'Q_batch': (N, M, M), 'z0_batch': (N, M)}
            ground_truth: {'zs_batch': (N, M), 'fs_batch': (N,)}
        """
        data_source_dir = self.cfg.get('data_source_dir', None)
        if data_source_dir is not None:
            loaded = _load_and_subsample_quad(
                os.path.join(data_source_dir, 'validation_set.npz'),
                N, self.cfg.out_of_sample_val_seed, 'validation',
            )
            if loaded is not None:
                Q_batch, z0_batch, zs_batch, fs_batch = loaded
                return (
                    {'Q_batch': Q_batch, 'z0_batch': z0_batch},
                    {'zs_batch': zs_batch, 'fs_batch': fs_batch},
                )
        # Fresh-sampling fallback: same distribution as training
        return self.sample_training_batch(key, N)

    def sample_test_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Generate N in-distribution test quadratic problem instances.

        When `data_source_dir` is configured and `test_set.npz` is present
        under it, load N rows from there (seeded by `out_of_sample_test_seed`).
        Otherwise fall back to fresh sampling from the training distribution.
        """
        data_source_dir = self.cfg.get('data_source_dir', None)
        if data_source_dir is not None:
            loaded = _load_and_subsample_quad(
                os.path.join(data_source_dir, 'test_set.npz'),
                N, self.cfg.out_of_sample_test_seed, 'test',
            )
            if loaded is not None:
                Q_batch, z0_batch, zs_batch, fs_batch = loaded
                return (
                    {'Q_batch': Q_batch, 'z0_batch': z0_batch},
                    {'zs_batch': zs_batch, 'fs_batch': fs_batch},
                )
        return self.sample_training_batch(key, N)

    def get_trajectory_fn(self, alg: str) -> Callable:
        """Return trajectory function for the specified algorithm.

        Args:
            alg: Algorithm name ('vanilla_gd' or 'nesterov_fgm')

        Returns:
            Trajectory function with signature:
                (stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation) -> (G, F) or trajectories
        """
        if alg == 'vanilla_gd':
            return problem_data_to_gd_trajectories
        elif alg == 'nesterov_fgm':
            return problem_data_to_nesterov_fgm_trajectories
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
        if alg == 'vanilla_gd':
            return pep_data_fn_gd
        elif alg == 'nesterov_fgm':
            return pep_data_fn_fgm
        else:
            raise ValueError(f"Unknown algorithm: {alg}")

    def compute_L_mu_R(self, samples: ProblemData | None = None) -> Tuple[float, float, float]:
        """Return problem parameters from configuration.

        For quadratic problems, L, mu, R are fixed by configuration.

        Args:
            samples: Unused for quadratic problems.

        Returns:
            Tuple of (L, mu, R).
        """
        return (self.L_val, self.mu_val, self.R_val)

    def get_initial_stepsizes(self, alg: str, K: int, L: float, mu: float) -> Stepsizes:
        """Get algorithm-specific stepsize initialization.

        Args:
            alg: Algorithm name.
            K: Number of algorithm iterations (K_max).
            L: Lipschitz constant (smoothness).
            mu: Strong convexity parameter.

        Returns:
            Tuple of stepsize arrays:
                GD: (t,) where t is scalar or (K,) vector
                FGM: (t, beta) where beta has K elements
        """
        is_vector = self.cfg.stepsize_type == "vector"

        if alg == 'nesterov_fgm':
            # Nesterov FGM uses t = 1.5/L (scalar or vector of same value)
            t = jnp.full(K, 1.5 / L) if is_vector else jnp.array(1.5 / L)
            beta = jax_get_nesterov_fgm_beta_sequence(mu, L, K)
            return (t, beta)
        elif alg == 'vanilla_gd':
            if is_vector:
                if self.cfg.vector_init == "silver":
                    t = jnp.array(get_strongly_convex_silver_stepsizes(K, mu=mu, L=L))
                else:  # "fixed"
                    t_scalar = 2.0 / (mu + L) if mu > 0 else 1.5 / L
                    t = jnp.full(K, t_scalar)
            else:
                t_scalar = 2.0 / (mu + L) if mu > 0 else 1.5 / L
                t = jnp.array(t_scalar)
            return (t,)
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
        raw_grad_norms: list[float] | None = None,
    ) -> pd.DataFrame:
        """Build DataFrame from stepsizes history for CSV saving.

        Args:
            stepsizes_history: List of stepsizes tuples, one per SGD iteration.
            K_max: Number of algorithm iterations.
            alg: Algorithm name.
            training_losses: Optional list of training loss values per iteration.
            validation_losses: Optional list of validation loss values per iteration.
            times: Optional list of iteration times in seconds.
            raw_grad_norms: Optional list of pre-clip gradient norms w.r.t.
                sqrt-reparameterized params (the params SGD actually steps on).

        Returns:
            DataFrame with columns for iteration, stepsizes, losses, times,
            and raw_grad_norm.
        """
        # Determine stepsize structure
        t_sample = stepsizes_history[0][0]
        is_vector_t = jnp.ndim(t_sample) > 0
        has_beta = len(stepsizes_history[0]) > 1

        data = {'iteration': list(range(len(stepsizes_history)))}

        # Add training loss column if provided
        # Losses now include initial loss, so they align with stepsizes_history
        if training_losses is not None:
            data['training_loss'] = [float(l) for l in training_losses]

        # Add validation loss column if provided
        if validation_losses is not None:
            data['validation_loss'] = [float(l) for l in validation_losses]

        # Add timing column if provided
        if times is not None:
            data['iter_time'] = [float(t) for t in times]

        # Add raw (pre-clip) grad-norm column (w.r.t. sqrt-reparameterized params)
        if raw_grad_norms is not None:
            data['raw_grad_norm'] = [float(g) for g in raw_grad_norms]

        # Extract t values (first element of each stepsizes tuple)
        if is_vector_t:
            for k in range(K_max):
                data[f't{k}'] = [float(ss[0][k]) for ss in stepsizes_history]
        else:
            data['t'] = [float(ss[0]) for ss in stepsizes_history]

        # Extract beta values if present (second element of each stepsizes tuple)
        if has_beta:
            for k in range(K_max):
                data[f'beta{k}'] = [float(ss[1][k]) for ss in stepsizes_history]

        return pd.DataFrame(data)

    def get_batched_parameters(self) -> ParameterNames:
        """Return names of parameters that vary across the batch.

        For quadratic problems, all parameters vary per sample.

        Returns:
            ('Q', 'z0', 'zs', 'fs')
        """
        return ('Q', 'z0', 'zs', 'fs')

    def get_fixed_parameters(self) -> ParameterNames:
        """Return names of parameters that are fixed across the batch.

        For quadratic problems, no parameters are fixed.

        Returns:
            ()
        """
        return ()

    def get_ground_truth_keys(self) -> ParameterNames:
        """Return names of ground truth keys.

        Returns:
            ('zs', 'fs')
        """
        return ('zs', 'fs')

    def get_gram_dimensions(self, alg: str, K: int) -> Tuple[int, int]:
        """Return Gram matrix dimensions for the algorithm.

        Args:
            alg: Algorithm name.
            K: Number of iterations (K_max).

        Returns:
            Tuple of (dimG, dimF) for Gram matrix dimensions.
        """
        # For both GD and FGM: dimG = K+2, dimF = K+1
        return (K + 2, K + 1)

    def create_metric_fn(
        self, trajectories: Any, problem_data: ProblemData, ground_truth: GroundTruth, pep_obj: str
    ) -> Callable[[int], float]:
        """Create a metric function for trajectory loss computation.

        Args:
            trajectories: Algorithm trajectory data (z_iter, g_iter, f_iter).
            problem_data: {'Q': (M, M)}
            ground_truth: {'zs': (M,), 'fs': scalar}
            pep_obj: Metric type ('obj_val', 'grad_sq_norm', 'opt_dist_sq_norm').

        Returns:
            Callable with signature metric_fn(k: int) -> scalar.
        """
        z_iter = trajectories[0]  # Shape (d, K+1), columns are z_0, ..., z_K
        Q = problem_data['Q']
        zs = ground_truth['zs']
        fs = ground_truth['fs']

        # pep_obj is a Python string - branch happens at trace time, not runtime
        if pep_obj == 'obj_val':
            def metric_fn(k):
                z_k = z_iter[:, k] + zs  # Convert from shifted to absolute
                f_zk = 0.5 * jnp.dot(z_k, Q @ z_k)
                return f_zk - fs
        elif pep_obj == 'opt_dist_sq_norm':
            def metric_fn(k):
                # z_iter stores shifted coordinates (z_k - zs)
                return jnp.sum(z_iter[:, k] ** 2)
        elif pep_obj == 'grad_sq_norm':
            def metric_fn(k):
                z_k = z_iter[:, k] + zs
                grad = Q @ z_k
                return jnp.sum(grad ** 2)
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

        Args:
            stepsizes: Algorithm stepsizes tuple.
            batched_data: Dict with 'Q', 'z0', 'zs', 'fs' arrays (with batch dim).
            fixed_data: Dict of fixed parameters (empty for quadratic).
            traj_fn: Trajectory function from get_trajectory_fn().
            K_max: Number of algorithm iterations.

        Returns:
            Tuple of (G_batch, F_batch) Gram representations.
        """
        # traj_fn signature: (stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation)
        batch_GF_func = jax.vmap(
            lambda Q, z0, zs, fs: traj_fn(stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True),
            in_axes=(0, 0, 0, 0)
        )

        Q_batch = batched_data['Q']
        z0_batch = batched_data['z0']
        zs_batch = batched_data['zs']
        fs_batch = batched_data['fs']

        return batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)

    def generate_out_of_sample_data(
        self, key: jax.Array
    ) -> Dict[str, Tuple[ProblemData, GroundTruth]]:
        """Generate validation, test, and out-of-distribution problem sets.

        Args:
            key: JAX random key for reproducible sampling.

        Returns:
            Dict with keys 'validation', 'test', 'ood', each mapping to
            (problem_data, ground_truth) tuple.
        """
        N_val = self.cfg.out_of_sample_val_N
        N_test = self.cfg.out_of_sample_test_N
        N_ood = self.cfg.out_of_dist_N

        # Split key for each independent dataset
        key, val_key, test_key, ood_key = jax.random.split(key, 4)

        # In-distribution validation and test sets
        val_data = self.sample_validation_batch(val_key, N_val)
        test_data = self.sample_test_batch(test_key, N_test)

        # Out-of-distribution set (different eigenvalue distribution)
        ood_data = self._sample_ood_batch(ood_key, N_ood)

        return {
            'validation': val_data,
            'test': test_data,
            'ood': ood_data,
        }

    def _sample_ood_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Sample out-of-distribution problems with different eigenvalue distribution.

        When `data_source_dir` is configured and `ood_set.npz` is present under
        it, load N rows from there (seeded by `out_of_dist_seed`). Otherwise
        sample fresh via the Beta-eigenvalue OOD sampler.

        Args:
            key: JAX random key (unused on load path).
            N: Number of samples.

        Returns:
            (problem_data, ground_truth) tuple.
        """
        data_source_dir = self.cfg.get('data_source_dir', None)
        if data_source_dir is not None:
            loaded = _load_and_subsample_quad(
                os.path.join(data_source_dir, 'ood_set.npz'),
                N, self.cfg.out_of_dist_seed, 'ood',
            )
            if loaded is not None:
                Q_batch, z0_batch, zs_batch, fs_batch = loaded
                return (
                    {'Q_batch': Q_batch, 'z0_batch': z0_batch},
                    {'zs_batch': zs_batch, 'fs_batch': fs_batch},
                )

        # Fresh-sampling path
        key, k1, k2 = jax.random.split(key, 3)
        Q_subkeys = jax.random.split(k1, N)
        z0_subkeys = jax.random.split(k2, N)

        # OOD Q sampler (Beta distribution eigenvalues + random rotation)
        Q_batch = get_out_of_dist_Q_samples(Q_subkeys, self.M_val, self.mu_val, self.L_val)
        z0_batch = get_z0_samples(z0_subkeys, self.M_val, self.R_val)

        zs_batch = jnp.zeros(z0_batch.shape)
        fs_batch = jnp.zeros(N)

        return (
            {'Q_batch': Q_batch, 'z0_batch': z0_batch},
            {'zs_batch': zs_batch, 'fs_batch': fs_batch}
        )

    def get_supported_algorithms(self) -> list[str]:
        """Return list of algorithms supported by quadratic problems.

        Returns:
            ['vanilla_gd', 'nesterov_fgm']
        """
        return ['vanilla_gd', 'nesterov_fgm']

    def validate_config(self) -> None:
        """Validate configuration for quadratic problems.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Check for invalid combination: nesterov_fgm with silver stepsizes
        if self.cfg.alg == 'nesterov_fgm' and self.cfg.vector_init == "silver":
            raise ValueError(
                "Silver stepsizes are not compatible with nesterov_fgm algorithm. "
                "Use 'fixed' vector_init instead."
            )


# =============================================================================
# Entry point functions
# =============================================================================

def quad_run(cfg):
    """Run learning experiment for quadratic functions.

    Loops over K_max values, runs training for each K, and saves per-K progress CSV.
    Algorithm and learning framework are selected via config.

    Args:
        cfg: Hydra configuration object.
    """
    log.info(cfg)

    # Initialize random key
    key = jax.random.PRNGKey(cfg.seed)

    # Create problem module
    problem_module = QuadProblemModule(cfg)

    # Ensure output directory exists
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create the trainer once; pre-sample training + validation data once.
    # Both are K-independent, so sharing across K avoids redundant work.
    problem_module.validate_config()
    key, train_key = jax.random.split(key)
    trainer = UnifiedTrainer(problem_module, cfg, train_key)
    trainer.prepare_data(save_dir=output_dir)

    # Loop over K_max values
    for K in cfg.K_max:
        log.info(f"=== Starting training for K={K} ===")

        # Create output directory for this K
        K_output_dir = os.path.join(output_dir, f"K_{K}")
        os.makedirs(K_output_dir, exist_ok=True)
        csv_path = os.path.join(K_output_dir, "progress.csv")

        result = trainer.train(K, csv_path, K_output_dir)

        # Log final stepsizes
        t = result.stepsizes[0]
        is_vector = jnp.ndim(t) > 0
        t_str = str(t.tolist()) if is_vector else f'{float(t):.6f}'
        log.info(f'K={K} complete. Final t={t_str}. Saved to {csv_path}')

    log.info("=== Experiment complete ===")


def quad_sample_creation_run(cfg):
    """Generate and save all Quad problem-instance sets in a unified format.

    Produces four sets, each as a single .npz with keys
    `Q_batch`, `z0_batch`, `zs_batch`, `fs_batch`:
        training_set.npz   (in-distribution, size cfg.training_sample_N)
        validation_set.npz (in-distribution, size cfg.out_of_sample_val_N)
        test_set.npz       (in-distribution, size cfg.out_of_sample_test_N)
        ood_set.npz        (out-of-distribution, size cfg.out_of_dist_N;
                            Beta-eigenvalue Q with random rotation, matrix
                            width M_val to match in-dist)

    Plus split files for the plot pipeline:
        Q_test_samples.npz, z0_test_samples.npz
        Q_out_of_dist_samples.npz, z0_out_of_dist_samples.npz
        out_of_sample_metadata.npz

    For Quad, the optimum is always at the origin (zs=0, fs=0), so no per-
    instance solver is required.

    Args:
        cfg: Hydra configuration object.
    """
    log.info("=" * 60)
    log.info("Generating Quad sample-creation problem sets")
    log.info("=" * 60)
    log.info(cfg)

    # Extract config values
    d_val = cfg.dim
    mu_val = cfg.mu
    L_val = cfg.L
    R_val = cfg.R
    training_sample_N = cfg.training_sample_N
    training_seed = cfg.training_seed
    out_of_sample_val_N = cfg.out_of_sample_val_N
    out_of_sample_test_N = cfg.out_of_sample_test_N
    out_of_sample_val_seed = cfg.out_of_sample_val_seed
    out_of_sample_test_seed = cfg.out_of_sample_test_seed
    out_of_dist_N = cfg.out_of_dist_N
    out_of_dist_seed = cfg.out_of_dist_seed

    # Matrix width for both in-dist (Marchenko-Pastur) and OOD samplers.
    r_val = (np.sqrt(L_val) - np.sqrt(mu_val)) ** 2 / (np.sqrt(L_val) + np.sqrt(mu_val)) ** 2
    M_val = int(np.round(r_val * d_val))
    log.info(f"Precomputed matrix width M: {M_val} (used for both in-dist and OOD)")

    def _build_set(name, N, seed, filename, ood=False):
        log.info(f"Generating {N} {name} problems (seed={seed})...")
        key = jax.random.PRNGKey(seed)
        key, k1, k2 = jax.random.split(key, 3)
        Q_subkeys = jax.random.split(k1, N)
        z0_subkeys = jax.random.split(k2, N)

        if ood:
            Q_batch = get_out_of_dist_Q_samples(Q_subkeys, M_val, mu_val, L_val)
            z0_batch = get_z0_samples(z0_subkeys, M_val, R_val)
        else:
            Q_batch = get_Q_samples(Q_subkeys, d_val, mu_val, L_val, M_val)
            z0_batch = get_z0_samples(z0_subkeys, M_val, R_val)

        Q_np = np.array(Q_batch)
        z0_np = np.array(z0_batch)
        zs_np = np.zeros(z0_np.shape, dtype=z0_np.dtype)
        fs_np = np.zeros(N, dtype=z0_np.dtype)

        np.savez_compressed(
            filename,
            Q_batch=Q_np,
            z0_batch=z0_np,
            zs_batch=zs_np,
            fs_batch=fs_np,
        )
        log.info(f"Saved {filename} (Q {Q_np.shape}, z0 {z0_np.shape})")

    _build_set("training",   training_sample_N,    training_seed,           "training_set.npz")
    _build_set("validation", out_of_sample_val_N,  out_of_sample_val_seed,  "validation_set.npz")
    _build_set("test",       out_of_sample_test_N, out_of_sample_test_seed, "test_set.npz")
    _build_set("ood",        out_of_dist_N,        out_of_dist_seed,        "ood_set.npz", ood=True)

    # Split files for experiment_plots_icml/quad/create_test_plots.py
    test = np.load("test_set.npz")
    np.savez_compressed("Q_test_samples.npz",  Q=test["Q_batch"])
    np.savez_compressed("z0_test_samples.npz", z0=test["z0_batch"])

    ood = np.load("ood_set.npz")
    np.savez_compressed("Q_out_of_dist_samples.npz",  Q=ood["Q_batch"])
    np.savez_compressed("z0_out_of_dist_samples.npz", z0=ood["z0_batch"])

    metadata = {
        'training_sample_N': training_sample_N,
        'training_seed': training_seed,
        'out_of_sample_val_N': out_of_sample_val_N,
        'out_of_sample_test_N': out_of_sample_test_N,
        'out_of_sample_val_seed': out_of_sample_val_seed,
        'out_of_sample_test_seed': out_of_sample_test_seed,
        'out_of_dist_N': out_of_dist_N,
        'out_of_dist_seed': out_of_dist_seed,
        'dim': d_val,
        'mu': mu_val,
        'L': L_val,
        'R': R_val,
        'M': M_val,
    }
    np.savez_compressed("out_of_sample_metadata.npz", **metadata)

    log.info("=== Quad sample-creation complete ===")
