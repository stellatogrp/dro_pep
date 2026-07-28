"""Logistic regression problem module for the unified learning framework.

Implements LogRegProblemModule using the ProblemModule ABC and UnifiedTrainer,
porting the pre-refactor pipeline in `old/logreg.py`.

The problem is UNREGULARIZED logistic regression (delta = 0): smooth convex
but not strongly convex, so mu = 0 and the smooth-convex PEP interpolation
conditions apply. Each sample carries its own data matrix A and label vector
b; the algorithm starts at x0 = 0 (z0 = -x_opt in the shifted coordinates
used by the trajectory functions). Supported algorithms: vanilla_gd and
nesterov_fgm (learned step sizes t and, for FGM, momentum beta).
"""

import diffcp_patch  # noqa: F401  # Apply COO -> CSC fix for diffcp
import cvxpy as cp
import jax
import jax.numpy as jnp
import logging
import numpy as np
import os
import pandas as pd
from functools import partial
from tqdm import trange
from typing import Any, Callable, Dict, Tuple

from learning.problem_module import ProblemModule, ProblemData, GroundTruth, Stepsizes, ParameterNames
from learning.unified_trainer import UnifiedTrainer
from learning.pep_constructions import construct_gd_pep_data, construct_fgm_pep_data
from learning.trajectories.logreg_gd_fgm import (
    create_logreg_traj_fn_gd,
    create_logreg_traj_fn_fgm,
)
from learning.silver_stepsizes import get_nonstrongly_convex_silver_stepsizes
from learning.acceleration_stepsizes import jax_get_nesterov_fgm_beta_sequence

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)


# =============================================================================
# Sampling helpers (ported from old/logreg.py)
# =============================================================================

def generate_single_A_logreg_jax(key, N_data, n, A_std):
    """Generate a single (N_data, n) data matrix; last column is the intercept."""
    A_features = jax.random.normal(key, shape=(N_data, n - 1)) * A_std
    A_intercept = jnp.ones((N_data, 1))
    return jnp.hstack([A_features, A_intercept])


@partial(jax.jit, static_argnames=['N_batch', 'N_data', 'n'])
def generate_batch_A_logreg_jax(key, N_batch, N_data, n, A_std):
    """Generate a batch of A matrices, shape (N_batch, N_data, n)."""
    keys = jax.random.split(key, N_batch)
    sample_one = partial(generate_single_A_logreg_jax,
                         N_data=N_data, n=n, A_std=A_std)
    return jax.vmap(sample_one)(keys)


@partial(jax.jit, static_argnames=['n', 'p_beta_nonzero'])
def generate_true_beta_jax(key, n, p_beta_nonzero, beta_scale):
    """Generate a sparse true coefficient vector."""
    key1, key2 = jax.random.split(key)
    beta = jax.random.uniform(key1, (n,), minval=-beta_scale, maxval=beta_scale)
    mask = jax.random.bernoulli(key2, p=p_beta_nonzero, shape=(n,)).astype(jnp.float64)
    return beta * mask


@jax.jit
def generate_labels_jax(key, A, beta_true, eps_std):
    """Binary labels: y = 1{A @ beta + eps > 0}."""
    N_data = A.shape[0]
    noise = eps_std * jax.random.normal(key, (N_data,))
    Abeta_noise = A @ beta_true + noise
    return jnp.where(Abeta_noise > 0, 1.0, 0.0)


@partial(jax.jit, static_argnames=['n', 'p_beta_nonzero'])
def generate_single_logreg_labels_jax(key, A, n, p_beta_nonzero, beta_scale, eps_std):
    """Generate labels for a fixed A (fresh sparse beta per instance)."""
    key1, key2 = jax.random.split(key)
    beta_true = generate_true_beta_jax(key1, n, p_beta_nonzero, beta_scale)
    return generate_labels_jax(key2, A, beta_true, eps_std)


def solve_logreg_cvxpy(A_np, b_np, delta):
    """Solve one logistic regression instance with CVXPY/CLARABEL.

    min 1/m * sum(-y_i * (A @ beta)_i + log(1 + exp((A @ beta)_i)))
        + delta/2 * ||beta||^2
    """
    m, n = A_np.shape
    beta = cp.Variable(n)
    log_likelihood = cp.sum(
        cp.multiply(b_np, A_np @ beta) - cp.logistic(A_np @ beta)
    )
    obj = -1 / m * log_likelihood + 0.5 * delta * cp.sum_squares(beta)
    problem = cp.Problem(cp.Minimize(obj))
    problem.solve(solver='CLARABEL')
    return beta.value, problem.value


def sample_logreg_batch(key, N, N_data, n, A_std, p_beta_nonzero,
                        beta_scale, eps_std, delta):
    """Sample N logreg instances (per-instance A and b) and solve each.

    Returns:
        A_batch: (N, N_data, n), b_batch: (N, N_data),
        x_opt_batch: (N, n), f_opt_batch: (N,)
    """
    k1, k2 = jax.random.split(key)
    A_batch = generate_batch_A_logreg_jax(k1, N, N_data, n, A_std)
    b_keys = jax.random.split(k2, N)

    def generate_b_for_A(key, A):
        return generate_single_logreg_labels_jax(
            key, A, n, p_beta_nonzero, beta_scale, eps_std
        )

    b_batch = jax.vmap(generate_b_for_A)(b_keys, A_batch)

    A_batch_np = np.array(A_batch)
    b_batch_np = np.array(b_batch)
    x_opt_batch_np = np.zeros((N, n))
    f_opt_batch_np = np.zeros(N)

    iterator = trange(N, desc="Solving logreg problems") if N >= 50 else range(N)
    for i in iterator:
        x_opt, f_opt = solve_logreg_cvxpy(A_batch_np[i], b_batch_np[i], delta)
        x_opt_batch_np[i] = x_opt
        f_opt_batch_np[i] = f_opt

    return A_batch, b_batch, jnp.array(x_opt_batch_np), jnp.array(f_opt_batch_np)


def compute_logreg_L_single(A, delta):
    """Smoothness constant for one instance: lambda_max(A^T A) / (4m) + delta."""
    m = A.shape[0]
    lambd_max = jnp.max(jnp.linalg.eigvalsh(A.T @ A))
    return lambd_max / (4 * m) + delta


def compute_logreg_L_worst_case(cfg):
    """Worst-case L over cfg.L_sample_size fresh A matrices (fallback path)."""
    log.info(f"Computing worst-case L from {cfg.L_sample_size} A matrices...")
    key = jax.random.PRNGKey(cfg.L_seed)
    A_batch = generate_batch_A_logreg_jax(
        key, cfg.L_sample_size, cfg.N_data, cfg.n, cfg.A_std
    )
    L_vals = jax.vmap(lambda A: compute_logreg_L_single(A, cfg.delta))(A_batch)
    L_max = float(jnp.max(L_vals))
    log.info(f"L distribution: mean={float(jnp.mean(L_vals)):.6f}, "
             f"std={float(jnp.std(L_vals)):.6f}, max={L_max:.6f}")
    return L_max, float(cfg.delta)


def compute_sample_radius_logreg(cfg):
    """R = max ||x_opt|| over cfg.R_sample_size fresh instances (fallback path)."""
    log.info(f"Computing R from {cfg.R_sample_size} (A, b) samples...")
    key = jax.random.PRNGKey(cfg.R_seed)
    _, _, x_opt_batch, _ = sample_logreg_batch(
        key, cfg.R_sample_size, cfg.N_data, cfg.n, cfg.A_std,
        cfg.p_beta_nonzero, cfg.beta_scale, cfg.eps_std, cfg.delta,
    )
    R_max = float(jnp.max(jnp.linalg.norm(x_opt_batch, axis=1)))
    log.info(f"Computed R = {R_max:.6f}")
    return R_max


# =============================================================================
# data_source_dir loader (mirror of quad._load_and_subsample_quad)
# =============================================================================

def _load_and_subsample_logreg(npz_path, N, seed, dataset_label):
    """Load a saved logreg problem-instance bundle and subsample N aligned rows.

    Returns (A, b, z0, x_opt, f_opt) as jnp.ndarrays, or None if the file is
    missing (caller falls back to fresh sampling).
    """
    if not os.path.isfile(npz_path):
        return None
    d = np.load(npz_path)
    total = int(d['A_batch'].shape[0])
    if N >= total:
        if N > total:
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
        jnp.asarray(d['A_batch'][idx]),
        jnp.asarray(d['b_batch'][idx]),
        jnp.asarray(d['z0_batch'][idx]),
        jnp.asarray(d['x_opt_batch'][idx]),
        jnp.asarray(d['f_opt_batch'][idx]),
    )


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
# LogRegProblemModule class
# =============================================================================

class LogRegProblemModule(ProblemModule):
    """Problem module for (unregularized) logistic regression.

    Every sample carries its own (A, b); delta (the L2 weight) is fixed by
    config and baked into the trajectory functions, so it never enters the
    batched-parameter plumbing.
    """

    def __init__(self, cfg: Any):
        super().__init__(cfg)

        self.n_val = cfg.n
        self.N_data_val = cfg.N_data
        self.delta_val = float(cfg.delta)
        self.A_std_val = cfg.A_std
        self.p_beta_nonzero_val = cfg.p_beta_nonzero
        self.beta_scale_val = cfg.beta_scale
        self.eps_std_val = cfg.eps_std

        # Factory-built trajectory fns with delta baked in (stable jit identity)
        self._traj_fn_gd = create_logreg_traj_fn_gd(self.delta_val)
        self._traj_fn_fgm = create_logreg_traj_fn_fgm(self.delta_val)

        # Cache for compute_L_mu_R (data-dependent, computed once)
        self._L_mu_R = None

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def _sample_fresh_batch(self, key: jax.Array, N: int, A_std: float,
                            eps_std: float | None = None) -> Tuple[ProblemData, GroundTruth]:
        """Fresh (A, b) sampling + CVXPY solves; z0 = -x_opt (x0 = 0)."""
        if eps_std is None:
            eps_std = self.eps_std_val
        A_batch, b_batch, x_opt_batch, f_opt_batch = sample_logreg_batch(
            key, N, self.N_data_val, self.n_val, A_std,
            self.p_beta_nonzero_val, self.beta_scale_val,
            eps_std, self.delta_val,
        )
        z0_batch = -x_opt_batch
        return (
            {'A_batch': A_batch, 'b_batch': b_batch, 'z0_batch': z0_batch},
            {'x_opt_batch': x_opt_batch, 'f_opt_batch': f_opt_batch},
        )

    def _load_set(self, filename: str, N: int, seed, label: str):
        data_source_dir = self.cfg.get('data_source_dir', None)
        if data_source_dir is None:
            return None
        loaded = _load_and_subsample_logreg(
            os.path.join(data_source_dir, filename), N, seed, label,
        )
        if loaded is None:
            return None
        A_batch, b_batch, z0_batch, x_opt_batch, f_opt_batch = loaded
        return (
            {'A_batch': A_batch, 'b_batch': b_batch, 'z0_batch': z0_batch},
            {'x_opt_batch': x_opt_batch, 'f_opt_batch': f_opt_batch},
        )

    def sample_training_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        loaded = self._load_set('training_set.npz', N,
                                self.cfg.get('training_seed', 40000), 'training')
        if loaded is not None:
            return loaded
        return self._sample_fresh_batch(key, N, self.A_std_val)

    def sample_validation_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        loaded = self._load_set('validation_set.npz', N,
                                self.cfg.out_of_sample_val_seed, 'validation')
        if loaded is not None:
            return loaded
        return self._sample_fresh_batch(key, N, self.A_std_val)

    def sample_test_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        loaded = self._load_set('test_set.npz', N,
                                self.cfg.out_of_sample_test_seed, 'test')
        if loaded is not None:
            return loaded
        return self._sample_fresh_batch(key, N, self.A_std_val)

    def _sample_ood_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """OOD: A_std AND eps_std scaled by ood_std_multiplier.

        Scaling both preserves the margin-to-noise ratio (same label
        distribution, so instances stay decisively non-separable and x_opt
        bounded, which unregularized logistic regression requires) while the
        smoothness constant grows by multiplier^2 — the L-shift robustness
        axis, mirroring the Quad OOD design.
        """
        loaded = self._load_set('ood_set.npz', N,
                                self.cfg.out_of_dist_seed, 'ood')
        if loaded is not None:
            return loaded
        multiplier = self.cfg.get('ood_std_multiplier', 1.25)
        return self._sample_fresh_batch(
            key, N, self.A_std_val * multiplier, self.eps_std_val * multiplier
        )

    def generate_out_of_sample_data(
        self, key: jax.Array
    ) -> Dict[str, Tuple[ProblemData, GroundTruth]]:
        key, val_key, test_key, ood_key = jax.random.split(key, 4)
        return {
            'validation': self.sample_validation_batch(val_key, self.cfg.out_of_sample_val_N),
            'test': self.sample_test_batch(test_key, self.cfg.out_of_sample_test_N),
            'ood': self._sample_ood_batch(ood_key, self.cfg.out_of_dist_N),
        }

    # -------------------------------------------------------------------------
    # Algorithm plumbing
    # -------------------------------------------------------------------------

    def get_trajectory_fn(self, alg: str) -> Callable:
        if alg == 'vanilla_gd':
            return self._traj_fn_gd
        elif alg == 'nesterov_fgm':
            return self._traj_fn_fgm
        raise ValueError(f"Unknown algorithm: {alg}")

    def get_pep_data_fn(self, alg: str) -> Callable:
        if alg == 'vanilla_gd':
            return pep_data_fn_gd
        elif alg == 'nesterov_fgm':
            return pep_data_fn_fgm
        raise ValueError(f"Unknown algorithm: {alg}")

    def compute_L_mu_R(self, samples: ProblemData | None = None) -> Tuple[float, float, float]:
        """Compute (L, mu, R) from the pinned training set when available.

        L = max_i lambda_max(A_i^T A_i) / (4m) + delta over the training
        instances, R = max_i ||x_opt_i|| (valid IC bound since z0 = -x_opt),
        mu = delta. Falls back to cfg L/R, then to fresh-sample estimates.
        """
        if self._L_mu_R is not None:
            return self._L_mu_R

        mu = self.delta_val
        data_source_dir = self.cfg.get('data_source_dir', None)
        train_path = (os.path.join(data_source_dir, 'training_set.npz')
                      if data_source_dir else None)
        if train_path is not None and os.path.isfile(train_path):
            d = np.load(train_path)
            A_batch = jnp.asarray(d['A_batch'])
            x_opt_batch = jnp.asarray(d['x_opt_batch'])
            L_vals = jax.vmap(
                lambda A: compute_logreg_L_single(A, self.delta_val)
            )(A_batch)
            L = float(jnp.max(L_vals))
            R = float(jnp.max(jnp.linalg.norm(x_opt_batch, axis=1)))
            log.info(
                f"Computed from {train_path}: L={L:.6f} (cfg {self.cfg.get('L', None)}), "
                f"R={R:.6f} (cfg {self.cfg.get('R', None)}), mu={mu}"
            )
        else:
            L_cfg = self.cfg.get('L', None)
            R_cfg = self.cfg.get('R', None)
            L = float(L_cfg) if L_cfg is not None else compute_logreg_L_worst_case(self.cfg)[0]
            R = float(R_cfg) if R_cfg is not None else compute_sample_radius_logreg(self.cfg)
            log.info(f"Using L={L:.6f}, R={R:.6f}, mu={mu} (no training_set.npz)")

        self._L_mu_R = (L, mu, R)
        return self._L_mu_R

    def get_initial_stepsizes(self, alg: str, K: int, L: float, mu: float) -> Stepsizes:
        """Initialization: t = 1/L for both algorithms (mu = 0 regime);
        FGM additionally gets the standard Nesterov beta sequence."""
        is_vector = self.cfg.stepsize_type == "vector"
        t_scalar = 1.0 / L if mu == 0 else 2.0 / (mu + L)

        if alg == 'nesterov_fgm':
            t = jnp.full(K, t_scalar) if is_vector else jnp.array(t_scalar)
            beta = jax_get_nesterov_fgm_beta_sequence(mu, L, K)
            return (t, beta)
        elif alg == 'vanilla_gd':
            if is_vector:
                if self.cfg.vector_init == "silver":
                    t = jnp.array(get_nonstrongly_convex_silver_stepsizes(K, L=L))
                else:  # "fixed"
                    t = jnp.full(K, t_scalar)
            else:
                t = jnp.array(t_scalar)
            return (t,)
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
        lrs: list[float] | None = None,
    ) -> pd.DataFrame:
        """CSV columns follow the Quad convention: t0..t{K-1}, beta0..beta{K-1}."""
        t_sample = stepsizes_history[0][0]
        is_vector_t = jnp.ndim(t_sample) > 0
        has_beta = len(stepsizes_history[0]) > 1

        data = {'iteration': list(range(len(stepsizes_history)))}
        if training_losses is not None:
            data['training_loss'] = [float(l) for l in training_losses]
        if validation_losses is not None:
            data['validation_loss'] = [float(l) for l in validation_losses]
        if times is not None:
            data['iter_time'] = [float(t) for t in times]
        if raw_grad_norms is not None:
            data['raw_grad_norm'] = [float(g) for g in raw_grad_norms]
        if lrs is not None:
            data['lr'] = [float(x) for x in lrs]

        if is_vector_t:
            for k in range(K_max):
                data[f't{k}'] = [float(ss[0][k]) for ss in stepsizes_history]
        else:
            data['t'] = [float(ss[0]) for ss in stepsizes_history]

        if has_beta:
            for k in range(K_max):
                data[f'beta{k}'] = [float(ss[1][k]) for ss in stepsizes_history]

        return pd.DataFrame(data)

    # -------------------------------------------------------------------------
    # Parameter structure
    # -------------------------------------------------------------------------

    def get_batched_parameters(self) -> ParameterNames:
        return ('A', 'b', 'z0', 'x_opt', 'f_opt')

    def get_fixed_parameters(self) -> ParameterNames:
        # delta is baked into the trajectory functions
        return ()

    def get_ground_truth_keys(self) -> ParameterNames:
        return ('x_opt', 'f_opt')

    def get_gram_dimensions(self, alg: str, K: int) -> Tuple[int, int]:
        # For both GD and FGM: dimG = K+2, dimF = K+1
        return (K + 2, K + 1)

    # -------------------------------------------------------------------------
    # Metric / trajectory computation
    # -------------------------------------------------------------------------

    def create_metric_fn(
        self, trajectories: Any, problem_data: ProblemData, ground_truth: GroundTruth, pep_obj: str
    ) -> Callable[[int], float]:
        """Metric via the trajectory stacks (already in shifted coordinates).

        For both GD and FGM the f-stack has K+1 entries with the last one
        being f(x_K) - f_opt, and the gradient stack has K+1 columns; indexing
        those directly avoids the FGM y-iterate stack having only K columns.
        opt_dist_sq_norm is only valid for vanilla_gd (FGM stores y-points).
        """
        f_stack = trajectories[2]
        g_stack = trajectories[1]
        z_stack = trajectories[0]

        if pep_obj == 'obj_val':
            def metric_fn(k):
                return f_stack[k]
        elif pep_obj == 'grad_sq_norm':
            def metric_fn(k):
                return jnp.sum(g_stack[:, k] ** 2)
        elif pep_obj == 'opt_dist_sq_norm':
            if self.cfg.alg == 'nesterov_fgm':
                raise NotImplementedError(
                    "opt_dist_sq_norm is not supported for nesterov_fgm: the "
                    "FGM trajectory stack stores y_0..y_{K-1} (K columns), so "
                    "the final iterate x_K is not addressable. Use obj_val."
                )

            def metric_fn(k):
                return jnp.sum(z_stack[:, k] ** 2)
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
        batch_GF_func = jax.vmap(
            lambda A, b, z0, x_opt, f_opt: traj_fn(
                stepsizes, A, b, z0, x_opt, f_opt, K_max,
                return_Gram_representation=True,
            ),
            in_axes=(0, 0, 0, 0, 0),
        )
        return batch_GF_func(
            batched_data['A'],
            batched_data['b'],
            batched_data['z0'],
            batched_data['x_opt'],
            batched_data['f_opt'],
        )

    def get_supported_algorithms(self) -> list[str]:
        return ['vanilla_gd', 'nesterov_fgm']

    def validate_config(self) -> None:
        if self.cfg.alg == 'nesterov_fgm' and self.cfg.vector_init == "silver":
            raise ValueError(
                "Silver stepsizes are not compatible with nesterov_fgm. "
                "Use 'fixed' vector_init instead."
            )
        if self.delta_val != 0.0:
            raise ValueError(
                f"LogReg experiment is the unregularized (delta=0, mu=0) "
                f"smooth-convex benchmark; got delta={self.delta_val}. "
                "Remove the override or change this guard deliberately."
            )


# =============================================================================
# Entry point functions
# =============================================================================

def logreg_run(cfg):
    """Run learning experiment for logistic regression.

    Loops over K_max values, runs training for each K, and saves per-K
    progress CSV. Algorithm and learning framework are selected via config.
    """
    log.info(cfg)

    key = jax.random.PRNGKey(cfg.seed)
    problem_module = LogRegProblemModule(cfg)

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    problem_module.validate_config()
    key, train_key = jax.random.split(key)
    trainer = UnifiedTrainer(problem_module, cfg, train_key)
    trainer.prepare_data(save_dir=output_dir)

    for K in cfg.K_max:
        log.info(f"=== Starting training for K={K} ===")
        K_output_dir = os.path.join(output_dir, f"K_{K}")
        os.makedirs(K_output_dir, exist_ok=True)
        csv_path = os.path.join(K_output_dir, "progress.csv")

        result = trainer.train(K, csv_path, K_output_dir)

        t = result.stepsizes[0]
        is_vector = jnp.ndim(t) > 0
        t_str = str(t.tolist()) if is_vector else f'{float(t):.6f}'
        log.info(f'K={K} complete. Final t={t_str}. Saved to {csv_path}')

    log.info("=== Experiment complete ===")


def logreg_sample_creation_run(cfg):
    """Generate and save all LogReg problem-instance sets in a unified format.

    Produces four sets, each as a single .npz with keys
    `A_batch`, `b_batch`, `z0_batch`, `x_opt_batch`, `f_opt_batch`:
        training_set.npz   (in-distribution, size cfg.training_sample_N)
        validation_set.npz (in-distribution, size cfg.out_of_sample_val_N)
        test_set.npz       (in-distribution, size cfg.out_of_sample_test_N)
        ood_set.npz        (out-of-distribution, size cfg.out_of_dist_N;
                            A_std AND eps_std scaled by cfg.ood_std_multiplier,
                            raising L by multiplier^2 at unchanged label
                            distribution)

    z0 = -x_opt for every instance (x0 = 0 in original coordinates). Every
    instance is solved with CVXPY/CLARABEL. Per-set L_max and R_max are
    recorded in out_of_sample_metadata.npz.
    """
    log.info("=" * 60)
    log.info("Generating LogReg sample-creation problem sets")
    log.info("=" * 60)
    log.info(cfg)

    module = LogRegProblemModule(cfg)
    multiplier = cfg.get('ood_std_multiplier', 1.25)
    delta = float(cfg.delta)

    metadata = {
        'n': cfg.n, 'N_data': cfg.N_data, 'delta': delta,
        'A_std': cfg.A_std, 'p_beta_nonzero': cfg.p_beta_nonzero,
        'beta_scale': cfg.beta_scale, 'eps_std': cfg.eps_std,
        'ood_std_multiplier': multiplier,
        'training_sample_N': cfg.training_sample_N,
        'training_seed': cfg.training_seed,
        'out_of_sample_val_N': cfg.out_of_sample_val_N,
        'out_of_sample_val_seed': cfg.out_of_sample_val_seed,
        'out_of_sample_test_N': cfg.out_of_sample_test_N,
        'out_of_sample_test_seed': cfg.out_of_sample_test_seed,
        'out_of_dist_N': cfg.out_of_dist_N,
        'out_of_dist_seed': cfg.out_of_dist_seed,
    }

    def _build_set(name, N, seed, filename, A_std, eps_std):
        log.info(f"Generating {N} {name} problems "
                 f"(seed={seed}, A_std={A_std}, eps_std={eps_std})...")
        key = jax.random.PRNGKey(int(seed))
        problem_data, ground_truth = module._sample_fresh_batch(key, N, A_std, eps_std)

        A_np = np.array(problem_data['A_batch'])
        b_np = np.array(problem_data['b_batch'])
        z0_np = np.array(problem_data['z0_batch'])
        x_opt_np = np.array(ground_truth['x_opt_batch'])
        f_opt_np = np.array(ground_truth['f_opt_batch'])

        np.savez_compressed(
            filename,
            A_batch=A_np, b_batch=b_np, z0_batch=z0_np,
            x_opt_batch=x_opt_np, f_opt_batch=f_opt_np,
        )

        L_vals = jax.vmap(
            lambda A: compute_logreg_L_single(A, delta)
        )(jnp.asarray(A_np))
        L_max = float(jnp.max(L_vals))
        R_max = float(np.max(np.linalg.norm(x_opt_np, axis=1)))
        metadata[f'{name}_L_max'] = L_max
        metadata[f'{name}_R_max'] = R_max
        log.info(f"Saved {filename} (A {A_np.shape}); L_max={L_max:.6f}, R_max={R_max:.6f}")

    _build_set("training", cfg.training_sample_N, cfg.training_seed,
               "training_set.npz", cfg.A_std, cfg.eps_std)
    _build_set("validation", cfg.out_of_sample_val_N, cfg.out_of_sample_val_seed,
               "validation_set.npz", cfg.A_std, cfg.eps_std)
    _build_set("test", cfg.out_of_sample_test_N, cfg.out_of_sample_test_seed,
               "test_set.npz", cfg.A_std, cfg.eps_std)
    _build_set("ood", cfg.out_of_dist_N, cfg.out_of_dist_seed,
               "ood_set.npz", cfg.A_std * multiplier, cfg.eps_std * multiplier)

    np.savez_compressed("out_of_sample_metadata.npz", **metadata)
    log.info("=== LogReg sample-creation complete ===")
