"""Unified training orchestrator for all three learning pipelines.

This module consolidates LPEP, L2O, and LDRO-PEP training loops into a single
UnifiedTrainer class, eliminating ~500-800 lines of duplicated code across
problem-specific experiment classes.

The key insight is that the three pipelines differ only in loss construction,
not in training loop structure:
- LPEP: Deterministic worst-case PEP → wc_pep_scs_solve(pep_data)
- L2O: Stochastic trajectory-based → risk_measure(trajectory_losses)
- LDRO-PEP: Stochastic DRO SDP → dro_scs_solve(pep_data, G_batch, F_batch)
"""

import jax
import jax.numpy as jnp
import logging
import os
import time
from typing import Callable, Dict, Tuple, List, Any
from omegaconf import DictConfig
import numpy as np

from learning.training_result import TrainingResult, Stepsizes
from learning.problem_module import ProblemModule
from learning.adam_optimizers import AdamWMin
from learning.jax_scs_layer import wc_pep_scs_solve, dro_scs_solve, compute_preconditioner_from_samples

log = logging.getLogger(__name__)


class UnifiedTrainer:
    """Unified training orchestrator for all three learning pipelines.

    Responsibilities:
    - Training loop management (minibatch iteration, checkpointing)
    - Loss function construction based on learning_framework
    - Optimizer management (AdamWMin, vanilla_sgd, sgd_wd)
    - Stepsize initialization and projection
    - Progress tracking and CSV saving

    The trainer delegates problem-specific operations to the ProblemModule:
    - Training data sampling
    - Trajectory computation functions
    - PEP constraint construction
    - Stepsize initialization
    - CSV output formatting
    """

    def __init__(self, problem_module: ProblemModule, cfg: DictConfig, key: jax.Array):
        """Initialize trainer with problem module and configuration.

        Args:
            problem_module: Problem-specific module implementing ProblemModule interface.
            cfg: Hydra configuration object containing all training parameters.
            key: JAX random key for reproducible sampling.
        """
        self.problem_module = problem_module
        self.cfg = cfg
        self.key = key

        # Extract common parameters from config
        self.learning_framework = cfg.learning_framework  # 'lpep', 'l2o', or 'ldro-pep'
        self.optimizer_type = cfg.get('optimizer_type', 'vanilla_sgd')
        self.sgd_iters = cfg.sgd_iters
        self.eta_t = cfg.eta_t
        self.alg = cfg.alg
        self.pep_obj = cfg.pep_obj
        self.stepsize_type = cfg.stepsize_type  # 'scalar' or 'vector'
        self.loss_type_composition = cfg.get('loss_type_composition', 'final')  # For L2O

        # DRO-specific parameters (only used for ldro-pep and l2o with risk measures)
        self.eps = cfg.get('eps', 0.5)
        self.alpha = cfg.get('alpha', 0.1)
        self.dro_obj = cfg.get('dro_obj', 'expectation')  # 'expectation' or 'cvar'
        self.risk_type = 'cvar' if self.dro_obj == 'cvar' else 'expectation'

        # Training data parameters (for l2o and ldro-pep)
        self.N_batch = cfg.get('N', 50)  # Minibatch size
        self.training_sample_N = cfg.get('training_sample_N', 500)

        # Validation data parameters (for monitoring generalization during training)
        # Support both old and new naming conventions
        self.validation_sample_N = cfg.get('out_of_sample_val_N', cfg.get('out_of_sample_N', 100))
        self.validation_seed = cfg.get('out_of_sample_val_seed', cfg.get('out_of_sample_seed', 10000))

        # LDRO-PEP specific parameters
        self.precond_type = cfg.get('precond_type', 'average')
        self.dro_canon_backend = cfg.get('dro_canon_backend', 'manual_jax')

        # Optimizer parameters
        self.weight_decay = cfg.get('weight_decay', 1e-2)
        self.learn_beta = cfg.get('learn_beta', True)

        # State
        self.optimizer = None
        self.training_data = None
        self.validation_data = None
        self.n_minibatches = 0
        self.precond_inv = None

    def train(self, K: int, csv_path: str, K_output_dir: str = None) -> TrainingResult:
        """Main training entry point.

        Args:
            K: Number of algorithm iterations (K_max).
            csv_path: Path to save progress CSV.
            K_output_dir: Optional directory to save additional outputs (training data).

        Returns:
            TrainingResult with final stepsizes, history, losses, and times.
        """
        log.info(f"=== Starting {self.learning_framework.upper()} training for K={K} ===")

        # Step 1: Compute problem parameters
        L, mu, R = self.problem_module.compute_L_mu_R()
        log.info(f"Problem parameters: L={L:.4f}, mu={mu:.4f}, R={R:.4f}")

        # Step 2: Initialize stepsizes
        stepsizes = self.problem_module.get_initial_stepsizes(self.alg, K, L, mu)
        log.info(f"Initial stepsizes: {stepsizes}")

        # Step 3: Pre-sample training data (for L2O/LDRO-PEP only)
        if self.learning_framework in ['l2o', 'ldro-pep']:
            self._presample_training_data(K, K_output_dir)

        # Step 4: Pre-sample validation data (for all frameworks)
        self._presample_validation_data()

        # Step 5: Build loss function
        loss_fn = self._build_loss_function(K, L, mu, R, stepsizes)

        # Step 6: Build validation loss function
        val_loss_fn = self._build_validation_loss_function(K)

        # Step 7: Initialize optimizer
        self._initialize_optimizer(stepsizes)

        # Step 8: Run training loop
        result = self._run_training_loop(loss_fn, val_loss_fn, stepsizes, K, csv_path)

        log.info(f"=== Training complete for K={K} ===")
        return result

    def _build_loss_function(self, K: int, L: float, mu: float, R: float,
                            initial_stepsizes: Stepsizes) -> Callable:
        """Factory method dispatching to pipeline-specific loss builders.

        Args:
            K: Number of algorithm iterations.
            L: Lipschitz constant (smoothness).
            mu: Strong convexity parameter.
            R: Initial radius bound.
            initial_stepsizes: Initial stepsizes (used for preconditioner in LDRO-PEP).

        Returns:
            JIT-compiled loss function with signature:
                - LPEP: loss_fn(stepsizes) -> scalar
                - L2O/LDRO-PEP: loss_fn(stepsizes, minibatch_idx) -> scalar
        """
        if self.learning_framework == 'lpep':
            return self._build_lpep_loss(K, L, mu, R)
        elif self.learning_framework == 'l2o':
            return self._build_l2o_loss(K)
        elif self.learning_framework == 'ldro-pep':
            return self._build_ldro_pep_loss(K, L, mu, R, initial_stepsizes)
        else:
            raise ValueError(f"Unknown learning_framework: {self.learning_framework}")

    def _build_lpep_loss(self, K: int, L: float, mu: float, R: float) -> Callable:
        """Construct deterministic PEP loss (no samples).

        Uses wc_pep_scs_solve for worst-case performance estimation.

        Args:
            K: Number of algorithm iterations.
            L, mu, R: Problem parameters.

        Returns:
            Loss function: stepsizes -> scalar
        """
        pep_data_fn = self.problem_module.get_pep_data_fn(self.alg)

        def lpep_loss(stepsizes):
            """Compute worst-case PEP objective."""
            # Construct PEP constraint matrices
            pep_data = pep_data_fn(stepsizes, mu, L, R, K, self.pep_obj)
            A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]

            # Solve worst-case PEP SDP
            return wc_pep_scs_solve(A_obj, b_obj, A_vals, b_vals, c_vals)

        return jax.jit(lpep_loss)

    def _build_l2o_loss(self, K: int) -> Callable:
        """Construct trajectory-based loss with risk measure.

        Uses problem_module.create_metric_fn() for metric computation at each k,
        supporting multiple loss composition types via cfg.loss_type_composition.

        Args:
            K: Number of algorithm iterations.

        Returns:
            Loss function: (stepsizes, minibatch_idx) -> scalar
        """
        traj_fn = self.problem_module.get_trajectory_fn(self.alg)

        def l2o_loss(stepsizes, minibatch_idx):
            """Compute trajectory-based loss with risk measure."""
            # Extract minibatch
            minibatch = self._get_minibatch(minibatch_idx)

            # Compute loss for each sample in batch
            losses = self._compute_batched_trajectory_losses(
                stepsizes, minibatch, traj_fn, K
            )

            # Apply risk measure
            return self._apply_risk_measure(losses)

        return jax.jit(l2o_loss)

    def _build_ldro_pep_loss(self, K: int, L: float, mu: float, R: float,
                            initial_stepsizes: Stepsizes) -> Callable:
        """Construct DRO SDP loss combining trajectories + PEP constraints.

        Supports two backends:
        - manual_jax: Direct diffcp with JAX autodiff (faster, lower memory)
        - cvxpylayers: CvxpyLayers wrapper (slower, higher memory)

        Args:
            K: Number of algorithm iterations.
            L, mu, R: Problem parameters.
            initial_stepsizes: Initial stepsizes for preconditioner computation.

        Returns:
            Loss function: (stepsizes, minibatch_idx) -> scalar
        """
        # Compute preconditioner from training data (once)
        self._compute_preconditioner(K, initial_stepsizes)

        pep_data_fn = self.problem_module.get_pep_data_fn(self.alg)
        traj_fn = self.problem_module.get_trajectory_fn(self.alg)

        if self.dro_canon_backend == 'manual_jax':
            def ldro_pep_loss(stepsizes, minibatch_idx):
                """LDRO-PEP pipeline using manual JAX canonicalization."""
                # Extract minibatch
                minibatch = self._get_minibatch(minibatch_idx)

                # Compute trajectories for all samples (Gram representations)
                G_batch, F_batch = self._compute_batched_gram_matrices(
                    stepsizes, minibatch, traj_fn, K
                )

                # Compute PEP constraint matrices (depend on stepsizes)
                pep_data = pep_data_fn(stepsizes, mu, L, R, K, self.pep_obj)
                A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]

                # Solve DRO SDP
                return dro_scs_solve(
                    A_obj, b_obj, A_vals, b_vals, c_vals,
                    G_batch, F_batch,
                    self.eps, self.precond_inv,
                    risk_type=self.risk_type,
                    alpha=self.alpha,
                )

            return jax.jit(ldro_pep_loss)

        elif self.dro_canon_backend == 'cvxpylayers':
            raise NotImplementedError(
                "cvxpylayers backend not yet implemented in UnifiedTrainer. "
                "Use dro_canon_backend='manual_jax' instead."
            )
        else:
            raise ValueError(
                f"Unknown dro_canon_backend: {self.dro_canon_backend}. "
                "Must be 'manual_jax' or 'cvxpylayers'."
            )

    def _presample_training_data(self, K: int, K_output_dir: str = None):
        """Pre-sample training set and set up minibatch access.

        Args:
            K: Number of algorithm iterations (for logging).
            K_output_dir: Optional directory to save training data.
        """
        log.info(f'Pre-sampling {self.training_sample_N} training problems...')

        # Validate divisibility
        assert self.training_sample_N % self.N_batch == 0, \
            f"training_sample_N ({self.training_sample_N}) must be divisible by N_batch ({self.N_batch})"

        # Sample training set from problem module
        self.key, sample_key = jax.random.split(self.key)
        problem_data, ground_truth = self.problem_module.sample_training_batch(
            sample_key, self.training_sample_N
        )

        # Merge into single dict for convenience
        self.training_data = {**problem_data, **ground_truth}

        # Compute number of minibatches
        self.n_minibatches = self.training_sample_N // self.N_batch
        log.info(f'Number of minibatches per epoch: {self.n_minibatches}')

        # Optionally save training set to disk
        if K_output_dir is not None:
            self._save_training_data(K_output_dir)

    def _save_training_data(self, K_output_dir: str):
        """Save pre-sampled training data to disk for reproducibility.

        Args:
            K_output_dir: Directory to save training data.
        """
        # Save all training data arrays
        train_data_path = os.path.join(K_output_dir, 'training_set.npz')
        # Convert JAX arrays to numpy for saving
        np_data = {k: np.array(v) for k, v in self.training_data.items()}
        np.savez_compressed(train_data_path, **np_data)
        log.info(f'Saved training set to {train_data_path}')

    def _presample_validation_data(self):
        """Pre-sample validation set for tracking generalization during training.

        The validation set is held fixed throughout training for consistent monitoring.
        Validation loss is always computed using the final iterate metric (k=K),
        regardless of training loss composition.
        """
        log.info(f'Pre-sampling {self.validation_sample_N} validation problems...')

        # Create separate key for validation set using configured seed
        val_key = jax.random.PRNGKey(self.validation_seed)

        # Sample validation set from problem module
        problem_data, ground_truth = self.problem_module.sample_validation_batch(
            val_key, self.validation_sample_N
        )

        # Merge into single dict for convenience
        self.validation_data = {**problem_data, **ground_truth}
        log.info(f'Validation set sampled with seed {self.validation_seed}')

    def _get_minibatch(self, minibatch_idx: int) -> Dict[str, jnp.ndarray]:
        """Extract minibatch using sliding window.

        Args:
            minibatch_idx: Index of minibatch (cycles through n_minibatches).

        Returns:
            Dict of minibatch data with '_batch' suffix stripped.
        """
        # Cycle through minibatches
        minibatch_idx = minibatch_idx % self.n_minibatches
        start = minibatch_idx * self.N_batch
        end = start + self.N_batch

        # Extract minibatch slices, removing '_batch' suffix from keys
        minibatch = {}
        for k, v in self.training_data.items():
            # Strip '_batch' suffix if present
            key_name = k[:-6] if k.endswith('_batch') else k
            minibatch[key_name] = v[start:end]

        return minibatch

    def _compute_preconditioner(self, K: int, initial_stepsizes: Stepsizes):
        """Compute preconditioner from training data (for LDRO-PEP only).

        Uses all training samples with initial stepsizes to compute preconditioning
        factors that improve numerical conditioning of the DRO SDP.

        Args:
            K: Number of algorithm iterations.
            initial_stepsizes: Initial stepsizes to use for trajectory computation.
        """
        log.info(f'Computing preconditioner from {self.training_sample_N} training samples...')

        traj_fn = self.problem_module.get_trajectory_fn(self.alg)

        # Prepare batched and fixed data for vmap
        batched_params = self.problem_module.get_batched_parameters()
        fixed_params = self.problem_module.get_fixed_parameters()

        batched_data = {k: self.training_data.get(k + '_batch', self.training_data.get(k))
                       for k in batched_params}
        fixed_data = {k: self.training_data[k] for k in fixed_params if k in self.training_data}

        # Compute G, F for all training samples
        G_batch, F_batch = self.problem_module.compute_batched_trajectories(
            initial_stepsizes, batched_data, fixed_data, traj_fn, K
        )

        # Compute preconditioner based on sample statistics
        self.precond_inv = compute_preconditioner_from_samples(
            G_batch, F_batch, precond_type=self.precond_type
        )
        log.info(f'Computed preconditioner using type: {self.precond_type}')

    def _compute_batched_gram_matrices(
        self, stepsizes: Stepsizes, minibatch: Dict[str, jnp.ndarray],
        traj_fn: Callable, K: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Gram matrices for a minibatch using vmap.

        Args:
            stepsizes: Algorithm stepsizes.
            minibatch: Dict of minibatch data (without '_batch' suffix).
            traj_fn: Trajectory function from problem_module.
            K: Number of algorithm iterations.

        Returns:
            Tuple of (G_batch, F_batch) Gram representations.
        """
        # Separate batched and fixed parameters
        batched_params = self.problem_module.get_batched_parameters()
        fixed_params = self.problem_module.get_fixed_parameters()

        batched_data = {k: minibatch[k] for k in batched_params if k in minibatch}
        fixed_data = {k: minibatch[k] for k in fixed_params if k in minibatch}

        # Use problem module's compute_batched_trajectories
        return self.problem_module.compute_batched_trajectories(
            stepsizes, batched_data, fixed_data, traj_fn, K
        )

    def _compute_batched_trajectory_losses(
        self, stepsizes: Stepsizes, minibatch: Dict[str, jnp.ndarray],
        traj_fn: Callable, K: int
    ) -> jnp.ndarray:
        """Compute trajectory losses for a minibatch.

        Uses problem_module.create_metric_fn() to compute losses based on
        the loss_type_composition configuration.

        Args:
            stepsizes: Algorithm stepsizes.
            minibatch: Dict of minibatch data.
            traj_fn: Trajectory function.
            K: Number of algorithm iterations.

        Returns:
            Array of losses, shape (N_batch,)
        """
        # Prepare batched and fixed data for vmap
        batched_params = self.problem_module.get_batched_parameters()
        fixed_params = self.problem_module.get_fixed_parameters()

        batched_data = {k: minibatch[k] for k in batched_params if k in minibatch}
        fixed_data = {k: minibatch[k] for k in fixed_params if k in minibatch}

        # Build vmap function for computing losses per sample
        def compute_single_loss(**sample_data):
            """Compute loss for a single problem instance."""
            # Merge fixed data with current sample data
            full_data = {**fixed_data, **sample_data}

            # Compute trajectories (not Gram representation)
            trajectories = traj_fn(stepsizes, **full_data, K_max=K, return_Gram_representation=False)

            # Extract problem data and ground truth for metric computation
            ground_truth_keys = self.problem_module.get_ground_truth_keys()
            problem_data = {k: v for k, v in full_data.items()
                          if k not in ground_truth_keys}
            ground_truth = {k: v for k, v in full_data.items()
                          if k in ground_truth_keys}

            # Create metric function
            metric_fn = self.problem_module.create_metric_fn(
                trajectories, problem_data, ground_truth, self.pep_obj
            )

            # Compute loss based on composition type
            return self._compute_loss_from_metric(metric_fn, K)

        # Vmap over batched parameters
        in_axes_dict = {k: 0 for k in batched_params}
        vmapped_loss = jax.vmap(compute_single_loss, in_axes=in_axes_dict)

        # Call with only batched data (fixed data is captured in closure)
        return vmapped_loss(**batched_data)

    def _compute_loss_from_metric(self, metric_fn: Callable[[int], float], K: int) -> float:
        """Compute loss from metric function using specified composition type.

        Supports multiple formulations via cfg.loss_type_composition:
        - 'final': Only final iterate (original, may cause uniform gradients)
        - 'cumulative': Mean of losses at all iterates
        - 'weighted': Exponentially weighted sum (emphasizes later iterations)
        - 'per_step': Per-step loss improvement
        - 'distance_cumulative': Cumulative distance to optimum

        Args:
            metric_fn: Function computing metric at iteration k (k=0..K).
            K: Number of algorithm iterations.

        Returns:
            Scalar loss value.
        """
        if self.loss_type_composition == 'final':
            # Original: only final iterate
            return metric_fn(K)

        elif self.loss_type_composition == 'cumulative':
            # Mean of losses at all iterates
            all_metrics = jnp.array([metric_fn(k) for k in range(K + 1)])
            return jnp.mean(all_metrics)

        elif self.loss_type_composition == 'weighted':
            # Exponentially weighted sum (emphasizes later iterations)
            decay_rate = self.cfg.get('l2o_decay_rate', 0.9)
            all_metrics = jnp.array([metric_fn(k) for k in range(K + 1)])
            weights = jnp.array([decay_rate ** (K - k) for k in range(K + 1)])
            weights = weights / jnp.sum(weights)  # Normalize
            return jnp.sum(weights * all_metrics)

        elif self.loss_type_composition == 'per_step':
            # Per-step loss improvement (sum of improvements)
            all_metrics = jnp.array([metric_fn(k) for k in range(K + 1)])
            improvements = all_metrics[:-1] - all_metrics[1:]  # Positive = improvement
            return -jnp.sum(improvements)  # Negative to minimize

        elif self.loss_type_composition == 'distance_cumulative':
            # Cumulative distance to optimum (assumes metric is distance)
            all_metrics = jnp.array([metric_fn(k) for k in range(K + 1)])
            return jnp.sum(all_metrics)

        else:
            raise ValueError(f"Unknown loss_type_composition: {self.loss_type_composition}")

    def _apply_risk_measure(self, losses: jnp.ndarray) -> float:
        """Apply risk measure to batch of losses.

        Args:
            losses: Array of losses, shape (N_batch,).

        Returns:
            Scalar risk value (expectation or CVaR).
        """
        if self.risk_type == 'expectation':
            return jnp.mean(losses)
        elif self.risk_type == 'cvar':
            # CVaR: average of worst alpha fraction
            N = losses.shape[0]
            k = max(int(jnp.ceil(self.alpha * N)), 1)
            sorted_losses = jnp.sort(losses)[::-1]  # Descending order
            return jnp.mean(sorted_losses[:k])
        else:
            raise ValueError(f"Unknown risk_type: {self.risk_type}")

    def _build_validation_loss_function(self, K: int) -> Callable:
        """Build validation loss function that computes final iterate metric only.

        CRITICAL: Validation loss is ALWAYS the final iterate metric (k=K),
        regardless of training loss composition. This ensures unbiased
        generalization performance measurement.

        The validation loss:
        1. Computes metric at k=K (final iterate) for each validation sample
        2. Applies the configured risk measure (expectation or CVaR)

        Args:
            K: Number of algorithm iterations.

        Returns:
            Validation loss function: stepsizes -> scalar
        """
        traj_fn = self.problem_module.get_trajectory_fn(self.alg)

        # Prepare batched and fixed parameters for vmap
        batched_params = self.problem_module.get_batched_parameters()
        fixed_params = self.problem_module.get_fixed_parameters()

        batched_data = {k: self.validation_data.get(k + '_batch', self.validation_data.get(k))
                       for k in batched_params}
        fixed_data = {k: self.validation_data[k] for k in fixed_params if k in self.validation_data}

        def val_loss_fn(stepsizes):
            """Compute validation loss on held-out validation set."""

            def compute_single_val_metric(**sample_data):
                """Compute final iterate metric for a single validation sample."""
                # Merge fixed data with current sample data
                full_data = {**fixed_data, **sample_data}

                # Compute trajectories (not Gram representation)
                trajectories = traj_fn(stepsizes, **full_data, K_max=K, return_Gram_representation=False)

                # Extract problem data and ground truth for metric computation
                ground_truth_keys = self.problem_module.get_ground_truth_keys()
                problem_data = {k: v for k, v in full_data.items()
                              if k not in ground_truth_keys}
                ground_truth = {k: v for k, v in full_data.items()
                              if k in ground_truth_keys}

                # Create metric function
                metric_fn = self.problem_module.create_metric_fn(
                    trajectories, problem_data, ground_truth, self.pep_obj
                )

                # ONLY final iterate, always
                return metric_fn(K)

            # Vmap over batched parameters
            in_axes_dict = {k: 0 for k in batched_params}
            vmapped_metric = jax.vmap(compute_single_val_metric, in_axes=in_axes_dict)

            # Compute metrics for all validation samples
            val_metrics = vmapped_metric(**batched_data)

            # Apply risk measure (same as training)
            return self._apply_risk_measure(val_metrics)

        return jax.jit(val_loss_fn)

    def _initialize_optimizer(self, stepsizes: Stepsizes):
        """Set up optimizer based on cfg.optimizer_type.

        Supports:
        - vanilla_sgd: Manual SGD with projection
        - adamw: AdamWMin from learning.adam_optimizers
        - sgd_wd: SGD with weight decay

        Args:
            stepsizes: Initial stepsizes for optimizer state initialization.
        """
        update_mask = self._get_update_mask(stepsizes)

        if self.optimizer_type == 'adamw':
            self.optimizer = AdamWMin(
                x_params=[jnp.array(s) for s in stepsizes],
                lr=self.eta_t,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.weight_decay,
                update_mask=update_mask,
            )
            log.info(f'Initialized AdamWMin optimizer with lr={self.eta_t}, wd={self.weight_decay}')
        elif self.optimizer_type in ['vanilla_sgd', 'sgd_wd']:
            self.optimizer = None  # Manual update in _optimizer_step
            log.info(f'Using {self.optimizer_type} with lr={self.eta_t}')
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

    def _get_update_mask(self, stepsizes: Stepsizes) -> List[bool] | None:
        """Determine which parameters to update (for learn_beta=False).

        Args:
            stepsizes: Stepsizes tuple to determine structure.

        Returns:
            List of booleans indicating which parameters to update, or None to update all.
        """
        has_beta = len(stepsizes) > 1
        if has_beta and not self.learn_beta:
            # Update t, keep beta fixed
            log.info('learn_beta=False: beta will NOT be updated during optimization')
            return [True, False]
        return None

    def _run_training_loop(
        self, loss_fn: Callable, val_loss_fn: Callable, stepsizes: Stepsizes, K: int, csv_path: str
    ) -> TrainingResult:
        """Core unified training loop.

        LPEP uses deterministic GD (no minibatch sampling).
        L2O/LDRO-PEP use stochastic SGD with minibatch sampling.

        Args:
            loss_fn: JIT-compiled training loss function.
            val_loss_fn: JIT-compiled validation loss function.
            stepsizes: Initial stepsizes.
            K: Number of algorithm iterations.
            csv_path: Path to save progress CSV.

        Returns:
            TrainingResult with final stepsizes, history, losses, val_losses, and times.
        """
        # Track history
        all_stepsizes_vals = [stepsizes]
        all_losses = []
        all_val_losses = []
        all_times = []

        # Determine update mask for manual optimizers
        update_mask = self._get_update_mask(stepsizes)

        # Create value_and_grad function
        value_and_grad_fn = jax.value_and_grad(loss_fn)

        # Training iterations
        n_iters = self.sgd_iters
        for iter_num in range(n_iters):
            # Log progress
            self._log_iteration(iter_num, stepsizes, K)

            # Compute loss and gradients
            iter_start_time = time.perf_counter()

            if self.learning_framework == 'lpep':
                # LPEP: deterministic, no minibatch
                loss, grads = value_and_grad_fn(stepsizes)
            else:
                # L2O/LDRO-PEP: stochastic, with minibatch
                loss, grads = value_and_grad_fn(stepsizes, iter_num)

            iter_time = time.perf_counter() - iter_start_time

            log.info(f'  loss: {float(loss):.6f}, iter_time: {iter_time:.3f}s')

            # Store loss and timing
            all_losses.append(float(loss))
            all_times.append(iter_time)

            # Optimizer step
            stepsizes = self._optimizer_step(stepsizes, grads, update_mask)

            # Compute validation loss with updated stepsizes
            val_loss = float(val_loss_fn(stepsizes))
            all_val_losses.append(val_loss)
            log.info(f'  val_loss: {val_loss:.6f}')

            # Store updated stepsizes
            all_stepsizes_vals.append(stepsizes)

            # Save checkpoint
            self._save_checkpoint(all_stepsizes_vals, K, all_losses, all_val_losses, all_times, csv_path)

        # Return result
        return TrainingResult(
            stepsizes=stepsizes,
            stepsizes_history=all_stepsizes_vals,
            losses=all_losses,
            val_losses=all_val_losses,
            times=all_times,
        )

    def _log_iteration(self, iter_num: int, stepsizes: Stepsizes, K: int):
        """Log current iteration progress.

        Args:
            iter_num: Current iteration number.
            stepsizes: Current stepsizes.
            K: Number of algorithm iterations.
        """
        t = stepsizes[0]
        is_vector_t = jnp.ndim(t) > 0
        has_beta = len(stepsizes) > 1

        t_log = f'{t:.5f}' if not is_vector_t else '[' + ', '.join(f'{x:.5f}' for x in t.tolist()) + ']'

        if has_beta:
            beta = stepsizes[1]
            beta_log = '[' + ', '.join(f'{x:.5f}' for x in beta.tolist()) + ']'
            log.info(f'K={K}, iter={iter_num}, t={t_log}, beta={beta_log}')
        else:
            log.info(f'K={K}, iter={iter_num}, t={t_log}')

    def _optimizer_step(
        self, stepsizes: Stepsizes, grads: Stepsizes, update_mask: List[bool] | None
    ) -> Stepsizes:
        """Execute one optimizer step with projection.

        Args:
            stepsizes: Current stepsizes.
            grads: Gradients w.r.t. stepsizes.
            update_mask: Optional mask for selective parameter updates.

        Returns:
            Updated stepsizes (projected to nonnegative).
        """
        if self.optimizer_type == 'vanilla_sgd':
            # Manual SGD with projection
            if update_mask is None:
                stepsizes = tuple(
                    jnp.maximum(s - self.eta_t * ds, 1e-6)
                    for s, ds in zip(stepsizes, grads)
                )
            else:
                stepsizes = tuple(
                    jnp.maximum(s - self.eta_t * ds, 1e-6) if should_update else s
                    for s, ds, should_update in zip(stepsizes, grads, update_mask)
                )

        elif self.optimizer_type == 'adamw':
            # AdamWMin optimizer
            x_params = [jnp.array(s) for s in stepsizes]
            grads_x = list(grads)
            x_new = self.optimizer.step(
                x_params=x_params,
                grads_x=grads_x,
                proj_x_fn=self._project_stepsizes,
            )
            stepsizes = tuple(x_new)

        elif self.optimizer_type == 'sgd_wd':
            # SGD with weight decay
            if update_mask is None:
                stepsizes = tuple(
                    jnp.maximum(s - self.eta_t * (ds + self.weight_decay * s), 1e-6)
                    for s, ds in zip(stepsizes, grads)
                )
            else:
                stepsizes = tuple(
                    jnp.maximum(s - self.eta_t * (ds + self.weight_decay * s), 1e-6) if should_update else s
                    for s, ds, should_update in zip(stepsizes, grads, update_mask)
                )

        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

        return stepsizes

    def _project_stepsizes(self, stepsizes: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """Project stepsizes to be nonnegative.

        Args:
            stepsizes: List of stepsize arrays.

        Returns:
            Projected stepsizes (list of arrays).
        """
        return [jnp.maximum(s, 1e-6) for s in stepsizes]

    def _save_checkpoint(
        self, stepsizes_history: List[Stepsizes], K_max: int,
        losses: List[float], val_losses: List[float], times: List[float], csv_path: str
    ):
        """Save progress to CSV via problem_module.build_stepsizes_dataframe().

        Args:
            stepsizes_history: Full history of stepsizes including initialization.
            K_max: Number of algorithm iterations.
            losses: List of training loss values.
            val_losses: List of validation loss values.
            times: List of iteration times in seconds.
            csv_path: Path to save CSV.
        """
        df = self.problem_module.build_stepsizes_dataframe(
            stepsizes_history=stepsizes_history,
            K_max=K_max,
            alg=self.alg,
            training_losses=losses,
            validation_losses=val_losses,
            times=times,
        )
        df.to_csv(csv_path, index=False)
