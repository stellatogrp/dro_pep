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
import optax
import os
import time
from typing import Callable, Dict, Tuple, List, Any
from omegaconf import DictConfig
import numpy as np

from learning.training_result import TrainingResult, Stepsizes
from learning.problem_module import ProblemModule
from learning.jax_scs_layer import (
    wc_pep_scs_solve,
    dro_scs_solve,
    compute_preconditioner_from_samples,
    compute_C_d_matrices,
    dro_expectation_setup_static,
    dro_expectation_canon_to_bcoo,
    scs_solve_wrapper_sparse,
)

log = logging.getLogger(__name__)


class UnifiedTrainer:
    """Unified training orchestrator for all three learning pipelines.

    Responsibilities:
    - Training loop management (minibatch iteration, checkpointing)
    - Loss function construction based on learning_framework
    - Optimizer management (optax-based: vanilla_sgd, sgd_wd, adamw)
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
        self.training_loss_type_composition = cfg.get('training_loss_type_composition', 'final')
        self.validation_loss_type_composition = cfg.get('validation_loss_type_composition', 'final')
        self.decay_rate = cfg.get('decay_rate', 0.9)

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
        # Max L2 norm across the full gradient tuple before the optimizer step.
        # Norms above this are scaled to exactly this value (direction preserved).
        self.grad_clip_norm = cfg.get('grad_clip_norm', 1.0)

        # State
        self.optimizer = None
        self.opt_state = None
        self.scheduler = None
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
        sqrt_stepsizes = tuple(jnp.sqrt(jnp.asarray(s)) for s in stepsizes)

        # Step 3+4: Pre-sample training + validation data (idempotent).
        # Data is K-independent; callers that hoist prepare_data() above the
        # K-loop pay this cost once per experiment, not once per K.
        self.prepare_data(save_dir=K_output_dir)

        # Step 5: Build loss function (probe with raw stepsizes for static PSD dims)
        loss_fn = self._build_loss_function(K, L, mu, R, stepsizes)

        # Step 6: Build validation loss function
        val_loss_fn = self._build_validation_loss_function(K)

        # Step 7: Initialize optimizer on sqrt-reparameterized params
        self._initialize_optimizer(sqrt_stepsizes)

        # Step 8: Run training loop
        result = self._run_training_loop(loss_fn, val_loss_fn, sqrt_stepsizes, K, csv_path)

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
            return self._build_lpep_loss(K, L, mu, R, initial_stepsizes)
        elif self.learning_framework == 'l2o':
            return self._build_l2o_loss(K)
        elif self.learning_framework == 'ldro-pep':
            return self._build_ldro_pep_loss(K, L, mu, R, initial_stepsizes)
        else:
            raise ValueError(f"Unknown learning_framework: {self.learning_framework}")

    def _build_lpep_loss(self, K: int, L: float, mu: float, R: float,
                         initial_stepsizes: Stepsizes) -> Callable:
        """Construct deterministic PEP loss (no samples).

        Uses wc_pep_scs_solve for worst-case performance estimation.

        For PEP constructions that produce PSD blocks (CP / PDLP), those
        blocks are forwarded so the SDP is structurally complete. PSD
        dimensions are precomputed once outside jit (same static-closure
        pattern used in LDRO-PEP) so `compute_C_d_matrices` sees concrete
        Python ints under nested jit.

        Args:
            K: Number of algorithm iterations.
            L, mu, R: Problem parameters.
            initial_stepsizes: Used to probe pep_data_fn once at build time
                to capture the static PSD block dimensions.

        Returns:
            Loss function: stepsizes -> scalar
        """
        pep_data_fn = self.problem_module.get_pep_data_fn(self.alg)

        _init_pep_data = pep_data_fn(
            initial_stepsizes, mu, L, R, K, self.pep_obj,
            composition_type=self.training_loss_type_composition,
            decay_rate=self.decay_rate,
        )
        psd_mat_dims_static = tuple(int(s) for s in _init_pep_data[8])

        def lpep_loss(sqrt_stepsizes):
            """Compute worst-case PEP objective."""
            stepsizes = tuple(s ** 2 for s in sqrt_stepsizes)
            pep_data = pep_data_fn(
                stepsizes, mu, L, R, K, self.pep_obj,
                composition_type=self.training_loss_type_composition,
                decay_rate=self.decay_rate,
            )
            (A_obj, b_obj, A_vals, b_vals, c_vals,
             PSD_A_vals, PSD_b_vals, PSD_c_vals, _) = pep_data

            return wc_pep_scs_solve(
                A_obj, b_obj, A_vals, b_vals, c_vals,
                PSD_A_vals=PSD_A_vals,
                PSD_b_vals=PSD_b_vals,
                PSD_c_vals=PSD_c_vals,
                PSD_mat_dims=psd_mat_dims_static,
            )

        return jax.jit(lpep_loss)

    def _build_l2o_loss(self, K: int) -> Callable:
        """Construct trajectory-based loss with risk measure.

        Uses problem_module.create_metric_fn() for metric computation at each k,
        supporting multiple loss composition types via cfg.loss_type_composition.

        Args:
            K: Number of algorithm iterations.

        Returns:
            Loss function: (stepsizes, minibatch) -> scalar
            Note: minibatch is a dict of data arrays, NOT an index.
        """
        traj_fn = self.problem_module.get_trajectory_fn(self.alg)

        def l2o_loss(sqrt_stepsizes, minibatch):
            """Compute trajectory-based loss with risk measure."""
            stepsizes = tuple(s ** 2 for s in sqrt_stepsizes)
            # minibatch is already extracted outside JIT boundary

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

        The preconditioner is computed per-step from minibatch data, enabling
        gradient flow through the stepsize parameters.

        Supports two backends:
        - manual_jax: Direct diffcp with JAX autodiff (faster, lower memory)
        - cvxpylayers: CvxpyLayers wrapper (slower, higher memory)

        Args:
            K: Number of algorithm iterations.
            L, mu, R: Problem parameters.
            initial_stepsizes: Initial stepsizes (unused, kept for interface compatibility).

        Returns:
            Loss function: (stepsizes, minibatch) -> scalar
            Note: minibatch is a dict of data arrays, NOT an index.
        """
        pep_data_fn = self.problem_module.get_pep_data_fn(self.alg)
        traj_fn = self.problem_module.get_trajectory_fn(self.alg)

        # Closure over precond_type for use inside JIT-compiled function
        precond_type = self.precond_type

        # Pre-compute the static PSD-block dimensions ONCE outside JIT.
        # Under a nested-jit call, Python ints returned from the inner jit
        # (`PSD_shapes`) get traced into jax int arrays, which breaks
        # `compute_C_d_matrices` (needs `int(dim)`). By capturing the static
        # dims at build time and closing over them as a Python tuple, we keep
        # them concrete. CP returns a non-empty PSD_shapes list whose values
        # depend only on K (static), so this is stable across SGD iterations.
        _init_pep_data = pep_data_fn(
            initial_stepsizes, mu, L, R, K, self.pep_obj,
            composition_type=self.training_loss_type_composition,
            decay_rate=self.decay_rate,
        )
        psd_mat_dims_static = tuple(int(s) for s in _init_pep_data[8])

        if self.dro_canon_backend == 'manual_jax':
            # === Hoist plumbing ===
            # The DRO SDP solve uses jax.pure_callback (Clarabel via diffcp),
            # which JAX marks as cacheable=False. Wrapping the entire loss in
            # @jax.jit therefore *embeds* the callback in the cache key and
            # forces a full recompile every process (~50 s on PDLP). To make
            # the bulk of the loss cacheable, we split:
            #
            #   (a) build_inputs:  trajectory + preconditioner + canon + BCOO
            #                       sparsify -> A_data, A_indices, b, c
            #                       This is JIT-compiled and contains NO callback;
            #                       cache hits across processes.
            #   (b) sdp_solve:     scs_solve_wrapper_sparse(...) -> obj_val.
            #                       Custom_vjp around the pure_callback. Runs
            #                       outside any compile artifact.
            #   (c) ldro_pep_loss: thin Python orchestrator combining (a)+(b).
            #                       NOT jit-wrapped, but the work it dispatches
            #                       is dominated by the cached jit in (a) plus
            #                       the un-cached but fast SDP solve.
            #
            # Static SCS dimensions (cone shapes, A_shape, nse upper bound) are
            # derivable from the probe pep_data and the static minibatch size
            # — compute them once here and close over them.
            (_A_obj0, _b_obj0, _A_vals0, _b_vals0, _c_vals0,
             _PSD_A_vals0, _PSD_b_vals0, _PSD_c_vals0, _) = _init_pep_data

            _N = int(self.N_batch)
            _M = int(_A_vals0.shape[0])
            _V = int(_b_obj0.shape[0])
            _S_mat = int(_A_obj0.shape[0])

            if _PSD_A_vals0 is not None and len(_PSD_A_vals0) > 0:
                _, _, _h_vec_dims_static, _ = compute_C_d_matrices(
                    _PSD_A_vals0, _PSD_b_vals0, psd_mat_dims_static
                )
                _h_vec_dims_static = list(_h_vec_dims_static)
            else:
                _h_vec_dims_static = []

            _static_data, _A_shape, _nse_upper = dro_expectation_setup_static(
                _N, _M, _V, _S_mat, psd_mat_dims_static, _h_vec_dims_static,
            )

            log.info(
                f"[ldro-pep hoist] static: N={_N} M={_M} V={_V} S_mat={_S_mat} "
                f"A_shape={_A_shape} nse_upper={_nse_upper}"
            )

            eps_local = self.eps  # bind for closure
            pep_obj_local = self.pep_obj
            train_comp = self.training_loss_type_composition
            decay = self.decay_rate

            @jax.jit
            def _build_inputs(sqrt_stepsizes, minibatch):
                stepsizes = tuple(s ** 2 for s in sqrt_stepsizes)
                G_batch, F_batch = self._compute_batched_gram_matrices(
                    stepsizes, minibatch, traj_fn, K
                )
                precond_inv = compute_preconditioner_from_samples(
                    G_batch, F_batch, precond_type
                )
                pep_data = pep_data_fn(
                    stepsizes, mu, L, R, K, pep_obj_local,
                    composition_type=train_comp, decay_rate=decay,
                )
                (A_obj, b_obj, A_vals, b_vals, c_vals,
                 PSD_A_vals, PSD_b_vals, PSD_c_vals, _) = pep_data
                return dro_expectation_canon_to_bcoo(
                    A_obj, b_obj, A_vals, b_vals, c_vals,
                    G_batch, F_batch,
                    eps_local, precond_inv,
                    PSD_A_vals, PSD_b_vals,
                    psd_mat_dims_static, _nse_upper,
                )

            def ldro_pep_loss(sqrt_stepsizes, minibatch):
                A_data, A_indices, b, c = _build_inputs(sqrt_stepsizes, minibatch)
                return scs_solve_wrapper_sparse(
                    _static_data, A_data, A_indices, _A_shape, b, c,
                )

            return ldro_pep_loss

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

    def prepare_data(self, save_dir: str = None) -> None:
        """One-time pre-sampling of training + validation data. Idempotent.

        Training data and validation data are K-independent, so this method
        should be called ONCE per experiment — hoisted above the K-loop by
        the runners. `train(K)` also calls this as a safety net, but on
        subsequent calls both branches become no-ops because self.training_data
        and self.validation_data are already populated.

        Args:
            save_dir: Optional directory to save training data to
                (as `training_set.npz`). Pass the top-level `output_dir`.
        """
        if self.learning_framework in ['l2o', 'ldro-pep'] and self.training_data is None:
            self._presample_training_data(save_dir=save_dir)
        if self.validation_data is None:
            self._presample_validation_data()

    def _presample_training_data(self, save_dir: str = None):
        """Pre-sample training set and set up minibatch access."""
        log.info(f'Pre-sampling {self.training_sample_N} training problems...')

        assert self.training_sample_N % self.N_batch == 0, \
            f"training_sample_N ({self.training_sample_N}) must be divisible by N_batch ({self.N_batch})"

        self.key, sample_key = jax.random.split(self.key)
        problem_data, ground_truth = self.problem_module.sample_training_batch(
            sample_key, self.training_sample_N
        )
        self.training_data = {**problem_data, **ground_truth}

        self.n_minibatches = self.training_sample_N // self.N_batch
        log.info(f'Number of minibatches per epoch: {self.n_minibatches}')

        if save_dir is not None:
            self._save_training_data(save_dir)

    def _save_training_data(self, save_dir: str):
        """Save pre-sampled training data to disk for reproducibility."""
        train_data_path = os.path.join(save_dir, 'training_set.npz')
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

        # Fix the keys order for vmap (must use positional args, not **kwargs)
        batched_keys = tuple(batched_params)
        in_axes = tuple(0 for _ in batched_keys)

        # Build vmap function for computing losses per sample
        def compute_single_loss(*args):
            """Compute loss for a single problem instance."""
            # Reconstruct sample_data dict from positional args
            sample_data = dict(zip(batched_keys, args))

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

        # Vmap over batched parameters using positional args
        vmapped_loss = jax.vmap(compute_single_loss, in_axes=in_axes)

        # Call with batched data in correct order
        return vmapped_loss(*[batched_data[k] for k in batched_keys])

    def _compute_loss_from_metric(
        self, metric_fn: Callable[[int], float], K: int, loss_type: str = None
    ) -> float:
        """Compute loss from metric function using specified composition type.

        Supports multiple formulations:
        - 'final': Only final iterate (original, may cause uniform gradients)
        - 'cumulative': Mean of losses at all iterates
        - 'weighted': Exponentially weighted sum (emphasizes later iterations)
        - 'per_step': Per-step loss improvement
        - 'distance_cumulative': Cumulative distance to optimum

        Args:
            metric_fn: Function computing metric at iteration k (k=0..K).
            K: Number of algorithm iterations.
            loss_type: Loss composition type. Defaults to training_loss_type_composition.

        Returns:
            Scalar loss value.
        """
        if loss_type is None:
            loss_type = self.training_loss_type_composition

        if loss_type == 'final':
            # Original: only final iterate
            return metric_fn(K)

        elif loss_type == 'cumulative':
            # Mean of losses at all iterates
            all_metrics = jnp.array([metric_fn(k) for k in range(K + 1)])
            return jnp.mean(all_metrics)

        elif loss_type == 'weighted':
            # Exponentially weighted sum (emphasizes later iterations)
            decay_rate = self.cfg.get('l2o_decay_rate', 0.9)
            all_metrics = jnp.array([metric_fn(k) for k in range(K + 1)])
            weights = jnp.array([decay_rate ** (K - k) for k in range(K + 1)])
            weights = weights / jnp.sum(weights)  # Normalize
            return jnp.sum(weights * all_metrics)

        elif loss_type == 'per_step':
            # Per-step loss improvement (sum of improvements)
            all_metrics = jnp.array([metric_fn(k) for k in range(K + 1)])
            improvements = all_metrics[:-1] - all_metrics[1:]  # Positive = improvement
            return -jnp.sum(improvements)  # Negative to minimize

        elif loss_type == 'distance_cumulative':
            # Cumulative distance to optimum (assumes metric is distance)
            all_metrics = jnp.array([metric_fn(k) for k in range(K + 1)])
            return jnp.sum(all_metrics)

        else:
            raise ValueError(f"Unknown loss_type_composition: {loss_type}")

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
            # Use numpy (not jax) for static computation at trace time
            N = losses.shape[0]
            k = max(int(np.ceil(self.alpha * N)), 1)
            sorted_losses = jnp.sort(losses)[::-1]  # Descending order
            return jnp.mean(sorted_losses[:k])
        else:
            raise ValueError(f"Unknown risk_type: {self.risk_type}")

    def _build_validation_loss_function(self, K: int) -> Callable:
        """Build validation loss function using configured loss composition.

        The validation loss:
        1. Computes metric using validation_loss_type_composition for each sample
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

        # Fix the keys order for vmap (must use positional args, not **kwargs)
        batched_keys = tuple(batched_params)
        in_axes = tuple(0 for _ in batched_keys)

        # Capture validation loss type for use in closure
        val_loss_type = self.validation_loss_type_composition

        def val_loss_fn(sqrt_stepsizes):
            """Compute validation loss on held-out validation set."""
            stepsizes = tuple(s ** 2 for s in sqrt_stepsizes)

            def compute_single_val_metric(*args):
                """Compute metric for a single validation sample."""
                # Reconstruct sample_data dict from positional args
                sample_data = dict(zip(batched_keys, args))

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

                # Use validation loss type composition
                return self._compute_loss_from_metric(metric_fn, K, loss_type=val_loss_type)

            # Vmap over batched parameters using positional args
            vmapped_metric = jax.vmap(compute_single_val_metric, in_axes=in_axes)

            # Compute metrics for all validation samples
            val_metrics = vmapped_metric(*[batched_data[k] for k in batched_keys])

            # Apply risk measure (same as training)
            return self._apply_risk_measure(val_metrics)

        return jax.jit(val_loss_fn)

    def _initialize_optimizer(self, stepsizes: Stepsizes):
        """Set up an optax optimizer with warmup-cosine LR schedule and global-norm clipping.

        Builds: optax.chain(clip_by_global_norm, <optimizer>(scheduler)).
        The scheduler linearly warms up from LR_INIT to self.eta_t over the
        first WARMUP_FRAC of training, then cosine-decays to LR_END by the
        final iteration.

        Args:
            stepsizes: Initial stepsizes (sqrt-reparameterized) for optimizer state init.
        """
        LR_INIT, LR_END, WARMUP_FRAC = 1e-6, 1e-6, 0.1
        warmup_steps = int(WARMUP_FRAC * self.sgd_iters)

        self.scheduler = optax.warmup_cosine_decay_schedule(
            init_value=LR_INIT,
            peak_value=self.eta_t,
            warmup_steps=warmup_steps,
            decay_steps=self.sgd_iters,
            end_value=LR_END,
        )
        clip = optax.clip_by_global_norm(self.grad_clip_norm)

        if self.optimizer_type == 'vanilla_sgd':
            opt = optax.sgd(learning_rate=self.scheduler)
        elif self.optimizer_type == 'sgd_wd':
            opt = optax.chain(
                optax.add_decayed_weights(self.weight_decay),
                optax.sgd(learning_rate=self.scheduler),
            )
        elif self.optimizer_type == 'adamw':
            opt = optax.adamw(
                learning_rate=self.scheduler,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

        self.optimizer = optax.chain(clip, opt)
        self.opt_state = self.optimizer.init(stepsizes)
        log.info(
            f'Initialized {self.optimizer_type} with warmup_cosine schedule '
            f'(peak={self.eta_t}, warmup={warmup_steps}/{self.sgd_iters}, '
            f'wd={self.weight_decay}, clip={self.grad_clip_norm})'
        )

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
        self, loss_fn: Callable, val_loss_fn: Callable, sqrt_stepsizes: Stepsizes, K: int, csv_path: str
    ) -> TrainingResult:
        """Core unified training loop.

        The optimizer holds sqrt_stepsizes; the loss functions square them
        before passing to the algorithm. History, checkpoints, and the
        returned TrainingResult contain the actual (squared) stepsizes so
        external consumers see unchanged semantics.

        LPEP uses deterministic GD (no minibatch sampling).
        L2O/LDRO-PEP use stochastic SGD with minibatch sampling.

        Args:
            loss_fn: JIT-compiled training loss function (expects sqrt params).
            val_loss_fn: JIT-compiled validation loss function (expects sqrt params).
            sqrt_stepsizes: Initial sqrt-reparameterized params.
            K: Number of algorithm iterations.
            csv_path: Path to save progress CSV.

        Returns:
            TrainingResult with final actual stepsizes, history, losses, val_losses, and times.
        """
        def to_actual(sqrt_s):
            return tuple(s ** 2 for s in sqrt_s)

        # Track history (in actual stepsize form for external consumers)
        all_stepsizes_vals = [to_actual(sqrt_stepsizes)]

        # Compute initial losses for the starting stepsizes (before any updates)
        log.info("Computing initial losses for starting stepsizes...")
        initial_start_time = time.perf_counter()

        if self.learning_framework == 'lpep':
            initial_loss_arr = loss_fn(sqrt_stepsizes)
        else:
            # Use first minibatch for initial loss computation
            initial_minibatch = self._get_minibatch(0)
            initial_loss_arr = loss_fn(sqrt_stepsizes, initial_minibatch)

        initial_val_loss_arr = val_loss_fn(sqrt_stepsizes)
        jax.block_until_ready((initial_loss_arr, initial_val_loss_arr))
        initial_time = time.perf_counter() - initial_start_time
        initial_loss = float(initial_loss_arr)
        initial_val_loss = float(initial_val_loss_arr)

        log.info(f'  initial_loss: {initial_loss:.6f}, initial_val_loss: {initial_val_loss:.6f}')

        all_losses = [initial_loss]
        all_val_losses = [initial_val_loss]
        all_times = [initial_time]
        # No grad is computed for the initial-loss probe; pad so this list
        # aligns with stepsizes_history / losses / times. Stores the raw
        # (pre-clip) norm so the CSV records the diagnostic signal.
        all_raw_grad_norms = [float('nan')]
        # Schedule LR pads with NaN for the initial-loss probe (no step taken).
        all_lrs = [float('nan')]

        # Determine update mask for manual optimizers
        update_mask = self._get_update_mask(sqrt_stepsizes)

        # Create value_and_grad function (gradients are w.r.t. sqrt_stepsizes)
        value_and_grad_fn = jax.value_and_grad(loss_fn)

        # Training iterations
        n_iters = self.sgd_iters
        for iter_num in range(n_iters):
            # Shuffle training data at the start of each epoch (stochastic frameworks only).
            # Without this, _get_minibatch cycles through the same fixed slices every epoch.
            if self.learning_framework != 'lpep' and iter_num % self.n_minibatches == 0:
                log.info(f'Epoch {iter_num // self.n_minibatches}: shuffling training data')
                self.key, subkey = jax.random.split(self.key)
                perm = jax.random.permutation(subkey, self.training_sample_N)
                self.training_data = {k: v[perm] for k, v in self.training_data.items()}

            # Log progress (shows actual stepsize = sqrt_stepsize ** 2)
            self._log_iteration(iter_num, sqrt_stepsizes, K)

            # The scheduler is a pure function of step count; the optax chain's
            # internal counter starts at 0 and is incremented inside .update(),
            # so scheduler(iter_num) is the LR actually applied this step.
            current_lr = float(self.scheduler(iter_num))

            # Compute loss and gradients
            iter_start_time = time.perf_counter()

            if self.learning_framework == 'lpep':
                # LPEP: deterministic, no minibatch
                loss, grads = value_and_grad_fn(sqrt_stepsizes)
            else:
                # L2O/LDRO-PEP: stochastic, with minibatch
                # Extract minibatch OUTSIDE JIT boundary to avoid traced indexing
                minibatch = self._get_minibatch(iter_num)
                loss, grads = value_and_grad_fn(sqrt_stepsizes, minibatch)

            jax.block_until_ready((loss, grads))
            iter_time = time.perf_counter() - iter_start_time

            # Raw (pre-clip) gradient norm — single global L2 across the whole
            # tuple, matching the per-element-squared-and-summed convention.
            # Clipping is now handled inside the optax chain via
            # clip_by_global_norm, so this remains a pure diagnostic.
            raw_grad_norm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads)))

            log.info(f'  loss: {float(loss):.6f}, raw_grad_norm: {raw_grad_norm:.6f}, '
                     f'lr: {current_lr:.6e}, iter_time: {iter_time:.3f}s')

            # Store loss, timing, raw grad norm, LR (w.r.t. sqrt-reparameterized params)
            all_losses.append(float(loss))
            all_times.append(iter_time)
            all_raw_grad_norms.append(raw_grad_norm)
            all_lrs.append(current_lr)

            # Optimizer step (optax chain applies clip + scheduled LR internally)
            sqrt_stepsizes = self._optimizer_step(sqrt_stepsizes, grads, update_mask)

            # Compute validation loss with updated stepsizes
            val_loss = float(val_loss_fn(sqrt_stepsizes))
            all_val_losses.append(val_loss)
            log.info(f'  val_loss: {val_loss:.6f}')

            # Store updated stepsizes (actual form)
            all_stepsizes_vals.append(to_actual(sqrt_stepsizes))

            # Save checkpoint
            self._save_checkpoint(all_stepsizes_vals, K, all_losses, all_val_losses, all_times, all_raw_grad_norms, all_lrs, csv_path)

        # Return result (actual stepsize form)
        return TrainingResult(
            stepsizes=to_actual(sqrt_stepsizes),
            stepsizes_history=all_stepsizes_vals,
            losses=all_losses,
            val_losses=all_val_losses,
            times=all_times,
        )

    def _log_iteration(self, iter_num: int, sqrt_stepsizes: Stepsizes, K: int):
        """Log current iteration progress.

        Logs the algorithmic stepsize (sqrt_stepsize ** 2), not the raw param.

        Stepsize tuple length determines the algorithm convention:
            1 -> (t,)              (e.g. ISTA)
            2 -> (t, beta)         (e.g. GD/FGM with momentum, FISTA)
            3 -> (tau, sigma, theta)  (Chambolle-Pock / PDHG)

        Args:
            iter_num: Current iteration number.
            sqrt_stepsizes: Current sqrt-reparameterized params.
            K: Number of algorithm iterations.
        """
        actual_stepsizes = tuple(s ** 2 for s in sqrt_stepsizes)

        def _fmt(s: jnp.ndarray) -> str:
            if jnp.ndim(s) > 0:
                return '[' + ', '.join(f'{x:.5f}' for x in s.tolist()) + ']'
            return f'{float(s):.5f}'

        n = len(actual_stepsizes)
        if n == 3:
            tau, sigma, theta = actual_stepsizes
            log.info(
                f'K={K}, iter={iter_num}, '
                f'tau={_fmt(tau)}, sigma={_fmt(sigma)}, theta={_fmt(theta)}'
            )
        elif n == 2:
            t, beta = actual_stepsizes
            log.info(f'K={K}, iter={iter_num}, t={_fmt(t)}, beta={_fmt(beta)}')
        else:
            log.info(f'K={K}, iter={iter_num}, t={_fmt(actual_stepsizes[0])}')

    def _optimizer_step(
        self, sqrt_stepsizes: Stepsizes, grads: Stepsizes, update_mask: List[bool] | None
    ) -> Stepsizes:
        """Execute one optax step on sqrt-reparameterized params.

        Selective masking is implemented by zeroing the gradients of frozen
        entries before passing them through the optax chain. No projection is
        needed: the loss functions square sqrt_stepsizes before use, so
        nonnegativity of the algorithmic stepsize is guaranteed regardless of
        the sign of sqrt_stepsize.

        Args:
            sqrt_stepsizes: Current sqrt-reparameterized params.
            grads: Gradients w.r.t. sqrt_stepsizes.
            update_mask: Optional mask for selective parameter updates.

        Returns:
            Updated sqrt_stepsizes.
        """
        if update_mask is not None:
            grads = tuple(
                g if should_update else jnp.zeros_like(g)
                for g, should_update in zip(grads, update_mask)
            )

        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, sqrt_stepsizes,
        )
        return tuple(optax.apply_updates(sqrt_stepsizes, updates))

    def _save_checkpoint(
        self, stepsizes_history: List[Stepsizes], K_max: int,
        losses: List[float], val_losses: List[float], times: List[float],
        raw_grad_norms: List[float], lrs: List[float], csv_path: str
    ):
        """Save progress to CSV via problem_module.build_stepsizes_dataframe().

        Args:
            stepsizes_history: Full history of stepsizes including initialization.
            K_max: Number of algorithm iterations.
            losses: List of training loss values.
            val_losses: List of validation loss values.
            times: List of iteration times in seconds.
            raw_grad_norms: List of pre-clip gradient norms w.r.t.
                sqrt-reparameterized params.
            lrs: List of scheduled learning rates per iteration.
            csv_path: Path to save CSV.
        """
        df = self.problem_module.build_stepsizes_dataframe(
            stepsizes_history=stepsizes_history,
            K_max=K_max,
            alg=self.alg,
            training_losses=losses,
            validation_losses=val_losses,
            times=times,
            raw_grad_norms=raw_grad_norms,
            lrs=lrs,
        )
        df.to_csv(csv_path, index=False)
