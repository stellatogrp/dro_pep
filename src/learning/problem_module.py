"""Abstract base class for problem-specific modules in the unified learning framework.

This module defines the ProblemModule ABC that encapsulates all problem-specific
mathematics and data handling. Concrete implementations exist for Quad, Lasso,
LogReg, and PDLP problem types.

The key design insight is the batched vs fixed parameters interface, which guides
how LearningFramework creates batched loss functions. Each problem module declares:
- `get_batched_parameters()`: Names of parameters that vary across the batch (vmapped)
- `get_fixed_parameters()`: Names of parameters that are fixed across the batch

This is more flexible than rigid strategy enums because:
1. New problem types can define arbitrary parameter structures
2. The learning framework can dynamically construct vmap calls based on these tuples
3. It's self-documenting - you can see exactly what varies vs what's fixed

| Problem | Batched Parameters              | Fixed Parameters |
|---------|---------------------------------|------------------|
| Quad    | ('Q', 'z0', 'zs', 'fs')         | ()               |
| Lasso   | ('b', 'x_opt', 'f_opt')         | ('A',)           |
| LogReg  | ('A', 'b', 'x_opt', 'f_opt')    | ('delta',)       |
| PDLP    | ('c', 'A_eq', 'b_eq', ...)      | ()               |
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import pandas as pd


# Type aliases for clarity
ProblemData = Dict[str, jnp.ndarray]
GroundTruth = Dict[str, jnp.ndarray]
Stepsizes = Tuple[jnp.ndarray, ...]
ParameterNames = Tuple[str, ...]


class ProblemModule(ABC):
    """Abstract base class for problem-specific mathematics and data handling.

    Each problem type (Quad, Lasso, LogReg, PDLP) implements this interface
    to provide:
    - Training data sampling
    - Trajectory computation functions
    - PEP constraint construction
    - Stepsize initialization and management
    - Output formatting

    The batched/fixed parameters interface is critical for enabling the
    LearningFramework to correctly batch operations across samples with
    different data structures. Each subclass declares which parameters
    vary across the batch (batched) vs which are shared (fixed).
    """

    def __init__(self, cfg: Any):
        """Initialize problem module with configuration.

        Args:
            cfg: Hydra configuration object containing problem parameters.
        """
        self.cfg = cfg

    @abstractmethod
    def sample_training_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Generate N training problem instances.

        Args:
            key: JAX random key for reproducible sampling.
            N: Number of problem instances to generate.

        Returns:
            problem_data: Problem-specific dict containing batch data.
                For Quad: {'Q_batch': (N, d, d), 'z0_batch': (N, d)}
                For Lasso: {'A': (m, n), 'b_batch': (N, n)}  # A is fixed
                For LogReg: {'A_batch': (N, m, n), 'b_batch': (N, m)}
                For PDLP: {'c_batch': ..., 'A_eq_batch': ..., etc.}
            ground_truth: Dict with optimal solutions and values.
                {'x_opt_batch': (N, d), 'f_opt_batch': (N,)}
        """
        pass

    @abstractmethod
    def sample_validation_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Generate N validation problem instances.

        Used for tracking validation loss during training. The validation loss
        is computed using the true PEP objective (not modified training losses),
        providing an unbiased estimate of generalization performance.

        The validation set should be sampled from the same distribution as
        training but held fixed throughout training for consistent monitoring.

        Args:
            key: JAX random key for reproducible sampling.
            N: Number of problem instances to generate.

        Returns:
            problem_data: Problem-specific dict containing batch data.
                Same structure as sample_training_batch().
            ground_truth: Dict with optimal solutions and values.
                {'x_opt_batch': (N, d), 'f_opt_batch': (N,)}
        """
        pass

    @abstractmethod
    def get_trajectory_fn(self, alg: str) -> Callable:
        """Return trajectory function for the specified algorithm.

        The returned function computes algorithm iterates given problem data
        and stepsizes. It may return:
        - Direct function: problem_data_to_gd_trajectories
        - Partial function: partial(ista_traj, A=self.A, lambd=self.lambd)
        - Factory-generated: create_logreg_traj_fn_gd(self.delta)

        Args:
            alg: Algorithm name (e.g., 'vanilla_gd', 'nesterov_fgm', 'ista', 'fista', 'pdhg')

        Returns:
            Callable that computes trajectories for this problem and algorithm.
        """
        pass

    @abstractmethod
    def get_pep_data_fn(self, alg: str) -> Callable:
        """Return PEP constraint construction function for the algorithm.

        The returned function constructs the PEP SDP data (A_obj, b_obj,
        A_vals, b_vals, c_vals, etc.) given stepsizes and problem parameters.

        Args:
            alg: Algorithm name.

        Returns:
            Callable with signature:
                pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj) -> pep_data_tuple
        """
        pass

    @abstractmethod
    def compute_L_mu_R(self, samples: ProblemData | None = None) -> Tuple[float, float, float]:
        """Compute problem parameters (smoothness, strong convexity, init radius).

        For some problems (Quad, Lasso), these are fixed by configuration.
        For others (LogReg), they may be computed from training samples.

        Args:
            samples: Optional problem data for data-dependent parameter computation.

        Returns:
            Tuple of (L, mu, R):
                L: Lipschitz constant of gradient (smoothness)
                mu: Strong convexity parameter
                R: Initial radius bound
        """
        pass

    @abstractmethod
    def get_initial_stepsizes(self, alg: str, K: int, L: float, mu: float) -> Stepsizes:
        """Get algorithm-specific stepsize initialization.

        Supports various initialization schemes:
        - Fixed scalar/vector (e.g., 1/L for GD)
        - Silver stepsizes for strongly convex GD
        - Nesterov acceleration sequences
        - Problem-specific defaults

        Args:
            alg: Algorithm name.
            K: Number of algorithm iterations (K_max).
            L: Lipschitz constant (smoothness).
            mu: Strong convexity parameter.

        Returns:
            Tuple of stepsize arrays. Structure depends on algorithm:
                GD: (t,) where t is scalar or (K,) vector
                FGM: (t, beta) where beta has K elements
                ISTA: (gamma,) scalar or vector
                FISTA: (gamma, beta) tuple
                PDHG: (tau, sigma, theta) 3-tuple
        """
        pass

    @abstractmethod
    def build_stepsizes_dataframe(
        self,
        stepsizes_history: list[Stepsizes],
        K_max: int,
        alg: str,
        training_losses: list[float] | None = None,
        validation_losses: list[float] | None = None,
        times: list[float] | None = None,
    ) -> pd.DataFrame:
        """Build problem-specific DataFrame for CSV output.

        Formats stepsize history into a DataFrame suitable for saving
        to CSV. The format depends on the algorithm (scalar vs vector,
        presence of beta/momentum parameters).

        Args:
            stepsizes_history: List of stepsizes tuples, one per SGD iteration.
            K_max: Number of algorithm iterations.
            alg: Algorithm name.
            training_losses: Optional list of training loss values per iteration.
            validation_losses: Optional list of validation loss values per iteration
                (computed using true PEP objective on held-out data).
            times: Optional list of iteration times in seconds.

        Returns:
            DataFrame with columns for iteration, stepsizes, training_loss,
            validation_loss, and times.
        """
        pass

    @abstractmethod
    def get_batched_parameters(self) -> ParameterNames:
        """Return names of parameters that vary across the batch (vmapped).

        These parameters will have a batch dimension and be vmapped over
        when computing trajectories. The names should match keys in the
        problem_data dict returned by sample_training_batch(), with '_batch'
        suffix for batched arrays.

        Examples:
            Quad: ('Q', 'z0', 'zs', 'fs')
            Lasso: ('b', 'x_opt', 'f_opt')
            LogReg: ('A', 'b', 'x_opt', 'f_opt')
            PDLP: ('c', 'A_eq', 'b_eq', 'A_ineq', 'b_ineq', 'x_opt', 'f_opt')

        Returns:
            Tuple of parameter names that vary across the batch.
        """
        pass

    @abstractmethod
    def get_fixed_parameters(self) -> ParameterNames:
        """Return names of parameters that are fixed across the batch.

        These parameters are shared across all samples in a batch and
        are not vmapped. The names should match keys in the problem_data
        dict returned by sample_training_batch().

        Examples:
            Quad: ()  # All parameters vary per sample
            Lasso: ('A',)  # A matrix is fixed, only b varies
            LogReg: ('delta',)  # Regularization parameter is fixed
            PDLP: ()  # All constraint data varies per sample

        Returns:
            Tuple of parameter names that are fixed across the batch.
        """
        pass

    @abstractmethod
    def get_ground_truth_keys(self) -> ParameterNames:
        """Return names of ground truth keys (optimal solutions and values).

        These keys identify the ground truth data that should be separated
        from problem parameters when computing metrics. They are returned
        by sample_training_batch() and sample_validation_batch() in the
        ground_truth dict.

        The keys should NOT include the '_batch' suffix. For example, if
        the ground_truth dict contains 'x_opt_batch', this method should
        return 'x_opt'.

        Examples:
            Quad: ('zs', 'fs')  # z-star, f-star (optimal point and value)
            Lasso: ('x_opt', 'f_opt')  # standard optimal solution
            LogReg: ('x_opt', 'f_opt')
            PDLP: ('x_opt', 'f_opt')

        Returns:
            Tuple of ground truth key names (without '_batch' suffix).
        """
        pass

    def compute_batched_trajectories(
        self,
        stepsizes: Stepsizes,
        batched_data: Dict[str, jnp.ndarray],
        fixed_data: Dict[str, jnp.ndarray],
        traj_fn: Callable,
        K_max: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute trajectories for a batch of problems.

        This method handles the vmapping logic for batched trajectory
        computation. Subclasses can override this for custom vmapping
        strategies, but the default implementation should work for most cases.

        Args:
            stepsizes: Algorithm stepsizes tuple.
            batched_data: Dict of batched parameters (with batch dimension).
            fixed_data: Dict of fixed parameters (no batch dimension).
            traj_fn: Trajectory function from get_trajectory_fn().
            K_max: Number of algorithm iterations.

        Returns:
            Tuple of (G_batch, F_batch) Gram representations.
        """
        raise NotImplementedError("compute_batched_trajectories must be implemented in subclass")

    @abstractmethod
    def generate_out_of_sample_data(
        self, key: jax.Array
    ) -> Dict[str, Tuple[ProblemData, GroundTruth]]:
        """Generate validation, test, and out-of-distribution problem sets.

        Used for evaluating learned stepsizes on held-out data. Different
        problem types have different OOD generation strategies:
        - Quad: Different eigenvalue distributions
        - Lasso: Different b vector scaling
        - LogReg: Different A matrix statistics
        - PDLP: Different problem instances

        Args:
            key: JAX random key for reproducible sampling.

        Returns:
            Dict with keys 'validation', 'test', 'ood', each mapping to
            (problem_data, ground_truth) tuple.
        """
        pass

    def get_supported_algorithms(self) -> list[str]:
        """Return list of algorithms supported by this problem type.

        Override in subclasses to restrict available algorithms.

        Returns:
            List of algorithm names (e.g., ['vanilla_gd', 'nesterov_fgm']).
        """
        return ["vanilla_gd", "nesterov_fgm"]

    def validate_config(self) -> None:
        """Validate configuration for this problem type.

        Override in subclasses to add problem-specific validation.
        Called during initialization to catch configuration errors early.

        Raises:
            ValueError: If configuration is invalid.
        """
        pass

    @abstractmethod
    def get_gram_dimensions(self, alg: str, K: int) -> Tuple[int, int]:
        """Return Gram matrix dimensions for the algorithm.

        Used for DRO SDP setup. Dimensions vary by algorithm:
        - GD/FGM: (K+2, K+2) for G, (K+1,) for F
        - ISTA: (2K+5, 2K+5) for composite structure
        - PDHG: (4K+11, 4K+11) for primal-dual

        Args:
            alg: Algorithm name.
            K: Number of iterations (K_max).

        Returns:
            Tuple of (dimG, dimF) for Gram matrix dimensions.
        """
        pass

    def create_metric_fn(
        self, trajectories: Any, problem_data: ProblemData, ground_truth: GroundTruth, pep_obj: str
    ) -> Callable[[int], float]:
        """Create a metric function for trajectory loss computation.

        The metric function computes the specified performance measure
        (objective value, gradient norm, or distance to optimum) at
        iteration k of the trajectory.

        This enables unified loss computation across problem types
        via metric function injection in TrajectoryLoss.

        Args:
            trajectories: Algorithm trajectory data (problem-specific format).
            problem_data: Problem-specific data dict.
            ground_truth: Optimal solution and value.
            pep_obj: Metric type ('obj_val', 'grad_sq_norm', 'opt_dist_sq_norm').

        Returns:
            Callable with signature metric_fn(k: int) -> scalar.
        """
        raise NotImplementedError(
            "create_metric_fn must be implemented in subclass for L2O support"
        )
