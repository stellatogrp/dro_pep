"""PDLP (Primal-Dual for Linear Programs) problem module.

Implements `PDLPProblemModule` using the ProblemModule ABC and UnifiedTrainer.
The underlying algorithm is Chambolle-Pock / PDHG on a capacitated
facility-location LP relaxation:

    min_x c^T x   s.t.   A_ineq x <= b_ineq,  A_eq x = b_eq,  l <= x <= u

cast as a saddle-point problem with K_mat = [-A_ineq; A_eq],  q = [-b_ineq; b_eq]:

    min_x max_y  L(x, y) = (c^T x + ind_{[l,u]}(x))
                           + <K x, y>
                           - (-q^T y + ind_{R^{m1}_+ × R^{m2}}(y))

This module uses:
- The verified-correct trajectory function
  `learning/trajectories/cp_lp.py::problem_data_to_cp_lp_trajectories`.
- The verified-correct PEP construction
  `learning/pep_constructions/chambolle_pock.py::construct_chambolle_pock_pep_data`.
- A lifted `FacilityLocationDPP` cvxpy DPP solver for efficient batched LP solving
  (originally from `learning_experiment_classes/old/pdlp.py`; its sign convention
  is confirmed to align with the verified CP stationarity).

A matching reference construction is locked down in
`tests/test_chambolle_pock_facility_location.py` and
`tests/test_cp_lp_trajectory_module.py`.
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

from learning.problem_module import (
    ProblemModule, ProblemData, GroundTruth, Stepsizes, ParameterNames,
)
from learning.unified_trainer import UnifiedTrainer
from learning.pep_constructions import construct_chambolle_pock_pep_data
from learning.trajectories import problem_data_to_cp_lp_trajectories

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)


# =============================================================================
# Facility-location problem generation (config-driven)
# =============================================================================

def generate_facility_location_problem(cfg, n_facilities, key, n_customers):
    """Generate a random Capacitated Facility Location problem instance."""
    k_fixed, k_demand, k_trans = jax.random.split(key, 3)

    fixed_costs = jax.random.uniform(
        k_fixed, shape=(n_facilities,),
        minval=cfg.fixed_costs.l, maxval=cfg.fixed_costs.u,
    )
    demands = jax.random.uniform(
        k_demand, shape=(n_customers,),
        minval=cfg.demands.l, maxval=cfg.demands.u,
    )
    transportation_costs = jax.random.uniform(
        k_trans, shape=(n_facilities, n_customers),
        minval=cfg.transportation_costs.l, maxval=cfg.transportation_costs.u,
    )

    avg_demand_per_facility = jnp.sum(demands) / n_facilities
    base_capacities = cfg.base_capacity.base * jnp.ones(n_facilities)
    capacities = base_capacities * avg_demand_per_facility * cfg.base_capacity.scaling

    return {
        "fixed_costs": fixed_costs,
        "capacities": capacities,
        "demands": demands,
        "transportation_costs": transportation_costs,
    }


@partial(jax.jit, static_argnames=['n_facilities', 'n_customers'])
def extract_constraint_matrices(fixed_costs, capacities, demands,
                                 transportation_costs, n_facilities, n_customers):
    """Extract (c, A_eq, b_eq, A_ineq, b_ineq, l, u) from problem parameters.

    Variable ordering: [y_1, ..., y_m, x_{11}, ..., x_{mn}].
    """
    m = n_facilities
    n = n_customers

    c = jnp.concatenate([fixed_costs, transportation_costs.flatten()])

    # Equality (demand satisfaction): sum_i x_{ij} = 1 for all j
    A_eq_y = jnp.zeros((n, m))
    A_eq_x = jnp.tile(jnp.eye(n), m)
    A_eq = jnp.hstack([A_eq_y, A_eq_x])
    b_eq = jnp.ones(n)

    # Capacity: sum_j d_j x_{ij} - s_i y_i <= 0 for all i
    A_cap_y = -jnp.diag(capacities)
    A_cap_x = jnp.kron(jnp.eye(m), demands.reshape(1, -1))
    A_cap = jnp.hstack([A_cap_y, A_cap_x])
    b_cap = jnp.zeros(m)

    # Linking: x_{ij} - y_i <= 0 for all i, j
    A_link_y = -jnp.kron(jnp.eye(m), jnp.ones((n, 1)))
    A_link_x = jnp.eye(m * n)
    A_link = jnp.hstack([A_link_y, A_link_x])
    b_link = jnp.zeros(m * n)

    A_ineq = jnp.vstack([A_cap, A_link])
    b_ineq = jnp.concatenate([b_cap, b_link])

    n_vars = m + m * n
    l = jnp.zeros(n_vars)
    u = jnp.ones(n_vars)
    return c, A_eq, b_eq, A_ineq, b_ineq, l, u


# =============================================================================
# FacilityLocationDPP — cvxpy DPP solver (lifted from old/pdlp.py:186-333).
# Sign convention here matches the verified CP stationarity identity, as
# verified in test_chambolle_pock_facility_location.py.
# =============================================================================

class FacilityLocationDPP:
    """CVXPY DPP-parametrized LP solver for fast batched facility-location solves."""

    def __init__(self, n, m1, m2):
        self.n = n
        self.m1 = m1
        self.m2 = m2

        self.c_param = cp.Parameter(n)
        self.Aineq_param = cp.Parameter((m1, n))
        self.bineq_param = cp.Parameter(m1)
        self.Aeq_param = cp.Parameter((m2, n))
        self.beq_param = cp.Parameter(m2)
        self.l_param = cp.Parameter(n)
        self.u_param = cp.Parameter(n)

        self.x = cp.Variable(n)

        self.obj = self.c_param.T @ self.x
        self.constraints = [
            -self.Aineq_param @ self.x >= -self.bineq_param,
            self.Aeq_param @ self.x == self.beq_param,
            self.x >= self.l_param,
            -self.x >= -self.u_param,
        ]

        self.prob = cp.Problem(cp.Minimize(self.obj), self.constraints)

    def solve(self, c_np, Aineq_np, bineq_np, Aeq_np, beq_np, l_np, u_np):
        """Solve LP; return (x_opt, y_opt) with PDHG sign convention.

        y_opt is stacked as [inequality_duals (>=0); -equality_duals].
        This aligns with K_mat = [-A_ineq; A_eq] and q = [-b_ineq; b_eq]
        used by the verified CP trajectory module.
        """
        self.c_param.value = c_np
        self.Aineq_param.value = Aineq_np
        self.bineq_param.value = bineq_np
        self.Aeq_param.value = Aeq_np
        self.beq_param.value = beq_np
        self.l_param.value = l_np
        self.u_param.value = u_np

        self.prob.solve(solver='CLARABEL')

        x_opt = self.x.value

        # Dual extraction, PDHG sign convention: keep inequality duals as-is,
        # negate equality duals. See old/pdlp.py:258-287 for derivation.
        y_ineq = self.constraints[0].dual_value
        nu_eq = -self.constraints[1].dual_value
        y_opt = np.concatenate([y_ineq, nu_eq])
        return x_opt, y_opt

    def solve_batch(self, c_batch_np, Aineq_batch_np, bineq_batch_np,
                    Aeq_batch_np, beq_batch_np, l_batch_np, u_batch_np):
        """Solve N LPs in batch. Returns (x_opt_batch, y_opt_batch)."""
        N = c_batch_np.shape[0]
        x_opt_batch = np.zeros((N, self.n))
        y_opt_batch = np.zeros((N, self.m1 + self.m2))
        for i in range(N):
            x_opt, y_opt = self.solve(
                c_batch_np[i], Aineq_batch_np[i], bineq_batch_np[i],
                Aeq_batch_np[i], beq_batch_np[i], l_batch_np[i], u_batch_np[i],
            )
            x_opt_batch[i] = x_opt
            y_opt_batch[i] = y_opt
        return x_opt_batch, y_opt_batch


# =============================================================================
# data_source_dir loader (mirror of lasso._load_and_subsample)
# =============================================================================

def _load_and_subsample_pdlp(npz_path, N, seed, dataset_label):
    """Load a saved PDLP problem-instance bundle and subsample N aligned rows.

    Samples indices once with a seeded np.random.Generator and applies them
    uniformly across c_batch, K_mat_batch, q_batch, x_opt_batch, y_opt_batch
    so row alignment is preserved. If N exceeds the available rows, returns
    all rows and logs a warning.

    Returns ({c_batch, K_mat_batch, q_batch}, {x_opt_batch, y_opt_batch}) as
    jnp.ndarrays — the same tuple shape produced by
    _sample_facility_batch_and_solve. Returns None if the file is missing
    (caller falls back to fresh sampling).
    """
    if not os.path.isfile(npz_path):
        return None
    d = np.load(npz_path)
    total = int(d['c_batch'].shape[0])
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
        {
            'c_batch': jnp.asarray(d['c_batch'][idx]),
            'K_mat_batch': jnp.asarray(d['K_mat_batch'][idx]),
            'q_batch': jnp.asarray(d['q_batch'][idx]),
        },
        {
            'x_opt_batch': jnp.asarray(d['x_opt_batch'][idx]),
            'y_opt_batch': jnp.asarray(d['y_opt_batch'][idx]),
        },
    )


# =============================================================================
# Batched problem generation / stacking (K_mat, q)
# =============================================================================

def _extract_batched_matrices(problem_batch, n_facilities, n_customers):
    """Vmapped matrix extraction for a batch of facility-location problems."""
    extractor = jax.vmap(
        partial(extract_constraint_matrices,
                n_facilities=n_facilities, n_customers=n_customers),
        in_axes=(0, 0, 0, 0),
    )
    return extractor(
        problem_batch["fixed_costs"],
        problem_batch["capacities"],
        problem_batch["demands"],
        problem_batch["transportation_costs"],
    )


def _sample_facility_batch_and_solve(
    key, cfg, n_facilities, n_customers, N, dpp_solver,
):
    """Sample N facility-location instances, extract matrices, solve LPs.

    Returns batched per-sample data:
      c_batch       (N, n_vars)
      K_mat_batch   (N, m1+m2, n_vars)   = stack([-A_ineq, A_eq]) per sample
      q_batch       (N, m1+m2)           = concat([-b_ineq, b_eq]) per sample
      x_opt_batch   (N, n_vars)
      y_opt_batch   (N, m1+m2)
    """
    keys = jax.random.split(key, N)
    generate_one = partial(
        generate_facility_location_problem,
        cfg=cfg, n_facilities=n_facilities, n_customers=n_customers,
    )
    problem_batch = jax.vmap(generate_one)(key=keys)

    c_b, Aeq_b, beq_b, Aineq_b, bineq_b, lb_b, ub_b = _extract_batched_matrices(
        problem_batch, n_facilities, n_customers,
    )

    # Solve LPs in batch via DPP (CPU / numpy)
    x_opt_b_np, y_opt_b_np = dpp_solver.solve_batch(
        np.asarray(c_b), np.asarray(Aineq_b), np.asarray(bineq_b),
        np.asarray(Aeq_b), np.asarray(beq_b), np.asarray(lb_b), np.asarray(ub_b),
    )

    # Stack K_mat = [-A_ineq; A_eq] and q = [-b_ineq; b_eq] per sample.
    K_mat_b = jnp.concatenate([-Aineq_b, Aeq_b], axis=1)
    q_b = jnp.concatenate([-bineq_b, beq_b], axis=1)

    return {
        'c_batch': c_b,
        'K_mat_batch': K_mat_b,
        'q_batch': q_b,
    }, {
        'x_opt_batch': jnp.asarray(x_opt_b_np),
        'y_opt_batch': jnp.asarray(y_opt_b_np),
    }


# =============================================================================
# Module-level trajectory / PEP wrappers
# =============================================================================

def _make_cp_lp_traj_fn(l, u, m1, m2):
    """Create a trajectory function for the unified trainer.

    Closes over the box bounds (l, u) and the inequality/equality row counts
    (m1, m2). The returned fn takes exactly the per-sample batched params:
    (c, K_mat, q, x_opt, y_opt) plus K_max.
    """
    m_total = m1 + m2

    # Fixed interior initial point — always strictly in the primal box and the
    # dual nonneg cone (for first m1 coords), so gf1_0 = c and gh_0 = -q are
    # valid subgradients (matches the verified facility-location test).
    x0 = 0.5 * (l + u)
    y0 = jnp.concatenate([0.1 * jnp.ones(m1), jnp.zeros(m2)])

    def wrapped_traj_fn(stepsizes, c, K_mat, q, x_opt, y_opt,
                        K_max, return_Gram_representation=True):
        return problem_data_to_cp_lp_trajectories(
            stepsizes, c, K_mat, q, l, u, x_opt, y_opt, x0, y0,
            K_max, m1,
            return_Gram_representation=return_Gram_representation,
        )

    return wrapped_traj_fn


def pep_data_fn_cp(stepsizes, mu, L, R, K_max, pep_obj,
                   composition_type='final', decay_rate=0.9):
    """Adapter for the uniform UnifiedTrainer pep_data_fn signature.

    For CP, `L` is repurposed as the operator-norm bound M = ||K||_op
    (a strictly-upper-bounding scalar), while `mu` and `pep_obj` are unused
    (the CP objective is fixed to the duality gap). `R` is the Lyapunov
    radius for the P-norm IC.
    """
    tau, sigma, theta = stepsizes
    return construct_chambolle_pock_pep_data(
        tau=tau, sigma=sigma, theta=theta, M=L, R=R, K_max=K_max,
        composition_type=composition_type, decay_rate=decay_rate,
    )


# =============================================================================
# PDLPProblemModule
# =============================================================================

class PDLPProblemModule(ProblemModule):
    """Problem module for PDLP (CP/PDHG on facility-location LP relaxations).

    Handles:
    - Facility-location LP generation (via cfg ranges).
    - Batched LP solving via cvxpy DPP.
    - Stacking (K_mat, q) per sample.
    - Per-sample trajectories via problem_data_to_cp_lp_trajectories.
    - Per-sample PEP SDP via construct_chambolle_pock_pep_data.

    Parameters bound via closure (not in training_data):
    - Box bounds l, u (always zeros and ones for facility location).
    - Inequality / equality row counts m1, m2.

    Batched per-sample parameters: c, K_mat, q, x_opt, y_opt.
    """

    def __init__(self, cfg: Any):
        super().__init__(cfg)

        self.n_facilities = int(cfg.n_facilities)
        self.n_customers = int(cfg.n_customers)
        self.n_vars = self.n_facilities + self.n_facilities * self.n_customers

        # Row counts of the mixed LP.
        self.m1 = self.n_facilities + self.n_facilities * self.n_customers  # capacity + linking
        self.m2 = self.n_customers                                           # demand
        self.m_total = self.m1 + self.m2

        log.info(
            f"PDLP: n_facilities={self.n_facilities}, n_customers={self.n_customers}; "
            f"n_vars={self.n_vars}, m1={self.m1}, m2={self.m2}"
        )

        # Box bounds are instance-independent for facility-location [0, 1].
        self.l = jnp.zeros(self.n_vars)
        self.u = jnp.ones(self.n_vars)

        # DPP solver (reused across all samples).
        self.dpp_solver = FacilityLocationDPP(self.n_vars, self.m1, self.m2)
        log.info("Created FacilityLocationDPP solver for batched LP solves")

        # M_val and R_val: prefer cached values from a sample-creation run when
        # `data_source_dir` is configured, since the mr-estimation pool itself
        # costs `mr_estimation_size` CVXPY solves. Fall back to fresh pool
        # estimation otherwise.
        data_source_dir = cfg.get('data_source_dir', None)
        meta_path = (os.path.join(data_source_dir, 'out_of_sample_metadata.npz')
                     if data_source_dir is not None else None)
        if meta_path is not None and os.path.isfile(meta_path):
            meta = np.load(meta_path)
            if 'M_val' in meta.files and 'R_val' in meta.files:
                self.M_val = float(meta['M_val'])
                self.R_val = float(meta['R_val'])
                log.info(
                    f"Loaded M_val={self.M_val:.6f} R_val={self.R_val:.6f} "
                    f"from cached metadata at {meta_path}"
                )
                return

        # Pre-sample a pool to compute M_val = max ||K_mat||_op and R_val.
        mr_N = int(cfg.get('mr_estimation_size', 100))
        mr_seed = int(cfg.get('mr_estimation_seed', 20260421))
        log.info(f"Pre-sampling {mr_N} instances to estimate M_val and R_val...")
        mr_key = jax.random.PRNGKey(mr_seed)
        pool_problem_data, pool_ground_truth = _sample_facility_batch_and_solve(
            mr_key, cfg,
            self.n_facilities, self.n_customers, mr_N, self.dpp_solver,
        )

        # M_val: upper bound on operator norm across ALL future samples.
        # The pool gives us an empirical max, but the random-LP distribution
        # has support beyond any finite pool, so we apply a safety factor.
        # If any actual training/val sample has ||K||_op > M_val, the PEP's
        # operator-PSD constraint ||K u||^2 <= M^2 ||u||^2 is violated by
        # that sample, which makes the DRO SDP unbounded.
        K_mat_pool = np.asarray(pool_problem_data['K_mat_batch'])
        pool_op_norms = np.array([
            np.linalg.norm(K_mat_pool[i], ord=2) for i in range(mr_N)
        ])
        m_safety = float(cfg.get('m_safety_factor', 1.3))
        self.M_val = float(pool_op_norms.max() * m_safety)
        log.info(
            f"M_val = {self.M_val:.6f}  "
            f"(pool max ||K||_op = {pool_op_norms.max():.6f}, "
            f"min = {pool_op_norms.min():.6f}, safety = {m_safety})"
        )

        # R_val: Euclidean radius such that all sample trajectories fit inside
        # the PEP's IC ball `||dx||^2 + ||du||^2 <= R^2`. The IC is stepsize-
        # and K-independent (verified bounded via the PEPit probe in
        # tests/test_cp_ic_boundedness_probe.py), so this computation runs
        # once at setup and never needs updating when stepsizes move during
        # LDRO-PEP training.
        x_opt_pool = np.asarray(pool_ground_truth['x_opt_batch'])
        y_opt_pool = np.asarray(pool_ground_truth['y_opt_batch'])
        x0_ref = 0.5 * np.ones(self.n_vars)
        y0_ref = np.concatenate([0.1 * np.ones(self.m1), np.zeros(self.m2)])

        pool_euc_sq = np.zeros(mr_N)
        for i in range(mr_N):
            dx_i = x0_ref - x_opt_pool[i]
            dy_i = y0_ref - y_opt_pool[i]
            pool_euc_sq[i] = dx_i @ dx_i + dy_i @ dy_i
        r_safety = float(cfg.get('r_safety_factor', 1.2))
        max_euc_sq = float(np.max(pool_euc_sq))
        self.R_val = float(np.sqrt(max_euc_sq) * r_safety)
        log.info(
            f"R_val = {self.R_val:.6f}  "
            f"(max pool Euclidean^2 = {max_euc_sq:.4f}, sqrt × safety {r_safety})"
        )

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    def sample_training_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Generate + solve N training facility-location instances.

        When `data_source_dir` is configured and `training_set.npz` is present
        under it, load N rows from there (seeded by `training_seed`). Otherwise
        sample fresh and solve via CVXPY.
        """
        data_source_dir = self.cfg.get('data_source_dir', None)
        if data_source_dir is not None:
            loaded = _load_and_subsample_pdlp(
                os.path.join(data_source_dir, 'training_set.npz'),
                N, self.cfg.training_seed, 'training',
            )
            if loaded is not None:
                return loaded
        return _sample_facility_batch_and_solve(
            key, self.cfg,
            self.n_facilities, self.n_customers, N, self.dpp_solver,
        )

    def sample_validation_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Same distribution as training; loads from data_source_dir when set."""
        data_source_dir = self.cfg.get('data_source_dir', None)
        if data_source_dir is not None:
            loaded = _load_and_subsample_pdlp(
                os.path.join(data_source_dir, 'validation_set.npz'),
                N, self.cfg.out_of_sample_val_seed, 'validation',
            )
            if loaded is not None:
                return loaded
        return self.sample_training_batch(key, N)

    def sample_test_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """In-distribution test set; loads from data_source_dir when set."""
        data_source_dir = self.cfg.get('data_source_dir', None)
        if data_source_dir is not None:
            loaded = _load_and_subsample_pdlp(
                os.path.join(data_source_dir, 'test_set.npz'),
                N, self.cfg.out_of_sample_test_seed, 'test',
            )
            if loaded is not None:
                return loaded
        return self.sample_training_batch(key, N)

    def _sample_ood_batch(self, key: jax.Array, N: int) -> Tuple[ProblemData, GroundTruth]:
        """Out-of-distribution set. Currently OOD = same distribution, different
        seed (no distribution shift); a real shift is a future work item."""
        data_source_dir = self.cfg.get('data_source_dir', None)
        if data_source_dir is not None:
            loaded = _load_and_subsample_pdlp(
                os.path.join(data_source_dir, 'ood_set.npz'),
                N, self.cfg.out_of_dist_seed, 'ood',
            )
            if loaded is not None:
                return loaded
        return _sample_facility_batch_and_solve(
            key, self.cfg,
            self.n_facilities, self.n_customers, N, self.dpp_solver,
        )

    # -----------------------------------------------------------------------
    # Trajectory / PEP wiring
    # -----------------------------------------------------------------------

    def get_trajectory_fn(self, alg: str) -> Callable:
        if alg != 'cp':
            raise ValueError(f"PDLP supports only alg='cp'; got {alg!r}")
        return _make_cp_lp_traj_fn(self.l, self.u, self.m1, self.m2)

    def get_pep_data_fn(self, alg: str) -> Callable:
        if alg != 'cp':
            raise ValueError(f"PDLP supports only alg='cp'; got {alg!r}")
        return pep_data_fn_cp

    # -----------------------------------------------------------------------
    # Problem parameters / stepsizes
    # -----------------------------------------------------------------------

    def compute_L_mu_R(self, samples: ProblemData | None = None) -> Tuple[float, float, float]:
        """Return (M_val, 0.0, R_val). `mu` is a placeholder (no convexity)."""
        return (self.M_val, 0.0, self.R_val)

    def get_initial_stepsizes(self, alg: str, K: int, L: float, mu: float) -> Stepsizes:
        """CP stepsizes satisfying tau * sigma * M^2 <= 1 strictly.

        Default: tau = sigma = 0.9 / M, theta = 1.0 (standard CP).
        """
        if alg != 'cp':
            raise ValueError(f"PDLP supports only alg='cp'; got {alg!r}")
        M = L
        tau_scalar = 0.9 / M
        sigma_scalar = 0.9 / M
        theta_scalar = 1.0

        is_vector = self.cfg.stepsize_type == "vector"
        if is_vector:
            tau = jnp.full(K, tau_scalar)
            sigma = jnp.full(K, sigma_scalar)
            theta = jnp.full(K, theta_scalar)
        else:
            tau = jnp.array(tau_scalar)
            sigma = jnp.array(sigma_scalar)
            theta = jnp.array(theta_scalar)

        return (tau, sigma, theta)

    # -----------------------------------------------------------------------
    # DataFrame formatting
    # -----------------------------------------------------------------------

    def build_stepsizes_dataframe(
        self,
        stepsizes_history: list[Stepsizes],
        K_max: int,
        alg: str,
        training_losses: list[float] | None = None,
        validation_losses: list[float] | None = None,
        times: list[float] | None = None,
    ) -> pd.DataFrame:
        """Build CSV rows with per-K tau/sigma/theta columns.

        Scalar stepsizes → single tau/sigma/theta columns.
        Vector stepsizes → tau_k, sigma_k, theta_k for k = 0 .. K_max-1.
        """
        tau_sample = stepsizes_history[0][0]
        is_vector = jnp.ndim(tau_sample) > 0

        data = {'iteration': list(range(len(stepsizes_history)))}

        if training_losses is not None:
            data['training_loss'] = [float(l) for l in training_losses]
        if validation_losses is not None:
            data['validation_loss'] = [float(l) for l in validation_losses]
        if times is not None:
            data['iter_time'] = [float(t) for t in times]

        if is_vector:
            for k in range(K_max):
                data[f'tau_{k}'] = [float(ss[0][k]) for ss in stepsizes_history]
                data[f'sigma_{k}'] = [float(ss[1][k]) for ss in stepsizes_history]
                data[f'theta_{k}'] = [float(ss[2][k]) for ss in stepsizes_history]
        else:
            data['tau'] = [float(ss[0]) for ss in stepsizes_history]
            data['sigma'] = [float(ss[1]) for ss in stepsizes_history]
            data['theta'] = [float(ss[2]) for ss in stepsizes_history]

        return pd.DataFrame(data)

    # -----------------------------------------------------------------------
    # Parameter / ground-truth key declarations
    # -----------------------------------------------------------------------

    def get_batched_parameters(self) -> ParameterNames:
        return ('c', 'K_mat', 'q', 'x_opt', 'y_opt')

    def get_fixed_parameters(self) -> ParameterNames:
        return ()

    def get_ground_truth_keys(self) -> ParameterNames:
        return ('x_opt', 'y_opt')

    def get_gram_dimensions(self, alg: str, K: int) -> Tuple[int, int]:
        return (4 * K + 11, 2 * (K + 2))

    # -----------------------------------------------------------------------
    # Batched trajectory computation (vmap)
    # -----------------------------------------------------------------------

    def compute_batched_trajectories(
        self,
        stepsizes: Stepsizes,
        batched_data: Dict[str, jnp.ndarray],
        fixed_data: Dict[str, jnp.ndarray],
        traj_fn: Callable,
        K_max: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Vmap the trajectory fn over (c, K_mat, q, x_opt, y_opt)."""
        batch_GF_func = jax.vmap(
            lambda c, K_mat, q, x_opt, y_opt: traj_fn(
                stepsizes, c, K_mat, q, x_opt, y_opt, K_max,
                return_Gram_representation=True,
            ),
            in_axes=(0, 0, 0, 0, 0),
        )
        return batch_GF_func(
            batched_data['c'], batched_data['K_mat'], batched_data['q'],
            batched_data['x_opt'], batched_data['y_opt'],
        )

    # -----------------------------------------------------------------------
    # Metric function
    # -----------------------------------------------------------------------

    def create_metric_fn(
        self, trajectories: Any, problem_data: ProblemData,
        ground_truth: GroundTruth, pep_obj: str,
    ) -> Callable[[int], float]:
        """Return metric_fn(k) → CP duality gap at iterate k.

        Gap_k = L(v_k, y_s) - L(v_s, y_k)  with L(v, y) = c^T v - y^T K v + q^T y.
        Only 'obj_val' is supported — other metrics don't have an obvious
        saddle-problem analog.
        """
        if pep_obj != 'obj_val':
            raise NotImplementedError(
                f"PDLP only supports pep_obj='obj_val' (duality gap); got {pep_obj!r}"
            )

        c = problem_data['c']
        K_mat = problem_data['K_mat']
        q = problem_data['q']
        x_opt = ground_truth['x_opt']
        y_opt = ground_truth['y_opt']

        # trajectories = (v_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter)
        v_iter = trajectories[0]  # shape (K_max+1, n_vars)
        y_iter = trajectories[1]  # shape (K_max+1, m1+m2)

        def L(vv, yy):
            return c @ vv - yy @ K_mat @ vv + q @ yy

        def metric_fn(k):
            v_k = v_iter[k]
            y_k = y_iter[k]
            return L(v_k, y_opt) - L(x_opt, y_k)

        return metric_fn

    # -----------------------------------------------------------------------
    # Out-of-sample generation
    # -----------------------------------------------------------------------

    def generate_out_of_sample_data(
        self, key: jax.Array,
    ) -> Dict[str, Tuple[ProblemData, GroundTruth]]:
        """Validation / test / OOD sets. OOD currently shares the in-distribution
        sampler with a different seed."""
        N_val = int(self.cfg.get('out_of_sample_val_N', 20))
        N_test = int(self.cfg.get('out_of_sample_test_N', 50))
        N_ood = int(self.cfg.get('out_of_dist_N', 50))

        key, val_key, test_key, ood_key = jax.random.split(key, 4)
        val = self.sample_validation_batch(val_key, N_val)
        test = self.sample_test_batch(test_key, N_test)
        ood = self._sample_ood_batch(ood_key, N_ood)
        return {'validation': val, 'test': test, 'ood': ood}

    # -----------------------------------------------------------------------
    # Algorithm support / validation
    # -----------------------------------------------------------------------

    def get_supported_algorithms(self) -> list[str]:
        return ['cp']

    def validate_config(self) -> None:
        alg = self.cfg.get('alg', 'cp')
        if alg != 'cp':
            raise ValueError(
                f"PDLP supports only alg='cp'; got {alg!r}"
            )


# =============================================================================
# Entry point
# =============================================================================

def pdlp_run(cfg):
    """Run learning experiment for PDLP (Chambolle-Pock on facility-location LP).

    Loops over K_max values, runs training for each K, saves per-K progress CSV.
    """
    log.info("=" * 60)
    log.info("Starting PDLP learning experiment")
    log.info("=" * 60)
    log.info(cfg)

    key = jax.random.PRNGKey(cfg.sgd_seed)

    problem_module = PDLPProblemModule(cfg)
    problem_module.validate_config()

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    key, train_key = jax.random.split(key)
    trainer = UnifiedTrainer(problem_module, cfg, train_key)
    trainer.prepare_data(save_dir=output_dir)

    for K in cfg.K_max:
        log.info(f"=== Starting training for K={K} ===")

        K_output_dir = os.path.join(output_dir, f"K_{K}")
        os.makedirs(K_output_dir, exist_ok=True)
        csv_path = os.path.join(K_output_dir, "progress.csv")

        result = trainer.train(K, csv_path, K_output_dir)

        tau0 = result.stepsizes[0]
        is_vector = jnp.ndim(tau0) > 0
        tau_str = str(tau0.tolist()) if is_vector else f'{float(tau0):.6f}'
        log.info(f'K={K} complete. Final tau={tau_str}. Saved to {csv_path}')

    log.info("=== PDLP experiment complete ===")


def pdlp_sample_creation_run(cfg):
    """Generate and save all PDLP problem-instance sets in a unified format.

    Produces four bundles, each as a single .npz with keys
    `c_batch`, `K_mat_batch`, `q_batch`, `x_opt_batch`, `y_opt_batch`:
        training_set.npz   (cfg.training_sample_N,    seed cfg.training_seed)
        validation_set.npz (cfg.out_of_sample_val_N,  seed cfg.out_of_sample_val_seed)
        test_set.npz       (cfg.out_of_sample_test_N, seed cfg.out_of_sample_test_seed)
        ood_set.npz        (cfg.out_of_dist_N,        seed cfg.out_of_dist_seed)

    Plus split files for the (future) plot pipeline (one file per per-instance
    array, key = array name without `_batch`):
        c_test_samples.npz, K_mat_test_samples.npz, q_test_samples.npz,
        x_opt_test_samples.npz, y_opt_test_samples.npz
        c_out_of_dist_samples.npz, K_mat_out_of_dist_samples.npz,
        q_out_of_dist_samples.npz, x_opt_out_of_dist_samples.npz,
        y_opt_out_of_dist_samples.npz

    Plus `out_of_sample_metadata.npz` with M_val, R_val, problem-shape arrays
    (l, u), and the full set of seeds + sample sizes.

    Args:
        cfg: Hydra configuration object.
    """
    log.info("=" * 60)
    log.info("Generating PDLP sample-creation problem sets")
    log.info("=" * 60)
    log.info(cfg)

    n_facilities = int(cfg.n_facilities)
    n_customers = int(cfg.n_customers)
    n_vars = n_facilities + n_facilities * n_customers
    m1 = n_facilities + n_facilities * n_customers
    m2 = n_customers
    log.info(
        f"PDLP shape: n_facilities={n_facilities}, n_customers={n_customers}; "
        f"n_vars={n_vars}, m1={m1}, m2={m2}"
    )

    dpp_solver = FacilityLocationDPP(n_vars, m1, m2)

    training_sample_N = int(cfg.training_sample_N)
    training_seed = int(cfg.training_seed)
    out_of_sample_val_N = int(cfg.out_of_sample_val_N)
    out_of_sample_val_seed = int(cfg.out_of_sample_val_seed)
    out_of_sample_test_N = int(cfg.out_of_sample_test_N)
    out_of_sample_test_seed = int(cfg.out_of_sample_test_seed)
    out_of_dist_N = int(cfg.out_of_dist_N)
    out_of_dist_seed = int(cfg.out_of_dist_seed)

    def _build_set(name, N, seed, filename):
        log.info(f"Generating {N} {name} problems (seed={seed})...")
        key = jax.random.PRNGKey(seed)
        problem_data, ground_truth = _sample_facility_batch_and_solve(
            key, cfg, n_facilities, n_customers, N, dpp_solver,
        )
        np.savez_compressed(
            filename,
            c_batch=np.asarray(problem_data['c_batch']),
            K_mat_batch=np.asarray(problem_data['K_mat_batch']),
            q_batch=np.asarray(problem_data['q_batch']),
            x_opt_batch=np.asarray(ground_truth['x_opt_batch']),
            y_opt_batch=np.asarray(ground_truth['y_opt_batch']),
        )
        log.info(f"Saved {filename}")
        return problem_data, ground_truth

    train_pd, train_gt = _build_set(
        "training",   training_sample_N,    training_seed,           "training_set.npz")
    _build_set(
        "validation", out_of_sample_val_N,  out_of_sample_val_seed,  "validation_set.npz")
    _build_set(
        "test",       out_of_sample_test_N, out_of_sample_test_seed, "test_set.npz")
    _build_set(
        "ood",        out_of_dist_N,        out_of_dist_seed,        "ood_set.npz")

    # -------------------------------------------------------------------------
    # Split files for plot consumers (mirrors lasso/quad split-file convention).
    # One file per per-instance array; key = array name without `_batch` suffix.
    # -------------------------------------------------------------------------
    def _split(bundle_path, suffix):
        bundle = np.load(bundle_path)
        for batched_key in ('c_batch', 'K_mat_batch', 'q_batch',
                            'x_opt_batch', 'y_opt_batch'):
            plain_key = batched_key[:-len('_batch')]
            np.savez_compressed(
                f"{plain_key}_{suffix}_samples.npz",
                **{plain_key: bundle[batched_key]},
            )
        log.info(f"Wrote split files for {suffix} ({bundle_path})")

    _split("test_set.npz", "test")
    _split("ood_set.npz",  "out_of_dist")

    # -------------------------------------------------------------------------
    # M_val and R_val computed from the training pool, mirroring __init__.
    # Saved into metadata so a future training run pointed at this dir can skip
    # the mr-estimation pool entirely.
    # -------------------------------------------------------------------------
    K_mat_train = np.asarray(train_pd['K_mat_batch'])
    pool_op_norms = np.array([
        np.linalg.norm(K_mat_train[i], ord=2) for i in range(training_sample_N)
    ])
    m_safety = float(cfg.get('m_safety_factor', 1.3))
    M_val = float(pool_op_norms.max() * m_safety)
    log.info(
        f"M_val = {M_val:.6f}  (training-pool max ||K||_op = "
        f"{pool_op_norms.max():.6f}, safety = {m_safety})"
    )

    x_opt_train = np.asarray(train_gt['x_opt_batch'])
    y_opt_train = np.asarray(train_gt['y_opt_batch'])
    x0_ref = 0.5 * np.ones(n_vars)
    y0_ref = np.concatenate([0.1 * np.ones(m1), np.zeros(m2)])
    pool_euc_sq = np.zeros(training_sample_N)
    for i in range(training_sample_N):
        dx_i = x0_ref - x_opt_train[i]
        dy_i = y0_ref - y_opt_train[i]
        pool_euc_sq[i] = dx_i @ dx_i + dy_i @ dy_i
    r_safety = float(cfg.get('r_safety_factor', 1.2))
    max_euc_sq = float(np.max(pool_euc_sq))
    R_val = float(np.sqrt(max_euc_sq) * r_safety)
    log.info(
        f"R_val = {R_val:.6f}  (max training-pool Euclidean^2 = "
        f"{max_euc_sq:.4f}, safety = {r_safety})"
    )

    np.savez_compressed(
        "out_of_sample_metadata.npz",
        # PEP / IC-radius parameters (loaded by __init__ to skip mr-estimation)
        M_val=M_val,
        R_val=R_val,
        m_safety_factor=m_safety,
        r_safety_factor=r_safety,
        # Problem shape (consumed by future plot scripts; box bounds explicit)
        n_facilities=n_facilities,
        n_customers=n_customers,
        n_vars=n_vars,
        m1=m1,
        m2=m2,
        l=np.zeros(n_vars),
        u=np.ones(n_vars),
        # Sizes
        training_sample_N=training_sample_N,
        out_of_sample_val_N=out_of_sample_val_N,
        out_of_sample_test_N=out_of_sample_test_N,
        out_of_dist_N=out_of_dist_N,
        # Seeds (also consumed by sample_*_batch on load)
        training_seed=training_seed,
        out_of_sample_val_seed=out_of_sample_val_seed,
        out_of_sample_test_seed=out_of_sample_test_seed,
        out_of_dist_seed=out_of_dist_seed,
    )

    log.info("=== PDLP sample-creation complete ===")
