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

from learning.trajectories_pdhg import problem_data_to_pdhg_trajectories
from learning.pep_construction_chambolle_pock_linop import (
    construct_chambolle_pock_pep_data,
    chambolle_pock_pep_data_to_numpy,
)

from learning.adam_optimizers import AdamWMin
from learning.trajectories_pdhg import problem_data_to_pdhg_trajectories
from learning.jax_scs_layer import dro_scs_solve, compute_preconditioner_from_samples

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=6, suppress=True)

log = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=['cfg', 'n_facilities', 'n_customers'])
def generate_facility_location_problem(
    cfg, 
    n_facilities: int,
    key: jax.Array,
    n_customers: int | None = None,
) -> dict[str, jax.Array]:
    """
    Generates a random Capacitated Facility Location problem instance.

    Args:
        n_facilities: Number of facilities.
        random_seed: Seed for the random number generator.
        n_customers: Number of customers (default: 10 * n_facilities).

    Returns:
        A dictionary containing fixed_costs, capacities, demands, and transportation_costs
        as JAX arrays.
    """

    k_fixed, k_demand, k_trans = jax.random.split(key, 3)
    
    if n_customers is None:
        n_customers = 10 * n_facilities
    
    fixed_costs = jax.random.uniform(
        k_fixed, 
        shape=(n_facilities,), 
        minval=cfg.fixed_costs.l, 
        maxval=cfg.fixed_costs.u
    )
    
    demands = jax.random.uniform(
        k_demand, 
        shape=(n_customers,), 
        minval=cfg.demands.l, 
        maxval=cfg.demands.u
    )
    
    transportation_costs = jax.random.uniform(
        k_trans, 
        shape=(n_facilities, n_customers), 
        minval=cfg.transportation_costs.l, 
        maxval=cfg.transportation_costs.u
    )

    # Calculate capacities using JAX numpy (jnp)
    avg_demand_per_facility = jnp.sum(demands) / n_facilities
    base_capacities = cfg.base_capacity.base * jnp.ones(n_facilities)
    
    # Scale capacities
    capacities = base_capacities * avg_demand_per_facility * cfg.base_capacity.scaling

    return {
        "fixed_costs": fixed_costs,
        "capacities": capacities,
        "demands": demands,
        "transportation_costs": transportation_costs,
        "n_facilities": n_facilities,
        "n_customers": n_customers,
    }


@partial(jax.jit, static_argnames=['n_facilities', 'n_customers'])
def extract_constraint_matrices(
    fixed_costs: jax.Array,
    capacities: jax.Array,
    demands: jax.Array,
    transportation_costs: jax.Array,
    n_facilities: int,
    n_customers: int,
):
    """
    Extract constraint matrices from facility location problem parameters.
    
    This function is designed to be used with JAX autodiff. You can create a closure
    by partially applying the dimension arguments:
    
        make_matrices = partial(extract_constraint_matrices, 
                                n_facilities=m, n_customers=n)
        # Now make_matrices(fixed_costs, capacities, demands, transportation_costs)
        # is differentiable w.r.t. these arrays
    
    Variable ordering: [y_1, ..., y_m, x_{11}, x_{12}, ..., x_{1n}, x_{21}, ..., x_{mn}]
    
    Constraints:
        1. Equality: sum_{i} x_{ij} = 1 for all j (demand satisfaction)
        2. Inequality: sum_{j} d_j * x_{ij} - s_i * y_i <= 0 for all i (capacity)
        3. Inequality: x_{ij} - y_i <= 0 for all i,j (linking)
        4. Bounds: 0 <= y_i, x_{ij} <= 1
    
    Args:
        fixed_costs: Fixed costs for opening facilities (m,)
        capacities: Capacity of each facility (m,)
        demands: Demand of each customer (n,)
        transportation_costs: Cost to serve customer j from facility i (m, n)
        n_facilities: Number of facilities (m)
        n_customers: Number of customers (n)
        
    Returns:
        FacilityLocationMatrices NamedTuple with all constraint matrices.
    """
    m = n_facilities
    n = n_customers
    n_vars = m + m * n
    
    # ==========================================================================
    # Objective vector c
    # ==========================================================================
    c = jnp.concatenate([fixed_costs, transportation_costs.flatten()])
    
    # ==========================================================================
    # Equality constraints: sum_{i} x_{ij} = 1 for all j (demand satisfaction)
    # ==========================================================================
    # For customer j, sum over facilities: x_{1j} + x_{2j} + ... + x_{mj} = 1
    # The x variables are ordered as [x_{11},...,x_{1n}, x_{21},...,x_{2n}, ...]
    # This is: I_n tiled m times horizontally
    A_eq_y = jnp.zeros((n, m))
    A_eq_x = jnp.tile(jnp.eye(n), m)  # (n, m*n)
    A_eq = jnp.hstack([A_eq_y, A_eq_x])
    b_eq = jnp.ones(n)
    
    # ==========================================================================
    # Inequality constraints: A_ineq @ x <= b_ineq
    # ==========================================================================
    # --- Capacity constraints (m rows) ---
    # For facility i: d^T @ x_i - s_i * y_i <= 0
    # y part: -diag(capacities)  shape (m, m)
    # x part: kron(I_m, demands^T) shape (m, m*n)
    A_cap_y = -jnp.diag(capacities)
    A_cap_x = jnp.kron(jnp.eye(m), demands.reshape(1, -1))
    A_cap = jnp.hstack([A_cap_y, A_cap_x])
    b_cap = jnp.zeros(m)
    
    # --- Linking constraints (m*n rows) ---
    # x_{ij} - y_i <= 0 for all i,j
    # y part: -kron(I_m, ones(n,1))
    # x part: I_{m*n}
    A_link_y = -jnp.kron(jnp.eye(m), jnp.ones((n, 1)))
    A_link_x = jnp.eye(m * n)
    A_link = jnp.hstack([A_link_y, A_link_x])
    b_link = jnp.zeros(m * n)
    
    # Stack all inequality constraints
    A_ineq = jnp.vstack([A_cap, A_link])
    b_ineq = jnp.concatenate([b_cap, b_link])
    
    # ==========================================================================
    # Bound constraints: lb <= x <= ub
    # ==========================================================================
    lb = jnp.zeros(n_vars)
    ub = jnp.ones(n_vars)

    return c, A_eq, b_eq, A_ineq, b_ineq, lb, ub


class FacilityLocationDPP:

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

        # Reformulate box constraints as inequalities to include them in K matrix
        # x >= l becomes I @ x >= l
        # x <= u becomes -I @ x >= -u
        # Stack these with A_ineq to form the full inequality system
        self.obj = self.c_param.T @ self.x
        self.constraints = [
            -self.Aineq_param @ self.x >= -self.bineq_param,  # A_ineq @ x <= b_ineq
            self.Aeq_param @ self.x == self.beq_param,
            self.x >= self.l_param,  # Will extract dual for lower bound
            -self.x >= -self.u_param,  # Will extract dual for upper bound
        ]

        self.prob = cp.Problem(cp.Minimize(self.obj), self.constraints)
    
    def solve(self, c_np, Aineq_np, bineq_np, Aeq_np, beq_np, l_np, u_np):
        """
        Solve LP and return primal + dual variables for PDLP formulation.

        Returns:
            x_opt: (n,) optimal primal solution
            y_opt: (m1 + m2 + 2*n,) optimal dual variables for K = [A_ineq; A_eq; I; -I]
                   where y_opt = [y_ineq; y_eq; y_lower; y_upper]
        """
        # Update parameter values
        self.c_param.value = c_np
        self.Aineq_param.value = Aineq_np
        self.bineq_param.value = bineq_np
        self.Aeq_param.value = Aeq_np
        self.beq_param.value = beq_np
        self.l_param.value = l_np
        self.u_param.value = u_np

        # Solve (reuses problem structure)
        self.prob.solve(solver='CLARABEL')

        x_opt = self.x.value

        # Extract dual variables with correct signs for PDLP formulation
        # CVX convention: constraint dual_value is the Lagrange multiplier
        # PDLP form: min_x max_y c^T x + <Kx - q, y> where K = [A_ineq; I; -I; A_eq]
        # Order: all inequalities first, then equalities
        #
        # Constraint [0]: -A_ineq @ x >= -b_ineq  =>  dual for (A_ineq @ x <= b_ineq)
        # Constraint [1]: A_eq @ x == b_eq  =>  dual for equality
        # Constraint [2]: x >= l  =>  dual for lower bound
        # Constraint [3]: -x >= -u  =>  dual for upper bound

        # Extract dual variables for PDHG formulation K = [-A_ineq; A_eq]
        # Box constraints are handled in f1's indicator function, not in K
        #
        # CVX solves with constraint -A_ineq @ x >= -b_ineq
        # For PDHG with K = [-A_ineq; A_eq], we convert from CVX's Lagrangian duals to saddle point duals
        # The conversion depends on whether we're maximizing or minimizing over y
        #
        # PDHG: min_x max_y c^T x + <Kx - q, y>
        # KKT: c + K^T y = 0  =>  K^T y = -c
        #
        # CVX standard form has KKT: c - (constraint gradients)^T λ = 0
        # For constraint -A_ineq x >= -b_ineq: gradient is -A_ineq
        # So: c - (-A_ineq)^T λ_ineq - A_eq^T λ_eq = 0
        # =>  c + A_ineq^T λ_ineq + A_eq^T λ_eq = 0
        #
        # Comparing with PDHG KKT: c + (-A_ineq)^T y_ineq + A_eq^T y_eq = 0
        # We get: y_ineq = -λ_ineq, y_eq = λ_eq

        # Extract dual variables for PDHG formulation K = [-A_ineq; A_eq]
        # Box constraints are handled implicitly in f1's indicator function, not in K
        #
        # CVX solves:
        #   min c^T x s.t. -A_ineq @ x >= -b_ineq, A_eq @ x == b_eq, l <= x <= u
        # CVX KKT: c - (-A_ineq)^T λ_ineq - A_eq^T λ_eq = 0
        #       => c + A_ineq^T λ_ineq - A_eq^T λ_eq = 0
        #
        # PDHG with K = [-A_ineq; A_eq], q = [-b_ineq; b_eq]:
        #   L(x,y) = c^T x + <Kx - q, y> with y = [s; ν]
        # PDHG KKT: c + K^T y = c + (-A_ineq)^T s + A_eq^T ν = 0
        #
        # Matching terms: -A_ineq^T s = A_ineq^T λ_ineq  =>  s = -λ_ineq (but s must be >= 0)
        #                 A_eq^T ν = -A_eq^T λ_eq         =>  ν = -λ_eq
        #
        # User clarification: Do NOT negate inequality duals (they must be non-negative)
        # Only negate equality duals

        y_ineq = self.constraints[0].dual_value    # Keep as is: s >= 0
        ν_eq = -self.constraints[1].dual_value     # Negate: ν = -λ_eq

        # Return dual variables: [s; ν] (no λ for box constraints)
        y_opt = np.concatenate([y_ineq, ν_eq])

        return x_opt, y_opt
    
    def solve_batch(self,
                     c_batch_np,
                     Aineq_batch_np,
                     bineq_batch_np, 
                     Aeq_batch_np,
                     beq_batch_np,
                     l_batch_np,
                     u_batch_np,
                     x_opt_benchmark=0,
                     y_opt_benchmark=0,
        ):
        N = c_batch_np.shape[0]
        n = self.c_param.shape[0]
        m1 = self.Aineq_param.shape[0]
        m2 = self.Aeq_param.shape[0]

        x_opt_batch = np.zeros((N, n))
        y_opt_batch = np.zeros((N, m1 + m2))
        R_max = 0.0
        Dnorm_max = 0.0

        for i in range(N):
            x_opt, y_opt = self.solve(c_batch_np[i],
                                      Aineq_batch_np[i],
                                      bineq_batch_np[i],
                                      Aeq_batch_np[i],
                                      beq_batch_np[i],
                                      l_batch_np[i],
                                      u_batch_np[i],
            )
            x_opt_batch[i] = x_opt
            y_opt_batch[i] = y_opt

            R = np.linalg.norm(np.concatenate([x_opt - x_opt_benchmark, y_opt - y_opt_benchmark]))
            Dnorm = np.linalg.norm(np.vstack([-Aineq_batch_np[i], Aeq_batch_np[i]]), ord=2)
            
            R_max = max(R_max, R)
            Dnorm_max = max(Dnorm_max, Dnorm)
        
        return x_opt_batch, y_opt_batch, R_max, Dnorm_max


def generate_batch_problem_jax(key, n_facilities, n_customers, N, cfg):
    """
    Generate a batch of Facility Location problems.
    
    Args:
        key: JAX random key
        n_facilities: Number of facilities
        n_customers: Number of customers
        N: Batch size
        cfg: Configuration object
        
    Returns:
        Dictionary of batched JAX arrays
    """
    keys = jax.random.split(key, N)
    generate_one = partial(
        generate_facility_location_problem,
        cfg=cfg,
        n_facilities=n_facilities,
        n_customers=n_customers
    )
    batch_problem = jax.vmap(generate_one)(key=keys)
    return batch_problem


def sample_pdlp_batch(key, n_facilities, n_customers, N, cfg, pdlp_dpp=None, x_opt_benchmark=0, y_opt_benchmark=0):
    """
    Sample a batch of problems and solve them.
    
    Args:
        key: JAX random key
        n_facilities, n_customers: Problem dimensions
        N: Batch size
        cfg: Config
        pdlp_dpp: Optional DPP solver instance
        
    Returns:
        problem_batch, x_opt_batch, y_opt_batch
    """
    # Generate batch of problems
    problem_batch = generate_batch_problem_jax(key, n_facilities, n_customers, N, cfg)
    
    # Solve batch
    # We need to compute m1, m2 to initialize/use DPP
    # Extract matrices for first problem to get dimensions
    # All problems in batch have same structure, just different data
    # But wait - extracts_constraint_matrices returns (c, A_eq, b_eq, A_ineq, b_ineq, lb, ub)
    # Be careful: extract_constraint_matrices supports vmap if inputs are batched
    
    # Vectorized extraction
    extractor = jax.vmap(
        partial(extract_constraint_matrices, n_facilities=n_facilities, n_customers=n_customers),
        in_axes=(0, 0, 0, 0)
    )
    
    c_b, Aeq_b, beq_b, Aineq_b, bineq_b, lb_b, ub_b = extractor(
        problem_batch["fixed_costs"],
        problem_batch["capacities"],
        problem_batch["demands"],
        problem_batch["transportation_costs"]
    )
    
    # Initialize DPP if needed
    if pdlp_dpp is None:
        m1 = Aineq_b.shape[1]
        m2 = Aeq_b.shape[1]
        n_vars = c_b.shape[1]
        pdlp_dpp = FacilityLocationDPP(n_vars, m1, m2)
        
    # Solve batch using CVXPY / DPP
    # We need to convert JAX arrays to Numpy for CVXPY
    x_opt_b, y_opt_b, R_max, Dnorm_max = pdlp_dpp.solve_batch(
        np.array(c_b),
        np.array(Aineq_b),
        np.array(bineq_b),
        np.array(Aeq_b),
        np.array(beq_b),
        np.array(lb_b),
        np.array(ub_b),
        x_opt_benchmark=x_opt_benchmark,
        y_opt_benchmark=y_opt_benchmark,
    )
    
    return problem_batch, jnp.array(x_opt_b), jnp.array(y_opt_b), R_max, Dnorm_max


def build_stepsizes_df_pdlp(all_stepsizes_vals, K_max, is_vector, all_losses=None, all_times=None):
    """Build a DataFrame from PDLP stepsizes history for CSV saving."""
    data = {'iteration': list(range(len(all_stepsizes_vals)))}

    # Add losses if provided
    if all_losses is not None:
        padded_losses = all_losses + [np.nan] * (len(all_stepsizes_vals) - len(all_losses))
        data['loss'] = padded_losses

    # Add times if provided
    if all_times is not None:
        padded_times = all_times + [np.nan] * (len(all_stepsizes_vals) - len(all_times))
        data['iter_time'] = padded_times

    # Add tau, sigma, theta values (grouped by parameter type)
    if is_vector:
        # All taus together
        for k in range(K_max):
            data[f'tau_{k}'] = [float(ss[0][k]) for ss in all_stepsizes_vals]
        # All sigmas together
        for k in range(K_max):
            data[f'sigma_{k}'] = [float(ss[1][k]) for ss in all_stepsizes_vals]
        # All thetas together
        for k in range(K_max):
            data[f'theta_{k}'] = [float(ss[2][k]) for ss in all_stepsizes_vals]
    else:
        # Scalar case
        def get_scalar(ss, idx):
            val = ss[idx]
            if hasattr(val, 'item') and val.size == 1:
                return float(val.item())
            else:
                return float(val)
        data['tau'] = [get_scalar(ss, 0) for ss in all_stepsizes_vals]
        data['sigma'] = [get_scalar(ss, 1) for ss in all_stepsizes_vals]
        data['theta'] = [get_scalar(ss, 2) for ss in all_stepsizes_vals]

    return pd.DataFrame(data)


def run_sgd_for_K_pdlp(cfg, K_max, key, stepsizes_init, sgd_iters, eta_t,
                       eps, alpha, optimizer_type, N_val, csv_path, precond_inv,
                       pdlp_dpp, x_ws, y_ws, m1, n_vars, Dnorm_max, R_max):
    """
    Run SGD training loop for PDLP step size learning.

    Args:
        cfg: Hydra config
        K_max: Number of PDHG iterations
        key: JAX random key
        stepsizes_init: Tuple (tau_init, sigma_init, theta_init)
        sgd_iters: Number of SGD iterations
        eta_t: Learning rate
        eps: Wasserstein radius
        alpha: CVaR confidence level
        optimizer_type: 'vanilla_sgd', 'adamw', or 'sgd_wd'
        N_val: Batch size
        csv_path: Path to save progress CSV
        precond_inv: Precomputed preconditioner inverse
        pdlp_dpp: FacilityLocationDPP instance for solving
        x_ws: Warmstart primal point
        y_ws: Warmstart dual point
        m1: Number of inequality constraints
        n_vars: Number of decision variables
        Dnorm_max: Maximum constraint matrix norm (for step size scaling)
        R_max: Maximum radius from samples
    """
    log.info(f"=== Running SGD for K={K_max} ===")

    learning_framework = cfg.learning_framework
    n_facilities = cfg.n_facilities
    n_customers = cfg.n_customers

    if learning_framework == 'ldro-pep':
        dro_canon_backend = cfg.get('dro_canon_backend', 'manual_jax')
        if dro_canon_backend != 'manual_jax':
            raise ValueError(f"Only 'manual_jax' dro_canon_backend is supported. Got: {dro_canon_backend}")
    elif learning_framework == 'l2o':
        pass  # L2O doesn't use DRO backend
    else:
        raise ValueError(f"Only 'ldro-pep' or 'l2o' learning_framework is supported. Got: {learning_framework}")

    # Helper to sample a batch of PDLP problems
    def sample_batch(sample_key):
        batch_key, next_key = jax.random.split(sample_key)
        problem_batch, x_opt_batch, y_opt_batch, _, _ = sample_pdlp_batch(
            batch_key, n_facilities, n_customers, N_val, cfg,
            pdlp_dpp=pdlp_dpp, x_opt_benchmark=x_ws, y_opt_benchmark=y_ws
        )
        return next_key, problem_batch, x_opt_batch, y_opt_batch

    # Extractor for constraint matrices (vmapped)
    extractor_vmap = jax.vmap(
        partial(extract_constraint_matrices, n_facilities=n_facilities, n_customers=n_customers),
        in_axes=(0, 0, 0, 0)
    )

    # Helper to compute Lagrangian optimal value
    def compute_lagrangian_opt(c, A_ineq, b_ineq, A_eq, b_eq, x, y):
        D = jnp.vstack([A_ineq, A_eq])
        q = jnp.concatenate([b_ineq, b_eq])
        return jnp.dot(c, x) + jnp.dot(y, q) - jnp.dot(y, D @ x)

    compute_f_opt_vmap = jax.vmap(compute_lagrangian_opt)

    risk_type = 'cvar' if cfg.dro_obj == 'cvar' else 'expectation'

    # M is the coupling matrix norm used in PEP construction
    # For PDHG on LP, M = ||D||_2 (spectral norm of constraint matrix)
    M = Dnorm_max
    R = R_max

    if learning_framework == 'ldro-pep':
        # DRO pipeline: compute Gram representation, then solve DRO SDP
        def pdlp_dro_pipeline(stepsizes_tuple, problem_batch, x0_batch, y0_batch, x_opt_batch, y_opt_batch, f_opt_batch):
            """Full DRO pipeline for PDLP using manual JAX canonicalization."""
            tau, sigma, theta = stepsizes_tuple

            # Extract constraint matrices from problem batch
            c_batch, Aeq_batch, beq_batch, Aineq_batch, bineq_batch, lb_batch, ub_batch = extractor_vmap(
                problem_batch["fixed_costs"],
                problem_batch["capacities"],
                problem_batch["demands"],
                problem_batch["transportation_costs"]
            )

            # Stack D and q for each sample
            def make_D_q(A_ineq, b_ineq, A_eq, b_eq):
                D = jnp.vstack([A_ineq, A_eq])
                q = jnp.concatenate([b_ineq, b_eq])
                return D, q

            make_D_q_vmap = jax.vmap(make_D_q)
            D_batch, q_batch = make_D_q_vmap(Aineq_batch, bineq_batch, Aeq_batch, beq_batch)

            # Compute trajectories for all samples
            def traj_fn_single(c, D, q, l, u, x0, y0, x_opt, y_opt):
                return problem_data_to_pdhg_trajectories(
                    stepsizes_tuple, c, D, q, l, u, x0, y0, x_opt, y_opt,
                    K_max=K_max, m1=m1, M=M
                )

            batch_GF_fn = jax.vmap(traj_fn_single, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))
            G_batch, F_batch = batch_GF_fn(
                c_batch, D_batch, q_batch, lb_batch, ub_batch,
                x0_batch, y0_batch, x_opt_batch, y_opt_batch
            )

            # Compute PEP constraint matrices (depend on stepsizes)
            pep_data = construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K_max)
            A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]
            PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes = pep_data[5:]

            # Convert PSD_shapes (Python list) to JAX array for API compatibility
            PSD_mat_dims = jnp.array(PSD_shapes) if PSD_shapes else None

            return dro_scs_solve(
                A_obj, b_obj, A_vals, b_vals, c_vals,
                G_batch, F_batch,
                eps, precond_inv,
                risk_type=risk_type,
                alpha=alpha,
                PSD_A_vals=PSD_A_vals,
                PSD_b_vals=PSD_b_vals,
                PSD_c_vals=PSD_c_vals,
                PSD_mat_dims=PSD_mat_dims,
            )

        value_and_grad_fn = jax.value_and_grad(pdlp_dro_pipeline, argnums=0)

    elif learning_framework == 'l2o':
        # L2O pipeline: compute PEP objectives directly without DRO SDP
        # TODO: Implement L2O pipeline for PDLP
        # This would compute duality gap or other metrics directly from trajectories
        raise NotImplementedError("L2O pipeline for PDLP not yet implemented. Please use 'ldro-pep'.")

    # Initialize stepsizes
    stepsizes = stepsizes_init
    tau, sigma, theta = stepsizes
    is_vector = jnp.ndim(tau) > 0

    all_stepsizes_vals = [stepsizes]
    all_losses = []
    all_times = []

    # Determine update mask for learning specific step sizes
    learn_tau = cfg.get('learn_tau', True)
    learn_sigma = cfg.get('learn_sigma', True)
    learn_theta = cfg.get('learn_theta', False)  # Often fixed to 1
    update_mask = [learn_tau, learn_sigma, learn_theta]

    if not all(update_mask):
        log.info(f'Update mask: tau={learn_tau}, sigma={learn_sigma}, theta={learn_theta}')

    def proj_stepsizes(x):
        """Project step sizes to be non-negative."""
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
            update_mask=update_mask if not all(update_mask) else None,
        )

    # SGD iterations
    for iter_num in range(sgd_iters):
        tau, sigma, theta = stepsizes

        if is_vector:
            tau_log = '[' + ', '.join(f'{x:.5f}' for x in tau.tolist()) + ']'
            sigma_log = '[' + ', '.join(f'{x:.5f}' for x in sigma.tolist()) + ']'
            theta_log = '[' + ', '.join(f'{x:.5f}' for x in theta.tolist()) + ']'
        else:
            tau_log = f'{float(tau):.5f}'
            sigma_log = f'{float(sigma):.5f}'
            theta_log = f'{float(theta):.5f}'

        log.info(f'K={K_max}, iter={iter_num}, tau={tau_log}, sigma={sigma_log}, theta={theta_log}')

        # Sample new batch
        iter_start_time = time.perf_counter()
        log.info('sampling...')
        key, problem_batch, x_opt_batch, y_opt_batch = sample_batch(key)
        log.info('samples found')

        # Create initial points (copies of warm-start solution)
        x0_batch = jnp.tile(x_ws, (N_val, 1))
        y0_batch = jnp.tile(y_ws, (N_val, 1))

        # Compute f_opt for each sample
        c_batch, Aeq_batch, beq_batch, Aineq_batch, bineq_batch, _, _ = extractor_vmap(
            problem_batch["fixed_costs"],
            problem_batch["capacities"],
            problem_batch["demands"],
            problem_batch["transportation_costs"]
        )
        f_opt_batch = compute_f_opt_vmap(
            c_batch, Aineq_batch, bineq_batch, Aeq_batch, beq_batch,
            x_opt_batch, y_opt_batch
        )

        # Compute loss and gradients
        log.info('calling value_and_grad_fn...')
        loss, d_stepsizes = value_and_grad_fn(
            stepsizes, problem_batch, x0_batch, y0_batch, x_opt_batch, y_opt_batch, f_opt_batch
        )
        log.info('value_and_grad_fn returned, materializing results...')

        # Force materialization of JAX arrays
        loss_val = float(loss)
        d_stepsizes_materialized = tuple(jnp.array(ds) for ds in d_stepsizes)
        log.info(f'results materialized, loss={loss_val:.6f}')
        log.info(f'd_stepsizes (tau, sigma, theta): {d_stepsizes_materialized}')

        iter_time = time.perf_counter() - iter_start_time
        log.info(f'  iter_time (finding optimal sols + solving SDP): {iter_time:.3f}s')

        all_losses.append(loss_val)
        all_times.append(iter_time)

        # SGD step
        if optimizer_type == "vanilla_sgd":
            new_stepsizes = []
            for i, (s, ds) in enumerate(zip(stepsizes, d_stepsizes)):
                if update_mask[i]:
                    new_stepsizes.append(jax.nn.relu(s - eta_t * ds))
                else:
                    new_stepsizes.append(s)
            stepsizes = tuple(new_stepsizes)
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
            new_stepsizes = []
            for i, (s, ds) in enumerate(zip(stepsizes, d_stepsizes)):
                if update_mask[i]:
                    new_stepsizes.append(jax.nn.relu(s - eta_t * (ds + weight_decay * s)))
                else:
                    new_stepsizes.append(s)
            stepsizes = tuple(new_stepsizes)
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

        all_stepsizes_vals.append(stepsizes)

        # Save progress
        df = build_stepsizes_df_pdlp(all_stepsizes_vals, K_max, is_vector, all_losses, all_times)
        df.to_csv(csv_path, index=False)

    # Final save
    df = build_stepsizes_df_pdlp(all_stepsizes_vals, K_max, is_vector, all_losses, all_times)
    tau, sigma, theta = stepsizes
    if is_vector:
        tau_str = '[' + ', '.join(f'{x:.5f}' for x in tau.tolist()) + ']'
        sigma_str = '[' + ', '.join(f'{x:.5f}' for x in sigma.tolist()) + ']'
        theta_str = '[' + ', '.join(f'{x:.5f}' for x in theta.tolist()) + ']'
    else:
        tau_str = f'{float(tau):.6f}'
        sigma_str = f'{float(sigma):.6f}'
        theta_str = f'{float(theta):.6f}'

    log.info(f'K={K_max} complete. Final tau={tau_str}, sigma={sigma_str}, theta={theta_str}. Saved to {csv_path}')
    df.to_csv(csv_path, index=False)


def pdlp_run(cfg):
    log.info("=" * 60)
    log.info("Starting PDLP learning experiment")
    log.info("=" * 60)
    log.info(cfg)

    optimizer_type = cfg.optimizer_type
    sgd_iters = cfg.sgd_iters
    eta_t = cfg.eta_t
    eps = cfg.eps
    alpha = cfg.alpha
    N_val = cfg.N

    seed = cfg.sgd_seed
    key = jax.random.PRNGKey(seed)

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for K in cfg.K_max:
        K_output_dir = os.path.join(output_dir, f"K_{K}")
        os.makedirs(K_output_dir, exist_ok=True)
        csv_path = os.path.join(K_output_dir, "progress.csv")
        
        # Select run function based on learning framework
        learning_framework = cfg.learning_framework

        if learning_framework == 'lpep':
            # LPEP: deterministic PEP minimization (no samples, no DRO)
            # run_gd_for_K_lpep_lasso(
            #     cfg, K, problem_data, gamma_init, sgd_iters, eta_t,
            #     alg, csv_path
            # )
            raise NotImplementedError
        elif learning_framework == 'l2o':
            # L2O doesn't need the preconditioner
            precond_inv = None
        elif learning_framework == 'ldro-pep':
            ws_key, key = jax.random.split(key)
            ws_problem = generate_batch_problem_jax(key, cfg.n_facilities, cfg.n_customers, 1, cfg)
            c_b, Aeq_b, beq_b, Aineq_b, bineq_b, lb_b, ub_b = extract_constraint_matrices(
                ws_problem["fixed_costs"][0],
                ws_problem["capacities"][0],
                ws_problem["demands"][0],
                ws_problem["transportation_costs"][0],
                cfg.n_facilities,
                cfg.n_customers,
            )
            # Infer variable dimensions from extracted matrices
            m1 = Aineq_b.shape[0]
            m2 = Aeq_b.shape[0]
            n_vars = c_b.shape[0]
            
            pdlp_dpp = FacilityLocationDPP(n_vars, m1, m2)

            if cfg.warmstart:
                x_ws, y_ws = pdlp_dpp.solve(
                    np.array(c_b), 
                    np.array(Aineq_b), 
                    np.array(bineq_b), 
                    np.array(Aeq_b), 
                    np.array(beq_b), 
                    np.array(lb_b), 
                    np.array(ub_b)
                )
            else:
                x_ws = jnp.zeros(n_vars)
                y_ws = jnp.zeros(m1 + m2)

            precond_sample_size = cfg.get('precond_sample_size', 100)
            log.info(f"Precomputing preconditioner using {precond_sample_size} samples...")

            precond_key, key = jax.random.split(key)
            problem_batch, x_opt_batch, y_opt_batch, R_max, Dnorm_max = sample_pdlp_batch(
                key,
                cfg.n_facilities,
                cfg.n_customers,
                precond_sample_size,
                cfg,
                pdlp_dpp=pdlp_dpp,
                x_opt_benchmark=x_ws,
                y_opt_benchmark=y_ws,
            )
            log.info(f'R_max: {R_max}')
            log.info(f'Dnorm_max: {Dnorm_max}')

            # Create batches of initial points (copies of warm-start solution)
            x0_batch_precond = jnp.tile(x_ws, (precond_sample_size, 1))
            y0_batch_precond = jnp.tile(y_ws, (precond_sample_size, 1))

            tau_0 = 0.75 / Dnorm_max
            sigma_0 = 0.75 / Dnorm_max
            theta_0 = 1.0
            stepsizes_precond_tuple = (jnp.tile(tau_0, K), jnp.tile(sigma_0, K), jnp.tile(theta_0, K))

            def traj_fn_precond(c, A_ineq, b_ineq, A_eq, b_eq, l, u, x0, y0, x_opt, y_opt):
                # Stack D and q
                D = jnp.vstack([A_ineq, A_eq])
                q = jnp.concatenate([b_ineq, b_eq])
                return problem_data_to_pdhg_trajectories(
                    stepsizes_precond_tuple, c, D, q, l, u, x0, y0, x_opt, y_opt,
                    K_max=cfg.K_max[0], m1=m1, M=Dnorm_max
                )
            
            extractor_vmap = jax.vmap(
                partial(extract_constraint_matrices, n_facilities=cfg.n_facilities, n_customers=cfg.n_customers),
                in_axes=(0, 0, 0, 0)
            )
            c_batch, Aeq_batch, beq_batch, Aineq_batch, bineq_batch, lb_batch, ub_batch = extractor_vmap(
                problem_batch["fixed_costs"],
                problem_batch["capacities"],
                problem_batch["demands"],
                problem_batch["transportation_costs"]
            )
            
            # Compute trajectories
            batch_traj_fn = jax.vmap(traj_fn_precond, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

            G_batch, F_batch = batch_traj_fn(
                c_batch, Aineq_batch, bineq_batch, Aeq_batch, beq_batch, lb_batch, ub_batch,
                x0_batch_precond, y0_batch_precond, x_opt_batch, y_opt_batch
            )

            # Compute preconditioner
            precond_inv = compute_preconditioner_from_samples(G_batch, F_batch, cfg.precond_type)
            log.info(f"Preconditioner computed. (type={cfg.precond_type})")

            log.info(precond_inv)
            
            # Clean up
            del problem_batch, x_opt_batch, y_opt_batch, G_batch, F_batch
        else:
            raise ValueError(f"Unknown learning_framework: {learning_framework}")

        # Initialize step sizes
        # For PDHG convergence: tau * sigma * ||D||^2 < 1
        # Default: tau = sigma = 0.75 / ||D||, theta = 1.0
        is_vector = cfg.stepsize_type == "vector"
        vector_init = cfg.get('vector_init', 'fixed')

        tau_0 = 0.75 / Dnorm_max
        sigma_0 = 0.75 / Dnorm_max
        theta_0 = 1.0

        if is_vector:
            if vector_init == 'fixed':
                tau_init = jnp.full(K, tau_0)
                sigma_init = jnp.full(K, sigma_0)
                theta_init = jnp.full(K, theta_0)
                log.info(f"Using fixed uniform initialization: tau_k={tau_0:.6f}, sigma_k={sigma_0:.6f}, theta_k={theta_0:.6f}")
            else:
                # TODO: Add other initialization schemes (e.g., silver-like schedules for PDHG)
                raise NotImplementedError(f"Vector init '{vector_init}' not implemented for PDLP")
        else:
            tau_init = jnp.array([tau_0])
            sigma_init = jnp.array([sigma_0])
            theta_init = jnp.array([theta_0])

        stepsizes_init = (tau_init, sigma_init, theta_init)

        # Run SGD
        R_max *= 1.2
        run_sgd_for_K_pdlp(
            cfg, K, key,
            stepsizes_init, sgd_iters, eta_t,
            eps, alpha, optimizer_type,
            N_val, csv_path, precond_inv,
            pdlp_dpp, x_ws, y_ws, m1, n_vars, Dnorm_max, R_max
        )

    log.info("=== PDLP SGD experiment complete ===")
