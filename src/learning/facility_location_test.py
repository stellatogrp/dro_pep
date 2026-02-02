"""
Capacitated Facility Location Problem - LP Relaxation with Matrix Extraction (JAX)

This module generates the LP relaxation of the Capacitated Facility Location Problem
and extracts the constraint matrices in standard form, designed for JAX autodiff.

Problem formulation:
    minimize    sum_{i in F} f_i * y_i + sum_{i in F, j in C} c_{ij} * x_{ij}
    subject to  sum_{i in F} x_{ij} = 1                 for all j in C    (demand satisfaction)
                sum_{j in C} d_j * x_{ij} <= s_i * y_i  for all i in F    (capacity)
                x_{ij} <= y_i                           for all i,j       (linking)
                0 <= y_i <= 1                           for all i in F    (relaxed binary)
                0 <= x_{ij} <= 1                        for all i,j       (bounds)

Variable ordering: [y_1, ..., y_m, x_{11}, x_{12}, ..., x_{1n}, x_{21}, ..., x_{mn}]
                   where m = n_facilities, n = n_customers
                   Total variables: m + m*n
"""

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from typing import NamedTuple
from functools import partial

jnp.set_printoptions(suppress=True, precision=5)

class FacilityLocationMatrices(NamedTuple):
    """Container for the extracted constraint matrices in standard form."""
    # Objective
    c: jax.Array  # Cost vector (m + m*n,)
    
    # Equality constraints: A_eq @ x = b_eq
    A_eq: jax.Array  # (n_customers, m + m*n)
    b_eq: jax.Array  # (n_customers,)
    
    # Inequality constraints: A_ineq @ x <= b_ineq
    A_ineq: jax.Array  # (m + m*n, m + m*n)
    b_ineq: jax.Array  # (m + m*n,)
    
    # Bound constraints: lb <= x <= ub
    lb: jax.Array  # (m + m*n,)
    ub: jax.Array  # (m + m*n,)


def generate_facility_location_problem(
    n_facilities: int,
    random_seed: int,
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
    rng = np.random.default_rng(random_seed)
    
    if n_customers is None:
        n_customers = 10 * n_facilities
    
    fixed_costs = rng.uniform(1, 2, n_facilities).astype(np.float32)
    demands = rng.uniform(0.5, 1.5, n_customers).astype(np.float32)
    transportation_costs = rng.uniform(0.1, 1.0, (n_facilities, n_customers)).astype(np.float32)

    avg_demand_per_facility = (demands.sum() / n_facilities)
    # base_capacities = rng.uniform(1.0, 2.0, n_facilities).astype(np.float32)
    base_capacities = 2.0 * np.ones(n_facilities)
    # scale capacities
    capacities = base_capacities * avg_demand_per_facility * 1.5

    return {
        "fixed_costs": jnp.array(fixed_costs),
        "capacities": jnp.array(capacities),
        "demands": jnp.array(demands),
        "transportation_costs": jnp.array(transportation_costs),
        "n_facilities": n_facilities,
        "n_customers": n_customers,
    }


def extract_constraint_matrices(
    fixed_costs: jax.Array,
    capacities: jax.Array,
    demands: jax.Array,
    transportation_costs: jax.Array,
    n_facilities: int,
    n_customers: int,
) -> FacilityLocationMatrices:
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
    
    return FacilityLocationMatrices(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        lb=lb,
        ub=ub,
    )


def make_matrix_extractor(n_facilities: int, n_customers: int):
    """
    Create a closure for extracting constraint matrices with fixed dimensions.
    
    This returns a function that takes only the problem parameters and returns
    the constraint matrices. Useful for JAX transformations like vmap and grad.
    
    Example usage:
        extractor = make_matrix_extractor(n_facilities=5, n_customers=20)
        matrices = extractor(fixed_costs, capacities, demands, transportation_costs)
        
        # For batched problems:
        batch_extractor = jax.vmap(extractor)
        batch_matrices = batch_extractor(batch_fixed_costs, batch_capacities, 
                                          batch_demands, batch_transportation_costs)
    
    Args:
        n_facilities: Number of facilities (m)
        n_customers: Number of customers (n)
        
    Returns:
        A function (fixed_costs, capacities, demands, transportation_costs) -> matrices
    """
    return partial(
        extract_constraint_matrices,
        n_facilities=n_facilities,
        n_customers=n_customers,
    )


def solve_relaxed_lp(matrices: FacilityLocationMatrices, n_facilities: int, n_customers: int) -> dict:
    """
    Solve the LP relaxation using CVXPY (for verification, not JAX-compatible).
    
    Args:
        matrices: FacilityLocationMatrices from extract_constraint_matrices.
        n_facilities: Number of facilities.
        n_customers: Number of customers.
        
    Returns:
        Dictionary with objective value, y values, and x values.
    """
    import cvxpy as cp
    
    n_vars = n_facilities + n_facilities * n_customers
    x = cp.Variable(n_vars)
    
    # Convert JAX arrays to numpy for CVXPY
    c_np = np.asarray(matrices.c)
    A_eq_np = np.asarray(matrices.A_eq)
    b_eq_np = np.asarray(matrices.b_eq)
    A_ineq_np = np.asarray(matrices.A_ineq)
    b_ineq_np = np.asarray(matrices.b_ineq)
    lb_np = np.asarray(matrices.lb)
    ub_np = np.asarray(matrices.ub)
    
    objective = cp.Minimize(c_np @ x)
    
    constraints = [
        -A_ineq_np @ x >= -b_ineq_np,
        A_eq_np @ x == b_eq_np,
        x >= lb_np,
        x <= ub_np,
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    raw_y = np.concatenate([constraints[0].dual_value, -constraints[1].dual_value])

    print('raw LP x:', x.value)
    print('raw LP y:', raw_y)
    
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"LP solve failed with status: {prob.status}")
    
    m = n_facilities
    n = n_customers
    
    y_vals = x.value[:m]
    x_vals = x.value[m:].reshape(m, n)
    
    return {
        "objective_value": prob.value,
        "y": y_vals,
        "x": x_vals,
        'raw_x': x.value,
        'raw_y': raw_y,
    }


def verify_matrices(problem: dict, matrices: FacilityLocationMatrices) -> bool:
    """
    Verify the extracted matrices match a direct CVXPY formulation.
    
    Args:
        problem: Original problem dictionary.
        matrices: Extracted matrices.
        
    Returns:
        True if both formulations give the same objective value.
    """
    import cvxpy as cp
    
    m = problem["n_facilities"]
    n = problem["n_customers"]
    
    # Solve using extracted matrices
    sol_matrices = solve_relaxed_lp(matrices, m, n)
    
    # Solve directly with CVXPY
    fixed_costs = np.asarray(problem["fixed_costs"])
    capacities = np.asarray(problem["capacities"])
    demands = np.asarray(problem["demands"])
    transportation_costs = np.asarray(problem["transportation_costs"])
    
    y = cp.Variable(m)
    x = cp.Variable((m, n))
    
    objective = cp.Minimize(fixed_costs @ y + cp.sum(cp.multiply(transportation_costs, x)))
    
    constraints = []
    # Demand satisfaction
    for j in range(n):
        constraints.append(cp.sum(x[:, j]) == 1)
    # Capacity
    for i in range(m):
        constraints.append(demands @ x[i, :] <= capacities[i] * y[i])
        # Linking
        for j in range(n):
            constraints.append(x[i, j] <= y[i])
    # Bounds
    constraints.append(y >= 0)
    constraints.append(y <= 1)
    constraints.append(x >= 0)
    constraints.append(x <= 1)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    
    print(f"Matrix formulation objective: {sol_matrices['objective_value']:.6f}")
    print(f"Direct CVXPY objective:       {prob.value:.6f}")
    print(f"Difference:                   {abs(sol_matrices['objective_value'] - prob.value):.2e}")
    
    return np.isclose(sol_matrices['objective_value'], prob.value, rtol=1e-5)


def run_PHDG(c, G, h, A, b, l, u, raw_xs, raw_ys, n_facilites, n_customers):
    K = jnp.vstack([G, A])
    q = jnp.concatenate([h, b])
    m, n = K.shape
    m1 = G.shape[0]
    m2 = A.shape[0]
    assert m1 + m2 == m

    def satlin(v):
        return jnp.minimum(u, jnp.maximum(v, l))
    
    def partial_relu(v):
        return v.at[:m1].set(jax.nn.relu(v[:m1]))

    def lagrangian(x, y):
        return c.T @ x - y.T @ K @ x + q.T @ y

    def lagrangian_gap(xk, yk):
        primal = lagrangian(xk, raw_ys)
        dual = lagrangian(raw_xs, yk)
        return primal - dual

    xk, yk = get_benchmark_solution(n_facilites, n_customers)
    R = jnp.linalg.norm(jnp.concatenate([xk - raw_xs, yk - raw_ys]))
    print('benchmark to opt norm:', R)

    M = jnp.linalg.norm(K, ord=2)
    print('M:', M)
    tau = 0.9 / M
    sigma = 0.9 / M
    theta = 1

    print(K.shape, q.shape)

    K_max = 15

    for _ in trange(K_max):
        print('lagrangian gap loss:', lagrangian_gap(xk, yk))

        xkplus1 = satlin(xk - tau * (c - K.T @ yk))
        xbar = xkplus1 + theta * (xkplus1 - xk)
        ykplus1 = partial_relu(yk + sigma * (q - K @ xbar))

        xk = xkplus1
        yk = ykplus1
    
    # print('PDHG x:', xk)
    print(c.T @ xk)


def get_benchmark_solution(n_facilities, n_customers, random_seed=100):
    """
    Generates a problem, solves it via CVXPY, and returns the raw primal 
    and dual variables formatted for the PDHG solver.
    
    Args:
        n_facilities: Number of facilities.
        n_customers: Number of customers.
        random_seed: Seed for reproducibility.
        
    Returns:
        raw_x: The optimal primal solution (n_vars,).
        raw_y: The optimal dual solution (m_ineq + m_eq,), with equality duals negated.
    """
    # 1. Generate Data
    problem = generate_facility_location_problem(
        n_facilities=n_facilities, 
        n_customers=n_customers, 
        random_seed=random_seed
    )
    
    # 2. Extract Matrices (Reuse existing extractor)
    extractor = make_matrix_extractor(n_facilities, n_customers)
    matrices = extractor(
        problem["fixed_costs"],
        problem["capacities"],
        problem["demands"],
        problem["transportation_costs"]
    )
    
    sol = solve_relaxed_lp(matrices, n_facilities, n_customers)
    
    return sol['raw_x'], sol['raw_y']


if __name__ == "__main__":
    # Generate a small test problem
    print("=" * 60)
    print("Generating Facility Location Problem")
    print("=" * 60)

    n_facilities = 5
    n_customers = 20
    
    problem = generate_facility_location_problem(
        n_facilities=n_facilities,
        random_seed=42,
        n_customers=n_customers,
    )
    
    m = problem["n_facilities"]
    n = problem["n_customers"]
    
    print(f"Number of facilities: {m}")
    print(f"Number of customers:  {n}")
    
    # Extract matrices using the closure approach
    print("\n" + "=" * 60)
    print("Extracting Constraint Matrices (JAX)")
    print("=" * 60)
    
    # Create the extractor closure
    extractor = make_matrix_extractor(n_facilities=m, n_customers=n)
    
    # Extract matrices
    matrices = extractor(
        problem["fixed_costs"],
        problem["capacities"],
        problem["demands"],
        problem["transportation_costs"],
    )
    
    n_vars = m + m * n
    print(f"Number of variables:           {n_vars}")
    print(f"Cost vector c shape:           {matrices.c.shape}")
    print(f"Equality A_eq shape:           {matrices.A_eq.shape}")
    print(f"Equality b_eq shape:           {matrices.b_eq.shape}")
    print(f"Inequality A_ineq shape:       {matrices.A_ineq.shape}")
    print(f"Inequality b_ineq shape:       {matrices.b_ineq.shape}")
    print(f"Lower bounds lb shape:         {matrices.lb.shape}")
    print(f"Upper bounds ub shape:         {matrices.ub.shape}")
    
    # Verify JAX types
    print(f"\nArray types are JAX: {isinstance(matrices.c, jax.Array)}")
    
    # Verify matrices
    print("\n" + "=" * 60)
    print("Verifying Matrix Formulation")
    print("=" * 60)
    
    is_correct = verify_matrices(problem, matrices)
    print(f"Verification passed: {is_correct}")
    
    # Solve and display results
    print("\n" + "=" * 60)
    print("LP Relaxation Solution")
    print("=" * 60)
    
    solution = solve_relaxed_lp(matrices, m, n)
    print(f"Optimal objective: {solution['objective_value']:.2f}")
    print(f"\nFacility opening values (y):")
    for i, y_i in enumerate(solution['y']):
        print(f"  Facility {i}: {y_i:.4f}")
    
    # Demonstrate JAX autodiff compatibility
    print("\n" + "=" * 60)
    print("Testing JAX Autodiff Compatibility")
    print("=" * 60)
    
    # Simple test: gradient of sum of objective coefficients w.r.t. fixed_costs
    def sum_objective_coeffs(fixed_costs, capacities, demands, transportation_costs):
        matrices = extractor(fixed_costs, capacities, demands, transportation_costs)
        return jnp.sum(matrices.c)
    
    grad_fn = jax.grad(sum_objective_coeffs)
    grads = grad_fn(
        problem["fixed_costs"],
        problem["capacities"],
        problem["demands"],
        problem["transportation_costs"],
    )
    print(f"Gradient of sum(c) w.r.t. fixed_costs: {grads}")
    print("(Should be all 1s since c starts with fixed_costs)")

    print("\n" + "=" * 60)
    print("Testing PDHG")
    print("=" * 60)

    G = -matrices.A_ineq
    h = -matrices.b_ineq

    A = matrices.A_eq
    b = matrices.b_eq

    c = matrices.c
    l = matrices.lb
    u = matrices.ub

    run_PHDG(c, G, h, A, b, l, u, solution['raw_x'], solution['raw_y'], n_facilities, n_customers)
