"""
Capacitated Facility Location Problem - LP Relaxation with Matrix Extraction

This module generates the LP relaxation of the Capacitated Facility Location Problem
and extracts the constraint matrices in standard form.

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

import numpy as np
from typing import Any
from dataclasses import dataclass


@dataclass
class FacilityLocationMatrices:
    """Container for the extracted constraint matrices in standard form."""
    # Objective
    c: np.ndarray  # Cost vector (m + m*n,)
    
    # Equality constraints: A_eq @ x = b_eq
    A_eq: np.ndarray  # (n_customers, m + m*n)
    b_eq: np.ndarray  # (n_customers,)
    
    # Inequality constraints: A_ineq @ x <= b_ineq
    A_ineq: np.ndarray  # (m + m*n, m + m*n)
    b_ineq: np.ndarray  # (m + m*n,)
    
    # Bound constraints: lb <= x <= ub
    lb: np.ndarray  # (m + m*n,)
    ub: np.ndarray  # (m + m*n,)
    
    # Problem dimensions
    n_facilities: int
    n_customers: int
    n_vars: int


def generate_facility_location_problem(
    n_facilities: int,
    random_seed: int,
    n_customers: int | None = None,
) -> dict[str, Any]:
    """
    Generates a random Capacitated Facility Location problem instance.

    Args:
        n_facilities: Number of facilities.
        random_seed: Seed for the random number generator.
        n_customers: Number of customers (default: 10 * n_facilities).

    Returns:
        A dictionary containing fixed_costs, capacities, demands, and transportation_costs.
    """
    rng = np.random.default_rng(random_seed)
    
    if n_customers is None:
        n_customers = 10 * n_facilities
    
    fixed_costs = rng.uniform(100, 500, n_facilities)
    demands = rng.uniform(10, 40, n_customers)
    transportation_costs = rng.uniform(5, 30, (n_facilities, n_customers))
    capacities = rng.uniform(50, 150, n_facilities)
    
    # Scale capacities to ensure feasibility
    capacities *= 5.0 * demands.sum() / capacities.sum()

    return {
        "fixed_costs": fixed_costs,
        "capacities": capacities,
        "demands": demands,
        "transportation_costs": transportation_costs,
        "n_facilities": n_facilities,
        "n_customers": n_customers,
    }


def extract_constraint_matrices(problem: dict[str, Any]) -> FacilityLocationMatrices:
    """
    Extract constraint matrices from the facility location problem.
    
    Variable ordering: [y_1, ..., y_m, x_{11}, x_{12}, ..., x_{1n}, x_{21}, ..., x_{mn}]
    
    Constraints:
        1. Equality: sum_{i} x_{ij} = 1 for all j (demand satisfaction)
        2. Inequality: sum_{j} d_j * x_{ij} - s_i * y_i <= 0 for all i (capacity)
        3. Inequality: x_{ij} - y_i <= 0 for all i,j (linking)
        4. Bounds: 0 <= y_i, x_{ij} <= 1
    
    Args:
        problem: Problem dictionary from generate_facility_location_problem.
        
    Returns:
        FacilityLocationMatrices dataclass with all extracted matrices.
    """
    fixed_costs = problem["fixed_costs"]
    capacities = problem["capacities"]
    demands = problem["demands"]
    transportation_costs = problem["transportation_costs"]
    m = problem["n_facilities"]  # Number of facilities
    n = problem["n_customers"]   # Number of customers
    
    n_vars = m + m * n  # Total variables: y_i's + x_{ij}'s
    
    # ==========================================================================
    # Objective vector c
    # ==========================================================================
    c = np.concatenate([fixed_costs, transportation_costs.flatten()])
    
    # ==========================================================================
    # Equality constraints: sum_{i} x_{ij} = 1 for all j (demand satisfaction)
    # ==========================================================================
    # For each customer j, sum over facilities i: x_{1j} + x_{2j} + ... + x_{mj} = 1
    # A_eq has shape (n, m + m*n)
    # The x variables are ordered as [x_{11},...,x_{1n}, x_{21},...,x_{2n}, ...]
    # For customer j, we need x_{1j}, x_{2j}, ..., x_{mj} which are at positions
    # m+j, m+n+j, m+2n+j, ..., m+(m-1)n+j
    # This is: I_n tiled m times horizontally
    A_eq_y = np.zeros((n, m))
    A_eq_x = np.tile(np.eye(n), m)  # (n, m*n)
    A_eq = np.hstack([A_eq_y, A_eq_x])
    b_eq = np.ones(n)
    
    # ==========================================================================
    # Inequality constraints: A_ineq @ x <= b_ineq
    # ==========================================================================
    # Two types:
    #   1. Capacity: sum_{j} d_j * x_{ij} - s_i * y_i <= 0  (m constraints)
    #   2. Linking:  x_{ij} - y_i <= 0                       (m * n constraints)
    
    # --- Capacity constraints (m rows) ---
    # For facility i: d^T @ x_i - s_i * y_i <= 0
    # y part: -diag(capacities)  shape (m, m)
    # x part: each row i has demands^T in positions [i*n : (i+1)*n]
    #         This is kron(I_m, demands^T) shape (m, m*n)
    A_cap_y = -np.diag(capacities)  # (m, m)
    A_cap_x = np.kron(np.eye(m), demands.reshape(1, -1))  # (m, m*n)
    A_cap = np.hstack([A_cap_y, A_cap_x])  # (m, m + m*n)
    b_cap = np.zeros(m)
    
    # --- Linking constraints (m*n rows) ---
    # x_{ij} - y_i <= 0 for all i,j
    # Ordered by: (i=0,j=0), (i=0,j=1), ..., (i=0,j=n-1), (i=1,j=0), ...
    # y part: -1 for y_i, repeated n times for each facility
    #         This is kron(I_m, ones(n,1)) with -1
    # x part: I_{m*n}
    A_link_y = -np.kron(np.eye(m), np.ones((n, 1)))  # (m*n, m)
    A_link_x = np.eye(m * n)  # (m*n, m*n)
    A_link = np.hstack([A_link_y, A_link_x])  # (m*n, m + m*n)
    b_link = np.zeros(m * n)
    
    # Stack all inequality constraints
    A_ineq = np.vstack([A_cap, A_link])  # (m + m*n, m + m*n)
    b_ineq = np.concatenate([b_cap, b_link])  # (m + m*n,)
    
    # ==========================================================================
    # Bound constraints: lb <= x <= ub
    # ==========================================================================
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    return FacilityLocationMatrices(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=A_ineq,
        b_ineq=b_ineq,
        lb=lb,
        ub=ub,
        n_facilities=m,
        n_customers=n,
        n_vars=n_vars,
    )


def solve_relaxed_lp(matrices: FacilityLocationMatrices) -> dict[str, Any]:
    """
    Solve the LP relaxation using CVXPY.
    
    Args:
        matrices: FacilityLocationMatrices from extract_constraint_matrices.
        
    Returns:
        Dictionary with objective value, y values, and x values.
    """
    import cvxpy as cp
    
    x = cp.Variable(matrices.n_vars)
    
    objective = cp.Minimize(matrices.c @ x)
    
    constraints = [
        matrices.A_eq @ x == matrices.b_eq,
        matrices.A_ineq @ x <= matrices.b_ineq,
        x >= matrices.lb,
        x <= matrices.ub,
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"LP solve failed with status: {prob.status}")
    
    m = matrices.n_facilities
    n = matrices.n_customers
    
    y_vals = x.value[:m]
    x_vals = x.value[m:].reshape(m, n)
    
    return {
        "objective_value": prob.value,
        "y": y_vals,
        "x": x_vals,
    }


def verify_matrices(problem: dict[str, Any], matrices: FacilityLocationMatrices) -> bool:
    """
    Verify the extracted matrices match a direct CVXPY formulation.
    
    Args:
        problem: Original problem dictionary.
        matrices: Extracted matrices.
        
    Returns:
        True if both formulations give the same objective value.
    """
    import cvxpy as cp
    
    # Solve using extracted matrices
    sol_matrices = solve_relaxed_lp(matrices)
    
    # Solve directly with CVXPY
    fixed_costs = problem["fixed_costs"]
    capacities = problem["capacities"]
    demands = problem["demands"]
    transportation_costs = problem["transportation_costs"]
    m = problem["n_facilities"]
    n = problem["n_customers"]
    
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


if __name__ == "__main__":
    # Generate a small test problem
    print("=" * 60)
    print("Generating Facility Location Problem")
    print("=" * 60)
    
    problem = generate_facility_location_problem(
        n_facilities=5,
        random_seed=42,
        n_customers=20,
    )
    
    print(f"Number of facilities: {problem['n_facilities']}")
    print(f"Number of customers:  {problem['n_customers']}")
    
    # Extract matrices
    print("\n" + "=" * 60)
    print("Extracting Constraint Matrices")
    print("=" * 60)
    
    matrices = extract_constraint_matrices(problem)
    
    print(f"Number of variables:           {matrices.n_vars}")
    print(f"Cost vector c shape:           {matrices.c.shape}")
    print(f"Equality A_eq shape:           {matrices.A_eq.shape}")
    print(f"Equality b_eq shape:           {matrices.b_eq.shape}")
    print(f"Inequality A_ineq shape:       {matrices.A_ineq.shape}")
    print(f"Inequality b_ineq shape:       {matrices.b_ineq.shape}")
    print(f"Lower bounds lb shape:         {matrices.lb.shape}")
    print(f"Upper bounds ub shape:         {matrices.ub.shape}")
    
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
    
    solution = solve_relaxed_lp(matrices)
    print(f"Optimal objective: {solution['objective_value']:.2f}")
    print(f"\nFacility opening values (y):")
    for i, y_i in enumerate(solution['y']):
        print(f"  Facility {i}: {y_i:.4f}")
