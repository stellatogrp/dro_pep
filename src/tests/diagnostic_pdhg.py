"""
Diagnostic script for PDHG PEP constraint analysis.

This script helps debug why trajectory Gram matrices may not satisfy PEP constraints.
Run with: python -m tests.diagnostic_pdhg
"""

import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp

jax.config.update('jax_enable_x64', True)

from learning.trajectories_pdhg import problem_data_to_pdhg_trajectories
from learning.pep_construction_chambolle_pock import (
    construct_chambolle_pock_pep_data,
    chambolle_pock_pep_data_to_numpy,
)


def generate_random_lp(n, m1, m2, seed=42):
    """Generate a random LP problem."""
    np.random.seed(seed)

    c = np.random.randn(n)
    G_ineq = np.random.randn(m1, n) / np.sqrt(m1)
    A_eq = np.random.randn(m2, n) / np.sqrt(m2)

    x_feas = np.random.rand(n)
    h = G_ineq @ x_feas - np.abs(np.random.randn(m1)) * 0.5
    b = A_eq @ x_feas

    l, u = np.zeros(n), np.ones(n)

    return c, G_ineq, h, A_eq, b, l, u


def solve_lp(c, G_ineq, h, A_eq, b, l, u):
    """Solve LP and return primal/dual solutions."""
    n = c.shape[0]
    x_var = cp.Variable(n)
    constraints = [
        G_ineq @ x_var >= h,
        A_eq @ x_var == b,
        x_var >= l,
        x_var <= u
    ]
    prob = cp.Problem(cp.Minimize(c @ x_var), constraints)
    prob.solve(solver='CLARABEL', verbose=False)

    x_opt = x_var.value
    # Our Lagrangian: L = c^T x + y^T(q - Kx) uses opposite sign for equality duals
    y_opt = np.concatenate([constraints[0].dual_value, -constraints[1].dual_value])
    f_opt = prob.value

    return x_opt, y_opt, f_opt


def run_diagnostic(n=5, m1=3, m2=2, K_iter=3, seed=42):
    """Run full diagnostic analysis."""
    print("=" * 70)
    print("PDHG PEP Constraint Diagnostic")
    print("=" * 70)

    # Generate LP
    c, G_ineq, h, A_eq, b, l, u = generate_random_lp(n, m1, m2, seed)
    K_mat = np.vstack([G_ineq, A_eq])
    q = np.concatenate([h, b])

    # Solve LP
    x_opt, y_opt, f_opt = solve_lp(c, G_ineq, h, A_eq, b, l, u)

    # Problem parameters
    M = np.linalg.norm(K_mat, ord=2)
    tau = 0.9 / M
    sigma = 0.9 / M
    theta = 1.0

    x0 = 0.5 * (l + u)
    y0 = np.zeros(m1 + m2)
    y0[:m1] = 0.1

    delta_x0 = x0 - x_opt
    delta_y0 = y0 - y_opt
    R = np.sqrt(np.dot(delta_x0, delta_x0) + np.dot(delta_y0, delta_y0))

    print(f"\nProblem Setup:")
    print(f"  n = {n}, m1 = {m1}, m2 = {m2}, K_iter = {K_iter}")
    print(f"  M = ||K||_2 = {M:.6f}")
    print(f"  tau = {tau:.6f}, sigma = {sigma:.6f}, theta = {theta:.6f}")
    print(f"  R = {R:.6f}")
    print(f"  tau * sigma * M^2 = {tau * sigma * M**2:.6f} (should be < 1)")

    # Get trajectory Gram matrix
    stepsizes = (jnp.array(tau), jnp.array(sigma), jnp.array(theta))
    G_traj, F_traj = problem_data_to_pdhg_trajectories(
        stepsizes,
        jnp.array(c), jnp.array(K_mat), jnp.array(q),
        jnp.array(l), jnp.array(u),
        jnp.array(x0), jnp.array(y0),
        jnp.array(x_opt), jnp.array(y_opt), f_opt,
        K_iter, m1, return_Gram_representation=True,
        M=M
    )

    # Get PEP constraints
    pep_data = construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K_iter)
    pep_data_np = chambolle_pock_pep_data_to_numpy(pep_data)
    A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data_np

    G_np = np.array(G_traj)
    F_np = np.array(F_traj)

    print(f"\nDimensions:")
    print(f"  G shape: {G_np.shape} (expected: {(2*K_iter+6, 2*K_iter+6)})")
    print(f"  F shape: {F_np.shape} (expected: {(2*(K_iter+2),)})")
    print(f"  A_vals shape: {A_vals.shape}")
    print(f"  b_vals shape: {b_vals.shape}")

    # Check G properties
    eigvals = np.linalg.eigvalsh(G_np)
    print(f"\nGram Matrix Properties:")
    print(f"  Symmetric: {np.allclose(G_np, G_np.T, atol=1e-10)}")
    print(f"  Eigenvalues: min = {eigvals.min():.6f}, max = {eigvals.max():.6f}")
    print(f"  PSD: {eigvals.min() >= -1e-8}")

    # Key G entries
    print(f"\nKey Gram Matrix Entries:")
    print(f"  G[0,0] = ||delta_x0||^2 = {G_np[0,0]:.6f}")
    print(f"    Expected: {np.dot(delta_x0, delta_x0):.6f}")
    print(f"  G[1,1] = ||delta_y0||^2 = {G_np[1,1]:.6f}")
    print(f"    Expected: {np.dot(delta_y0, delta_y0):.6f}")
    print(f"  G[2,2] = ||xs||^2 = {G_np[2,2]:.6f}")
    print(f"    Expected: {np.dot(x_opt, x_opt):.6f}")
    print(f"  G[3,3] = ||ys||^2 = {G_np[3,3]:.6f}")
    print(f"    Expected: {np.dot(y_opt, y_opt):.6f}")

    # Cross terms
    print(f"\nCross Inner Products (bilinear form W):")
    print(f"  G[0,1] = <delta_x0, delta_y0>_W = {G_np[0,1]:.6f}")
    expected_01 = -np.dot(delta_x0, K_mat.T @ delta_y0) / M
    print(f"    Expected: -{delta_x0}^T K^T {delta_y0} / M = {expected_01:.6f}")

    print(f"  G[0,3] = <delta_x0, ys>_W = {G_np[0,3]:.6f}")
    expected_03 = -np.dot(delta_x0, K_mat.T @ y_opt) / M
    print(f"    Expected: {expected_03:.6f}")

    print(f"  G[2,3] = <xs, ys>_W = {G_np[2,3]:.6f}")
    expected_23 = -np.dot(x_opt, K_mat.T @ y_opt) / M
    print(f"    Expected: {expected_23:.6f}")

    # Function values
    dimF1 = K_iter + 2
    F1 = F_np[:dimF1]
    F_h = F_np[dimF1:]

    print(f"\nFunction Values:")
    print(f"  F1 (f1 values): {F1}")
    print(f"  F_h (h values): {F_h}")
    print(f"  F1[-1] (at optimal): {F1[-1]:.6f} (should be 0)")
    print(f"  F_h[-1] (at optimal): {F_h[-1]:.6f} (should be 0)")

    # Check each constraint
    print(f"\n" + "=" * 70)
    print("Constraint Analysis")
    print("=" * 70)

    violations = []
    violated_indices = []
    for i in range(A_vals.shape[0]):
        val = np.trace(A_vals[i] @ G_np) + np.dot(b_vals[i], F_np) + c_vals[i]
        violations.append(val)
        if val > 1e-6:
            violated_indices.append(i)

    violations = np.array(violations)
    print(f"\nSummary:")
    print(f"  Total constraints: {len(violations)}")
    print(f"  Violated (> 1e-6): {len(violated_indices)}")
    print(f"  Max violation: {violations.max():.6f}")
    print(f"  Min value: {violations.min():.6f}")

    if violated_indices:
        print(f"\nViolated Constraints (showing first 10):")
        for i in violated_indices[:10]:
            val = violations[i]
            trace_part = np.trace(A_vals[i] @ G_np)
            bf_part = np.dot(b_vals[i], F_np)
            c_part = c_vals[i]

            print(f"\n  Constraint {i}: violation = {val:.6f}")
            print(f"    trace(A@G) = {trace_part:.6f}")
            print(f"    b@F = {bf_part:.6f}")
            print(f"    c = {c_part:.6f}")

            # Non-zero entries in A
            nonzero_A = np.argwhere(np.abs(A_vals[i]) > 1e-10)
            if len(nonzero_A) > 0:
                print(f"    Non-zero A[i,j] entries (up to 10):")
                for idx in nonzero_A[:10]:
                    print(f"      A[{idx[0]},{idx[1]}] = {A_vals[i][idx[0], idx[1]]:.6f}, G[{idx[0]},{idx[1]}] = {G_np[idx[0], idx[1]]:.6f}")

            # Non-zero entries in b
            nonzero_b = np.argwhere(np.abs(b_vals[i]) > 1e-10).flatten()
            if len(nonzero_b) > 0:
                print(f"    Non-zero b entries: {nonzero_b.tolist()}")
                for j in nonzero_b[:5]:
                    print(f"      b[{j}] = {b_vals[i][j]:.6f}, F[{j}] = {F_np[j]:.6f}")

    # Also check what happens with a working test case (smaller problem)
    print(f"\n" + "=" * 70)
    print("Comparison with Simpler Problem (n=4, m1=2, m2=1, K=2)")
    print("=" * 70)

    c2, G2, h2, A2, b2, l2, u2 = generate_random_lp(4, 2, 1, seed=100)
    K2 = np.vstack([G2, A2])
    q2 = np.concatenate([h2, b2])
    x_opt2, y_opt2, f_opt2 = solve_lp(c2, G2, h2, A2, b2, l2, u2)

    M2 = np.linalg.norm(K2, ord=2)
    tau2, sigma2, theta2 = 0.9/M2, 0.9/M2, 1.0
    x02 = 0.5 * (l2 + u2)
    y02 = np.zeros(3)
    y02[:2] = 0.1
    R2 = np.sqrt(np.sum((x02 - x_opt2)**2) + np.sum((y02 - y_opt2)**2))

    stepsizes2 = (jnp.array(tau2), jnp.array(sigma2), jnp.array(theta2))
    G_traj2, F_traj2 = problem_data_to_pdhg_trajectories(
        stepsizes2,
        jnp.array(c2), jnp.array(K2), jnp.array(q2),
        jnp.array(l2), jnp.array(u2),
        jnp.array(x02), jnp.array(y02),
        jnp.array(x_opt2), jnp.array(y_opt2), f_opt2,
        2, 2, return_Gram_representation=True,
        M=M2
    )

    pep_data2 = construct_chambolle_pock_pep_data(tau2, sigma2, theta2, M2, R2, 2)
    pep_data_np2 = chambolle_pock_pep_data_to_numpy(pep_data2)
    A_vals2, b_vals2, c_vals2 = pep_data_np2[2], pep_data_np2[3], pep_data_np2[4]

    G_np2 = np.array(G_traj2)
    F_np2 = np.array(F_traj2)

    violations2 = []
    for i in range(A_vals2.shape[0]):
        val = np.trace(A_vals2[i] @ G_np2) + np.dot(b_vals2[i], F_np2) + c_vals2[i]
        violations2.append(val)

    violations2 = np.array(violations2)
    print(f"  Total constraints: {len(violations2)}")
    print(f"  Violated (> 1e-6): {np.sum(violations2 > 1e-6)}")
    print(f"  Max violation: {violations2.max():.6f}")

    return G_np, F_np, A_vals, b_vals, c_vals, violations


if __name__ == '__main__':
    run_diagnostic()
