"""
Unit test for Chambolle-Pock linear operator interpolation inequalities.

Tests the PEP construction with linear operator interpolation inequalities from
https://arxiv.org/pdf/2302.08781, formulating and solving the resulting SDP
with cvxpy and CLARABEL.
"""
import pytest
import numpy as np
import jax.numpy as jnp
import cvxpy as cp

from learning.pep_constructions import construct_chambolle_pock_pep_data


def test_chambolle_pock_linop_interpolation():
    """
    Test Chambolle-Pock PEP with linear operator interpolation constraints.

    This test:
    1. Constructs interpolation inequalities for Chambolle-Pock
    2. Formulates the SDP in cvxpy
    3. Solves with CLARABEL solver
    4. Verifies the solution is feasible and bounded
    """
    # 1. Set up problem parameters
    K_max = 1  # Number of iterations
    M = 1.0    # Operator norm bound ||K|| <= M
    R = 1.0    # Initial distance bound

    # Step sizes (using standard choice for tau*sigma*M^2 < 1)
    tau = 0.5
    sigma = 0.5
    theta = 1.0

    print(f"\n=== Testing Chambolle-Pock PEP Construction ===")
    print(f"K_max={K_max}, M={M}, R={R}")
    print(f"tau={tau}, sigma={sigma}, theta={theta}")

    # 2. Construct PEP data (interpolation inequalities)
    print("\nConstructing PEP interpolation inequalities...")
    pep_data = construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K_max)

    A_obj, b_obj, A_vals, b_vals, c_vals, PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes = pep_data

    # Convert JAX arrays to numpy for cvxpy
    A_obj_np = np.array(A_obj)
    b_obj_np = np.array(b_obj)
    A_vals_np = np.array(A_vals)
    b_vals_np = np.array(b_vals)
    c_vals_np = np.array(c_vals)

    PSD_A_vals_np = [np.array(A) for A in PSD_A_vals]
    PSD_b_vals_np = [np.array(b) for b in PSD_b_vals]
    PSD_c_vals_np = [np.array(c) for c in PSD_c_vals]

    dimG = A_obj_np.shape[0]
    dimF = b_obj_np.shape[0]
    num_scalar_constraints = A_vals_np.shape[0]
    num_psd_constraints = len(PSD_A_vals_np)

    print(f"\nProblem dimensions:")
    print(f"  dimG (Gram matrix): {dimG} x {dimG}")
    print(f"  dimF (function values): {dimF}")
    print(f"  Scalar linear constraints: {num_scalar_constraints}")
    print(f"  PSD constraints: {num_psd_constraints}")
    for i, size in enumerate(PSD_shapes):
        print(f"    PSD constraint {i}: {size} x {size}")

    # 3. Formulate SDP in cvxpy
    print("\nFormulating SDP in cvxpy...")

    # Decision variables
    G = cp.Variable((dimG, dimG), PSD=True, name="G")
    F = cp.Variable(dimF, name="F")

    # Objective: maximize trace(A_obj @ G) + b_obj^T @ F (PEP finds worst-case)
    objective = cp.Maximize(cp.trace(A_obj_np @ G) + b_obj_np @ F)

    # Constraints
    constraints = []

    # Scalar linear constraints: trace(A_vals[i] @ G) + b_vals[i]^T @ F + c_vals[i] <= 0
    print(f"\nAdding {num_scalar_constraints} scalar inequality constraints...")
    for i in range(num_scalar_constraints):
        constraint = cp.trace(A_vals_np[i] @ G) + b_vals_np[i] @ F + c_vals_np[i] <= 0
        constraints.append(constraint)

    # PSD constraints: H(G, F) = A_psd ⊗ G + b_psd ⊗ F + c_psd ≽ 0
    print(f"Adding {num_psd_constraints} PSD constraints...")
    for idx in range(num_psd_constraints):
        A_psd = PSD_A_vals_np[idx]
        b_psd = PSD_b_vals_np[idx]
        c_psd = PSD_c_vals_np[idx]
        size_H = PSD_shapes[idx]

        print(f"  PSD constraint {idx}: shape {size_H} x {size_H}")
        print(f"    A_psd shape: {A_psd.shape}")
        print(f"    b_psd shape: {b_psd.shape}")
        print(f"    c_psd shape: {c_psd.shape}")

        # Build H(G, F) matrix as a linear combination
        # H[row, col] = sum_{i,j} A_psd[row, col, i, j] * G[i, j]
        #               + sum_k b_psd[row, col, k] * F[k]
        #               + c_psd[row, col]

        # Start with constant term
        H = c_psd.copy()

        # Add contributions from G: sum over all (i,j) of A_psd[:, :, i, j] * G[i, j]
        for i in range(dimG):
            for j in range(dimG):
                H = H + A_psd[:, :, i, j] * G[i, j]

        # Add contributions from F: sum over k of b_psd[:, :, k] * F[k]
        for k in range(dimF):
            H = H + b_psd[:, :, k] * F[k]

        # H must be PSD
        constraints.append(H >> 0)

    # 4. Solve with CLARABEL
    print(f"\nSolving SDP with CLARABEL...")
    print(f"Total constraints: {len(constraints)}")

    problem = cp.Problem(objective, constraints)

    try:
        result = problem.solve(solver=cp.CLARABEL, verbose=True)

        print(f"\n=== Solution Results ===")
        print(f"Status: {problem.status}")
        print(f"Optimal value: {result:.6e}")

        # Verify solution
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print("\n✓ SDP solved successfully!")

            # Check some basic properties
            G_sol = G.value
            F_sol = F.value

            print(f"\nSolution properties:")
            print(f"  G matrix: min eigenvalue = {np.min(np.linalg.eigvalsh(G_sol)):.6e}")
            print(f"  G matrix: max eigenvalue = {np.max(np.linalg.eigvalsh(G_sol)):.6e}")
            print(f"  G matrix: trace = {np.trace(G_sol):.6e}")
            print(f"  F vector: min = {np.min(F_sol):.6e}, max = {np.max(F_sol):.6e}")

            # Verify scalar inequality constraints (should be <= 0)
            max_violation = 0.0
            for i in range(num_scalar_constraints):
                val = np.trace(A_vals_np[i] @ G_sol) + b_vals_np[i] @ F_sol + c_vals_np[i]
                # Violation occurs if val > 0 (should be <= 0)
                max_violation = max(max_violation, val)
            print(f"  Max scalar constraint violation: {max_violation:.6e} (should be <= 0)")

            # Verify PSD constraints
            for idx in range(num_psd_constraints):
                A_psd = PSD_A_vals_np[idx]
                b_psd = PSD_b_vals_np[idx]
                c_psd = PSD_c_vals_np[idx]
                size_H = PSD_shapes[idx]

                H_check = np.zeros((size_H, size_H))
                for row in range(size_H):
                    for col in range(size_H):
                        H_check[row, col] = (
                            np.trace(A_psd[row, col, :, :] @ G_sol) +
                            b_psd[row, col, :] @ F_sol +
                            c_psd[row, col]
                        )

                eigvals = np.linalg.eigvalsh(H_check)
                min_eigval = np.min(eigvals)
                print(f"  PSD constraint {idx}: min eigenvalue = {min_eigval:.6e}")

                if min_eigval < -1e-6:
                    print(f"    WARNING: PSD constraint {idx} violated!")

            # The optimal value should be non-negative (worst-case gap)
            assert result >= -1e-6, f"Optimal value should be non-negative, got {result}"
            assert problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE

        else:
            pytest.fail(f"Solver failed with status: {problem.status}")

    except Exception as e:
        pytest.fail(f"Solver raised exception: {e}")

    print("\n=== Test Passed ===\n")


def test_chambolle_pock_dimension_consistency():
    """Test that dimensions are consistent across different K_max values."""
    print("\n=== Testing Dimension Consistency ===")

    for K_max in [3, 5, 10]:
        pep_data = construct_chambolle_pock_pep_data(
            tau=0.5, sigma=0.5, theta=1.0, M=1.0, R=1.0, K_max=K_max
        )

        A_obj, b_obj, A_vals, b_vals, c_vals, PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes = pep_data

        dimG = A_obj.shape[0]
        dimF = b_obj.shape[0]

        # Check objective consistency
        assert A_obj.shape == (dimG, dimG), f"A_obj shape mismatch for K={K_max}"
        assert b_obj.shape == (dimF,), f"b_obj shape mismatch for K={K_max}"

        # Check scalar constraints consistency
        assert A_vals.shape[1:] == (dimG, dimG), f"A_vals shape mismatch for K={K_max}"
        assert b_vals.shape[1:] == (dimF,), f"b_vals shape mismatch for K={K_max}"
        assert A_vals.shape[0] == b_vals.shape[0] == c_vals.shape[0], f"Scalar constraint count mismatch for K={K_max}"

        # Check PSD constraints consistency
        assert len(PSD_A_vals) == len(PSD_b_vals) == len(PSD_c_vals) == len(PSD_shapes), \
            f"PSD constraint count mismatch for K={K_max}"

        for i in range(len(PSD_A_vals)):
            size_H = PSD_shapes[i]
            assert PSD_A_vals[i].shape == (size_H, size_H, dimG, dimG), \
                f"PSD_A_vals[{i}] shape mismatch for K={K_max}"
            assert PSD_b_vals[i].shape == (size_H, size_H, dimF), \
                f"PSD_b_vals[{i}] shape mismatch for K={K_max}"
            assert PSD_c_vals[i].shape == (size_H, size_H), \
                f"PSD_c_vals[{i}] shape mismatch for K={K_max}"

        print(f"✓ K_max={K_max}: dimG={dimG}, dimF={dimF}, "
              f"scalar_constraints={A_vals.shape[0]}, "
              f"PSD_constraints={len(PSD_A_vals)}")

    print("=== Dimension Consistency Test Passed ===\n")


if __name__ == "__main__":
    test_chambolle_pock_dimension_consistency()
    test_chambolle_pock_linop_interpolation()
