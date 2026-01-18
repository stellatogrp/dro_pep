"""
Unit tests for SCS vectorization ordering.

Tests that the SCS lower-triangle column-major ordering is correct
and differs from Clarabel's upper-triangle column-major ordering.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

from learning.jax_scs_layer import (
    get_scs_lower_tri_indices,
    jax_scs_symm_vectorize,
)


# =============================================================================
# NumPy helper functions for testing (local to tests)
# =============================================================================

def np_get_scs_lower_tri_indices(n):
    """NumPy version using the same triu + flip trick."""
    rows, cols = np.triu_indices(n)
    return cols, rows


def np_scs_symm_vectorize(A, scale_factor):
    """NumPy version of jax_scs_symm_vectorize for testing."""
    n = A.shape[0]
    rows, cols = np_get_scs_lower_tri_indices(n)
    
    A_vec = A[rows, cols].copy()
    off_diag_mask = rows != cols
    A_vec[off_diag_mask] *= scale_factor
    
    return A_vec


def np_scs_mat_from_vec(vec, n, scale_factor):
    """Reconstruct symmetric matrix from SCS-format vector."""
    rows, cols = np_get_scs_lower_tri_indices(n)
    
    off_diag_mask = rows != cols
    vec_unscaled = vec.copy()
    vec_unscaled[off_diag_mask] /= scale_factor
    
    A = np.zeros((n, n))
    A[rows, cols] = vec_unscaled
    A[cols, rows] = vec_unscaled
    
    return A


def compare_ordering_clarabel_vs_scs(n):
    """Compare Clarabel vs SCS vectorization ordering."""
    A = np.arange(n * n).reshape(n, n).astype(float)
    A_sym = (A + A.T) / 2
    
    # Clarabel ordering (upper triangle, column-major via tril + flip)
    clarabel_rows, clarabel_cols = np.tril_indices(n)
    clarabel_rows, clarabel_cols = clarabel_cols, clarabel_rows
    clarabel_vec = A_sym[clarabel_rows, clarabel_cols]
    
    # SCS ordering (lower triangle, column-major via triu + flip)
    scs_rows, scs_cols = np.triu_indices(n)
    scs_rows, scs_cols = scs_cols, scs_rows
    scs_vec = A_sym[scs_rows, scs_cols]
    
    return {
        'A_sym': A_sym,
        'clarabel_indices': list(zip(clarabel_rows, clarabel_cols)),
        'clarabel_vec': clarabel_vec,
        'scs_indices': list(zip(scs_rows, scs_cols)),
        'scs_vec': scs_vec,
    }


class TestSCSVectorization:
    """Tests for SCS lower-triangle column-major vectorization."""
    
    def test_scs_indices_3x3(self):
        """Test that SCS indices for 3x3 give correct column-major lower triangle order."""
        # For a 3x3 matrix:
        # [a b c]     Lower triangle positions: (0,0)=a, (1,0)=d, (2,0)=g,
        # [d e f]                               (1,1)=e, (2,1)=h, (2,2)=i
        # [g h i]
        # 
        # Column-major order: go down column 0, then column 1, then column 2
        # Expected: (0,0), (1,0), (2,0), (1,1), (2,1), (2,2)
        
        rows, cols = np_get_scs_lower_tri_indices(3)
        
        expected_indices = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]
        actual_indices = list(zip(rows, cols))
        
        assert actual_indices == expected_indices, (
            f"Expected {expected_indices}, got {actual_indices}"
        )
    
    def test_scs_indices_4x4(self):
        """Test SCS indices for 4x4 matrix."""
        # Column-major lower triangle for 4x4:
        # Col 0: (0,0), (1,0), (2,0), (3,0)
        # Col 1: (1,1), (2,1), (3,1)
        # Col 2: (2,2), (3,2)
        # Col 3: (3,3)
        
        rows, cols = np_get_scs_lower_tri_indices(4)
        
        expected_indices = [
            (0, 0), (1, 0), (2, 0), (3, 0),  # Column 0
            (1, 1), (2, 1), (3, 1),           # Column 1
            (2, 2), (3, 2),                   # Column 2
            (3, 3),                           # Column 3
        ]
        actual_indices = list(zip(rows, cols))
        
        assert actual_indices == expected_indices
    
    def test_scs_vectorize_extracts_correct_elements(self):
        """Test that vectorization extracts correct elements in correct order."""
        # Create a 3x3 matrix with unique values
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)
        
        # Make symmetric for proper test
        A_sym = (A + A.T) / 2
        # A_sym = [[1, 3, 5],
        #          [3, 5, 7],
        #          [5, 7, 9]]
        
        vec = np_scs_symm_vectorize(A_sym, scale_factor=1.0)
        
        # Column-major lower triangle: (0,0)=1, (1,0)=3, (2,0)=5, (1,1)=5, (2,1)=7, (2,2)=9
        expected = np.array([1, 3, 5, 5, 7, 9], dtype=float)
        
        np.testing.assert_array_almost_equal(vec, expected)
    
    def test_scs_vectorize_scales_off_diagonals(self):
        """Test that off-diagonal elements are scaled correctly."""
        A = np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6]
        ], dtype=float)  # Already symmetric
        
        scale = np.sqrt(2)
        vec = np_scs_symm_vectorize(A, scale_factor=scale)
        
        # Diagonal elements: 1, 4, 6 (not scaled)
        # Off-diagonal elements: 2, 3, 5 (scaled by sqrt(2))
        # Order: (0,0)=1, (1,0)=2*sqrt(2), (2,0)=3*sqrt(2), (1,1)=4, (2,1)=5*sqrt(2), (2,2)=6
        expected = np.array([1, 2*scale, 3*scale, 4, 5*scale, 6])
        
        np.testing.assert_array_almost_equal(vec, expected)
    
    def test_mat_from_vec_inverts_vectorization(self):
        """Test that mat_from_vec correctly inverts vectorization."""
        # Create random symmetric matrix
        np.random.seed(42)
        n = 4
        A = np.random.randn(n, n)
        A_sym = (A + A.T) / 2
        
        scale = np.sqrt(2)
        vec = np_scs_symm_vectorize(A_sym, scale_factor=scale)
        A_recovered = np_scs_mat_from_vec(vec, n, scale_factor=scale)
        
        np.testing.assert_array_almost_equal(A_recovered, A_sym)
    
    def test_jax_matches_numpy(self):
        """Test that JAX version matches NumPy version."""
        np.random.seed(123)
        n = 5
        A = np.random.randn(n, n)
        A_sym = (A + A.T) / 2
        
        scale = np.sqrt(2)
        vec_np = np_scs_symm_vectorize(A_sym, scale_factor=scale)
        vec_jax = jax_scs_symm_vectorize(jnp.array(A_sym), scale_factor=scale)
        
        np.testing.assert_array_almost_equal(np.array(vec_jax), vec_np)
    
    def test_vector_length(self):
        """Test that vector has correct length n*(n+1)/2."""
        for n in [2, 3, 4, 5, 6]:
            A = np.eye(n)
            vec = np_scs_symm_vectorize(A, scale_factor=1.0)
            expected_len = n * (n + 1) // 2
            assert len(vec) == expected_len, f"n={n}: expected {expected_len}, got {len(vec)}"


class TestSCSvsClarabelOrdering:
    """Tests comparing SCS and Clarabel orderings."""
    
    def test_orderings_differ(self):
        """Test that SCS and Clarabel use different index orderings."""
        result = compare_ordering_clarabel_vs_scs(3)
        
        # For symmetric matrices, values can be same but indices are different
        # SCS: lower triangle, Clarabel: upper triangle
        # SCS indices: (row >= col), Clarabel indices: (row <= col)
        scs_indices = result['scs_indices']
        clarabel_indices = result['clarabel_indices']
        
        # They should NOT be the same indices
        assert scs_indices != clarabel_indices, (
            "SCS and Clarabel should have different index orderings"
        )
        
        # Verify the index differences
        # SCS (0,0), (1,0), (2,0), (1,1), (2,1), (2,2) - lower, col-major
        # Clarabel (0,0), (0,1), (1,1), (0,2), (1,2), (2,2) - upper, col-major
        assert scs_indices[1] == (1, 0), f"SCS second index should be (1,0), got {scs_indices[1]}"
        assert clarabel_indices[1] == (0, 1), f"Clarabel second index should be (0,1), got {clarabel_indices[1]}"
    
    def test_same_elements_different_order(self):
        """Test that both contain same elements, just reordered."""
        result = compare_ordering_clarabel_vs_scs(4)
        
        # Both should contain the same elements (sorted)
        scs_sorted = np.sort(result['scs_vec'])
        clarabel_sorted = np.sort(result['clarabel_vec'])
        
        np.testing.assert_array_almost_equal(scs_sorted, clarabel_sorted)
    
    def test_clarabel_is_upper_triangle(self):
        """Test that Clarabel indices are upper triangle."""
        result = compare_ordering_clarabel_vs_scs(4)
        
        for row, col in result['clarabel_indices']:
            assert row <= col, f"Clarabel index ({row}, {col}) is not upper triangle"
    
    def test_scs_is_lower_triangle(self):
        """Test that SCS indices are lower triangle."""
        result = compare_ordering_clarabel_vs_scs(4)
        
        for row, col in result['scs_indices']:
            assert row >= col, f"SCS index ({row}, {col}) is not lower triangle"
    
    def test_scs_column_major_order(self):
        """Test that SCS indices are in column-major order."""
        result = compare_ordering_clarabel_vs_scs(4)
        
        indices = result['scs_indices']
        
        # Column should be non-decreasing
        prev_col = -1
        prev_row = -1
        for row, col in indices:
            if col > prev_col:
                prev_col = col
                prev_row = row
            elif col == prev_col:
                assert row > prev_row, (
                    f"Within column {col}, rows should be increasing: "
                    f"prev_row={prev_row}, row={row}"
                )
                prev_row = row
            else:
                pytest.fail(f"Column went backwards: {prev_col} -> {col}")


class TestSCSCanonicalization:
    """Tests for SCS DRO expectation canonicalization."""
    
    def test_canonicalization_runs(self):
        """Test that canonicalization runs without error."""
        from learning.jax_scs_layer import jax_scs_canonicalize_dro_expectation
        
        # Small test problem
        N, M, S_mat, V = 2, 3, 4, 5
        S_vec = S_mat * (S_mat + 1) // 2
        
        # Create random test data
        np.random.seed(42)
        A_obj = np.random.randn(S_mat, S_mat)
        A_obj = (A_obj + A_obj.T) / 2
        b_obj = np.random.randn(V)
        A_vals = np.random.randn(M, S_mat, S_mat)
        A_vals = (A_vals + A_vals.transpose(0, 2, 1)) / 2
        b_vals = np.random.randn(M, V)
        c_vals = np.random.randn(M)
        
        G_batch = np.random.randn(N, S_mat, S_mat)
        G_batch = (G_batch + G_batch.transpose(0, 2, 1)) / 2
        F_batch = np.random.randn(N, V)
        
        eps = 0.1
        # G_precond_vec should be S_mat dimension (dimG), not S_vec
        G_precond_vec = np.random.randn(S_mat)
        F_precond = np.random.randn(V)
        precond_inv = (jnp.array(G_precond_vec), jnp.array(F_precond))
        
        # Run canonicalization
        A_dense, b, c_obj, x_dim, cone_info = jax_scs_canonicalize_dro_expectation(
            jnp.array(A_obj), jnp.array(b_obj), jnp.array(A_vals), 
            jnp.array(b_vals), jnp.array(c_vals),
            jnp.array(G_batch), jnp.array(F_batch),
            eps, precond_inv,
        )
        
        # Check dimensions
        expected_rows = cone_info['z'] + cone_info['l'] + sum(cone_info['q']) + sum(cone_info['s']) * (cone_info['s'][0] + 1) // 2
        assert A_dense.shape[0] == b.shape[0]
        assert A_dense.shape[1] == x_dim
        assert c_obj.shape[0] == x_dim
        
        # Check cone info
        assert cone_info['z'] == N * V
        assert cone_info['l'] == N + N * M
        assert len(cone_info['q']) == N
        assert len(cone_info['s']) == N
    
    def test_scs_vs_clarabel_same_objective(self):
        """Test that SCS and Clarabel canonicalizations give same objective when solved."""
        import scs
        import scipy.sparse as spa
        from learning.jax_scs_layer import jax_scs_canonicalize_dro_expectation
        from learning.jax_clarabel_layer import jax_canonicalize_dro_expectation
        from learning.autodiff_setup import (
            problem_data_to_gd_trajectories,
            compute_preconditioner_from_samples,
        )
        from learning.pep_construction import construct_gd_pep_data, pep_data_to_numpy
        import clarabel
        
        # Simple gradient descent PEP problem
        np.random.seed(42)
        K = 2  # 2 iterations
        N = 3  # 3 samples
        mu, L, R = 1.0, 10.0, 1.0
        dim = 10
        
        # Stepsize
        t = jnp.array([0.1, 0.1])
        
        # Build PEP data using the JIT-compatible constructor
        pep_data = construct_gd_pep_data(t, mu, L, R, K, pep_obj='obj_val')
        pep_data_np = pep_data_to_numpy(pep_data)
        A_obj, b_obj, A_vals, b_vals, c_vals = pep_data_np[:5]
        
        # Generate samples - quadratic functions
        G_batch = []
        F_batch = []
        for _ in range(N):
            # Random positive definite Hessian with eigenvalues in [mu, L]
            Q = np.random.randn(dim, dim)
            Q = Q @ Q.T / dim + np.eye(dim)
            Q = (Q / np.linalg.norm(Q, 2)) * (L - mu) + mu * np.eye(dim)
            
            # Initial point and optimal point
            x0 = np.random.randn(dim)
            x0 = x0 / np.linalg.norm(x0) * R
            xs = np.zeros(dim)  # Optimal point for quadratic
            fs = 0.0  # Optimal function value
            
            # Compute Gram representation
            G, F = problem_data_to_gd_trajectories((t,), Q, x0, xs, fs, K)
            G_batch.append(np.array(G))
            F_batch.append(np.array(F))
        
        G_batch = np.array(G_batch)
        F_batch = np.array(F_batch)
        
        # Compute preconditioner
        precond_inv_np = compute_preconditioner_from_samples(G_batch, F_batch, 'average')
        precond_inv = (jnp.array(precond_inv_np[0]), jnp.array(precond_inv_np[1]))
        
        eps = 0.1
        
        # ===== SCS canonicalization and solve =====
        A_scs, b_scs, c_scs, x_dim_scs, cone_info_scs = jax_scs_canonicalize_dro_expectation(
            jnp.array(A_obj), jnp.array(b_obj), jnp.array(A_vals), 
            jnp.array(b_vals), jnp.array(c_vals),
            jnp.array(G_batch), jnp.array(F_batch),
            eps, precond_inv,
        )
        
        # Build SCS cone dict
        scs_cone = {
            'z': int(cone_info_scs['z']),
            'l': int(cone_info_scs['l']),
            'q': [int(q) for q in cone_info_scs['q']],
            's': [int(s) for s in cone_info_scs['s']],
        }
        
        # Solve with SCS
        A_csc_scs = spa.csc_matrix(np.array(A_scs))
        data_scs = {
            'A': A_csc_scs,
            'b': np.array(b_scs),
            'c': np.array(c_scs),
        }
        solver_scs = scs.SCS(data_scs, scs_cone, verbose=False)
        sol_scs = solver_scs.solve()
        obj_scs = sol_scs['info']['pobj']
        
        # ===== Clarabel canonicalization and solve =====
        A_cla, b_cla, c_cla, x_dim_cla, cone_info_cla = jax_canonicalize_dro_expectation(
            jnp.array(A_obj), jnp.array(b_obj), jnp.array(A_vals), 
            jnp.array(b_vals), jnp.array(c_vals),
            jnp.array(G_batch), jnp.array(F_batch),
            eps, precond_inv,
        )
        
        # Build Clarabel cones
        cones_cla = []
        if cone_info_cla['zero'] > 0:
            cones_cla.append(clarabel.ZeroConeT(int(cone_info_cla['zero'])))
        if cone_info_cla['nonneg'] > 0:
            cones_cla.append(clarabel.NonnegativeConeT(int(cone_info_cla['nonneg'])))
        for q in cone_info_cla['soc']:
            cones_cla.append(clarabel.SecondOrderConeT(int(q)))
        for s in cone_info_cla['psd']:
            cones_cla.append(clarabel.PSDTriangleConeT(int(s)))
        
        # Solve with Clarabel
        P_cla = spa.csc_matrix((x_dim_cla, x_dim_cla))
        A_csc_cla = spa.csc_matrix(np.array(A_cla))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver_cla = clarabel.DefaultSolver(P_cla, np.array(c_cla), A_csc_cla, np.array(b_cla), cones_cla, settings)
        sol_cla = solver_cla.solve()
        obj_cla = sol_cla.obj_val
        
        # Check both solved successfully
        assert sol_scs['info']['status'] == 'solved', f"SCS failed: {sol_scs['info']['status']}"
        assert 'Solved' in str(sol_cla.status), f"Clarabel failed: {sol_cla.status}"
        
        # Compare objectives
        np.testing.assert_allclose(obj_scs, obj_cla, rtol=1e-2, atol=1e-3,
            err_msg=f"SCS obj={obj_scs}, Clarabel obj={obj_cla}")

    def test_finite_difference_derivatives_match(self):
        """Test that finite difference derivatives of SCS and Clarabel match.
        
        Uses scalar stepsize perturbation to compute numerical derivatives
        and verifies they are approximately equal.
        """
        import scs
        import scipy.sparse as spa
        from learning.jax_scs_layer import jax_scs_canonicalize_dro_expectation
        from learning.jax_clarabel_layer import jax_canonicalize_dro_expectation
        from learning.autodiff_setup import (
            problem_data_to_gd_trajectories,
            compute_preconditioner_from_samples,
        )
        from learning.pep_construction import construct_gd_pep_data, pep_data_to_numpy
        import clarabel
        
        def solve_scs(A_scs, b_scs, c_scs, cone_info_scs):
            """Solve SCS problem and return objective."""
            scs_cone = {
                'z': int(cone_info_scs['z']),
                'l': int(cone_info_scs['l']),
                'q': [int(q) for q in cone_info_scs['q']],
                's': [int(s) for s in cone_info_scs['s']],
            }
            A_csc = spa.csc_matrix(np.array(A_scs))
            data = {'A': A_csc, 'b': np.array(b_scs), 'c': np.array(c_scs)}
            solver = scs.SCS(data, scs_cone, verbose=False)
            sol = solver.solve()
            return sol['info']['pobj']
        
        def solve_clarabel(A_cla, b_cla, c_cla, x_dim, cone_info_cla):
            """Solve Clarabel problem and return objective."""
            cones = []
            if cone_info_cla['zero'] > 0:
                cones.append(clarabel.ZeroConeT(int(cone_info_cla['zero'])))
            if cone_info_cla['nonneg'] > 0:
                cones.append(clarabel.NonnegativeConeT(int(cone_info_cla['nonneg'])))
            for q in cone_info_cla['soc']:
                cones.append(clarabel.SecondOrderConeT(int(q)))
            for s in cone_info_cla['psd']:
                cones.append(clarabel.PSDTriangleConeT(int(s)))
            
            P = spa.csc_matrix((x_dim, x_dim))
            A_csc = spa.csc_matrix(np.array(A_cla))
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            solver = clarabel.DefaultSolver(P, np.array(c_cla), A_csc, np.array(b_cla), cones, settings)
            sol = solver.solve()
            return sol.obj_val
        
        # Problem parameters
        np.random.seed(42)
        K = 2
        N = 3
        mu, L, R = 1.0, 10.0, 1.0
        dim = 10
        eps = 0.1
        
        # Use scalar stepsize
        t_base = 0.1
        h = 1e-5  # Finite difference step
        
        # Generate fixed samples (don't change with stepsize)
        G_batch_list = []
        F_batch_list = []
        sample_data = []  # Store Q, x0, xs, fs for each sample
        
        for _ in range(N):
            Q = np.random.randn(dim, dim)
            Q = Q @ Q.T / dim + np.eye(dim)
            Q = (Q / np.linalg.norm(Q, 2)) * (L - mu) + mu * np.eye(dim)
            x0 = np.random.randn(dim)
            x0 = x0 / np.linalg.norm(x0) * R
            xs = np.zeros(dim)
            fs = 0.0
            sample_data.append((Q, x0, xs, fs))
        
        def compute_objectives(t_scalar):
            """Compute SCS and Clarabel objectives for given scalar stepsize."""
            t = jnp.array([t_scalar, t_scalar])
            
            # Build PEP data
            pep_data = construct_gd_pep_data(t, mu, L, R, K, pep_obj='obj_val')
            pep_data_np = pep_data_to_numpy(pep_data)
            A_obj, b_obj, A_vals, b_vals, c_vals = pep_data_np[:5]
            
            # Generate sample trajectories
            G_batch = []
            F_batch = []
            for Q, x0, xs, fs in sample_data:
                G, F = problem_data_to_gd_trajectories((t,), Q, x0, xs, fs, K)
                G_batch.append(np.array(G))
                F_batch.append(np.array(F))
            G_batch = np.array(G_batch)
            F_batch = np.array(F_batch)
            
            # Compute preconditioner
            precond_inv_np = compute_preconditioner_from_samples(G_batch, F_batch, 'average')
            precond_inv = (jnp.array(precond_inv_np[0]), jnp.array(precond_inv_np[1]))
            
            # SCS
            A_scs, b_scs, c_scs, _, cone_info_scs = jax_scs_canonicalize_dro_expectation(
                jnp.array(A_obj), jnp.array(b_obj), jnp.array(A_vals),
                jnp.array(b_vals), jnp.array(c_vals),
                jnp.array(G_batch), jnp.array(F_batch),
                eps, precond_inv,
            )
            obj_scs = solve_scs(A_scs, b_scs, c_scs, cone_info_scs)
            
            # Clarabel
            A_cla, b_cla, c_cla, x_dim_cla, cone_info_cla = jax_canonicalize_dro_expectation(
                jnp.array(A_obj), jnp.array(b_obj), jnp.array(A_vals),
                jnp.array(b_vals), jnp.array(c_vals),
                jnp.array(G_batch), jnp.array(F_batch),
                eps, precond_inv,
            )
            obj_cla = solve_clarabel(A_cla, b_cla, c_cla, x_dim_cla, cone_info_cla)
            
            return obj_scs, obj_cla
        
        # Compute objectives at t, t+h, t-h
        obj_scs_center, obj_cla_center = compute_objectives(t_base)
        obj_scs_plus, obj_cla_plus = compute_objectives(t_base + h)
        obj_scs_minus, obj_cla_minus = compute_objectives(t_base - h)
        
        # Compute central difference derivatives
        deriv_scs = (obj_scs_plus - obj_scs_minus) / (2 * h)
        deriv_cla = (obj_cla_plus - obj_cla_minus) / (2 * h)
        
        # Check that center objectives match (sanity check)
        np.testing.assert_allclose(obj_scs_center, obj_cla_center, rtol=1e-2, atol=1e-3,
            err_msg=f"Center objectives differ: SCS={obj_scs_center}, Clarabel={obj_cla_center}")
        
        # Check that numerical derivatives match
        np.testing.assert_allclose(deriv_scs, deriv_cla, rtol=0.1, atol=1e-3,
            err_msg=f"Numerical derivatives differ: SCS={deriv_scs}, Clarabel={deriv_cla}")

    def test_scs_autodiff_vs_finite_difference(self):
        """Test that SCS autodiff gradient matches finite difference approximation.
        
        Uses jax.grad on dro_scs_solve and compares to central difference.
        """
        from learning.jax_scs_layer import dro_scs_solve, jax_scs_canonicalize_dro_expectation
        from learning.autodiff_setup import (
            problem_data_to_gd_trajectories,
            compute_preconditioner_from_samples,
        )
        from learning.pep_construction import construct_gd_pep_data
        
        # Problem parameters
        np.random.seed(42)
        K = 2
        N = 3
        mu, L, R = 1.0, 10.0, 1.0
        dim = 10
        eps = 0.1
        
        # Use scalar stepsize
        t_base = 0.1
        h = 1e-5  # Finite difference step
        
        # Generate fixed samples (don't change with stepsize)
        sample_data = []
        for _ in range(N):
            Q = np.random.randn(dim, dim)
            Q = Q @ Q.T / dim + np.eye(dim)
            Q = (Q / np.linalg.norm(Q, 2)) * (L - mu) + mu * np.eye(dim)
            x0 = np.random.randn(dim)
            x0 = x0 / np.linalg.norm(x0) * R
            xs = np.zeros(dim)
            fs = 0.0
            sample_data.append((jnp.array(Q), jnp.array(x0), jnp.array(xs), fs))
        
        def compute_objective(t_scalar):
            """Compute SCS objective for given scalar stepsize using full diffcp pipeline."""
            t = jnp.array([t_scalar, t_scalar])
            
            # Build PEP data - returns JAX arrays directly
            pep_data = construct_gd_pep_data(t, mu, L, R, K, pep_obj='obj_val')
            # pep_data is tuple: (A_obj, b_obj, A_vals, b_vals, c_vals, ...)
            A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]
            
            # Generate sample trajectories (keep in JAX)
            G_list = []
            F_list = []
            for Q, x0, xs, fs in sample_data:
                G, F = problem_data_to_gd_trajectories((t,), Q, x0, xs, fs, K)
                G_list.append(G)
                F_list.append(F)
            G_batch = jnp.stack(G_list)
            F_batch = jnp.stack(F_list)
            
            # Compute preconditioner from average of diagonals
            # Note: using JAX operations only to stay traceable
            G_diag_sqrt = jnp.sqrt(jnp.mean(jnp.array([jnp.diag(G) for G in G_list]), axis=0))
            F_avg = jnp.mean(F_batch, axis=0)
            precond_G = 1.0 / G_diag_sqrt
            precond_F = 1.0 / jnp.sqrt(jnp.maximum(F_avg, 1e-10))
            precond_inv = (precond_G, precond_F)
            
            # Use the full differentiable pipeline
            obj = dro_scs_solve(
                A_obj, b_obj, A_vals, b_vals, c_vals,
                G_batch, F_batch,
                eps, precond_inv,
            )
            
            return obj
        
        # Compute gradient using JAX autodiff
        grad_fn = jax.grad(compute_objective)
        autodiff_grad = grad_fn(t_base)
        
        # Compute gradient using finite difference
        obj_plus = compute_objective(t_base + h)
        obj_minus = compute_objective(t_base - h)
        finite_diff_grad = (obj_plus - obj_minus) / (2 * h)
        
        # Check gradients match
        np.testing.assert_allclose(
            autodiff_grad, finite_diff_grad, 
            rtol=0.1, atol=1e-3,
            err_msg=f"SCS autodiff gradient ({autodiff_grad}) differs from finite difference ({finite_diff_grad})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
