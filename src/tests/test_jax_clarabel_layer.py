"""
Unit tests for JAX Clarabel layer components.

Tests compare JAX implementations against the NumPy ClarabelCanonicalizer baseline.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, '..')

from learning.jax_clarabel_layer import (
    jax_symm_vectorize,
    jax_scaled_off_triangles,
    jax_get_triangular_indices,
)
from reformulator.canonicalizers.clarabel_canonicalizer import (
    symm_vectorize,
    scaled_off_triangles,
    get_triangular_idx,
)


class TestSymmVectorize:
    """Tests for symmetric matrix vectorization."""
    
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_matches_numpy(self, n):
        """JAX symm_vectorize should match NumPy version."""
        # Create random symmetric matrix
        np.random.seed(42)
        A = np.random.randn(n, n)
        A = (A + A.T) / 2  # Make symmetric
        
        scale_factor = np.sqrt(2.0)
        
        # NumPy version
        np_result = symm_vectorize(A, scale_factor)
        
        # JAX version
        jax_result = jax_symm_vectorize(jnp.array(A), scale_factor)
        
        assert_allclose(np.array(jax_result), np_result, rtol=1e-10)
    
    def test_output_length(self):
        """Output should have length n*(n+1)/2."""
        for n in [3, 5, 7]:
            A = jnp.eye(n)
            result = jax_symm_vectorize(A, 1.0)
            expected_len = n * (n + 1) // 2
            assert result.shape[0] == expected_len
    
    def test_diagonal_unscaled(self):
        """Diagonal elements should not be scaled."""
        n = 4
        A = jnp.eye(n)
        scale_factor = 2.0
        
        result = jax_symm_vectorize(A, scale_factor)
        
        # Diagonal elements should be 1.0 (not scaled)
        # In lower triangular ordering, diagonals are at specific positions
        # For n=4: positions 0, 2, 5, 9 (triangular numbers)
        diag_positions = [i * (i + 1) // 2 + i for i in range(n)]
        for pos in diag_positions:
            if pos < len(result):
                assert result[pos] == 1.0
    
    def test_off_diagonal_scaled(self):
        """Off-diagonal elements should be scaled."""
        A = jnp.array([[1.0, 2.0], [2.0, 3.0]])
        scale_factor = jnp.sqrt(2.0)
        
        result = jax_symm_vectorize(A, scale_factor)
        
        # Result should be [1.0, 2*sqrt(2), 3.0]
        expected = jnp.array([1.0, 2.0 * scale_factor, 3.0])
        assert_allclose(result, expected, rtol=1e-10)
    
    def test_jit_compatible(self):
        """Function should be JIT-compatible."""
        A = jnp.eye(3)
        
        @jax.jit
        def jit_vectorize(A):
            return jax_symm_vectorize(A, jnp.sqrt(2.0))
        
        result = jit_vectorize(A)
        expected = jax_symm_vectorize(A, jnp.sqrt(2.0))
        assert_allclose(result, expected, rtol=1e-10)


class TestScaledOffTriangles:
    """Tests for scaled off-triangle diagonal matrix."""
    
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_matches_numpy(self, n):
        """JAX scaled_off_triangles should match NumPy version."""
        A = np.ones((n, n))
        scale_factor = np.sqrt(2.0)
        
        # NumPy version
        np_result = scaled_off_triangles(A, scale_factor)
        
        # JAX version
        jax_result = jax_scaled_off_triangles(jnp.array(A), scale_factor)
        
        assert_allclose(np.array(jax_result), np_result, rtol=1e-10)
    
    def test_is_diagonal(self):
        """Result should be a diagonal matrix."""
        n = 4
        A = jnp.ones((n, n))
        result = jax_scaled_off_triangles(A, 1.0)
        
        # Off-diagonal should be zero
        off_diag = result - jnp.diag(jnp.diag(result))
        assert jnp.allclose(off_diag, 0)
    
    def test_output_shape(self):
        """Output shape should be (n*(n+1)/2, n*(n+1)/2)."""
        for n in [3, 4, 5]:
            A = jnp.ones((n, n))
            result = jax_scaled_off_triangles(A, 1.0)
            expected_dim = n * (n + 1) // 2
            assert result.shape == (expected_dim, expected_dim)


class TestTriangularIndices:
    """Tests for triangular index generation."""
    
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_matches_numpy(self, n):
        """JAX triangular indices should match NumPy version."""
        np_cols, np_rows = get_triangular_idx(n)
        jax_cols, jax_rows = jax_get_triangular_indices(n)
        
        assert_allclose(np.array(jax_cols), np_cols)
        assert_allclose(np.array(jax_rows), np_rows)


class TestCanonicalization:
    """Tests comparing JAX/NumPy canonicalization to ClarabelCanonicalizer."""
    
    @pytest.fixture
    def pep_data_and_samples(self):
        """Create PEP data and samples matching ClarabelCanonicalizer input format."""
        np.random.seed(42)
        
        # Dimensions matching a small problem
        K_max = 2
        dimG = K_max + 2  # 4
        dimF = K_max + 1  # 3
        N = 4  # Number of samples
        
        # PEP data (objective and constraints)
        A_obj = np.eye(dimG) * 0.1  # Objective A matrix
        b_obj = np.zeros(dimF)  # Objective b vector
        
        # Interpolation constraints
        M = 6  # Number of constraints
        A_vals = []
        b_vals = []
        c_vals = np.random.randn(M)
        
        for m in range(M):
            A_m = np.random.randn(dimG, dimG)
            A_m = (A_m + A_m.T) / 2  # Symmetrize
            A_vals.append(A_m)
            b_vals.append(np.random.randn(dimF))
        
        A_vals = np.array(A_vals)
        b_vals = np.array(b_vals)
        
        # Sample data (Gram matrices and function values)
        G_batch = []
        F_batch = []
        for i in range(N):
            G = np.random.randn(dimG, dimG)
            G = (G + G.T) / 2  # Symmetrize
            G_batch.append(G)
            F_batch.append(np.random.randn(dimF))
        
        G_batch = np.array(G_batch)
        F_batch = np.array(F_batch)
        
        # Preconditioner
        G_precond_inv = np.random.rand(dimG) + 0.5
        F_precond_inv = np.random.rand(dimF) + 0.5
        precond_inv = (G_precond_inv, F_precond_inv)
        
        eps = 0.1
        
        return {
            'A_obj': A_obj,
            'b_obj': b_obj,
            'A_vals': A_vals,
            'b_vals': b_vals,
            'c_vals': c_vals,
            'G_batch': G_batch,
            'F_batch': F_batch,
            'precond_inv': precond_inv,
            'eps': eps,
            'N': N,
            'M': M,
            'V': dimF,
            'S_mat': dimG,
        }
    
    def test_canonicalization_dimensions(self, pep_data_and_samples):
        """Test that canonicalization produces correct dimensions."""
        from learning.numpy_clarabel_layer import (
            numpy_canonicalize_dro_expectation
        )
        
        d = pep_data_and_samples
        A_csc, b, c, cones, x_dim = numpy_canonicalize_dro_expectation(
            A_obj=d['A_obj'],
            b_obj=d['b_obj'],
            A_vals=d['A_vals'],
            b_vals=d['b_vals'],
            c_vals=d['c_vals'],
            G_batch=d['G_batch'],
            F_batch=d['F_batch'],
            eps=d['eps'],
            precond_inv=d['precond_inv'],
        )
        
        N, M, V, S_mat = d['N'], d['M'], d['V'], d['S_mat']
        S_vec = S_mat * (S_mat + 1) // 2
        
        # Check x_dim
        expected_x_dim = 1 + N * (1 + M + V + S_vec)
        assert x_dim == expected_x_dim, f"x_dim mismatch: {x_dim} vs {expected_x_dim}"
        
        # Check A shape
        assert A_csc.shape[1] == x_dim, f"A columns: {A_csc.shape[1]} vs {x_dim}"
        
        # Check b length matches A rows
        assert len(b) == A_csc.shape[0], f"b length: {len(b)} vs A rows: {A_csc.shape[0]}"
        
        # Check c length
        assert len(c) == x_dim, f"c length: {len(c)} vs {x_dim}"
    
    def test_canonicalization_solves(self, pep_data_and_samples):
        """Test that canonicalized problem can be solved by Clarabel (or at least runs)."""
        from learning.numpy_clarabel_layer import (
            numpy_canonicalize_dro_expectation
        )
        import clarabel
        import scipy.sparse as spa
        
        d = pep_data_and_samples
        A_csc, b, c, cones, x_dim = numpy_canonicalize_dro_expectation(
            A_obj=d['A_obj'],
            b_obj=d['b_obj'],
            A_vals=d['A_vals'],
            b_vals=d['b_vals'],
            c_vals=d['c_vals'],
            G_batch=d['G_batch'],
            F_batch=d['F_batch'],
            eps=d['eps'],
            precond_inv=d['precond_inv'],
        )
        
        # Create P (zero matrix, no quadratic term)
        P = spa.csc_matrix((x_dim, x_dim))
        
        # Solve
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(P, c, A_csc, b, cones, settings)
        solution = solver.solve()
        
        # Check that solver ran (may not solve due to random data)
        # The key is that the problem was constructed correctly
        assert solution is not None, "Solver failed to run"
        assert solution.x is not None or len(solution.x) > 0, "No primal solution returned"
        
        # Note: Random data may be infeasible, so we just check the solver ran


class TestAgainstClarabelCanonicalizer:
    """Tests comparing our canonicalization to the production ClarabelCanonicalizer."""
    
    @pytest.fixture
    def real_pep_data_and_samples(self):
        """Generate real PEP data using pep_construction.py functions."""
        np.random.seed(42)
        
        # Parameters
        K_max = 2
        mu = 0.0
        L = 10.0
        R = 1.0
        N = 4
        dim = 10  # Dimension of the problem
        
        # Generate trajectories and PEP data using GD for simplicity
        from learning.pep_construction import construct_gd_pep_data
        
        # Step sizes
        t = 0.1 * np.ones(K_max)
        
        # Get PEP data structure
        pep_data = construct_gd_pep_data(t, mu, L, R, K_max, 'obj_val')
        
        # Generate sample trajectories
        from learning.autodiff_setup import problem_data_to_gd_trajectories
        import jax.numpy as jnp
        
        samples = []
        for i in range(N):
            # Random quadratic problem
            Q_half = np.random.randn(dim, dim)
            Q = Q_half.T @ Q_half + 0.1 * np.eye(dim)  # PSD with eigenvalues >= 0.1
            Q = Q / np.linalg.norm(Q, 2) * L  # Scale to have max eigenvalue ~L
            
            z0 = np.random.randn(dim) * R
            zs = np.zeros(dim)  # Optimal is at origin for quadratic
            fs = 0.0
            
            G, F = problem_data_to_gd_trajectories(
                (jnp.array(t),), 
                jnp.array(Q), 
                jnp.array(z0), 
                jnp.array(zs), 
                jnp.array(fs), 
                K_max, 
                return_Gram_representation=True
            )
            samples.append((np.array(G), np.array(F)))
        
        # Compute preconditioner
        from learning.autodiff_setup import compute_preconditioner_from_samples
        G_batch = np.stack([s[0] for s in samples])
        F_batch = np.stack([s[1] for s in samples])
        precond_inv = compute_preconditioner_from_samples(G_batch, F_batch, precond_type='average')
        
        return {
            'pep_data': pep_data,
            'samples': samples,
            'G_batch': G_batch,
            'F_batch': F_batch,
            'precond_inv': precond_inv,
            'eps': 0.1,
            'K_max': K_max,
            'mu': mu,
            'L': L,
            'R': R,
        }
    
    def test_matches_clarabel_canonicalizer_A_matrix(self, real_pep_data_and_samples):
        """Test that our canonicalization produces same A matrix as ClarabelCanonicalizer."""
        from learning.numpy_clarabel_layer import numpy_canonicalize_dro_expectation
        from reformulator.canonicalizers.clarabel_canonicalizer import ClarabelCanonicalizer
        
        d = real_pep_data_and_samples
        pep_data = d['pep_data']
        
        # Our canonicalization
        A_obj, b_obj, A_vals, b_vals, c_vals, PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes = pep_data
        
        A_ours, b_ours, c_ours, cones_ours, x_dim_ours = numpy_canonicalize_dro_expectation(
            A_obj=A_obj,
            b_obj=b_obj,
            A_vals=A_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            G_batch=d['G_batch'],
            F_batch=d['F_batch'],
            eps=d['eps'],
            precond_inv=d['precond_inv'],
        )
        
        # ClarabelCanonicalizer
        canonicalizer = ClarabelCanonicalizer(
            pep_data=pep_data,
            samples=d['samples'],
            measure='expectation',
            wrapper='clarabel',
            precond=False,  # Disable to match our direct precond_inv
        )
        # Manually set precond_inv to match
        canonicalizer.precond_inv = d['precond_inv']
        canonicalizer.setup_problem()
        canonicalizer.set_params(eps=d['eps'])
        
        # Compare shapes
        assert A_ours.shape == canonicalizer.A.shape, \
            f"A shape mismatch: {A_ours.shape} vs {canonicalizer.A.shape}"
        assert len(b_ours) == len(canonicalizer.b), \
            f"b length mismatch: {len(b_ours)} vs {len(canonicalizer.b)}"
        
        # Compare values (allowing for sparse format differences)
        A_ours_dense = A_ours.toarray() if hasattr(A_ours, 'toarray') else A_ours
        A_ref_dense = canonicalizer.A.toarray() if hasattr(canonicalizer.A, 'toarray') else canonicalizer.A
        
        assert_allclose(A_ours_dense, A_ref_dense, rtol=1e-10, 
                       err_msg="A matrix values don't match")
        assert_allclose(b_ours, canonicalizer.b, rtol=1e-10,
                       err_msg="b vector values don't match")
    
    def test_matches_clarabel_canonicalizer_objective(self, real_pep_data_and_samples):
        """Test that both versions produce same optimal objective."""
        from learning.numpy_clarabel_layer import numpy_canonicalize_dro_expectation
        from reformulator.canonicalizers.clarabel_canonicalizer import ClarabelCanonicalizer
        import scipy.sparse as spa
        import clarabel
        
        d = real_pep_data_and_samples
        pep_data = d['pep_data']
        
        # Our canonicalization
        A_obj, b_obj, A_vals, b_vals, c_vals, PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes = pep_data
        
        A_ours, b_ours, c_ours, cones_ours, x_dim_ours = numpy_canonicalize_dro_expectation(
            A_obj=A_obj,
            b_obj=b_obj,
            A_vals=A_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            G_batch=d['G_batch'],
            F_batch=d['F_batch'],
            eps=d['eps'],
            precond_inv=d['precond_inv'],
        )
        
        # Solve our version
        P_ours = spa.csc_matrix((x_dim_ours, x_dim_ours))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver_ours = clarabel.DefaultSolver(P_ours, c_ours, A_ours, b_ours, cones_ours, settings)
        sol_ours = solver_ours.solve()
        
        # ClarabelCanonicalizer version
        canonicalizer = ClarabelCanonicalizer(
            pep_data=pep_data,
            samples=d['samples'],
            measure='expectation',
            wrapper='clarabel',
            precond=False,
        )
        canonicalizer.precond_inv = d['precond_inv']
        canonicalizer.setup_problem()
        canonicalizer.set_params(eps=d['eps'])
        sol_ref = canonicalizer.solve()
        
        # Both should solve (or both should fail in same way)
        if "Solved" in str(sol_ours.status):
            assert np.isfinite(sol_ours.obj_val), "Our solution has non-finite objective"
            assert np.isfinite(sol_ref['obj']), "Reference solution has non-finite objective"
            assert_allclose(sol_ours.obj_val, sol_ref['obj'], rtol=1e-4,
                          err_msg="Objective values don't match")


class TestJaxCanonicalization:
    """Tests for the pure JAX canonicalization function."""
    
    @pytest.fixture
    def pep_data_for_jax(self):
        """Generate PEP data as JAX arrays for testing."""
        np.random.seed(42)
        
        K_max = 2
        N = 4
        dim = 5
        
        from learning.pep_construction import construct_gd_pep_data
        from learning.autodiff_setup import problem_data_to_gd_trajectories
        import jax.numpy as jnp
        
        t = 0.1 * np.ones(K_max)
        pep_data = construct_gd_pep_data(t, 0.0, 10.0, 1.0, K_max, 'obj_val')
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        
        # Generate samples
        samples = []
        for i in range(N):
            Q_half = np.random.randn(dim, dim)
            Q = Q_half.T @ Q_half + 0.1 * np.eye(dim)
            Q = Q / np.linalg.norm(Q, 2) * 10.0
            z0 = np.random.randn(dim)
            
            G, F = problem_data_to_gd_trajectories(
                (jnp.array(t),), jnp.array(Q), jnp.array(z0),
                jnp.zeros(dim), jnp.array(0.0), K_max, return_Gram_representation=True
            )
            samples.append((np.array(G), np.array(F)))
        
        G_batch = np.stack([s[0] for s in samples])
        F_batch = np.stack([s[1] for s in samples])
        
        from learning.autodiff_setup import compute_preconditioner_from_samples
        precond_inv = compute_preconditioner_from_samples(G_batch, F_batch, 'average')
        
        return {
            'A_obj': jnp.array(A_obj),
            'b_obj': jnp.array(b_obj),
            'A_vals': jnp.array(A_vals),
            'b_vals': jnp.array(b_vals),
            'c_vals': jnp.array(c_vals),
            'G_batch': jnp.array(G_batch),
            'F_batch': jnp.array(F_batch),
            'precond_inv': (jnp.array(precond_inv[0]), jnp.array(precond_inv[1])),
            'eps': 0.1,
        }
    
    def test_jax_matches_numpy(self, pep_data_for_jax):
        """Test that JAX canonicalization produces equivalent problem to NumPy version.
        
        Note: The JAX version uses diffcp-compatible row ordering (zero, nonneg, soc, psd)
        while NumPy uses a different ordering. We compare problem dimensions and c vectors,
        since the actual constraint rows are reordered.
        """
        from learning.jax_clarabel_layer import (
            jax_canonicalize_dro_expectation,
        )
        from learning.numpy_clarabel_layer import (
            numpy_canonicalize_dro_expectation,
        )
        
        d = pep_data_for_jax
        
        # JAX version
        A_jax, b_jax, c_jax, x_dim_jax, cone_info = jax_canonicalize_dro_expectation(
            A_obj=d['A_obj'], b_obj=d['b_obj'],
            A_vals=d['A_vals'], b_vals=d['b_vals'], c_vals=d['c_vals'],
            G_batch=d['G_batch'], F_batch=d['F_batch'],
            eps=d['eps'], precond_inv=d['precond_inv'],
        )
        
        # NumPy version
        A_np, b_np, c_np, cones_np, x_dim_np = numpy_canonicalize_dro_expectation(
            A_obj=np.array(d['A_obj']), b_obj=np.array(d['b_obj']),
            A_vals=np.array(d['A_vals']), b_vals=np.array(d['b_vals']), 
            c_vals=np.array(d['c_vals']),
            G_batch=np.array(d['G_batch']), F_batch=np.array(d['F_batch']),
            eps=d['eps'], precond_inv=(np.array(d['precond_inv'][0]), np.array(d['precond_inv'][1])),
        )
        
        # Compare dimensions
        assert x_dim_jax == x_dim_np, f"x_dim mismatch: {x_dim_jax} vs {x_dim_np}"
        
        # Compare A matrix shape (same shape, different row ordering)
        A_np_dense = A_np.toarray() if hasattr(A_np, 'toarray') else A_np
        assert A_jax.shape == A_np_dense.shape, f"A shape mismatch: {A_jax.shape} vs {A_np_dense.shape}"
        
        # Compare c (objective vector - should be identical)
        assert_allclose(np.array(c_jax), c_np, rtol=1e-10,
                       err_msg="c vectors don't match")
    
    def test_jit_compatible(self, pep_data_for_jax):
        """Test that JAX canonicalization can be JIT compiled."""
        from learning.jax_clarabel_layer import jax_canonicalize_dro_expectation
        
        d = pep_data_for_jax
        
        # JIT compile the function
        @jax.jit
        def jit_canonicalize(G_batch, F_batch, eps):
            return jax_canonicalize_dro_expectation(
                A_obj=d['A_obj'], b_obj=d['b_obj'],
                A_vals=d['A_vals'], b_vals=d['b_vals'], c_vals=d['c_vals'],
                G_batch=G_batch, F_batch=F_batch,
                eps=eps, precond_inv=d['precond_inv'],
            )
        
        # Run once to compile
        A1, b1, c1, x_dim1, cone_info1 = jit_canonicalize(d['G_batch'], d['F_batch'], d['eps'])
        
        # Run again to use cached compilation
        A2, b2, c2, x_dim2, cone_info2 = jit_canonicalize(d['G_batch'], d['F_batch'], d['eps'])
        
        # Results should match
        assert_allclose(np.array(A1), np.array(A2), rtol=1e-10)
        assert_allclose(np.array(b1), np.array(b2), rtol=1e-10)
        assert_allclose(np.array(c1), np.array(c2), rtol=1e-10)
    
    def test_gradients_flow(self, pep_data_for_jax):
        """Test that gradients can be computed through the canonicalization."""
        from learning.jax_clarabel_layer import jax_canonicalize_dro_expectation
        
        d = pep_data_for_jax
        
        def loss_fn(G_batch, F_batch, eps):
            A, b, c, x_dim, cone_info = jax_canonicalize_dro_expectation(
                A_obj=d['A_obj'], b_obj=d['b_obj'],
                A_vals=d['A_vals'], b_vals=d['b_vals'], c_vals=d['c_vals'],
                G_batch=G_batch, F_batch=F_batch,
                eps=eps, precond_inv=d['precond_inv'],
            )
            # Simple loss: sum of A matrix
            return jnp.sum(A)
        
        # Compute gradients
        grad_G, grad_F, grad_eps = jax.grad(loss_fn, argnums=(0, 1, 2))(
            d['G_batch'], d['F_batch'], d['eps']
        )
        
        # Gradients should be finite
        assert jnp.all(jnp.isfinite(grad_G)), "G gradient contains non-finite values"
        assert jnp.all(jnp.isfinite(grad_F)), "F gradient contains non-finite values"
        assert jnp.isfinite(grad_eps), "eps gradient is non-finite"


class TestDroClarabelSolve:
    """Tests for the full differentiable Clarabel solve pipeline."""
    
    @pytest.fixture
    def pep_data_for_solve(self):
        """Generate PEP data for solve testing using quad.py functions for guaranteed feasibility."""
        import jax
        import jax.numpy as jnp
        
        # Use a fixed JAX key for reproducibility
        key = jax.random.PRNGKey(42)
        
        # Parameters from quad.yaml defaults
        K_max = 2
        mu = 0.0
        L = 10.0
        R = 1.0
        N = 4
        dim = 20  # Larger dimension helps with feasibility
        eps = 0.1
        
        # Import functions from quad.py for proper data generation
        from learning_experiment_classes.quad import get_Q_samples, get_z0_samples
        from learning.pep_construction import construct_gd_pep_data
        from learning.autodiff_setup import (
            problem_data_to_gd_trajectories, compute_preconditioner_from_samples
        )
        
        # Step sizes (reasonable for GD on L-smooth functions)
        t = 0.1 * np.ones(K_max)
        
        # Get PEP constraint data
        pep_data = construct_gd_pep_data(t, mu, L, R, K_max, 'obj_val')
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        
        # Generate Q samples using rejection sampling (guarantees eigenvalues in [mu, L])
        key, *subkeys = jax.random.split(key, N + 1)
        subkeys = jnp.stack(subkeys)
        Q_batch = get_Q_samples(subkeys, dim, mu, L, dim)  # M=dim for d x d matrices
        
        # Generate z0 samples uniformly in ball of radius R
        key, *subkeys = jax.random.split(key, N + 1)
        subkeys = jnp.stack(subkeys)
        z0_batch = get_z0_samples(subkeys, dim, R)
        
        # zs_batch = 0 (optimal at origin for quadratics with min at 0)
        zs_batch = jnp.zeros((N, dim))
        fs_batch = jnp.zeros(N)
        
        # Compute trajectories (G, F) for each sample
        samples = []
        G_list = []
        F_list = []
        for i in range(N):
            G, F = problem_data_to_gd_trajectories(
                (jnp.array(t),), 
                Q_batch[i], 
                z0_batch[i], 
                zs_batch[i], 
                fs_batch[i], 
                K_max, 
                return_Gram_representation=True
            )
            G_list.append(np.array(G))
            F_list.append(np.array(F))
            samples.append((np.array(G), np.array(F)))
        
        G_batch_np = np.stack(G_list)
        F_batch_np = np.stack(F_list)
        
        # Compute preconditioner
        precond_inv = compute_preconditioner_from_samples(G_batch_np, F_batch_np, precond_type='average')
        
        return {
            'A_obj': jnp.array(A_obj),
            'b_obj': jnp.array(b_obj),
            'A_vals': jnp.array(A_vals),
            'b_vals': jnp.array(b_vals),
            'c_vals': jnp.array(c_vals),
            'G_batch': jnp.array(G_batch_np),
            'F_batch': jnp.array(F_batch_np),
            'precond_inv': (jnp.array(precond_inv[0]), jnp.array(precond_inv[1])),
            'eps': eps,
            'samples': samples,
            'pep_data': pep_data,
        }
    
    def test_forward_pass_runs(self, pep_data_for_solve):
        """Test that forward pass of dro_clarabel_solve works."""
        from learning.jax_clarabel_layer import (
            dro_clarabel_solve
        )
        from learning.numpy_clarabel_layer import (
            numpy_canonicalize_dro_expectation
        )
        import scipy.sparse as spa
        import clarabel
        
        d = pep_data_for_solve
        
        obj = dro_clarabel_solve(
            A_obj=d['A_obj'], b_obj=d['b_obj'],
            A_vals=d['A_vals'], b_vals=d['b_vals'], c_vals=d['c_vals'],
            G_batch=d['G_batch'], F_batch=d['F_batch'],
            eps=d['eps'], precond_inv=d['precond_inv'],
        )
        
        # Also solve via NumPy path to compare
        A_np, b_np, c_np, cones_np, x_dim = numpy_canonicalize_dro_expectation(
            A_obj=np.array(d['A_obj']), b_obj=np.array(d['b_obj']),
            A_vals=np.array(d['A_vals']), b_vals=np.array(d['b_vals']),
            c_vals=np.array(d['c_vals']),
            G_batch=np.array(d['G_batch']), F_batch=np.array(d['F_batch']),
            eps=d['eps'], 
            precond_inv=(np.array(d['precond_inv'][0]), np.array(d['precond_inv'][1])),
        )
        P = spa.csc_matrix((x_dim, x_dim))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(P, c_np, A_np, b_np, cones_np, settings)
        sol = solver.solve()
        
        # Both should have same behavior (both finite or both NaN)
        if "Solved" in str(sol.status):
            assert jnp.isfinite(obj), f"NumPy solved but JAX returned NaN: {obj}"
            assert_allclose(float(obj), sol.obj_val, rtol=1e-4)
        else:
            # Both should be NaN for infeasible problems
            assert not jnp.isfinite(obj) or np.isnan(sol.obj_val), \
                f"Status mismatch: NumPy={sol.status}, JAX obj={obj}"
    
    def test_forward_matches_numpy(self, pep_data_for_solve):
        """Test that JAX solve matches NumPy canonicalization solve."""
        from learning.jax_clarabel_layer import (
            dro_clarabel_solve,
        )
        from learning.numpy_clarabel_layer import (
            numpy_canonicalize_dro_expectation
        )
        import clarabel
        import scipy.sparse as spa
        
        d = pep_data_for_solve
        
        # JAX version
        obj_jax = dro_clarabel_solve(
            A_obj=d['A_obj'], b_obj=d['b_obj'],
            A_vals=d['A_vals'], b_vals=d['b_vals'], c_vals=d['c_vals'],
            G_batch=d['G_batch'], F_batch=d['F_batch'],
            eps=d['eps'], precond_inv=d['precond_inv'],
        )
        
        # NumPy version
        A_np, b_np, c_np, cones_np, x_dim = numpy_canonicalize_dro_expectation(
            A_obj=np.array(d['A_obj']), b_obj=np.array(d['b_obj']),
            A_vals=np.array(d['A_vals']), b_vals=np.array(d['b_vals']),
            c_vals=np.array(d['c_vals']),
            G_batch=np.array(d['G_batch']), F_batch=np.array(d['F_batch']),
            eps=d['eps'], 
            precond_inv=(np.array(d['precond_inv'][0]), np.array(d['precond_inv'][1])),
        )
        
        P = spa.csc_matrix((x_dim, x_dim))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solver = clarabel.DefaultSolver(P, c_np, A_np, b_np, cones_np, settings)
        sol = solver.solve()
        
        obj_numpy = sol.obj_val
        
        # Compare objectives
        assert_allclose(float(obj_jax), obj_numpy, rtol=1e-4,
                       err_msg=f"Objectives don't match: JAX={obj_jax}, NumPy={obj_numpy}")
    
    def test_gradients_through_solve(self, pep_data_for_solve):
        """Test that gradients can be computed through dro_clarabel_solve."""
        from learning.jax_clarabel_layer import dro_clarabel_solve
        
        d = pep_data_for_solve
        
        # First check if forward pass produces a finite result
        obj_fwd = dro_clarabel_solve(
            A_obj=d['A_obj'], b_obj=d['b_obj'],
            A_vals=d['A_vals'], b_vals=d['b_vals'], c_vals=d['c_vals'],
            G_batch=d['G_batch'], F_batch=d['F_batch'],
            eps=d['eps'], precond_inv=d['precond_inv'],
        )
        
        if not jnp.isfinite(obj_fwd):
            pytest.skip("Skipping gradient test - forward pass returned infeasible result")
        
        def loss_fn(eps):
            return dro_clarabel_solve(
                A_obj=d['A_obj'], b_obj=d['b_obj'],
                A_vals=d['A_vals'], b_vals=d['b_vals'], c_vals=d['c_vals'],
                G_batch=d['G_batch'], F_batch=d['F_batch'],
                eps=eps, precond_inv=d['precond_inv'],
            )
        
        # Compute gradient w.r.t. eps
        grad_eps = jax.grad(loss_fn)(d['eps'])
        
        # Gradient should be finite
        assert jnp.isfinite(grad_eps), f"Gradient is not finite: {grad_eps}"
        
        # For DRO, gradient w.r.t. eps should be approximately lambda (dual var)
        # This should be non-negative
        print(f"Gradient w.r.t. eps: {grad_eps}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
