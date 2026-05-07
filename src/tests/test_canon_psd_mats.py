"""
Test PSD constraint support in JAX SCS layer.

This module tests the extension of the JAX SCS layer to support additional PSD
constraints specified via PSD_A_vals, PSD_b_vals, PSD_c_vals, and PSD_mat_dims.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import pytest

from learning.jax_scs_layer import (
    dro_scs_solve,
    compute_preconditioner_from_samples,
    compute_C_d_matrices,
)


def setup_test_data():
    """Create test data for PSD constraint tests."""
    np.random.seed(42)
    N, M, S_mat, V = 3, 4, 5, 6

    # PEP data
    A_obj = jnp.eye(S_mat) * 0.1
    b_obj = jnp.ones(V) * 0.5
    A_vals = jnp.array([jnp.eye(S_mat) * (0.1 + 0.01*i) for i in range(M)])
    b_vals = np.random.randn(M, V) * 0.1
    c_vals = jnp.ones(M) * 0.2

    # Sample data
    G_batch = jnp.array([jnp.eye(S_mat) * (1.0 + 0.1*i) for i in range(N)])
    F_batch = jnp.ones((N, V)) * 0.3

    # Parameters
    eps = 0.5
    alpha = 0.1
    precond_inv = compute_preconditioner_from_samples(G_batch, F_batch, 'average')

    return {
        'A_obj': A_obj,
        'b_obj': b_obj,
        'A_vals': A_vals,
        'b_vals': b_vals,
        'c_vals': c_vals,
        'G_batch': G_batch,
        'F_batch': F_batch,
        'eps': eps,
        'alpha': alpha,
        'precond_inv': precond_inv,
        'N': N,
        'M': M,
        'S_mat': S_mat,
        'V': V,
    }


def create_psd_constraint_data(M_psd, dim, S_mat, V):
    """Create PSD constraint test data."""
    PSD_mat_dims = jnp.array([dim] * M_psd)

    # Create PSD_A_vals: (M_psd, dim, dim, S_mat, S_mat)
    PSD_A_vals = jnp.zeros((M_psd, dim, dim, S_mat, S_mat))
    for m in range(M_psd):
        for i in range(dim):
            for j in range(dim):
                PSD_A_vals = PSD_A_vals.at[m, i, j].set(
                    jnp.eye(S_mat) * 0.01 * (i + j + 1)
                )

    # Create PSD_b_vals: (M_psd, dim, dim, V)
    PSD_b_vals = jnp.zeros((M_psd, dim, dim, V))
    for m in range(M_psd):
        for i in range(dim):
            for j in range(dim):
                PSD_b_vals = PSD_b_vals.at[m, i, j].set(
                    jnp.ones(V) * 0.01 * (i + j + 1)
                )

    PSD_c_vals = None

    return PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_mat_dims


class TestBackwardCompatibility:
    """Test that PSD_A_vals=None maintains backward compatibility."""

    def test_expectation_without_psd(self):
        """Test expectation risk measure without PSD constraints."""
        data = setup_test_data()

        obj_val = dro_scs_solve(
            data['A_obj'], data['b_obj'], data['A_vals'],
            data['b_vals'], data['c_vals'],
            data['G_batch'], data['F_batch'],
            data['eps'], data['precond_inv'],
            risk_type='expectation',
            PSD_A_vals=None,
        )

        assert jnp.isfinite(obj_val), "Objective value should be finite"
        assert isinstance(obj_val, jnp.ndarray), "Should return JAX array"

    def test_cvar_without_psd(self):
        """Test CVaR risk measure without PSD constraints."""
        data = setup_test_data()

        obj_val = dro_scs_solve(
            data['A_obj'], data['b_obj'], data['A_vals'],
            data['b_vals'], data['c_vals'],
            data['G_batch'], data['F_batch'],
            data['eps'], data['precond_inv'],
            risk_type='cvar',
            alpha=data['alpha'],
            PSD_A_vals=None,
        )

        assert jnp.isfinite(obj_val), "Objective value should be finite"
        assert isinstance(obj_val, jnp.ndarray), "Should return JAX array"


class TestComputeCdMatrices:
    """Test the helper function for computing C and d matrices."""

    def test_single_psd_constraint(self):
        """Test C and d matrix computation with M_psd=1."""
        M_psd = 1
        dim = 2
        S_mat = 5
        V = 6

        PSD_A_vals, PSD_b_vals, _, PSD_mat_dims = create_psd_constraint_data(
            M_psd, dim, S_mat, V
        )

        C_symvecs, d_vecs, H_vec_dims, PSD_mat_dims_list = compute_C_d_matrices(
            PSD_A_vals, PSD_b_vals, PSD_mat_dims
        )

        # Check dimensions
        assert len(C_symvecs) == M_psd, "Should have M_psd C matrices"
        assert len(d_vecs) == M_psd, "Should have M_psd d matrices"
        assert len(H_vec_dims) == M_psd, "Should have M_psd H_vec dimensions"

        # Check expected H_vec for dim=2: 2*(2+1)/2 = 3
        expected_H_vec = dim * (dim + 1) // 2
        assert H_vec_dims[0] == expected_H_vec, f"H_vec should be {expected_H_vec}"

        # Check C_symvecs shape: (S_vec, H_vec)
        S_vec = S_mat * (S_mat + 1) // 2
        assert C_symvecs[0].shape == (S_vec, expected_H_vec), \
            f"C_symvecs shape should be ({S_vec}, {expected_H_vec})"

        # Check d_vecs shape: (V, H_vec)
        assert d_vecs[0].shape == (V, expected_H_vec), \
            f"d_vecs shape should be ({V}, {expected_H_vec})"

        # Check PSD_mat_dims_list contains Python ints
        assert all(isinstance(d, int) for d in PSD_mat_dims_list), \
            "PSD_mat_dims_list should contain Python ints"

    def test_multiple_psd_constraints(self):
        """Test C and d matrix computation with M_psd=2."""
        M_psd = 2
        dim = 3
        S_mat = 5
        V = 6

        PSD_A_vals, PSD_b_vals, _, PSD_mat_dims = create_psd_constraint_data(
            M_psd, dim, S_mat, V
        )

        C_symvecs, d_vecs, H_vec_dims, PSD_mat_dims_list = compute_C_d_matrices(
            PSD_A_vals, PSD_b_vals, PSD_mat_dims
        )

        # Check dimensions
        assert len(C_symvecs) == M_psd, f"Should have {M_psd} C matrices"
        assert len(d_vecs) == M_psd, f"Should have {M_psd} d matrices"
        assert len(H_vec_dims) == M_psd, f"Should have {M_psd} H_vec dimensions"

        # Check expected H_vec for dim=3: 3*(3+1)/2 = 6
        expected_H_vec = dim * (dim + 1) // 2
        for i in range(M_psd):
            assert H_vec_dims[i] == expected_H_vec, \
                f"H_vec[{i}] should be {expected_H_vec}"


class TestPSDConstraints:
    """Test DRO solve with PSD constraints."""

    def test_expectation_with_psd_single_constraint(self):
        """Test expectation with single PSD constraint (M_psd=1, dim=2)."""
        data = setup_test_data()
        M_psd = 1
        dim = 2

        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_mat_dims = create_psd_constraint_data(
            M_psd, dim, data['S_mat'], data['V']
        )

        obj_val = dro_scs_solve(
            data['A_obj'], data['b_obj'], data['A_vals'],
            data['b_vals'], data['c_vals'],
            data['G_batch'], data['F_batch'],
            data['eps'], data['precond_inv'],
            risk_type='expectation',
            PSD_A_vals=PSD_A_vals,
            PSD_b_vals=PSD_b_vals,
            PSD_c_vals=PSD_c_vals,
            PSD_mat_dims=PSD_mat_dims,
        )

        assert jnp.isfinite(obj_val), "Objective value should be finite"
        assert isinstance(obj_val, jnp.ndarray), "Should return JAX array"

    def test_cvar_with_psd_single_constraint(self):
        """Test CVaR with single PSD constraint (M_psd=1, dim=2)."""
        data = setup_test_data()
        M_psd = 1
        dim = 2

        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_mat_dims = create_psd_constraint_data(
            M_psd, dim, data['S_mat'], data['V']
        )

        obj_val = dro_scs_solve(
            data['A_obj'], data['b_obj'], data['A_vals'],
            data['b_vals'], data['c_vals'],
            data['G_batch'], data['F_batch'],
            data['eps'], data['precond_inv'],
            risk_type='cvar',
            alpha=data['alpha'],
            PSD_A_vals=PSD_A_vals,
            PSD_b_vals=PSD_b_vals,
            PSD_c_vals=PSD_c_vals,
            PSD_mat_dims=PSD_mat_dims,
        )

        assert jnp.isfinite(obj_val), "Objective value should be finite"
        assert isinstance(obj_val, jnp.ndarray), "Should return JAX array"

    def test_expectation_with_psd_multiple_constraints(self):
        """Test expectation with multiple PSD constraints (M_psd=2, dim=3)."""
        data = setup_test_data()
        M_psd = 2
        dim = 3

        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_mat_dims = create_psd_constraint_data(
            M_psd, dim, data['S_mat'], data['V']
        )

        obj_val = dro_scs_solve(
            data['A_obj'], data['b_obj'], data['A_vals'],
            data['b_vals'], data['c_vals'],
            data['G_batch'], data['F_batch'],
            data['eps'], data['precond_inv'],
            risk_type='expectation',
            PSD_A_vals=PSD_A_vals,
            PSD_b_vals=PSD_b_vals,
            PSD_c_vals=PSD_c_vals,
            PSD_mat_dims=PSD_mat_dims,
        )

        assert jnp.isfinite(obj_val), "Objective value should be finite"
        assert isinstance(obj_val, jnp.ndarray), "Should return JAX array"


class TestJAXAutodiff:
    """Test that JAX autodiff still works with PSD constraints."""

    def test_gradient_expectation_with_psd(self):
        """Test that gradients can be computed with PSD constraints."""
        data = setup_test_data()
        M_psd = 1
        dim = 2

        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_mat_dims = create_psd_constraint_data(
            M_psd, dim, data['S_mat'], data['V']
        )

        def objective(eps_param):
            return dro_scs_solve(
                data['A_obj'], data['b_obj'], data['A_vals'],
                data['b_vals'], data['c_vals'],
                data['G_batch'], data['F_batch'],
                eps_param, data['precond_inv'],
                risk_type='expectation',
                PSD_A_vals=PSD_A_vals,
                PSD_b_vals=PSD_b_vals,
                PSD_c_vals=PSD_c_vals,
                PSD_mat_dims=PSD_mat_dims,
            )

        # Test forward pass
        obj_val = objective(data['eps'])
        assert jnp.isfinite(obj_val), "Forward pass should succeed"

        # Test gradient computation
        grad_fn = jax.grad(objective)
        grad_val = grad_fn(data['eps'])
        assert jnp.isfinite(grad_val), "Gradient should be finite"
        assert isinstance(grad_val, jnp.ndarray), "Gradient should be JAX array"

    def test_gradient_cvar_with_psd(self):
        """Test that gradients can be computed for CVaR with PSD constraints."""
        data = setup_test_data()
        M_psd = 1
        dim = 2

        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_mat_dims = create_psd_constraint_data(
            M_psd, dim, data['S_mat'], data['V']
        )

        def objective(alpha_param):
            return dro_scs_solve(
                data['A_obj'], data['b_obj'], data['A_vals'],
                data['b_vals'], data['c_vals'],
                data['G_batch'], data['F_batch'],
                data['eps'], data['precond_inv'],
                risk_type='cvar',
                alpha=alpha_param,
                PSD_A_vals=PSD_A_vals,
                PSD_b_vals=PSD_b_vals,
                PSD_c_vals=PSD_c_vals,
                PSD_mat_dims=PSD_mat_dims,
            )

        # Test forward pass
        obj_val = objective(data['alpha'])
        assert jnp.isfinite(obj_val), "Forward pass should succeed"

        # Test gradient computation
        grad_fn = jax.grad(objective)
        grad_val = grad_fn(data['alpha'])
        assert jnp.isfinite(grad_val), "Gradient should be finite"
        assert isinstance(grad_val, jnp.ndarray), "Gradient should be JAX array"


if __name__ == '__main__':
    """Run tests manually if executed as script."""
    print('='*60)
    print('Testing PSD Constraint Support in JAX SCS Layer')
    print('='*60)
    print()

    # Test backward compatibility
    print('TEST 1: Backward Compatibility')
    print('-'*60)
    test = TestBackwardCompatibility()
    test.test_expectation_without_psd()
    print('✓ Expectation without PSD constraints')
    test.test_cvar_without_psd()
    print('✓ CVaR without PSD constraints')
    print()

    # Test compute_C_d_matrices
    print('TEST 2: Compute C and d Matrices')
    print('-'*60)
    test = TestComputeCdMatrices()
    test.test_single_psd_constraint()
    print('✓ Single PSD constraint (M_psd=1, dim=2)')
    test.test_multiple_psd_constraints()
    print('✓ Multiple PSD constraints (M_psd=2, dim=3)')
    print()

    # Test PSD constraints
    print('TEST 3: DRO Solve with PSD Constraints')
    print('-'*60)
    test = TestPSDConstraints()
    test.test_expectation_with_psd_single_constraint()
    print('✓ Expectation with single PSD constraint')
    test.test_cvar_with_psd_single_constraint()
    print('✓ CVaR with single PSD constraint')
    test.test_expectation_with_psd_multiple_constraints()
    print('✓ Expectation with multiple PSD constraints')
    print()

    # Test JAX autodiff
    print('TEST 4: JAX Autodiff Compatibility')
    print('-'*60)
    test = TestJAXAutodiff()
    test.test_gradient_expectation_with_psd()
    print('✓ Gradient computation (expectation with PSD)')
    test.test_gradient_cvar_with_psd()
    print('✓ Gradient computation (CVaR with PSD)')
    print()

    print('='*60)
    print('✓✓✓ ALL TESTS PASSED ✓✓✓')
    print('='*60)
