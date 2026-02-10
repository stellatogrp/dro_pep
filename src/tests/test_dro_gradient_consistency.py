"""
Unit tests comparing cvxpylayer DRO (SCS) vs JAX/Clarabel DRO pipelines.

NOTE: This test avoids importing from quad.py to prevent diffcp_patch 
from being loaded at module level, which causes hangs in test environments.
"""

import pytest
import numpy as np
import os
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning.trajectories_gd_fgm import (
    problem_data_to_gd_trajectories,
    compute_preconditioner_from_samples,
    dro_pep_obj_jax,
)
from learning.cvxpylayers_setup import create_full_dro_exp_layer
from learning.pep_constructions import construct_gd_pep_data
from learning.jax_clarabel_layer import dro_clarabel_solve


def gd_pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj):
    """Wrapper for construct_gd_pep_data that takes stepsizes tuple."""
    t = stepsizes[0]
    return construct_gd_pep_data(t, mu, L, R, K_max, pep_obj)


def generate_samples(N, d, mu, L, R, seed=50):
    """Generate Q and z0 samples for testing (without quad.py/diffcp_patch)."""
    key = jax.random.PRNGKey(seed)
    Q_batch = []
    z0_batch = []
    
    for i in range(N):
        key, k1, k2, k3 = jax.random.split(key, 4)
        # Generate random orthonormal matrix
        V = jax.random.normal(k1, (d, d))
        V, _ = jnp.linalg.qr(V)
        # Eigenvalues uniformly between mu and L
        eigvals = jax.random.uniform(k2, (d,), minval=mu, maxval=L)
        Q = V @ jnp.diag(eigvals) @ V.T
        Q_batch.append(Q)
        # Initial point
        z0 = jax.random.normal(k3, (d,))
        z0 = z0 / jnp.linalg.norm(z0) * R * 0.9  # Inside ball
        z0_batch.append(z0)
    
    return jnp.stack(Q_batch), jnp.stack(z0_batch)


def get_problem_params():
    """Problem parameters matching working quad.yaml config."""
    return {
        'mu': 1.,
        'L': 10.,
        'R': 1.0,
        'K_max': 2,
        'N': 8,  # Smaller for tests
        'd': 10,  # Smaller for tests
        'eps': 0.1,
        'pep_obj': 'obj_val',
    }


def run_scs_pipeline(stepsizes, Q_batch, z0_batch, zs_batch, fs_batch, 
                     mu, L, R, K_max, eps, pep_obj, layer, N):
    """Run SCS pipeline (matching quad.py's full_SCS_pipeline structure)."""
    # 1. Compute sample Gram matrices
    batch_GF_func = jax.vmap(
        lambda Q, z0, zs, fs: problem_data_to_gd_trajectories(
            stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True
        ),
        in_axes=(0, 0, 0, 0)
    )
    G_batch, F_batch = batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)
    
    # 2. Compute constraint matrices
    pep_data = gd_pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj)
    A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
    
    # 3. Build parameter list
    M = A_vals.shape[0]
    params_list = (
        [A_vals[m] for m in range(M)] +
        [b_vals[m] for m in range(M)] +
        [A_obj, b_obj] +
        [G_batch[i] for i in range(N)] +
        [F_batch[i] for i in range(N)]
    )
    
    # 4. Call layer
    (lambd_star, s_star) = layer(*params_list)
    loss = dro_pep_obj_jax(eps, lambd_star, s_star)
    return loss


def run_clarabel_pipeline(stepsizes, Q_batch, z0_batch, zs_batch, fs_batch,
                          mu, L, R, K_max, eps, pep_obj, precond_inv):
    """Run Clarabel pipeline (matching quad.py's full_clarabel_pipeline structure)."""
    # 1. Compute sample Gram matrices
    batch_GF_func = jax.vmap(
        lambda Q, z0, zs, fs: problem_data_to_gd_trajectories(
            stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True
        ),
        in_axes=(0, 0, 0, 0)
    )
    G_batch, F_batch = batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)
    
    # 2. Compute constraint matrices
    pep_data = gd_pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj)
    A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
    
    # 3. Call dro_clarabel_solve
    loss = dro_clarabel_solve(
        A_obj=A_obj, b_obj=b_obj,
        A_vals=A_vals, b_vals=b_vals, c_vals=c_vals,
        G_batch=G_batch, F_batch=F_batch,
        eps=eps, precond_inv=precond_inv,
        dro_obj='expectation', alpha=0.1,
    )
    return loss


class TestDROGradientConsistency:
    """Compare cvxpylayer (SCS) and JAX/Clarabel DRO pipelines."""

    def test_scs_gradients_correct_sign(self):
        """Test that SCS gradient has correct sign (decreasing direction)."""
        import diffcp_patch  # Import here so test collection works
        
        params = get_problem_params()
        mu, L, R = params['mu'], params['L'], params['R']
        K_max, N, d, eps = params['K_max'], params['N'], params['d'], params['eps']
        pep_obj = params['pep_obj']
        
        print(f"Testing SCS with N={N}, d={d}, K={K_max}")
        
        Q_batch, z0_batch = generate_samples(N, d, mu, L, R)
        zs_batch = jnp.zeros((N, d))
        fs_batch = jnp.zeros(N)
        
        t_scalar = 2.0 / (L + mu)
        stepsizes = (jnp.array(t_scalar),)
        
        # Compute preconditioner
        batch_GF_func = jax.vmap(
            lambda Q, z0, zs, fs: problem_data_to_gd_trajectories(
                stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True
            ),
            in_axes=(0, 0, 0, 0)
        )
        G_batch, F_batch = batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)
        precond_inv = compute_preconditioner_from_samples(
            np.array(G_batch), np.array(F_batch), precond_type='identity'
        )
        
        # Create layer
        pep_data = gd_pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj)
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        M_constr = A_vals.shape[0]
        mat_shape = (A_obj.shape[0], A_obj.shape[1])
        vec_shape = (b_obj.shape[0],)
        
        layer = create_full_dro_exp_layer(
            M_constr, N, mat_shape, vec_shape, mat_shape, vec_shape,
            np.array(c_vals), precond_inv, eps
        )
        
        # Define loss function
        def loss_fn(t):
            stepsizes_var = (t,)
            return run_scs_pipeline(
                stepsizes_var, Q_batch, z0_batch, zs_batch, fs_batch,
                mu, L, R, K_max, eps, pep_obj, layer, N
            )
        
        t_test = jnp.array(t_scalar)
        
        # Compute autodiff gradient
        loss_val = float(loss_fn(t_test))
        grad_val = float(jax.grad(loss_fn)(t_test))
        
        # Compute numerical gradient
        delta = 1e-4
        loss_plus = float(loss_fn(t_test + delta))
        loss_minus = float(loss_fn(t_test - delta))
        numerical_grad = (loss_plus - loss_minus) / (2 * delta)
        
        print(f"Loss at t={t_scalar}: {loss_val}")
        print(f"Autodiff gradient: {grad_val}")
        print(f"Numerical gradient: {numerical_grad}")
        
        # Check signs match
        if abs(numerical_grad) > 1e-6:
            assert np.sign(grad_val) == np.sign(numerical_grad), \
                f"Sign mismatch: autodiff={grad_val}, numerical={numerical_grad}"
            np.testing.assert_allclose(grad_val, numerical_grad, rtol=0.2)
            print("SCS gradient test PASSED!")

    def test_scs_clarabel_objectives_match(self):
        """Test that SCS and Clarabel pipelines produce the same objective value."""
        import diffcp_patch  # Import here for SCS pipeline
        
        params = get_problem_params()
        mu, L, R = params['mu'], params['L'], params['R']
        K_max, N, d, eps = params['K_max'], params['N'], params['d'], params['eps']
        pep_obj = params['pep_obj']
        
        print(f"\nComparing SCS vs Clarabel with N={N}, d={d}, K={K_max}")
        
        Q_batch, z0_batch = generate_samples(N, d, mu, L, R)
        zs_batch = jnp.zeros((N, d))
        fs_batch = jnp.zeros(N)
        
        t_scalar = 2.0 / (L + mu)
        stepsizes = (jnp.array(t_scalar),)
        
        # Compute preconditioner
        batch_GF_func = jax.vmap(
            lambda Q, z0, zs, fs: problem_data_to_gd_trajectories(
                stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True
            ),
            in_axes=(0, 0, 0, 0)
        )
        G_batch, F_batch = batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)
        precond_inv_np = compute_preconditioner_from_samples(
            np.array(G_batch), np.array(F_batch), precond_type='identity'
        )
        precond_inv_jax = (jnp.array(precond_inv_np[0]), jnp.array(precond_inv_np[1]))
        
        # Create SCS layer
        pep_data = gd_pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj)
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        M_constr = A_vals.shape[0]
        mat_shape = (A_obj.shape[0], A_obj.shape[1])
        vec_shape = (b_obj.shape[0],)
        
        scs_layer = create_full_dro_exp_layer(
            M_constr, N, mat_shape, vec_shape, mat_shape, vec_shape,
            np.array(c_vals), precond_inv_np, eps
        )
        
        # Run SCS pipeline
        scs_loss = run_scs_pipeline(
            stepsizes, Q_batch, z0_batch, zs_batch, fs_batch,
            mu, L, R, K_max, eps, pep_obj, scs_layer, N
        )
        scs_val = float(scs_loss)
        
        # Run Clarabel pipeline
        clarabel_loss = run_clarabel_pipeline(
            stepsizes, Q_batch, z0_batch, zs_batch, fs_batch,
            mu, L, R, K_max, eps, pep_obj, precond_inv_jax
        )
        clarabel_val = float(clarabel_loss)
        
        print(f"SCS loss: {scs_val}")
        print(f"Clarabel loss: {clarabel_val}")
        
        # They should match approximately
        np.testing.assert_allclose(
            scs_val, clarabel_val, rtol=1e-2,
            err_msg=f"SCS: {scs_val}, Clarabel: {clarabel_val}"
        )
        print("Objectives match - PASSED!")

    def test_scs_clarabel_gradients_match(self):
        """Test that SCS and Clarabel pipelines produce the same gradient."""
        import diffcp_patch  # Import here for SCS pipeline
        
        params = get_problem_params()
        mu, L, R = params['mu'], params['L'], params['R']
        K_max, N, d, eps = params['K_max'], params['N'], params['d'], params['eps']
        pep_obj = params['pep_obj']
        
        print(f"\nComparing SCS vs Clarabel gradients with N={N}, d={d}, K={K_max}")
        
        Q_batch, z0_batch = generate_samples(N, d, mu, L, R)
        zs_batch = jnp.zeros((N, d))
        fs_batch = jnp.zeros(N)
        
        t_scalar = 2.0 / (L + mu)
        stepsizes = (jnp.array(t_scalar),)
        
        # Compute preconditioner
        batch_GF_func = jax.vmap(
            lambda Q, z0, zs, fs: problem_data_to_gd_trajectories(
                stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True
            ),
            in_axes=(0, 0, 0, 0)
        )
        G_batch, F_batch = batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)
        precond_inv_np = compute_preconditioner_from_samples(
            np.array(G_batch), np.array(F_batch), precond_type='identity'
        )
        precond_inv_jax = (jnp.array(precond_inv_np[0]), jnp.array(precond_inv_np[1]))
        
        # Create SCS layer
        pep_data = gd_pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj)
        A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
        M_constr = A_vals.shape[0]
        mat_shape = (A_obj.shape[0], A_obj.shape[1])
        vec_shape = (b_obj.shape[0],)
        
        scs_layer = create_full_dro_exp_layer(
            M_constr, N, mat_shape, vec_shape, mat_shape, vec_shape,
            np.array(c_vals), precond_inv_np, eps
        )
        
        # Define loss functions
        def scs_loss_fn(t):
            stepsizes_var = (t,)
            return run_scs_pipeline(
                stepsizes_var, Q_batch, z0_batch, zs_batch, fs_batch,
                mu, L, R, K_max, eps, pep_obj, scs_layer, N
            )
        
        def clarabel_loss_fn(t):
            stepsizes_var = (t,)
            return run_clarabel_pipeline(
                stepsizes_var, Q_batch, z0_batch, zs_batch, fs_batch,
                mu, L, R, K_max, eps, pep_obj, precond_inv_jax
            )
        
        t_test = jnp.array(t_scalar)
        
        # Compute gradients
        scs_grad = float(jax.grad(scs_loss_fn)(t_test))
        clarabel_grad = float(jax.grad(clarabel_loss_fn)(t_test))
        
        # Numerical gradient check for both
        delta = 1e-4
        scs_plus = float(scs_loss_fn(t_test + delta))
        scs_minus = float(scs_loss_fn(t_test - delta))
        scs_numerical = (scs_plus - scs_minus) / (2 * delta)
        
        clarabel_plus = float(clarabel_loss_fn(t_test + delta))
        clarabel_minus = float(clarabel_loss_fn(t_test - delta))
        clarabel_numerical = (clarabel_plus - clarabel_minus) / (2 * delta)
        
        print(f"SCS gradient (autodiff): {scs_grad}")
        print(f"SCS gradient (numerical): {scs_numerical}")
        print(f"Clarabel gradient (autodiff): {clarabel_grad}")
        print(f"Clarabel gradient (numerical): {clarabel_numerical}")
        
        ratio_autodiff = scs_grad / clarabel_grad if abs(clarabel_grad) > 1e-10 else float('inf')
        print(f"Ratio autodiff (SCS/Clarabel): {ratio_autodiff}")
        
        # Check numerical gradients match (they should!)
        print(f"\nNumerical gradients should be close:")
        print(f"  SCS numerical: {scs_numerical}")
        print(f"  Clarabel numerical: {clarabel_numerical}")
        
        # The NUMERICAL gradients should match since objectives match
        np.testing.assert_allclose(
            scs_numerical, clarabel_numerical, rtol=1e-2,
            err_msg=f"Numerical grads should match: SCS={scs_numerical}, Clarabel={clarabel_numerical}"
        )
        
        # Check autodiff matches numerical for each
        print(f"\nChecking autodiff vs numerical for each solver:")
        print(f"  SCS: autodiff={scs_grad}, numerical={scs_numerical}")
        print(f"  Clarabel: autodiff={clarabel_grad}, numerical={clarabel_numerical}")
        
        np.testing.assert_allclose(
            scs_grad, scs_numerical, rtol=0.2,
            err_msg=f"SCS autodiff doesn't match numerical"
        )
        np.testing.assert_allclose(
            clarabel_grad, clarabel_numerical, rtol=0.2,
            err_msg=f"Clarabel autodiff doesn't match numerical"
        )


if __name__ == '__main__':
    import sys
    t = TestDROGradientConsistency()
    try:
        t.test_scs_gradients_correct_sign()
        t.test_scs_clarabel_objectives_match()
        t.test_scs_clarabel_gradients_match()
        print("\n=== ALL TESTS PASSED ===")
    except Exception as e:
        print(f"\n=== TEST FAILED: {e} ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    os._exit(0)  # Force clean exit
