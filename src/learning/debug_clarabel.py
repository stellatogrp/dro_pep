"""
Diagnostic script to trace why Clarabel is returning -1.0 after diffcp upgrade.
Run from src/ directory.
"""

import os
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse as spa

# Import needed modules (avoid quad.py to prevent hang)
from learning.autodiff_setup import (
    problem_data_to_gd_trajectories,
    compute_preconditioner_from_samples,
)
from learning.pep_construction import construct_gd_pep_data
from learning.jax_clarabel_layer import dro_clarabel_solve


def gd_pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj):
    t = stepsizes[0]
    return construct_gd_pep_data(t, mu, L, R, K_max, pep_obj)


def generate_samples(N, d, mu, L, R, seed=50):
    """Generate Q and z0 samples."""
    key = jax.random.PRNGKey(seed)
    Q_batch = []
    z0_batch = []
    
    for i in range(N):
        key, k1, k2, k3 = jax.random.split(key, 4)
        V = jax.random.normal(k1, (d, d))
        V, _ = jnp.linalg.qr(V)
        eigvals = jax.random.uniform(k2, (d,), minval=mu, maxval=L)
        Q = V @ jnp.diag(eigvals) @ V.T
        Q_batch.append(Q)
        z0 = jax.random.normal(k3, (d,))
        z0 = z0 / jnp.linalg.norm(z0) * R * 0.9
        z0_batch.append(z0)
    
    return jnp.stack(Q_batch), jnp.stack(z0_batch)


def run_clarabel_pipeline(stepsizes, Q_batch, z0_batch, zs_batch, fs_batch,
                          mu, L, R, K_max, eps, pep_obj, precond_inv):
    """Run Clarabel pipeline."""
    batch_GF_func = jax.vmap(
        lambda Q, z0, zs, fs: problem_data_to_gd_trajectories(
            stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True
        ),
        in_axes=(0, 0, 0, 0)
    )
    G_batch, F_batch = batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)
    
    pep_data = gd_pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj)
    A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
    
    loss = dro_clarabel_solve(
        A_obj=A_obj, b_obj=b_obj,
        A_vals=A_vals, b_vals=b_vals, c_vals=c_vals,
        G_batch=G_batch, F_batch=F_batch,
        eps=eps, precond_inv=precond_inv,
        dro_obj='expectation', alpha=0.1,
    )
    return loss


def test_diffcp_directly():
    """Test diffcp.solve_and_derivative directly with Clarabel."""
    import diffcp
    
    print("=== Testing diffcp directly with Clarabel ===")
    print(f"diffcp version: {diffcp.__version__}")
    
    # Create test problem
    params = {
        'mu': 1., 'L': 10., 'R': 1.0, 'K_max': 2,
        'N': 8, 'd': 10, 'eps': 0.1, 'pep_obj': 'obj_val',
    }
    mu, L, R = params['mu'], params['L'], params['R']
    K_max, N, d, eps = params['K_max'], params['N'], params['d'], params['eps']
    pep_obj = params['pep_obj']
    
    Q_batch, z0_batch = generate_samples(N, d, mu, L, R)
    zs_batch = jnp.zeros((N, d))
    fs_batch = jnp.zeros(N)
    
    t_scalar = 2.0 / (L + mu)
    stepsizes = (jnp.array(t_scalar),)
    
    # Compute G_batch, F_batch
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
    
    print(f"\nProblem setup: N={N}, d={d}, K={K_max}, eps={eps}")
    print(f"G_batch shape: {G_batch.shape}")
    print(f"F_batch shape: {F_batch.shape}")
    
    # Now test clarabel pipeline
    print("\n=== Testing Clarabel pipeline ===")
    try:
        loss = run_clarabel_pipeline(
            stepsizes, Q_batch, z0_batch, zs_batch, fs_batch,
            mu, L, R, K_max, eps, pep_obj, precond_inv_jax
        )
        print(f"Clarabel loss: {float(loss)}")
    except Exception as e:
        print(f"Clarabel pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare with what cvxpylayers would produce
    print("\n=== Comparing with cvxpylayers internal data ===")
    import diffcp_patch  # Needed for cvxpylayers
    from learning.autodiff_setup import create_full_dro_exp_layer
    
    pep_data = gd_pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj)
    A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
    
    # Create cvxpylayer
    M_constr = A_vals.shape[0]
    mat_shape = (A_obj.shape[0], A_obj.shape[1])
    vec_shape = (b_obj.shape[0],)
    
    cvxpy_layer = create_full_dro_exp_layer(
        M_constr, N, mat_shape, vec_shape, mat_shape, vec_shape,
        np.array(c_vals), precond_inv_np, eps
    )
    
    # Get internal cvxpy problem structure
    print(f"cvxpylayer type: {type(cvxpy_layer)}")
    print(f"cvxpylayer attributes: {[a for a in dir(cvxpy_layer) if not a.startswith('_')]}")
    
    # Access the internal problem through the layer
    if hasattr(cvxpy_layer, 'layer'):
        prob = cvxpy_layer.layer.problem
    elif hasattr(cvxpy_layer, 'problem'):
        prob = cvxpy_layer.problem
    else:
        # Try to find the actual cvxpylayer
        prob = cvxpy_layer._problem if hasattr(cvxpy_layer, '_problem') else None
        if prob is None:
            # The wrapper stores the actual layer
            print("Trying to access inner layer...")
            inner = cvxpy_layer
            for attr in ['layer', 'cvxpylayer', 'cvxpy_layer']:
                if hasattr(inner, attr):
                    inner = getattr(inner, attr)
                    if hasattr(inner, 'problem'):
                        prob = inner.problem
                        break
    
    if prob is not None:
        print(f"  is_dcp: {prob.is_dcp()}")
        print(f"  variables: {[v.size for v in prob.variables()]}")
        
        # Get more info about the cone program generated by cvxpylayers
        from cvxpy import SCS
        
        # Try to get the data that would be sent to solver
        data, chain, inverse_data = prob.get_problem_data(solver=SCS)
        print(f"\nSCS problem data (from cvxpylayers):")
        print(f"  A shape: {data['A'].shape}")
        print(f"  b shape: {data['b'].shape}")
        print(f"  c shape: {data['c'].shape}")
        print(f"  dims: {data.get('dims', 'N/A')}")
    else:
        print("Could not access cvxpylayers problem")
    
    # Call the cvxpylayer with samples
    print("\n=== Calling cvxpylayer with SCS (should work) ===")
    batch_GF_func = jax.vmap(
        lambda Q, z0, zs, fs: problem_data_to_gd_trajectories(
            stepsizes, Q, z0, zs, fs, K_max, return_Gram_representation=True
        ),
        in_axes=(0, 0, 0, 0)
    )
    G_batch_t, F_batch_t = batch_GF_func(Q_batch, z0_batch, zs_batch, fs_batch)
    
    # Build params list
    params_list = (
        [A_vals[m] for m in range(M_constr)] +
        [b_vals[m] for m in range(M_constr)] +
        [A_obj, b_obj] +
        [G_batch_t[i] for i in range(N)] +
        [F_batch_t[i] for i in range(N)]
    )
    
    try:
        (lambd_star, s_star) = cvxpy_layer(*params_list)
        from learning.autodiff_setup import dro_pep_obj_jax
        cvxpy_loss = float(dro_pep_obj_jax(eps, lambd_star, s_star))
        print(f"cvxpylayer (SCS) loss: {cvxpy_loss}")
        print(f"  lambd_star: {float(lambd_star)}")
        print(f"  s_star sum: {float(jnp.sum(s_star))}")
    except Exception as e:
        print(f"cvxpylayer (SCS) call failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Now test cvxpylayer with CLARABEL backend (testing the diffcp fix)
    print("\n=== Calling cvxpylayer with CLARABEL (testing diffcp fix) ===")
    try:
        from cvxpy import CLARABEL
        from cvxpylayers.jax import CvxpyLayer
        
        # Create a new layer with Clarabel solver
        # Need to rebuild the problem since cvxpylayers needs solver at init time
        cvxpy_layer_clarabel = create_full_dro_exp_layer(
            M_constr, N, mat_shape, vec_shape, mat_shape, vec_shape,
            np.array(c_vals), precond_inv_np, eps
        )
        
        # The solver is passed via solver_args - check if this works
        (lambd_clar, s_clar) = cvxpy_layer_clarabel(*params_list, solver_args={'solver': 'CLARABEL'})
        cvxpy_clar_loss = float(dro_pep_obj_jax(eps, lambd_clar, s_clar))
        print(f"cvxpylayer (CLARABEL) loss: {cvxpy_clar_loss}")
        print(f"  lambd_star: {float(lambd_clar)}")
        print(f"  s_star sum: {float(jnp.sum(s_clar))}")
        
        # Compare with SCS
        print(f"\nComparison SCS vs CLARABEL:")
        print(f"  Loss diff: {abs(cvxpy_loss - cvxpy_clar_loss):.2e}")
    except Exception as e:
        print(f"cvxpylayer (CLARABEL) call failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Extract actual cvxpy problem data for comparison
    print("\n=== Extracting cvxpylayers internal data ===")
    
    # Build the same problem as create_full_dro_exp_layer but with actual values set
    import cvxpy as cp
    
    M = M_constr
    S_mat = int(A_obj.shape[0])
    
    # Variables
    lambd = cp.Variable()
    s = cp.Variable(N)
    y = cp.Variable((N, M))
    Gz = [cp.Variable((S_mat, S_mat), symmetric=True) for _ in range(N)]
    Fz = [cp.Variable(int(b_obj.shape[0])) for _ in range(N)]
    
    # Objective
    obj = lambd * eps + 1.0 / N * cp.sum(s)
    
    # Constraints
    constraints = [y >= 0]
    
    G_preconditioner = np.diag(precond_inv_np[0])
    F_preconditioner = precond_inv_np[1]
    
    for i in range(N):
        G_sample = np.array(G_batch_t[i])
        F_sample = np.array(F_batch_t[i])
        
        # Epigraph constraint
        constraints += [- np.array(c_vals).T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
        
        # SOC constraint
        constraints += [cp.SOC(lambd, cp.hstack([
            cp.vec(G_preconditioner @ Gz[i] @ G_preconditioner, order='F'),
            cp.multiply(F_preconditioner**2, Fz[i])
        ]))]
        
        # L* constraint
        LstarG = 0
        LstarF = 0
        for m in range(M):
            LstarG = LstarG + y[i, m] * np.array(A_vals[m])
            LstarF = LstarF + y[i, m] * np.array(b_vals[m])
        
        constraints += [LstarG - Gz[i] - np.array(A_obj) >> 0]
        constraints += [LstarF - Fz[i] - np.array(b_obj) == 0]
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    
    # Get data for SCS
    from cvxpy import SCS
    data, chain, inverse_data = prob.get_problem_data(solver=SCS)
    
    print(f"cvxpy problem (recreated) SCS data:")
    print(f"  A: {data['A'].shape}")
    print(f"  b: {data['b'].shape}")
    print(f"  c: {data['c'].shape}")
    print(f"  dims: {data.get('dims', 'N/A')}")
    
    # Compare with jax_canonicalize
    print(f"\njax_canonicalize shapes:")
    print(f"  A: {A_dense.shape}")
    print(f"  b: {b_can.shape}")
    print(f"  c: {c_can.shape}")
    
    # Numerical comparison
    A_cvxpy_dense = data['A'].toarray() if hasattr(data['A'], 'toarray') else np.array(data['A'])
    A_jax = np.array(A_dense)
    
    print(f"\nShape comparison:")
    print(f"  cvxpy A: {A_cvxpy_dense.shape}")
    print(f"  jax A: {A_jax.shape}")
    
    if A_cvxpy_dense.shape == A_jax.shape:
        A_diff = np.abs(A_cvxpy_dense - A_jax).max()
        print(f"  Max A difference: {A_diff}")
    else:
        print(f"  A shapes DON'T MATCH!")
    
    # Test diffcp directly with a simple LP
    print("\n=== Testing diffcp directly with simple LP ===")
    A = spa.csc_matrix(np.array([[-1., 0.], [0., -1.], [1., 1.]]))
    b = np.array([0., 0., 1.])
    c = np.array([1., 1.])
    cone_dict = {'l': 3}
    
    try:
        x, y, s, D, DT = diffcp.solve_and_derivative(
            A, b, c, cone_dict,
            solve_method='Clarabel',
            verbose=True,
        )
        print(f"Simple LP x: {x}")
        print(f"Simple LP objective: {c @ x}")
    except Exception as e:
        print(f"Simple LP failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Now test the DRO problem with verbose Clarabel
    print("\n=== Testing DRO problem with verbose diffcp/Clarabel ===")
    from learning.jax_clarabel_layer import jax_canonicalize_dro_expectation, ClarabelSolveData
    
    pep_data = gd_pep_data_fn(stepsizes, mu, L, R, K_max, pep_obj)
    A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
    
    A_dense, b_can, c_can, x_dim, cone_info = jax_canonicalize_dro_expectation(
        A_obj, b_obj, A_vals, b_vals, c_vals,
        G_batch, F_batch,
        eps, precond_inv_jax,
    )
    
    print(f"Canonicalized problem shapes:")
    print(f"  A: {A_dense.shape}")
    print(f"  b: {b_can.shape}")
    print(f"  c: {c_can.shape}")
    print(f"  cone_info: {cone_info}")
    
    # Build cone dict
    static_data = ClarabelSolveData(cone_info, A_dense.shape)
    print(f"  diffcp_cone_dict: {static_data.diffcp_cone_dict}")
    
    A_csc = spa.csc_matrix(np.array(A_dense))
    b_np = np.array(b_can)
    c_np = np.array(c_can)
    
    # Detailed row count verification
    print(f"\n=== Row count verification ===")
    print(f"Total A rows: {A_dense.shape[0]}")
    
    # Expected row counts based on diffcp CONES order: ZERO, POS, SOC, PSD
    zero_rows = int(cone_info['zero'])
    nonneg_rows = int(cone_info['nonneg'])
    socp_dims = [int(d) for d in cone_info['soc']]
    psd_dims = [int(d) for d in cone_info['psd']]
    
    soc_rows = sum(socp_dims)
    psd_rows_expected = sum(d * (d + 1) // 2 for d in psd_dims)
    
    total_expected = zero_rows + nonneg_rows + soc_rows + psd_rows_expected
    
    print(f"Expected row breakdown (diffcp order: ZERO, POS, SOC, PSD):")
    print(f"  ZERO (equality): {zero_rows} rows")
    print(f"  POS (nonneg): {nonneg_rows} rows")
    print(f"  SOC: {soc_rows} rows (dims: {socp_dims})")
    print(f"  PSD: {psd_rows_expected} rows (dims: {psd_dims}, svec dims: {[d*(d+1)//2 for d in psd_dims]})")
    print(f"  Total expected: {total_expected}")
    print(f"  Total actual: {A_dense.shape[0]}")
    
    if total_expected != A_dense.shape[0]:
        print(f"  *** MISMATCH! ***")
    else:
        print(f"  ✓ Row counts match")
    
    print(f"\nCalling diffcp.solve_and_derivative with Clarabel (verbose)...")
    try:
        x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
            A_csc, b_np, c_np,
            static_data.diffcp_cone_dict,
            solve_method='Clarabel',
            verbose=True,
        )
        obj = c_np @ x
        print(f"Clarabel solved! x[:5]: {x[:5]}")
        print(f"Clarabel objective c @ x: {obj}")
    except Exception as e:
        print(f"DRO solve (Clarabel) failed: {e}")
        import traceback
        traceback.print_exc()

    # Now test the SAME problem with SCS through diffcp
    print("\n=== Testing SAME DRO problem with SCS through diffcp ===")
    A_csc2 = spa.csc_matrix(np.array(A_dense))  # Fresh copy
    b_np2 = np.array(b_can)
    c_np2 = np.array(c_can)
    
    try:
        x_scs, y_scs, s_scs, _, adjoint_scs = diffcp.solve_and_derivative(
            A_csc2, b_np2, c_np2,
            static_data.diffcp_cone_dict,
            solve_method='SCS',
            verbose=True,
        )
        obj_scs = c_np2 @ x_scs
        print(f"SCS solved! x[:5]: {x_scs[:5]}")
        print(f"SCS objective c @ x: {obj_scs}")
    except Exception as e:
        print(f"DRO solve (SCS) failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_diffcp_directly()
    os._exit(0)
