"""
Compare A matrices between cvxpy canonicalization and jax_canonicalize.
Run from src/ directory.
"""

import os
os.chdir('/Users/vranjan/Research/2026/dro_pep/src')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import cvxpy as cp

from learning.jax_clarabel_layer import jax_canonicalize_dro_expectation, ClarabelSolveData
from learning.trajectories_gd_fgm import problem_data_to_gd_trajectories, compute_preconditioner_from_samples
from learning.pep_construction import construct_gd_pep_data


def generate_test_data():
    """Generate test data matching both pipelines."""
    N, d, K_max = 8, 10, 2
    mu, L, R, eps = 1., 10., 1., 0.1
    
    key = jax.random.PRNGKey(50)
    Q_batch, z0_batch = [], []
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
    Q_batch = jnp.stack(Q_batch)
    z0_batch = jnp.stack(z0_batch)
    zs_batch = jnp.zeros((N, d))
    fs_batch = jnp.zeros(N)
    
    t_scalar = 2.0 / (L + mu)
    stepsizes = (jnp.array(t_scalar),)
    
    # Compute G, F
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
    
    # Get PEP data
    pep_data = construct_gd_pep_data(stepsizes[0], mu, L, R, K_max, 'obj_val')
    A_obj, b_obj, A_vals, b_vals, c_vals, _, _, _, _ = pep_data
    
    return {
        'N': N, 'd': d, 'K_max': K_max, 'eps': eps,
        'mu': mu, 'L': L, 'R': R,
        'G_batch': G_batch, 'F_batch': F_batch,
        'A_obj': A_obj, 'b_obj': b_obj, 'A_vals': A_vals, 'b_vals': b_vals, 'c_vals': c_vals,
        'precond_inv_np': precond_inv_np, 'precond_inv_jax': precond_inv_jax,
    }


def build_cvxpy_problem_with_values(data):
    """Build and solve cvxpy problem with actual values set, then get problem data."""
    N = data['N']
    M = data['A_vals'].shape[0]
    S_mat = int(data['A_obj'].shape[0])
    V_dim = int(data['b_obj'].shape[0])
    eps = data['eps']
    
    G_batch = data['G_batch']
    F_batch = data['F_batch']
    A_obj = np.array(data['A_obj'])
    b_obj = np.array(data['b_obj'])
    A_vals = np.array(data['A_vals'])
    b_vals = np.array(data['b_vals'])
    c_vals = np.array(data['c_vals'])
    precond_inv_np = data['precond_inv_np']
    
    # Variables
    lambd = cp.Variable()
    s = cp.Variable(N)
    y = cp.Variable((N, M))
    Gz = [cp.Variable((S_mat, S_mat), symmetric=True) for _ in range(N)]
    Fz = [cp.Variable(V_dim) for _ in range(N)]
    
    # Objective
    obj = lambd * eps + 1.0 / N * cp.sum(s)
    
    # Constraints
    constraints = [y >= 0]
    
    G_preconditioner = np.diag(precond_inv_np[0])
    F_preconditioner = precond_inv_np[1]
    
    for i in range(N):
        G_sample = np.array(G_batch[i])
        F_sample = np.array(F_batch[i])
        
        # Epigraph constraint
        constraints += [- c_vals.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
        
        # SOC constraint
        constraints += [cp.SOC(lambd, cp.hstack([
            cp.vec(G_preconditioner @ Gz[i] @ G_preconditioner, order='F'),
            cp.multiply(F_preconditioner**2, Fz[i])
        ]))]
        
        # L* constraint
        LstarG = 0
        LstarF = 0
        for m in range(M):
            LstarG = LstarG + y[i, m] * A_vals[m]
            LstarF = LstarF + y[i, m] * b_vals[m]
        
        constraints += [LstarG - Gz[i] - A_obj >> 0]
        constraints += [LstarF - Fz[i] - b_obj == 0]
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    
    # Solve with CLARABEL to verify feasibility
    print("Solving cvxpy problem with CLARABEL...")
    result = prob.solve(solver=cp.CLARABEL, verbose=False)
    print(f"  Status: {prob.status}")
    print(f"  Optimal value: {result}")
    print(f"  lambd: {lambd.value}")
    
    # Get CLARABEL solver data for comparison
    print("\nExtracting CLARABEL problem data...")
    cvxpy_data, chain, inverse_data = prob.get_problem_data(solver=cp.CLARABEL)
    
    return cvxpy_data, prob


def get_jax_canonicalize_data(data):
    """Get A, b, c from jax_canonicalize."""
    A_dense, b_can, c_can, x_dim, cone_info = jax_canonicalize_dro_expectation(
        data['A_obj'], data['b_obj'], data['A_vals'], data['b_vals'], data['c_vals'],
        data['G_batch'], data['F_batch'],
        data['eps'], data['precond_inv_jax'],
    )
    return np.array(A_dense), np.array(b_can), np.array(c_can), cone_info


def compare_matrices(cvxpy_data, jax_A, jax_b, jax_c, jax_cone_info):
    """Compare matrices numerically."""
    cvxpy_A = cvxpy_data['A'].toarray() if hasattr(cvxpy_data['A'], 'toarray') else np.array(cvxpy_data['A'])
    cvxpy_b = np.array(cvxpy_data['b'])
    cvxpy_c = np.array(cvxpy_data['c'])
    
    print("\n=== Shape Comparison ===")
    print(f"cvxpy A: {cvxpy_A.shape}, jax A: {jax_A.shape}")
    
    if cvxpy_A.shape != jax_A.shape:
        print("A shapes don't match!")
        return
    
    # Compare overall
    A_diff = np.abs(cvxpy_A - jax_A)
    b_diff = np.abs(cvxpy_b - jax_b)
    c_diff = np.abs(cvxpy_c - jax_c)
    
    print("\n=== Numerical Comparison ===")
    print(f"A max diff: {A_diff.max():.6e}, c max diff: {c_diff.max():.6e}, b max diff: {b_diff.max():.6e}")
    
    # Block boundaries (based on diffcp CONES order: zero, nonneg, SOC, PSD)
    zero_dim = int(jax_cone_info['zero'])
    nonneg_dim = int(jax_cone_info['nonneg'])
    soc_dims = [int(d) for d in jax_cone_info['soc']]
    psd_dims = [int(d) for d in jax_cone_info['psd']]
    
    soc_total = sum(soc_dims)
    psd_total = sum(d * (d + 1) // 2 for d in psd_dims)
    
    zero_end = zero_dim
    nonneg_end = zero_end + nonneg_dim
    soc_end = nonneg_end + soc_total
    psd_end = soc_end + psd_total
    
    print(f"\n=== Block Analysis (rows) ===")
    print(f"zero [0:{zero_end}], nonneg [{zero_end}:{nonneg_end}], soc [{nonneg_end}:{soc_end}], psd [{soc_end}:{psd_end}]")
    
    blocks = [
        ("Zero/equality", 0, zero_end),
        ("Nonneg", zero_end, nonneg_end),
        ("SOC", nonneg_end, soc_end),
        ("PSD", soc_end, psd_end),
    ]
    
    for name, start, end in blocks:
        if end > start:
            A_block_diff = A_diff[start:end, :]
            b_block_diff = b_diff[start:end]
            print(f"\n{name} ({end-start} rows):")
            print(f"  A max diff: {A_block_diff.max():.6e}, b max diff: {b_block_diff.max():.6e}")
            
            # Show which rows have big differences
            row_max = A_block_diff.max(axis=1)
            big_rows = np.where(row_max > 0.1)[0]
            if len(big_rows) > 0:
                print(f"  Rows with A diff > 0.1: {len(big_rows)}")
    
    # Show specific row comparison for SOC (where we made changes)
    print(f"\n=== SOC Row Sample ===")
    soc_row = nonneg_end  # First SOC row (t-bound)
    print(f"First SOC row (row {soc_row}, t-bound):")
    cvxpy_nnz = np.nonzero(cvxpy_A[soc_row])[0]
    jax_nnz = np.nonzero(jax_A[soc_row])[0]
    print(f"  cvxpy nonzeros at cols: {cvxpy_nnz}")
    print(f"  jax nonzeros at cols: {jax_nnz}")
    print(f"  cvxpy vals: {cvxpy_A[soc_row, cvxpy_nnz]}")
    print(f"  jax vals: {jax_A[soc_row, jax_nnz]}")
    
    # Check a SOC row in the G part (after t and F)
    V_dim = int(jax_cone_info['V'])
    soc_G_row = nonneg_end + 1 + V_dim  # After t (1) and F part (V)
    print(f"\nFirst SOC G-part row (row {soc_G_row}):")
    cvxpy_nnz = np.nonzero(cvxpy_A[soc_G_row])[0]
    jax_nnz = np.nonzero(jax_A[soc_G_row])[0]
    print(f"  cvxpy nonzeros at cols: {cvxpy_nnz[:15]}")
    print(f"  jax nonzeros at cols: {jax_nnz[:15]}")
    print(f"  cvxpy vals: {cvxpy_A[soc_G_row, cvxpy_nnz[:8]]}")
    print(f"  jax vals: {jax_A[soc_G_row, jax_nnz[:8]]}")


if __name__ == '__main__':
    print("=== Generating test data ===")
    data = generate_test_data()
    
    print("\n=== Building cvxpy problem ===")
    import diffcp_patch  # Need for cvxpylayers
    cvxpy_data, prob = build_cvxpy_problem_with_values(data)
    
    print("\n=== Getting jax_canonicalize data ===")
    jax_A, jax_b, jax_c, jax_cone_info = get_jax_canonicalize_data(data)
    print(f"jax A: {jax_A.shape}")
    print(f"jax cone_info: soc={[int(d) for d in jax_cone_info['soc']]}")
    
    compare_matrices(cvxpy_data, jax_A, jax_b, jax_c, jax_cone_info)
    
    os._exit(0)
