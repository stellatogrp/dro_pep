import clarabel
import cvxpy as cp
import diffcp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sparse
np.set_printoptions(precision=5, suppress=True)

class SCSSolveData:
    """Container for static data needed by the SCS solver."""
    
    def __init__(self, cone_info, A_shape):
        self.cone_info = cone_info
        self.A_shape = A_shape
        # Build diffcp cone dict matching SCS ordering
        self.diffcp_cone_dict = self._build_diffcp_cone_dict(cone_info)
    
    def _build_diffcp_cone_dict(self, cone_info):
        """Build diffcp cone dictionary matching SCS cone ordering.
        
        SCS ordering: z (zero), l (nonneg), q (SOC), s (PSD)
        diffcp uses: ZERO, POS (nonneg), SOC, PSD
        """
        cone_dict = {}
        
        # Note: diffcp expects cones in specific order
        if cone_info['z'] > 0:
            cone_dict[diffcp.ZERO] = int(cone_info['z'])
        if cone_info['l'] > 0:
            cone_dict[diffcp.POS] = int(cone_info['l'])
        if cone_info['q']:
            cone_dict[diffcp.SOC] = [int(d) for d in cone_info['q']]
        if cone_info['s']:
            cone_dict[diffcp.PSD] = [int(d) for d in cone_info['s']]
        
        return cone_dict


def scs_solve_wrapper(static_data, A_dense, b, c):
    """
    Differentiable SCS solve using diffcp for both forward and backward.
    
    Uses diffcp.solve_and_derivative with SCS as the solver. The adjoint
    derivative is computed once during the forward pass and cached for backward.
    
    Args:
        static_data: SCSSolveData with cone info
        A_dense: Constraint matrix as JAX array
        b: RHS vector as JAX array
        c: Objective vector as JAX array
    
    Returns:
        Optimal objective value
    """
    # Module-level cache stores the adjoint_derivative function between forward and backward
    # since jax.custom_vjp doesn't support closures with non-JAX objects
    _adjoint_cache = {}
    
    @jax.custom_vjp
    def _solve(A_dense, b, c):
        # Forward pass: call diffcp via pure_callback
        def solve_impl(A_np, b_np, c_np):
            # Explicitly convert to numpy arrays (pure_callback passes JAX array views)
            A_arr = np.asarray(A_np)
            b_arr = np.asarray(b_np)
            c_arr = np.asarray(c_np)
            A_csc = sparse.csc_matrix(A_arr)
            try:
                # x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
                #     A_csc, b_arr, c_arr,
                #     static_data.diffcp_cone_dict,
                #     solve_method='CLARABEL',
                #     direct_solve_method=get_direct_solve_method(),
                #     verbose=False,
                # )
                x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
                    A_csc, b_arr, c_arr,
                    static_data.diffcp_cone_dict,
                    solve_method='SCS',
                    verbose=False,
                )
                obj = c_arr @ x
                # Store adjoint for backward pass
                _adjoint_cache['adjoint'] = adjoint_deriv
                _adjoint_cache['valid'] = True
            except Exception as e:
                log.warning(f"diffcp solve failed: {e}")
                x = np.zeros(c_arr.shape[0])
                obj = np.nan
                _adjoint_cache['valid'] = False
            
            return np.array([obj])
        
        result_shapes = jax.ShapeDtypeStruct((1,), jnp.float64)
        obj = jax.pure_callback(solve_impl, result_shapes, A_dense, b, c)
        return obj[0]
    
    def _solve_fwd(A_dense, b, c):
        # Solve using diffcp and cache solution + adjoint
        def solve_impl(A_np, b_np, c_np):
            # Explicitly convert to numpy arrays
            A_arr = np.asarray(A_np)
            b_arr = np.asarray(b_np)
            c_arr = np.asarray(c_np)
            A_csc = sparse.csc_matrix(A_arr)
            try:
                # x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
                #     A_csc, b_arr, c_arr,
                #     static_data.diffcp_cone_dict,
                #     solve_method='CLARABEL',
                #     # direct_solve_method=get_direct_solve_method(),
                #     verbose=False,
                # )
                x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
                    A_csc, b_arr, c_arr,
                    static_data.diffcp_cone_dict,
                    solve_method='SCS',
                    verbose=False,
                )
                obj = c_arr @ x
                # Store adjoint for backward pass
                _adjoint_cache['adjoint'] = adjoint_deriv
                _adjoint_cache['x'] = x
                _adjoint_cache['y'] = y
                _adjoint_cache['s'] = s
                _adjoint_cache['valid'] = True
            except Exception as e:
                log.warning(f"diffcp solve failed in fwd: {e}")
                x = np.zeros(c_arr.shape[0])
                y = np.zeros(b_arr.shape[0])
                s = np.zeros(b_arr.shape[0])
                obj = np.nan
                _adjoint_cache['valid'] = False
            
            return (
                np.array([obj]),
                x, y, s
            )
        
        result_shapes = (
            jax.ShapeDtypeStruct((1,), jnp.float64),
            jax.ShapeDtypeStruct((c.shape[0],), jnp.float64),
            jax.ShapeDtypeStruct((b.shape[0],), jnp.float64),
            jax.ShapeDtypeStruct((b.shape[0],), jnp.float64),
        )
        
        obj, x, y, s = jax.pure_callback(
            solve_impl, result_shapes, A_dense, b, c
        )
        
        # Return primal and residuals for backward (x needed for dc gradient)
        return obj[0], (A_dense, b, c, x)
    
    def _solve_bwd(res, g):
        A_dense, b, c, x = res
        d_obj = g  # Scalar gradient w.r.t. objective
        
        def compute_grads(A_np, b_np, c_np, x_np, d_obj_np):
            # Explicitly convert to numpy arrays
            A_arr = np.asarray(A_np)
            b_arr = np.asarray(b_np)
            c_arr = np.asarray(c_np)
            x_arr = np.asarray(x_np)
            d_obj_arr = np.asarray(d_obj_np)
            
            if not _adjoint_cache.get('valid', False):
                # If forward solve failed, return zero gradients
                return np.zeros_like(A_arr), np.zeros_like(b_arr), np.zeros_like(c_arr)
            
            adjoint_deriv = _adjoint_cache['adjoint']
            y_cached = _adjoint_cache['y']
            s_cached = _adjoint_cache['s']
            
            # The objective is c^T x.
            # d(c^T x) = c^T dx + x^T dc
            # Using adjoint: dx = d_obj * c propagates through D^T
            
            dx = d_obj_arr * c_arr
            dy = np.zeros_like(y_cached)
            ds = np.zeros_like(s_cached)
            
            dA_sol, db_sol, dc_sol = adjoint_deriv(dx, dy, ds)
            
            # Direct contribution to dc from x
            dc_direct = d_obj_arr * x_arr
            
            dA = dA_sol.toarray()
            db = db_sol
            dc = dc_sol + dc_direct
            
            return dA, db, dc
        
        result_shapes = (
            jax.ShapeDtypeStruct(A_dense.shape, jnp.float64),
            jax.ShapeDtypeStruct(b.shape, jnp.float64),
            jax.ShapeDtypeStruct(c.shape, jnp.float64),
        )
        
        dA, db, dc = jax.pure_callback(
            compute_grads, result_shapes,
            A_dense, b, c, x, jnp.array(d_obj)
        )

        _adjoint_cache.clear()
        
        return dA, db, dc
    
    _solve.defvjp(_solve_fwd, _solve_bwd)
    
    return _solve(A_dense, b, c)


def get_scs_lower_tri_indices(n):
    """Get indices for lower triangle in column-major order for SCS.
    
    SCS expects the lower triangle vectorized in column-major order:
    For a 3x3 matrix:
        [a b c]
        [d e f]  -> SCS vectorization: [a, d, g, e, h, i]
        [g h i]
    
    This is: (0,0), (1,0), (2,0), (1,1), (2,1), (2,2) in (row, col) format.
    
    The trick: triu_indices gives row-major upper triangle.
    Flipping (cols, rows) converts to column-major lower triangle.
    
    Args:
        n: Matrix dimension
        
    Returns:
        (rows, cols): Tuple of arrays giving lower triangle indices in column-major order
    """
    # triu_indices gives row-major upper triangle: (r, c) with r <= c
    # Flipping gives column-major lower triangle: (c, r) which is (row, col) with row >= col
    rows, cols = jnp.triu_indices(n)
    return cols, rows  # Flip to get column-major lower triangle


def permute_psd_rows(A: sparse.csc_matrix, b: np.ndarray, n: int, row_offset: int) -> sparse.csc_matrix:
    """
    Permutes rows of a sparse CSC constraint matrix A to switch from lower
    triangular order (SCS) to upper triangular order (Clarabel) for a PSD constraint.

    Args:
        A (csc_matrix): Constraint matrix in CSC format.
        n (int): Size of the PSD constraint matrix (n x n).
        row_offset (int): Row index where the PSD block starts.

    Returns:
        csc_matrix: New CSC matrix with permuted rows.
    """

    tril_rows, tril_cols = np.tril_indices(n)
    triu_rows, triu_cols = np.triu_indices(n)

    # Compute the permutation mapping
    tril_multi_index = np.ravel_multi_index((tril_cols, tril_rows), (n, n))
    triu_multi_index = np.ravel_multi_index((triu_cols, triu_rows), (n, n))
    postshuffle_from_preshuffle_perm = np.argsort(tril_multi_index) + row_offset
    preshuffle_from_postshuffle_perm = np.argsort(triu_multi_index) + row_offset
    n_rows = len(postshuffle_from_preshuffle_perm)

    # Apply row permutation
    data, rows, cols = A.data, A.indices, A.indptr
    new_rows = np.copy(rows)  # Create a new row index array
    # Identify affected rows
    mask = (rows >= row_offset) & (rows < (row_offset + n_rows))
    new_rows[mask] = postshuffle_from_preshuffle_perm[rows[mask] - row_offset]

    new_A = sparse.csc_matrix((data, new_rows, cols), shape=A.shape)

    new_b = np.copy(b)

    new_b[row_offset:row_offset+n_rows] = new_b[preshuffle_from_postshuffle_perm]

    return new_A, new_b


def scs_to_clarabel(A, b, cone_dict):
    cones = []
    # Given the difference in convention between SCS (lower triangluar columnwise)
    # and Clarabel (upper triangular columnwise), the rows of A may need to be permuted
    start_row = 0
    if "z" in cone_dict:
        v = cone_dict["z"]
        if v > 0:
            cones.append(clarabel.ZeroConeT(v))
            start_row += v
    if "f" in cone_dict:
        v = cone_dict["f"]
        if v > 0:
            cones.append(clarabel.ZeroConeT(v))
            start_row += v
    if "l" in cone_dict:
        v = cone_dict["l"]
        if v > 0:
            cones.append(clarabel.NonnegativeConeT(v))
            start_row += v
    if "q" in cone_dict:
        for v in cone_dict["q"]:
            cones.append(clarabel.SecondOrderConeT(v))
            start_row += v
    if "s" in cone_dict:
        for v in cone_dict["s"]:
            cones.append(clarabel.PSDTriangleConeT(v))
            A, b = permute_psd_rows(A, b, v, start_row)
            # start_row += v
            start_row += v * (v + 1) // 2  ## THIS IS THE FIXED VERSION
    if "ep" in cone_dict:
        v = cone_dict["ep"]
        cones += [clarabel.ExponentialConeT()] * v

    return A, b, cones



def jax_scs_symm_vectorize(A, scale_factor=jnp.sqrt(2)):
    n = A.shape[0]
    
    # Get SCS column-major lower triangle indices
    rows, cols = get_scs_lower_tri_indices(n)
    
    # Extract elements
    A_vec = A[rows, cols]
    
    # Scale off-diagonal elements
    off_diag_mask = rows != cols
    A_vec = jnp.where(off_diag_mask, A_vec * scale_factor, A_vec)
    
    return A_vec


def scs_data_from_cvxpy_problem(problem):
    data = problem.get_problem_data(cp.SCS)[0]
    cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(data[
                                                                                  "dims"])
    return data["A"], data["b"], data["c"], cone_dims


def main():
    n = 3
    A = np.array([
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6],
    ])
    B = np.array([
        [7, 8, 9],
        [8, 10, 11],
        [9, 11, 12],
    ])

    X = cp.Variable((n, n), symmetric=True)
    # y = np.array([1, 2, 3])
    y = cp.Variable(2)

    constraints = [y[0] * A + y[1] * B >> 0, X >> 0]
    constraints += [
        cp.trace(A @ X) == 1,
        y >= 0,
    ]

    print(jax_scs_symm_vectorize(A))

    obj = cp.Minimize(cp.trace(X) + np.ones(2) @ y)
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.SCS)
    print('from cvxpy:', res)
    # print(X.value, y.value)

    scs_A, scs_b, scs_c, scs_cones = scs_data_from_cvxpy_problem(prob)
    # print(scs_A.shape, scs_A.todense())
    # print(scs_cones)
    clarabel_A, clarabel_b, clarabel_cones = scs_to_clarabel(scs_A, scs_b, scs_cones)

    # print(clarabel_A.todense())

    settings = clarabel.DefaultSettings()
    settings.verbose = False

    P = sparse.csc_matrix((scs_c.size, scs_c.size))

    solver = clarabel.DefaultSolver(P,scs_c,clarabel_A,clarabel_b,clarabel_cones,settings)
    solution = solver.solve()
    print('from clarabel:', solution.status, solution.obj_val)

    A_csc = sparse.csc_matrix(scs_A)

    x, y, s, D, DT = diffcp.solve_and_derivative(A_csc, scs_b, scs_c,
        scs_cones,
        solve_method='SCS',
        verbose=False,
    )

    print('values from SCS pipeline')
    print(x)
    print(scs_c.T @ x)
    # print(DT(dx, dy, ds))

    dx = scs_c                # Gradient of J w.r.t x
    dy = np.zeros_like(y) # J doesn't depend on y
    ds = np.zeros_like(s) # J doesn't depend on s

    dA_manual, db_manual, dc_manual = DT(dx, dy, ds)

    # Convert sparse dA to dense for comparison
    dA_manual = dA_manual.toarray()

    # CRITICAL: Add the "Direct Term" to dc
    # The Product Rule: d(c'x)/dc = x + c' * (dx/dc)
    # DT only gives you the second term (indirect). Must add x manually.
    dc_manual = dc_manual + x

    print('values from custom SCS wrapper')

    static_data = SCSSolveData(scs_cones, scs_A.shape)

    obj_val = scs_solve_wrapper(static_data, scs_A.todense(), scs_b, scs_c)
    print(obj_val)

    grads = jax.grad(scs_solve_wrapper, argnums=(1, 2, 3))
    dA_jax, db_jax, dc_jax = grads(static_data, jnp.array(scs_A.todense()), jnp.array(scs_b), jnp.array(scs_c))

    print(f"Max Diff A: {np.max(np.abs(dA_jax - dA_manual)):.2e}")
    print(f"Max Diff b: {np.max(np.abs(db_jax - db_manual)):.2e}")
    print(f"Max Diff c: {np.max(np.abs(dc_jax - dc_manual)):.2e}")

    assert np.allclose(dA_jax, dA_manual, atol=1e-5)
    assert np.allclose(db_jax, db_manual, atol=1e-5)
    assert np.allclose(dc_jax, dc_manual, atol=1e-5)
    print("SUCCESS: JAX wrapper matches raw Diffcp derivatives.")

    x, y, s, D, DT = diffcp.solve_and_derivative(sparse.csc_matrix(scs_A), scs_b, scs_c,
        scs_cones,
        solve_method='CLARABEL',
        verbose=False,
    )

    print(x)
    print(scs_c.T @ x)
    # print(DT(dx, dy, ds))
    dA_manual_clarabel, db_manual_clarabel, dc_manual_clarabel = DT(dx, dy, ds)

    print(f"Max Diff A: {np.max(np.abs(dA_jax - dA_manual_clarabel.todense())):.2e}")
    print(f"Max Diff b: {np.max(np.abs(db_jax - db_manual_clarabel)):.2e}")
    print(f"Max Diff c: {np.max(np.abs(dc_jax - dc_manual_clarabel)):.2e}")

    print(dc_jax, dc_manual_clarabel)


if __name__ == '__main__':
    main()
