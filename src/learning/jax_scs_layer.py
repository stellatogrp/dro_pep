"""
JAX-compatible SCS solver layer with diffcp differentiation.

This module provides a differentiable interface to the SCS solver
using JAX custom VJP and diffcp for implicit differentiation.

Key difference from Clarabel:
- SCS uses LOWER triangle in column-major order for PSD cones
- Clarabel uses UPPER triangle in column-major order
- Both use sqrt(2) scaling for off-diagonal elements

The trick for efficient indexing:
- Clarabel: tril_indices + flip → column-major upper triangle
- SCS: triu_indices + flip → column-major lower triangle
"""

import jax
# Enable float64 for numerical precision (must be before any JAX imports use arrays)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from functools import partial
import scipy.sparse as spa
import logging

log = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=['precond_type'])
def compute_preconditioner_from_samples(G_batch, F_batch, precond_type='average'):
    """Compute preconditioning factors from sample Gram matrices.
    
    Computes inverse preconditioning factors used to scale the DRO constraints
    based on sample statistics. This improves numerical conditioning.
    
    This is a JAX-traceable version that can be JIT compiled and differentiated.
    
    Args:
        G_batch: Batch of Gram matrices (N, dimG, dimG)
        F_batch: Batch of function value vectors (N, dimF)
        precond_type: Type of preconditioning:
            - 'average': Use average of sample diagonals (default)
            - 'max': Use maximum values
            - 'min': Use minimum values
            - 'identity': No preconditioning
    
    Returns:
        precond_inv: Tuple (precond_inv_G, precond_inv_F) of inverse preconditioning factors
            - precond_inv_G: (dimG,) array for Gram matrix scaling
            - precond_inv_F: (dimF,) array for function value scaling
    """
    dimG = G_batch.shape[1]
    dimF = F_batch.shape[1]
    
    if precond_type == 'identity':
        return (jnp.ones(dimG), jnp.ones(dimF))
    
    # Compute sqrt of diagonals of each G matrix: shape (N, dimG)
    # Use vmap to extract diagonals efficiently
    G_diags = jax.vmap(jnp.diag)(G_batch)  # (N, dimG)
    G_diag_sqrt = jnp.sqrt(jnp.maximum(G_diags, 1e-10))  # Avoid sqrt of negative
    
    # F_batch is already (N, dimF)
    F_vals = F_batch
    
    if precond_type == 'average':
        avg_G = jnp.mean(G_diag_sqrt, axis=0)  # (dimG,)
        avg_F = jnp.mean(F_vals, axis=0)       # (dimF,)
        precond_G = 1.0 / jnp.maximum(avg_G, 1e-10)
        precond_F = 1.0 / jnp.sqrt(jnp.maximum(avg_F, 1e-10))
    elif precond_type == 'max':
        max_G = jnp.max(G_diag_sqrt, axis=0)
        max_F = jnp.max(F_vals, axis=0)
        precond_G = 1.0 / jnp.maximum(max_G, 1e-10)
        precond_F = 1.0 / jnp.sqrt(jnp.maximum(max_F, 1e-10))
    elif precond_type == 'min':
        min_G = jnp.min(G_diag_sqrt, axis=0)
        min_F = jnp.min(F_vals, axis=0)
        precond_G = 1.0 / jnp.maximum(min_G, 1e-10)
        precond_F = 1.0 / jnp.sqrt(jnp.maximum(min_F, 1e-10))
    else:
        # This branch won't be traced due to static_argnames, but needed for type checking
        precond_G = jnp.ones(dimG)
        precond_F = jnp.ones(dimF)
    
    # Apply scaling factors
    precond_G = precond_G * dimG
    precond_F = precond_F * jnp.sqrt(dimF)
    
    # Handle NaN/inf values by clipping
    precond_G = jnp.clip(precond_G, 1e-10, 1e10)
    precond_F = jnp.clip(precond_F, 1e-10, 1e10)
    
    # Return inverse preconditioner
    precond_inv_G = 1.0 / precond_G
    precond_inv_F = 1.0 / precond_F
    
    return (precond_inv_G, precond_inv_F)


# =============================================================================
# MKL Pardiso Detection (for faster linear solves)
# =============================================================================

_MKL_AVAILABLE = None  # Cached detection result

def _detect_mkl_pardiso():
    """Detect if MKL Pardiso is available in clarabel.
    
    Returns True if MKL Pardiso can be used as the direct solver.
    Result is cached after first call.
    """
    global _MKL_AVAILABLE
    if _MKL_AVAILABLE is not None:
        return _MKL_AVAILABLE
    
    try:
        import clarabel
        import scipy.sparse as spa
        import numpy as np
        
        # Try solving a trivial problem with MKL
        P = spa.csc_matrix((2, 2))
        q = np.array([1., 1.])
        A = spa.csc_matrix([[1., 0.], [0., 1.]])
        b = np.array([1., 1.])
        cones = [clarabel.NonnegativeConeT(2)]
        
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.direct_solve_method = 'mkl'
        
        solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
        solver.solve()
        _MKL_AVAILABLE = True
        log.info("MKL Pardiso detected and will be used for Clarabel solves")
    except Exception:
        _MKL_AVAILABLE = False
        # log.debug("MKL Pardiso not available, using default qdldl solver")
        log.info("MKL Pardiso not available, using default qdldl solver")
    
    return _MKL_AVAILABLE

def get_direct_solve_method():
    """Get the best available direct solve method for Clarabel.
    
    Returns 'mkl' if MKL Pardiso is available, otherwise 'qdldl'.
    """
    return 'mkl' if _detect_mkl_pardiso() else 'qdldl'


# =============================================================================
# SCS-specific vectorization helpers (lower triangle, column-major)
# =============================================================================

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


@jax.jit
def jax_scs_symm_vectorize(A, scale_factor):
    """Vectorize lower triangle of symmetric matrix in SCS format.
    
    Extracts lower triangle in column-major order with off-diagonal scaling.
    This matches SCS's expected PSD cone vectorization format.
    
    Args:
        A: (n, n) symmetric matrix
        scale_factor: Factor to scale off-diagonal elements (typically sqrt(2))
    
    Returns:
        Vector of length n*(n+1)/2 containing lower triangular elements
        in column-major order
    """
    n = A.shape[0]
    
    # Get SCS column-major lower triangle indices
    rows, cols = get_scs_lower_tri_indices(n)
    
    # Extract elements
    A_vec = A[rows, cols]
    
    # Scale off-diagonal elements
    off_diag_mask = rows != cols
    A_vec = jnp.where(off_diag_mask, A_vec * scale_factor, A_vec)
    
    return A_vec


@jax.jit 
def jax_scs_scaled_off_triangles(A, scale_factor):
    """Create diagonal matrix from SCS-format vectorization.
    
    Args:
        A: (n, n) matrix (typically ones matrix for identity scaling)
        scale_factor: Scale factor for off-diagonals
    
    Returns:
        Diagonal matrix of shape (n*(n+1)/2, n*(n+1)/2)
    """
    A_vec = jax_scs_symm_vectorize(A, scale_factor)
    return jnp.diag(A_vec)


# =============================================================================
# Pure JAX Canonicalization Function (JIT-compatible)
# =============================================================================

@jax.jit
def jax_scs_canonicalize_dro_expectation(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps, precond_inv,
):
    """
    Pure JAX canonicalization of DRO expectation problem to SCS form.
    
    This is a JAX-traceable version that can be JIT compiled.
    Uses SCS cone ordering and lower-triangle column-major vectorization for PSD cones.
    
    SCS cone ordering (in rows of A, b): zero, nonneg (l), SOC (q), PSD (s)
    
    Returns:
        A_dense: Dense constraint matrix (total_rows, x_dim) - JAX array
        b: RHS vector - JAX array
        c: Objective vector - JAX array
        x_dim: Dimension of decision variable
        cone_info: Dict with cone dimensions for SCS
    """
    # Get dimensions
    N = G_batch.shape[0]
    M = A_vals.shape[0]
    S_mat = A_obj.shape[0]
    V = b_obj.shape[0]
    S_vec = S_mat * (S_mat + 1) // 2
    
    # Decision variable dimension
    # x = [lambda, s_1..s_N, y_1..y_NM, Fz_1..Fz_NV, Gz_1..Gz_N*S_vec]
    x_dim = 1 + N * (1 + M + V + S_vec)
    
    # Index offsets
    lambd_idx = 0
    s_start = 1
    y_start = s_start + N
    Fz_start = y_start + N * M
    Gz_start = Fz_start + N * V
    
    # Build objective vector c (minimize lambda*eps + (1/N) sum s_i)
    c_obj = jnp.zeros(x_dim)
    c_obj = c_obj.at[lambd_idx].set(eps)
    c_obj = c_obj.at[s_start:s_start + N].set(1.0 / N)
    
    # Precompute symmetric vectorizations using SCS format
    A_obj_svec = jax_scs_symm_vectorize(A_obj, jnp.sqrt(2.0))  # (S_vec,)
    A_vals_svec = jax.vmap(lambda A: jax_scs_symm_vectorize(A, jnp.sqrt(2.0)))(A_vals)  # (M, S_vec)
    
    # Vectorize G_batch using SCS format
    G_batch_svec = jax.vmap(lambda G: jax_scs_symm_vectorize(G, 2.0))(G_batch)  # (N, S_vec)
    
    # Scaled identity for PSD cone (SCS format)
    scaledI = jax_scs_scaled_off_triangles(jnp.ones((S_mat, S_mat)), jnp.sqrt(2.0))  # (S_vec, S_vec)
    
    # Preconditioner scaled matrices
    G_precond_vec, F_precond = precond_inv
    F_precond_sq = F_precond ** 2
    scaled_G_vec_outer = jnp.outer(G_precond_vec, G_precond_vec)
    scaledG_mult = jax_scs_scaled_off_triangles(scaled_G_vec_outer, jnp.sqrt(2.0))  # (S_vec, S_vec)
    
    # =========================================================================
    # Build constraint blocks using block diagonal / broadcasting approach
    # =========================================================================
    
    # 1. Epigraph constraints (N rows): -c^T y_i - Tr(G_i @ Gz_i) - F_i @ Fz_i - s_i <= 0
    # Each row i has: -1 at s_i, -c_vals at y_i block, -F_i at Fz_i block, -G_svec at Gz_i block
    epi_rows = jnp.zeros((N, x_dim))
    # s coefficients: -1 on diagonal of s block
    epi_rows = epi_rows.at[jnp.arange(N), s_start + jnp.arange(N)].set(-1.0)
    # y coefficients: block diagonal with -c_vals for each sample
    for i in range(N):
        epi_rows = epi_rows.at[i, y_start + i * M:y_start + (i + 1) * M].set(-c_vals)
    # Fz coefficients: -F_batch
    for i in range(N):
        epi_rows = epi_rows.at[i, Fz_start + i * V:Fz_start + (i + 1) * V].set(-F_batch[i])
    # Gz coefficients: -G_batch_svec
    for i in range(N):
        epi_rows = epi_rows.at[i, Gz_start + i * S_vec:Gz_start + (i + 1) * S_vec].set(-G_batch_svec[i])
    epi_b = jnp.zeros(N)
    
    # 2. y >= 0 constraints (N*M rows): -I @ y <= 0
    y_nonneg = jnp.zeros((N * M, x_dim))
    y_nonneg = y_nonneg.at[:, y_start:y_start + N * M].set(-jnp.eye(N * M))
    y_nonneg_b = jnp.zeros(N * M)
    
    # 3. Equality constraints (N*V rows): -B^T y_i + Fz_i = -b_obj
    Bm_T = b_vals.T  # (V, M)
    eq_rows = jnp.zeros((N * V, x_dim))
    for i in range(N):
        eq_rows = eq_rows.at[i * V:(i + 1) * V, y_start + i * M:y_start + (i + 1) * M].set(-Bm_T)
        eq_rows = eq_rows.at[i * V:(i + 1) * V, Fz_start + i * V:Fz_start + (i + 1) * V].set(jnp.eye(V))
    eq_b = jnp.tile(-b_obj, N)
    
    # 4. PSD constraints (N*S_vec rows): -A^* y_i + Gz_i << -A_obj
    Am_T = A_vals_svec.T  # (S_vec, M)
    psd_rows = jnp.zeros((N * S_vec, x_dim))
    for i in range(N):
        psd_rows = psd_rows.at[i * S_vec:(i + 1) * S_vec, y_start + i * M:y_start + (i + 1) * M].set(-Am_T)
        psd_rows = psd_rows.at[i * S_vec:(i + 1) * S_vec, Gz_start + i * S_vec:Gz_start + (i + 1) * S_vec].set(scaledI)
    psd_b = jnp.tile(-A_obj_svec, N)
    
    # 5. SOCP constraints (N*(1+V+S_vec) rows): || [Gz_i, Fz_i] ||_precond <= lambda
    socp_dim = 1 + V + S_vec
    socp_rows = jnp.zeros((N * socp_dim, x_dim))
    for i in range(N):
        # -lambda coefficient
        socp_rows = socp_rows.at[i * socp_dim, lambd_idx].set(-1.0)
        # -diag(F_precond_sq) @ Fz_i
        socp_rows = socp_rows.at[i * socp_dim + 1:i * socp_dim + 1 + V, 
                                  Fz_start + i * V:Fz_start + (i + 1) * V].set(-jnp.diag(F_precond_sq))
        # -scaledG_mult @ Gz_i
        socp_rows = socp_rows.at[i * socp_dim + 1 + V:(i + 1) * socp_dim,
                                  Gz_start + i * S_vec:Gz_start + (i + 1) * S_vec].set(-scaledG_mult)
    socp_b = jnp.zeros(N * socp_dim)
    
    # =========================================================================
    # Combine all blocks
    # SCS cone ordering: zero (z), nonneg (l), SOC (q), PSD (s)
    # =========================================================================
    A_dense = jnp.vstack([
        eq_rows,       # N*V rows (zero cone / equality)
        epi_rows,      # N rows (nonneg)
        y_nonneg,      # N*M rows (nonneg)
        socp_rows,     # N*socp_dim rows (SOC)
        psd_rows,      # N*S_vec rows (PSD)
    ])
    
    b = jnp.concatenate([
        eq_b,
        epi_b,
        y_nonneg_b,
        socp_b,
        psd_b,
    ])
    
    # Cone info for SCS
    # SCS uses: z (zero), l (nonneg), q (SOC array), s (PSD array)
    cone_info = {
        'z': N * V,           # Equality constraints (first)
        'l': N + N * M,       # Epigraph + y >= 0 (second)
        'q': [socp_dim] * N,  # SOC cones (third)
        's': [S_mat] * N,     # PSD cones (fourth)
        'N': N,
        'M': M,
        'V': V,
        'S_mat': S_mat,
        'S_vec': S_vec,
    }
    
    return A_dense, b, c_obj, x_dim, cone_info


# =============================================================================
# High-level differentiable solve function
# =============================================================================

import numpy as np
import diffcp


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
            A_csc = spa.csc_matrix(A_arr)
            try:
                x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
                    A_csc, b_arr, c_arr,
                    static_data.diffcp_cone_dict,
                    solve_method='CLARABEL',
                    direct_solve_method=get_direct_solve_method(),
                    verbose=False,
                )
                # x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
                #     A_csc, b_arr, c_arr,
                #     static_data.diffcp_cone_dict,
                #     solve_method='SCS',
                #     verbose=False,
                # )
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
            A_csc = spa.csc_matrix(A_arr)
            try:
                x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
                    A_csc, b_arr, c_arr,
                    static_data.diffcp_cone_dict,
                    solve_method='CLARABEL',
                    direct_solve_method=get_direct_solve_method(),
                    verbose=False,
                )
                # x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
                #     A_csc, b_arr, c_arr,
                #     static_data.diffcp_cone_dict,
                #     solve_method='SCS',
                #     verbose=False,
                # )
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


# =============================================================================
# CVaR Canonicalization
# =============================================================================

@jax.jit
def jax_scs_canonicalize_dro_cvar(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps, alpha, precond_inv,
):
    """
    Pure JAX canonicalization of DRO CVaR problem to SCS form.
    
    This is a JAX-traceable version that can be JIT compiled.
    Uses SCS cone ordering and lower-triangle column-major vectorization for PSD cones.
    
    CVaR has doubled dual variables (y1, y2) and corresponding Wasserstein
    constraint variables (Fz1/Fz2, Gz1/Gz2). The VaR threshold is 't'.
    
    SCS cone ordering (in rows of A, b): zero (z), nonneg (l), SOC (q), PSD (s)
    
    Returns:
        A_dense: Constraint matrix as dense JAX array
        b: RHS vector
        c_obj: Objective vector
        x_dim: Dimension of decision variable
        cone_info: Dict with cone dimensions for SCS
    """
    alpha_inv = 1.0 / alpha
    
    N = G_batch.shape[0]
    M = A_vals.shape[0]  # Number of interpolation constraints
    V = b_obj.shape[0]  # Dimension of Fz
    S_mat = A_obj.shape[0]  # Dimension of main PSD constraint
    S_vec = S_mat * (S_mat + 1) // 2
    
    # Process PEP matrices using SCS format (lower triangle column-major)
    A_obj_svec = jax_scs_symm_vectorize(A_obj, jnp.sqrt(2.0))  # (S_vec,)
    A_vals_svec = jax.vmap(lambda A: jax_scs_symm_vectorize(A, jnp.sqrt(2.0)))(A_vals)  # (M, S_vec)
    Bm_T = b_vals.T  # (V, M)
    
    # Process sample matrices to vectors
    G_batch_svec = jax.vmap(lambda G: jax_scs_symm_vectorize(G, 2.0))(G_batch)  # (N, S_vec)
    
    # Preconditioner
    G_precond_vec, F_precond = precond_inv
    F_precond_sq = F_precond ** 2
    scaled_G_vec_outer = jnp.outer(G_precond_vec, G_precond_vec)
    scaledG_mult = jax_scs_scaled_off_triangles(scaled_G_vec_outer, jnp.sqrt(2.0))  # (S_vec, S_vec)
    scaledI = jax_scs_scaled_off_triangles(jnp.ones((S_mat, S_mat)), jnp.sqrt(2.0))  # (S_vec, S_vec)
    
    # Decision variable dimension
    # x = [lambda, t, s_1..s_N, y1_1..y1_NM, y2_1..y2_NM, Fz1_1..Fz1_NV, Fz2_1..Fz2_NV, Gz1_1..Gz1_NS, Gz2_1..Gz2_NS]
    x_dim = 2 + N * (1 + 2 * (M + V + S_vec))
    
    # Index helpers
    lambd_idx = 0
    t_idx = 1
    s_start = 2
    y1_start = s_start + N
    y2_start = y1_start + N * M
    Fz1_start = y2_start + N * M
    Fz2_start = Fz1_start + N * V
    Gz1_start = Fz2_start + N * V
    Gz2_start = Gz1_start + N * S_vec
    
    # Build objective
    c_obj = jnp.zeros(x_dim)
    c_obj = c_obj.at[lambd_idx].set(eps)
    c_obj = c_obj.at[s_start:s_start + N].set(1.0 / N)
    
    Am_T = A_vals_svec.T  # (S_vec, M)
    
    # =========================================================================
    # Build constraint blocks (in SCS order: zero, nonneg, soc, psd)
    # =========================================================================
    
    # --- ZERO CONE: Equality constraints (-B^T y1 + Fz1 = -b_obj, -B^T y2 + Fz2 = -alpha_inv * b_obj)
    eq1_rows = jnp.zeros((N * V, x_dim))
    for i in range(N):
        eq1_rows = eq1_rows.at[i * V:(i + 1) * V, y1_start + i * M:y1_start + (i + 1) * M].set(-Bm_T)
        eq1_rows = eq1_rows.at[i * V:(i + 1) * V, Fz1_start + i * V:Fz1_start + (i + 1) * V].set(jnp.eye(V))
    eq1_b = jnp.tile(-b_obj, N)
    
    eq2_rows = jnp.zeros((N * V, x_dim))
    for i in range(N):
        eq2_rows = eq2_rows.at[i * V:(i + 1) * V, y2_start + i * M:y2_start + (i + 1) * M].set(-Bm_T)
        eq2_rows = eq2_rows.at[i * V:(i + 1) * V, Fz2_start + i * V:Fz2_start + (i + 1) * V].set(jnp.eye(V))
    eq2_b = jnp.tile(-alpha_inv * b_obj, N)
    
    # --- NONNEGATIVE CONE: epi1, epi2, y1 >= 0, y2 >= 0
    # epi1: t - c^T y1 - Tr(G @ Gz1) - F @ Fz1 - s <= 0
    epi1_rows = jnp.zeros((N, x_dim))
    for i in range(N):
        epi1_rows = epi1_rows.at[i, t_idx].set(1.0)
        epi1_rows = epi1_rows.at[i, y1_start + i * M:y1_start + (i + 1) * M].set(-c_vals)
        epi1_rows = epi1_rows.at[i, s_start + i].set(-1.0)
        epi1_rows = epi1_rows.at[i, Fz1_start + i * V:Fz1_start + (i + 1) * V].set(-F_batch[i])
        epi1_rows = epi1_rows.at[i, Gz1_start + i * S_vec:Gz1_start + (i + 1) * S_vec].set(-G_batch_svec[i])
    epi1_b = jnp.zeros(N)
    
    # epi2: -(1/alpha - 1)*t - c^T y2 - Tr(G @ Gz2) - F @ Fz2 - s <= 0
    epi2_rows = jnp.zeros((N, x_dim))
    for i in range(N):
        epi2_rows = epi2_rows.at[i, t_idx].set(-(alpha_inv - 1))
        epi2_rows = epi2_rows.at[i, y2_start + i * M:y2_start + (i + 1) * M].set(-c_vals)
        epi2_rows = epi2_rows.at[i, s_start + i].set(-1.0)
        epi2_rows = epi2_rows.at[i, Fz2_start + i * V:Fz2_start + (i + 1) * V].set(-F_batch[i])
        epi2_rows = epi2_rows.at[i, Gz2_start + i * S_vec:Gz2_start + (i + 1) * S_vec].set(-G_batch_svec[i])
    epi2_b = jnp.zeros(N)
    
    # y1 >= 0, y2 >= 0
    y1_nonneg = jnp.zeros((N * M, x_dim))
    y1_nonneg = y1_nonneg.at[0:N*M, y1_start:y1_start + N*M].set(-jnp.eye(N * M))
    y1_nonneg_b = jnp.zeros(N * M)
    
    y2_nonneg = jnp.zeros((N * M, x_dim))
    y2_nonneg = y2_nonneg.at[0:N*M, y2_start:y2_start + N*M].set(-jnp.eye(N * M))
    y2_nonneg_b = jnp.zeros(N * M)
    
    # --- SOC CONES: 2N SOC cones for (Gz1, Fz1) and (Gz2, Fz2)
    socp_dim = 1 + V + S_vec
    socp1_rows = jnp.zeros((N * socp_dim, x_dim))
    for i in range(N):
        socp1_rows = socp1_rows.at[i * socp_dim, lambd_idx].set(-1.0)
        socp1_rows = socp1_rows.at[i * socp_dim + 1:i * socp_dim + 1 + V, 
                                    Fz1_start + i * V:Fz1_start + (i + 1) * V].set(-jnp.diag(F_precond_sq))
        socp1_rows = socp1_rows.at[i * socp_dim + 1 + V:(i + 1) * socp_dim,
                                    Gz1_start + i * S_vec:Gz1_start + (i + 1) * S_vec].set(-scaledG_mult)
    socp1_b = jnp.zeros(N * socp_dim)
    
    socp2_rows = jnp.zeros((N * socp_dim, x_dim))
    for i in range(N):
        socp2_rows = socp2_rows.at[i * socp_dim, lambd_idx].set(-1.0)
        socp2_rows = socp2_rows.at[i * socp_dim + 1:i * socp_dim + 1 + V, 
                                    Fz2_start + i * V:Fz2_start + (i + 1) * V].set(-jnp.diag(F_precond_sq))
        socp2_rows = socp2_rows.at[i * socp_dim + 1 + V:(i + 1) * socp_dim,
                                    Gz2_start + i * S_vec:Gz2_start + (i + 1) * S_vec].set(-scaledG_mult)
    socp2_b = jnp.zeros(N * socp_dim)
    
    # --- PSD CONES: 2N PSD cones for Gz1 and Gz2
    psd1_rows = jnp.zeros((N * S_vec, x_dim))
    for i in range(N):
        psd1_rows = psd1_rows.at[i * S_vec:(i + 1) * S_vec, y1_start + i * M:y1_start + (i + 1) * M].set(-Am_T)
        psd1_rows = psd1_rows.at[i * S_vec:(i + 1) * S_vec, Gz1_start + i * S_vec:Gz1_start + (i + 1) * S_vec].set(scaledI)
    psd1_b = jnp.tile(-A_obj_svec, N)
    
    psd2_rows = jnp.zeros((N * S_vec, x_dim))
    for i in range(N):
        psd2_rows = psd2_rows.at[i * S_vec:(i + 1) * S_vec, y2_start + i * M:y2_start + (i + 1) * M].set(-Am_T)
        psd2_rows = psd2_rows.at[i * S_vec:(i + 1) * S_vec, Gz2_start + i * S_vec:Gz2_start + (i + 1) * S_vec].set(scaledI)
    psd2_b = jnp.tile(-alpha_inv * A_obj_svec, N)
    
    # Combine all blocks (SCS order: zero, nonneg, soc, psd)
    A_dense = jnp.vstack([
        eq1_rows, eq2_rows,  # zero cone (2*N*V)
        epi1_rows, epi2_rows, y1_nonneg, y2_nonneg,  # nonneg cone (2*N + 2*N*M)
        socp1_rows, socp2_rows,  # soc cones (2*N*socp_dim)
        psd1_rows, psd2_rows,  # psd cones (2*N*S_vec)
    ])
    
    b = jnp.concatenate([
        eq1_b, eq2_b,
        epi1_b, epi2_b, y1_nonneg_b, y2_nonneg_b,
        socp1_b, socp2_b,
        psd1_b, psd2_b,
    ])
    
    # Cone info for SCS
    # SCS uses: z (zero), l (nonneg), q (SOC array), s (PSD array)
    cone_info = {
        'z': 2 * N * V,              # Equality constraints
        'l': 2 * N + 2 * N * M,      # Epigraph + y1 >= 0 + y2 >= 0
        'q': [socp_dim] * (2 * N),   # SOC cones (doubled)
        's': [S_mat] * (2 * N),      # PSD cones (doubled)
        'N': N,
        'M': M,
        'V': V,
        'S_mat': S_mat,
        'S_vec': S_vec,
    }
    
    return A_dense, b, c_obj, x_dim, cone_info


# =============================================================================
# Full Differentiable DRO Pipeline
# =============================================================================

def dro_expectation_scs_solve(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps, precond_inv,
):
    """
    Full differentiable DRO expectation solve using SCS + diffcp.
    
    Args:
        A_obj, b_obj, A_vals, b_vals, c_vals: PEP constraint data
        G_batch: (N, S_mat, S_mat) Gram matrices
        F_batch: (N, V) function values
        eps: Wasserstein radius
        precond_inv: (G_precond_inv, F_precond_inv) preconditioner
    
    Returns:
        obj_val: Optimal DRO expectation objective value (scalar, differentiable)
    """
    # Compute static dimensions from input shapes (not traced)
    N = G_batch.shape[0]
    M = A_vals.shape[0]
    S_mat = A_obj.shape[0]
    V = b_obj.shape[0]
    S_vec = S_mat * (S_mat + 1) // 2
    socp_dim = 1 + V + S_vec
    
    # Cone info computed from static dimensions
    cone_info = {
        'z': N * V,           # Equality constraints
        'l': N + N * M,       # Epigraph + y >= 0
        'q': [socp_dim] * N,  # SOC cones
        's': [S_mat] * N,     # PSD cones
    }
    
    # Calculate A shape from cone dimensions
    m = cone_info['z'] + cone_info['l'] + sum(cone_info['q']) + sum([s*(s+1)//2 for s in cone_info['s']])
    x_dim = 1 + N * (1 + M + V + S_vec)
    A_shape = (m, x_dim)
    
    # Create static data for cones BEFORE canonicalization
    static_data = SCSSolveData(cone_info, A_shape)
    
    # Canonicalize to SCS form (this can be traced)
    A_dense, b, c, _, _ = jax_scs_canonicalize_dro_expectation(
        A_obj, b_obj, A_vals, b_vals, c_vals,
        G_batch, F_batch,
        eps, precond_inv,
    )

    log.info('canon done')
    
    # Solve with custom VJP
    obj_val = scs_solve_wrapper(static_data, A_dense, b, c)
    
    return obj_val


def dro_cvar_scs_solve(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps, alpha, precond_inv,
):
    """
    Full differentiable DRO CVaR solve using SCS + diffcp.
    
    Args:
        A_obj, b_obj, A_vals, b_vals, c_vals: PEP constraint data
        G_batch: (N, S_mat, S_mat) Gram matrices
        F_batch: (N, V) function values
        eps: Wasserstein radius
        alpha: CVaR confidence level (e.g., 0.1 for CVaR at 10%)
        precond_inv: (G_precond_inv, F_precond_inv) preconditioner
    
    Returns:
        obj_val: Optimal DRO CVaR objective value (scalar, differentiable)
    """
    # Compute static dimensions from input shapes (not traced)
    N = G_batch.shape[0]
    M = A_vals.shape[0]
    S_mat = A_obj.shape[0]
    V = b_obj.shape[0]
    S_vec = S_mat * (S_mat + 1) // 2
    socp_dim = 1 + V + S_vec
    
    # Cone info computed from static dimensions (doubled for CVaR)
    cone_info = {
        'z': 2 * N * V,              # Equality constraints (doubled)
        'l': 2 * N + 2 * N * M,      # Epigraph + y1 >= 0 + y2 >= 0
        'q': [socp_dim] * (2 * N),   # SOC cones (doubled)
        's': [S_mat] * (2 * N),      # PSD cones (doubled)
    }
    
    # Calculate A shape from cone dimensions
    m = cone_info['z'] + cone_info['l'] + sum(cone_info['q']) + sum([s*(s+1)//2 for s in cone_info['s']])
    x_dim = 2 + N * (1 + 2 * (M + V + S_vec))
    A_shape = (m, x_dim)
    
    # Create static data for cones BEFORE canonicalization
    static_data = SCSSolveData(cone_info, A_shape)
    
    # Canonicalize to SCS form (this can be traced)
    A_dense, b, c, _, _ = jax_scs_canonicalize_dro_cvar(
        A_obj, b_obj, A_vals, b_vals, c_vals,
        G_batch, F_batch,
        eps, alpha, precond_inv,
    )

    # log.info('canon done')
    
    # Solve with custom VJP
    obj_val = scs_solve_wrapper(static_data, A_dense, b, c)
    
    return obj_val


def dro_scs_solve(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps,
    # Preconditioner type
    precond_type='average',
    # Risk measure selection
    risk_type='expectation',
    alpha=0.1,
):
    """
    Full differentiable DRO solve using SCS + diffcp.
    
    This is the main entry point that dispatches to the appropriate solver
    based on the risk_type.
    
    Args:
        A_obj, b_obj, A_vals, b_vals, c_vals: PEP constraint data
        G_batch: (N, S_mat, S_mat) Gram matrices
        F_batch: (N, V) function values
        eps: Wasserstein radius
        precond_inv: (G_precond_inv, F_precond_inv) preconditioner
        risk_type: 'expectation' or 'cvar'
        alpha: CVaR confidence level (only used when risk_type='cvar')
    
    Returns:
        obj_val: Optimal DRO objective value (scalar, differentiable)
    """
    precond_inv = compute_preconditioner_from_samples(G_batch, F_batch, precond_type=precond_type)

    if risk_type == 'expectation':
        return dro_expectation_scs_solve(
            A_obj, b_obj, A_vals, b_vals, c_vals,
            G_batch, F_batch,
            eps, precond_inv,
        )
    elif risk_type == 'cvar':
        return dro_cvar_scs_solve(
            A_obj, b_obj, A_vals, b_vals, c_vals,
            G_batch, F_batch,
            eps, alpha, precond_inv,
        )
    else:
        raise ValueError(f"Unknown risk_type: {risk_type}. Must be 'expectation' or 'cvar'.")
