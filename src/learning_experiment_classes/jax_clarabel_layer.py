"""
JAX-compatible Clarabel solver layer with diffcp differentiation.

This module provides a differentiable interface to the Clarabel solver
using JAX custom VJP and diffcp for implicit differentiation.
"""

import jax
# Enable float64 for numerical precision (must be before any JAX imports use arrays)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
import scipy.sparse as spa
import clarabel
import diffcp
import logging

log = logging.getLogger(__name__)


# =============================================================================
# JAX Helper Functions (vectorization for PSD cones)
# =============================================================================

def jax_symm_vectorize(A, scale_factor):
    """Vectorize lower triangle of symmetric matrix with off-diagonal scaling.
    
    Equivalent to the NumPy function in clarabel_canonicalizer.py.
    
    Args:
        A: (n, n) symmetric matrix
        scale_factor: Factor to scale off-diagonal elements (typically sqrt(2))
    
    Returns:
        Vector of length n*(n+1)/2 containing lower triangular elements
    """
    n = A.shape[0]
    # Get lower triangular indices
    rows, cols = jnp.tril_indices(n)
    
    # Extract elements
    A_vec = A[rows, cols]
    
    # Scale off-diagonal elements
    off_diag_mask = rows != cols
    A_vec = jnp.where(off_diag_mask, A_vec * scale_factor, A_vec)
    
    return A_vec


def jax_scaled_off_triangles(A, scale_factor):
    """Create diagonal matrix from scaled symmetric vectorization.
    
    Args:
        A: (n, n) matrix (typically ones matrix for identity scaling)
        scale_factor: Scale factor for off-diagonals
    
    Returns:
        Diagonal matrix of shape (n*(n+1)/2, n*(n+1)/2)
    """
    A_vec = jax_symm_vectorize(A, scale_factor)
    return jnp.diag(A_vec)


def jax_get_triangular_indices(n):
    """Get indices for lower triangle in column-major order.
    
    Returns (cols, rows) to match the NumPy version's flipped order.
    """
    rows, cols = jnp.tril_indices(n)
    return cols, rows  # Flipped for column-major order


# =============================================================================
# Pure JAX Canonicalization Function (JIT-compatible)
# =============================================================================

def jax_canonicalize_dro_expectation(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps, precond_inv,
):
    """
    Pure JAX canonicalization of DRO expectation problem to Clarabel form.
    
    This is a JAX-traceable version that can be JIT compiled.
    Uses lax.dynamic_update_slice for dynamic indexing within vmap.
    
    Returns:
        A_dense: Dense constraint matrix (total_rows, x_dim) - JAX array
        b: RHS vector - JAX array
        c: Objective vector - JAX array
        x_dim: Dimension of decision variable
        cone_info: Dict with cone dimensions for later conversion
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
    
    # Precompute symmetric vectorizations
    A_obj_svec = jax_symm_vectorize(A_obj, jnp.sqrt(2.0))  # (S_vec,)
    A_vals_svec = jax.vmap(lambda A: jax_symm_vectorize(A, jnp.sqrt(2.0)))(A_vals)  # (M, S_vec)
    
    # Vectorize G_batch
    G_batch_svec = jax.vmap(lambda G: jax_symm_vectorize(G, 2.0))(G_batch)  # (N, S_vec)
    
    # Scaled identity for PSD cone
    scaledI = jax_scaled_off_triangles(jnp.ones((S_mat, S_mat)), jnp.sqrt(2.0))  # (S_vec, S_vec)
    
    # Preconditioner scaled matrices
    G_precond_vec, F_precond = precond_inv
    F_precond_sq = F_precond ** 2
    scaled_G_vec_outer = jnp.outer(G_precond_vec, G_precond_vec)
    scaledG_mult = jax_scaled_off_triangles(scaled_G_vec_outer, jnp.sqrt(2.0))  # (S_vec, S_vec)
    
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
    # Order for diffcp compatibility: zero, nonneg (pos), SOC, PSD
    # This matches diffcp's required cone ordering
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
    
    # Cone info for later conversion to Clarabel/diffcp format
    # Order matches: zero, nonneg (pos), soc, psd
    cone_info = {
        'zero': N * V,        # Equality constraints (first)
        'nonneg': N + N * M,  # Epigraph + y >= 0 (second)
        'soc': [socp_dim] * N,  # SOC cones (third)
        'psd': [S_mat] * N,   # PSD cones (fourth)
        'N': N,
        'M': M,
        'V': V,
        'S_mat': S_mat,
        'S_vec': S_vec,
    }
    
    return A_dense, b, c_obj, x_dim, cone_info


def jax_to_clarabel_cones(cone_info):
    """Convert cone_info dict to Clarabel cone list.
    
    Order matches diffcp requirements: zero, nonneg (pos), soc, psd
    """
    cones = []
    # Zero cone first (equality constraints)
    if cone_info['zero'] > 0:
        cones.append(clarabel.ZeroConeT(int(cone_info['zero'])))
    # Nonnegative cone second
    if cone_info['nonneg'] > 0:
        cones.append(clarabel.NonnegativeConeT(int(cone_info['nonneg'])))
    # SOC cones third
    for soc_dim in cone_info['soc']:
        cones.append(clarabel.SecondOrderConeT(int(soc_dim)))
    # PSD cones fourth
    for psd_dim in cone_info['psd']:
        cones.append(clarabel.PSDTriangleConeT(int(psd_dim)))
    return cones



def numpy_canonicalize_dro_expectation(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps, precond_inv,
):
    """
    NumPy canonicalization of DRO expectation problem to Clarabel form.
    
    This mirrors ClarabelCanonicalizer.setup_expectation_problem() exactly.
    For now, we assume no auxiliary PSD constraints (M_psd = 0).
    
    Returns:
        A_csc: Constraint matrix (scipy CSC sparse)
        b: RHS vector (numpy array)
        c: Objective vector (numpy array) 
        cones: List of Clarabel cone objects
        x_dim: Dimension of decision variable
    """
    # Local helper functions that work on NumPy arrays
    def symm_vectorize(A, scale_factor):
        """Vectorize lower triangle of symmetric matrix with off-diagonal scaling."""
        A = np.asarray(A)  # Ensure NumPy array
        n = A.shape[0]
        rows, cols = np.tril_indices(n)
        A_vec = A[rows, cols].copy()  # Copy to avoid modifying original
        off_diag_mask = rows != cols
        A_vec[off_diag_mask] *= scale_factor
        return A_vec
    
    def scaled_off_triangles(A, scale_factor):
        """Create diagonal matrix from scaled symmetric vectorization."""
        A_vec = symm_vectorize(A, scale_factor)
        return np.diag(A_vec)
    
    N = len(G_batch)
    M = len(A_vals)
    V = b_obj.shape[0]
    S_mat = A_obj.shape[0]
    S_vec = S_mat * (S_mat + 1) // 2
    
    # No auxiliary PSD constraints for now
    M_psd = 0
    H_vec_sum = 0
    
    # Decision variable dimension
    x_dim = 1 + N * (1 + M + V + S_vec + H_vec_sum)
    
    # Index offsets
    lambd_idx = 0
    s_start = 1
    y_start = s_start + N
    Fz_start = y_start + N * M
    Gz_start = Fz_start + N * V
    
    def s_idx(i):
        return s_start + i
    
    def y_idx(i, j):
        return y_start + i * M + j
    
    def Fz_idx(i, j):
        return Fz_start + i * V + j
    
    def Gz_idx(i, j):
        return Gz_start + i * S_vec + j
    
    # Build objective vector
    c_obj = np.zeros(x_dim)
    c_obj[lambd_idx] = eps
    c_obj[s_start:s_start + N] = 1.0 / N
    
    A_blocks = []
    b_blocks = []
    cones = []
    
    # 1. Epigraph constraints: -c^T y_i - Tr(G_i @ Gz_i) - F_i @ Fz_i - s_i <= 0
    epi_constr = np.zeros((N, x_dim))
    for i in range(N):
        G_sample, F_sample = G_batch[i], F_batch[i]
        y_s, y_e = y_idx(i, 0), y_idx(i, M)
        epi_constr[i, y_s:y_e] = -c_vals
        epi_constr[i, s_idx(i)] = -1
        
        Fz_s, Fz_e = Fz_idx(i, 0), Fz_idx(i, V)
        epi_constr[i, Fz_s:Fz_e] = -F_sample
        
        Gz_s, Gz_e = Gz_idx(i, 0), Gz_idx(i, S_vec)
        epi_constr[i, Gz_s:Gz_e] = -symm_vectorize(G_sample, 2)
    
    A_blocks.append(spa.csc_matrix(epi_constr))
    b_blocks.append(np.zeros(N))
    
    # 2. y >= 0 constraints
    y_nonneg = np.zeros((N * M, x_dim))
    y_s = y_idx(0, 0)
    y_e = y_idx(N-1, M)
    y_nonneg[0:N*M, y_s:y_e] = -np.eye(N * M)
    
    A_blocks.append(spa.csc_matrix(y_nonneg))
    b_blocks.append(np.zeros(N * M))
    cones.append(clarabel.NonnegativeConeT(N + N * M))  # Coalesced
    
    # 3. Equality constraints: -B^T y_i + Fz_i = -b_obj
    Bm_T = np.array(b_vals).T  # (V, M)
    
    yB_rows = []
    yB_rhs = []
    for i in range(N):
        curr_lhs = np.zeros((V, x_dim))
        y_s, y_e = y_idx(i, 0), y_idx(i, M)
        curr_lhs[:, y_s:y_e] = -Bm_T
        
        Fz_s, Fz_e = Fz_idx(i, 0), Fz_idx(i, V)
        curr_lhs[:, Fz_s:Fz_e] = np.eye(V)
        
        yB_rows.append(spa.csc_matrix(curr_lhs))
        yB_rhs.append(-b_obj)
    
    A_blocks.append(spa.vstack(yB_rows))
    b_blocks.append(np.hstack(yB_rhs))
    cones.append(clarabel.ZeroConeT(V * N))
    
    # 4. PSD constraints: -A^* y_i + Gz_i << -A_obj
    A_obj_svec = symm_vectorize(A_obj, np.sqrt(2.0))
    Am_svec = np.array([symm_vectorize(A_vals[m], np.sqrt(2.0)) for m in range(M)])  # (M, S_vec)
    Am_T = Am_svec.T  # (S_vec, M)
    
    yA_rows = []
    yA_rhs = []
    scaledI = scaled_off_triangles(np.ones((S_mat, S_mat)), np.sqrt(2.0))
    
    for i in range(N):
        curr_lhs = np.zeros((S_vec, x_dim))
        y_s, y_e = y_idx(i, 0), y_idx(i, M)
        curr_lhs[:, y_s:y_e] = -Am_T
        
        Gz_s, Gz_e = Gz_idx(i, 0), Gz_idx(i, S_vec)
        curr_lhs[:, Gz_s:Gz_e] = scaledI
        
        yA_rows.append(spa.csc_matrix(curr_lhs))
        yA_rhs.append(-A_obj_svec)
    
    A_blocks.append(spa.vstack(yA_rows))
    b_blocks.append(np.hstack(yA_rhs))
    cones.extend([clarabel.PSDTriangleConeT(S_mat) for _ in range(N)])
    
    # 5. SOCP constraints: || [Gz_i, Fz_i] ||_precond <= lambda
    G_precond_vec, F_precond = precond_inv
    F_precond_sq = F_precond ** 2
    scaled_G_vec_outer_prod = np.outer(G_precond_vec, G_precond_vec)
    scaledG_mult = scaled_off_triangles(scaled_G_vec_outer_prod, np.sqrt(2.0))
    
    socp_rows = []
    socp_rhs = []
    for i in range(N):
        curr_lhs = np.zeros((1 + V + S_vec, x_dim))
        Fz_s, Fz_e = Fz_idx(i, 0), Fz_idx(i, V)
        Gz_s, Gz_e = Gz_idx(i, 0), Gz_idx(i, S_vec)
        
        curr_lhs[0, lambd_idx] = -1
        curr_lhs[1:V+1, Fz_s:Fz_e] = -np.diag(F_precond_sq)
        curr_lhs[V+1:, Gz_s:Gz_e] = -scaledG_mult
        
        socp_rows.append(spa.csc_matrix(curr_lhs))
        socp_rhs.append(np.zeros(1 + V + S_vec))
    
    A_blocks.append(spa.vstack(socp_rows))
    b_blocks.append(np.hstack(socp_rhs))
    cones.extend([clarabel.SecondOrderConeT(1 + V + S_vec) for _ in range(N)])
    
    # Combine all blocks
    A_csc = spa.vstack(A_blocks)
    b = np.hstack(b_blocks)
    
    return A_csc, b, c_obj, cones, x_dim


# =============================================================================
# High-level differentiable solve function
# =============================================================================

class ClarabelSolveData:
    """Container for static data needed by the Clarabel solver."""
    
    def __init__(self, cone_info, A_shape):
        self.cone_info = cone_info
        self.A_shape = A_shape
        # Use jax_to_clarabel_cones to match the ordering from jax_canonicalize_dro_expectation
        self.clarabel_cones = jax_to_clarabel_cones(cone_info)
        # Build diffcp cone dict with matching ordering
        self.diffcp_cone_dict = self._build_diffcp_cone_dict(cone_info)
    
    def _build_diffcp_cone_dict(self, cone_info):
        """Build diffcp cone dictionary matching the JAX canonicalization ordering."""
        N = cone_info['N']
        V = cone_info['V']
        S_mat = cone_info['S_mat']
        S_vec = cone_info['S_vec']
        
        # Order: nonnegative, zero, PSD, SOC - matching jax_to_clarabel_cones
        cone_dict = {}
        
        # For diffcp: POS = nonnegative, ZERO = equality, SOC = second order, PSD = semidefinite
        if cone_info['nonneg'] > 0:
            cone_dict[diffcp.POS] = int(cone_info['nonneg'])
        if cone_info['zero'] > 0:
            cone_dict[diffcp.ZERO] = int(cone_info['zero'])
        if cone_info['psd']:
            cone_dict[diffcp.PSD] = [int(d) for d in cone_info['psd']]
        if cone_info['soc']:
            cone_dict[diffcp.SOC] = [int(d) for d in cone_info['soc']]
        
        return cone_dict



def clarabel_solve_wrapper(static_data, A_dense, b, c):
    """
    Differentiable Clarabel solve using diffcp for both forward and backward.
    
    Uses diffcp.solve_and_derivative with Clarabel as the solver. The adjoint
    derivative is computed once during the forward pass and cached for backward.
    
    Args:
        static_data: ClarabelSolveData with cone info
        A_dense: Constraint matrix as JAX array
        b: RHS vector as JAX array
        c: Objective vector as JAX array
    
    Returns:
        Optimal objective value
    """
    # We use a module-level cache to store the adjoint_derivative function
    # since jax.custom_vjp doesn't support closures with non-JAX objects
    _adjoint_cache = {}
    
    @jax.custom_vjp
    def _solve(A_dense, b, c):
        # Forward pass: call diffcp via pure_callback
        def solve_impl(A_np, b_np, c_np):
            A_csc = spa.csc_matrix(A_np)
            try:
                x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
                    A_csc, b_np, c_np,
                    static_data.diffcp_cone_dict,
                    solve_method='Clarabel',
                    verbose=False,
                )
                obj = c_np @ x
                # Store adjoint for backward pass
                _adjoint_cache['adjoint'] = adjoint_deriv
                _adjoint_cache['valid'] = True
            except Exception as e:
                print(f"diffcp solve failed: {e}")
                x = np.zeros(c_np.shape[0])
                obj = np.nan
                _adjoint_cache['valid'] = False
            
            return np.array([obj])
        
        result_shapes = jax.ShapeDtypeStruct((1,), jnp.float64)
        obj = jax.pure_callback(solve_impl, result_shapes, A_dense, b, c)
        return obj[0]
    
    def _solve_fwd(A_dense, b, c):
        # Solve using diffcp and cache solution + adjoint
        def solve_impl(A_np, b_np, c_np):
            A_csc = spa.csc_matrix(A_np)
            try:
                x, y, s, _, adjoint_deriv = diffcp.solve_and_derivative(
                    A_csc, b_np, c_np,
                    static_data.diffcp_cone_dict,
                    solve_method='Clarabel',
                    verbose=False,
                )
                obj = c_np @ x
                # Store adjoint for backward pass
                _adjoint_cache['adjoint'] = adjoint_deriv
                _adjoint_cache['x'] = x
                _adjoint_cache['y'] = y
                _adjoint_cache['s'] = s
                _adjoint_cache['valid'] = True
            except Exception as e:
                print(f"diffcp solve failed in fwd: {e}")
                x = np.zeros(c_np.shape[0])
                y = np.zeros(b_np.shape[0])
                s = np.zeros(b_np.shape[0])
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
            if not _adjoint_cache.get('valid', False):
                # If forward solve failed, return zero gradients
                return np.zeros_like(A_np), np.zeros_like(b_np), np.zeros_like(c_np)
            
            adjoint_deriv = _adjoint_cache['adjoint']
            y_np = _adjoint_cache['y']
            s_np = _adjoint_cache['s']
            
            # The objective is c^T x.
            # d(c^T x) = c^T dx + x^T dc
            # Using adjoint: dx = d_obj * c propagates through D^T
            
            dx = d_obj_np * c_np
            dy = np.zeros_like(y_np)
            ds = np.zeros_like(s_np)
            
            dA_sol, db_sol, dc_sol = adjoint_deriv(dx, dy, ds)
            
            # Direct contribution to dc from x
            dc_direct = d_obj_np * x_np
            
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
        
        return dA, db, dc
    
    _solve.defvjp(_solve_fwd, _solve_bwd)
    
    return _solve(A_dense, b, c)



# =============================================================================
# Full Differentiable DRO Pipeline
# =============================================================================

def dro_clarabel_solve(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps, precond_inv,
):
    """
    Full differentiable DRO solve using Clarabel + diffcp.
    
    This function combines:
    1. JAX canonicalization (traceable, JIT-compatible)
    2. Clarabel solve (via pure_callback)
    3. diffcp adjoint derivative (via pure_callback in backward pass)
    
    Args:
        A_obj, b_obj, A_vals, b_vals, c_vals: PEP constraint data
        G_batch: (N, S_mat, S_mat) Gram matrices
        F_batch: (N, V) function values
        eps: Wasserstein radius
        precond_inv: (G_precond_inv, F_precond_inv) preconditioner
    
    Returns:
        obj_val: Optimal DRO objective value (scalar, differentiable)
    """
    # JAX canonicalization (differentiable)
    A_dense, b, c, x_dim, cone_info = jax_canonicalize_dro_expectation(
        A_obj, b_obj, A_vals, b_vals, c_vals,
        G_batch, F_batch,
        eps, precond_inv,
    )
    
    # Create static data for cones (non-traceable, used in callbacks)
    # Note: This creates Python objects, so the function isn't fully JIT-able
    # but the gradients will still flow through correctly
    static_data = ClarabelSolveData(cone_info, A_dense.shape)
    
    # Solve with custom VJP (non-traceable forward, diffcp adjoint backward)
    obj_val = clarabel_solve_wrapper(static_data, A_dense, b, c)
    
    return obj_val

