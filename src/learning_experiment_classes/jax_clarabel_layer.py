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
        log.debug("MKL Pardiso not available, using default qdldl solver")
    
    return _MKL_AVAILABLE

def get_direct_solve_method():
    """Get the best available direct solve method for Clarabel.
    
    Returns 'mkl' if MKL Pardiso is available, otherwise 'qdldl'.
    """
    return 'mkl' if _detect_mkl_pardiso() else 'qdldl'


# =============================================================================
# JAX Helper Functions (vectorization for PSD cones)
# =============================================================================

@jax.jit
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


@jax.jit
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

@jax.jit
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


@jax.jit
def jax_canonicalize_dro_cvar(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps, alpha, precond_inv,
):
    """
    Pure JAX canonicalization of DRO CVaR problem to Clarabel form.
    
    This is a JAX-traceable version that can be JIT compiled.
    
    CVaR has doubled dual variables (y1, y2) and corresponding Wasserstein
    constraint variables (Fz1/Fz2, Gz1/Gz2). The VaR threshold is 't'.
    
    Returns:
        A_dense: Constraint matrix as dense JAX array
        b: RHS vector
        c_obj: Objective vector
        x_dim: Dimension of decision variable
        cone_info: Dict with cone dimensions for later conversion
    """
    alpha_inv = 1.0 / alpha
    
    N = G_batch.shape[0]
    M = A_vals.shape[0]  # Number of interpolation constraints
    V = b_obj.shape[0]  # Dimension of Fz
    S_mat = A_obj.shape[0]  # Dimension of main PSD constraint
    S_vec = S_mat * (S_mat + 1) // 2
    
    # Process PEP matrices
    A_obj_svec = jax_symm_vectorize(A_obj, jnp.sqrt(2.0))  # (S_vec,)
    A_vals_svec = jax.vmap(lambda A: jax_symm_vectorize(A, jnp.sqrt(2.0)))(A_vals)  # (M, S_vec)
    Bm_T = b_vals.T  # (V, M)
    
    # Process sample matrices to vectors
    G_batch_svec = jax.vmap(lambda G: jax_symm_vectorize(G, 2.0))(G_batch)  # (N, S_vec)
    
    # Preconditioner
    G_precond_vec, F_precond = precond_inv
    F_precond_sq = F_precond ** 2
    scaled_G_vec_outer = jnp.outer(G_precond_vec, G_precond_vec)
    scaledG_mult = jax_scaled_off_triangles(scaled_G_vec_outer, jnp.sqrt(2.0))  # (S_vec, S_vec)
    scaledI = jax_scaled_off_triangles(jnp.ones((S_mat, S_mat)), jnp.sqrt(2.0))  # (S_vec, S_vec)
    
    # Decision variable dimension
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
    # Build constraint blocks (in diffcp order: zero, nonneg, soc, psd)
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
    
    # Combine all blocks (diffcp order: zero, nonneg, soc, psd)
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
    
    # Cone info for later conversion
    cone_info = {
        'zero': 2 * N * V,
        'nonneg': 2 * N + 2 * N * M,
        'soc': [socp_dim] * (2 * N),
        'psd': [S_mat] * (2 * N),
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

@jax.jit
def jax_canonicalize_pep(A_obj, b_obj, A_vals, b_vals, c_vals):
    """
    Pure JAX canonicalization of vanilla PEP (no DRO) to Clarabel form.
    
    This is a JAX-traceable version for differentiable PEP without samples.
    Simpler than DRO version - just the dual SDP with y >= 0, equality, and PSD constraints.
    
    The PEP dual problem is:
        min  c^T y
        s.t. B^T y = -b_obj   (equality)
             y >= 0           (non-negativity)
             A_obj - A^* y >> 0  (PSD)
    
    Args:
        A_obj: Objective matrix (S_mat, S_mat) - contribution to Gram matrix
        b_obj: Objective vector (V,) - contribution to function values
        A_vals: Constraint matrices (M, S_mat, S_mat)
        b_vals: Constraint vectors (M, V)
        c_vals: Constraint constants (M,)
    
    Returns:
        A_dense: Dense constraint matrix (V + M + S_vec, M) - JAX array
        b: RHS vector - JAX array  
        c: Objective vector - JAX array
        x_dim: Dimension of decision variable (= M)
        cone_info: Dict with cone dimensions for later conversion
    """
    # Get dimensions
    M = A_vals.shape[0]     # Number of constraints / dual variables
    V = b_obj.shape[0]      # Dimension of b_obj (function values)
    S_mat = A_obj.shape[0]  # Dimension of PSD constraint matrix
    S_vec = S_mat * (S_mat + 1) // 2  # Vectorized lower triangle size
    
    # Decision variable is just y with dimension M
    x_dim = M
    
    # Build constraint matrix rows
    # 1. Equality: -B^T y = -b_obj  =>  constraint matrix is -B^T
    Bm_T = b_vals.T  # (V, M) - transpose of b_vals
    
    # 2. Non-negativity: -y <= 0  =>  constraint matrix is -I
    eye_M = jnp.eye(M)
    
    # 3. PSD: -A^* y + slack = -A_obj (vectorized)
    # A^* y = sum_m y_m * A_vals[m], in svec form
    A_obj_svec = jax_symm_vectorize(A_obj, jnp.sqrt(2.0))  # (S_vec,)
    A_vals_svec = jax.vmap(lambda A: jax_symm_vectorize(A, jnp.sqrt(2.0)))(A_vals)  # (M, S_vec)
    Am_T = A_vals_svec.T  # (S_vec, M)
    
    # Stack constraint matrix (diffcp order: zero, nonneg, psd)
    A_dense = jnp.vstack([
        -Bm_T,    # (V, M) - equality constraints
        -eye_M,   # (M, M) - non-negativity constraints
        -Am_T,    # (S_vec, M) - PSD constraints
    ])
    
    # Build RHS vector b
    b = jnp.concatenate([
        -b_obj,           # (V,) - equality RHS
        jnp.zeros(M),     # (M,) - non-negativity RHS
        -A_obj_svec,      # (S_vec,) - PSD RHS
    ])
    
    # Objective: negate c_vals for correct sign (PEP primal is maximization)
    c = -c_vals
    
    # Cone info for later conversion to Clarabel cones
    cone_info = {
        'zero': V,          # V equality constraints
        'nonneg': M,        # M non-negativity constraints
        'soc': [],          # No SOC constraints in vanilla PEP
        'psd': [S_mat],     # One PSD constraint of size S_mat
        'M': M,
        'V': V,
        'S_mat': S_mat,
        'S_vec': S_vec,
    }
    
    return A_dense, b, c, x_dim, cone_info


# NumPy canonicalization functions have been moved to numpy_clarabel_layer.py
# Import them from there if needed for testing purposes



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
                    direct_solve_method=get_direct_solve_method(),
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
                    direct_solve_method=get_direct_solve_method(),
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
    # Risk measure
    dro_obj='expectation', alpha=0.1,
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
        dro_obj: Risk measure type ('expectation' or 'cvar')
        alpha: CVaR confidence level (only used when dro_obj='cvar')
    
    Returns:
        obj_val: Optimal DRO objective value (scalar, differentiable)
    """
    # Select canonicalization based on risk measure
    if dro_obj == 'expectation':
        A_dense, b, c, x_dim, cone_info = jax_canonicalize_dro_expectation(
            A_obj, b_obj, A_vals, b_vals, c_vals,
            G_batch, F_batch,
            eps, precond_inv,
        )
    elif dro_obj == 'cvar':
        A_dense, b, c, x_dim, cone_info = jax_canonicalize_dro_cvar(
            A_obj, b_obj, A_vals, b_vals, c_vals,
            G_batch, F_batch,
            eps, alpha, precond_inv,
        )
    else:
        raise ValueError(f"Unknown dro_obj: {dro_obj}")
    
    # Create static data for cones (non-traceable, used in callbacks)
    static_data = ClarabelSolveData(cone_info, A_dense.shape)
    
    # Solve with custom VJP (non-traceable forward, diffcp adjoint backward)
    obj_val = clarabel_solve_wrapper(static_data, A_dense, b, c)
    
    return obj_val


def pep_clarabel_solve(A_obj, b_obj, A_vals, b_vals, c_vals):
    """
    Differentiable vanilla PEP solve using Clarabel + diffcp.
    
    This is the non-DRO version - solves the standard PEP SDP without samples.
    Used for learning step sizes by minimizing the worst-case PEP bound.
    
    The function combines:
    1. JAX canonicalization (traceable, JIT-compatible)
    2. Clarabel solve (via pure_callback)
    3. diffcp adjoint derivative (via pure_callback in backward pass)
    
    Args:
        A_obj: Objective matrix (S_mat, S_mat) - Gram matrix contribution
        b_obj: Objective vector (V,) - function value contribution
        A_vals: Constraint matrices (M, S_mat, S_mat)
        b_vals: Constraint vectors (M, V)
        c_vals: Constraint constants (M,)
    
    Returns:
        obj_val: PEP worst-case bound (scalar, differentiable)
    """
    # JAX canonicalization (traceable)
    A_dense, b, c, x_dim, cone_info = jax_canonicalize_pep(
        A_obj, b_obj, A_vals, b_vals, c_vals
    )
    
    # Create static data for cones (non-traceable, used in callbacks)
    static_data = ClarabelSolveData(cone_info, A_dense.shape)
    
    # Solve with custom VJP (non-traceable forward, diffcp adjoint backward)
    obj_val = clarabel_solve_wrapper(static_data, A_dense, b, c)
    
    return obj_val
