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
