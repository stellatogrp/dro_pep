"""
NumPy-based Clarabel canonicalization functions.

This module provides NumPy implementations of DRO problem canonicalization
for use with the Clarabel solver. These are used for testing and comparison
against the JAX versions.
"""

import numpy as np
import scipy.sparse as spa
import clarabel
import logging

log = logging.getLogger(__name__)


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


def numpy_canonicalize_dro_cvar(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps, alpha, precond_inv,
):
    """
    NumPy canonicalization of DRO CVaR problem to Clarabel form.
    
    CVaR has doubled dual variables (y1, y2) and corresponding Wasserstein
    constraint variables (Fz1/Fz2, Gz1/Gz2). The VaR threshold is 't'.
    
    For now, we assume no auxiliary PSD constraints (M_psd = 0).
    
    Decision vector layout:
        [lambda, t, s_0, ..., s_{N-1}, 
         y1_0, ..., y1_{N-1},  # each y1_i has M components
         y2_0, ..., y2_{N-1},  # each y2_i has M components
         Fz1_0, ..., Fz1_{N-1},  # each Fz1_i has V components
         Fz2_0, ..., Fz2_{N-1},
         Gz1_0, ..., Gz1_{N-1},  # each Gz1_i has S_vec components
         Gz2_0, ..., Gz2_{N-1}]
    
    Returns:
        A_csc: Constraint matrix in CSC format
        b: RHS vector
        c_obj: Objective vector
        cones: List of Clarabel cones
        x_dim: Dimension of decision variable
    """
    alpha_inv = 1.0 / alpha
    
    N = G_batch.shape[0]
    M = len(A_vals)  # Number of interpolation constraints
    V = b_obj.shape[0]  # Dimension of Fz (b_obj vector size)
    S_mat = A_obj.shape[0]  # Dimension of main PSD constraint
    S_vec = S_mat * (S_mat + 1) // 2  # Vectorized size
    
    # Local helper functions (in-file, avoiding JAX immutability issues)
    def symm_vectorize(A, scale_factor):
        """Vectorize lower triangle of symmetric matrix with off-diagonal scaling."""
        n = A.shape[0]
        cols, rows = np.tril_indices(n)  # Column-major order (flipped)
        diag_mask = (rows == cols)
        v = np.zeros(len(rows))
        v[diag_mask] = A[rows[diag_mask], cols[diag_mask]]
        v[~diag_mask] = A[rows[~diag_mask], cols[~diag_mask]] * scale_factor
        return v
    
    def scaled_off_triangles(A, scale_factor):
        """Create diagonal matrix from scaled symmetric vectorization."""
        v = symm_vectorize(A, scale_factor)
        return np.diag(v)
    
    # Decision variable dimension: 
    # lambda + t + N*(s) + N*2*M (y1, y2) + N*2*V (Fz1, Fz2) + N*2*S_vec (Gz1, Gz2)
    x_dim = 2 + N * (1 + 2 * (M + V + S_vec))
    
    # Index functions
    lambd_idx = 0
    t_idx = 1
    
    s_start = 2
    def s_idx(i):
        return s_start + i
    
    y1_start = s_start + N
    def y1_idx(i, j):
        return y1_start + i * M + j
    
    y2_start = y1_start + N * M
    def y2_idx(i, j):
        return y2_start + i * M + j
    
    Fz1_start = y2_start + N * M
    def Fz1_idx(i, j):
        return Fz1_start + i * V + j
    
    Fz2_start = Fz1_start + N * V
    def Fz2_idx(i, j):
        return Fz2_start + i * V + j
    
    Gz1_start = Fz2_start + N * V
    def Gz1_idx(i, j):
        return Gz1_start + i * S_vec + j
    
    Gz2_start = Gz1_start + N * S_vec
    def Gz2_idx(i, j):
        return Gz2_start + i * S_vec + j
    
    # Build objective vector: minimize eps*lambda + (1/N)*sum(s)
    c_obj = np.zeros(x_dim)
    c_obj[lambd_idx] = eps
    c_obj[s_start:s_start + N] = 1.0 / N
    
    A_blocks = []
    b_blocks = []
    cones = []
    
    c = c_vals  # Shorthand
    scaledI = scaled_off_triangles(np.ones((S_mat, S_mat)), np.sqrt(2.0))
    
    # Precompute constraint matrices
    Bm_T = np.array(b_vals).T  # (V, M)
    A_obj_svec = symm_vectorize(A_obj, np.sqrt(2.0))
    Am_svec = np.array([symm_vectorize(A_vals[m], np.sqrt(2.0)) for m in range(M)])  # (M, S_vec)
    Am_T = Am_svec.T  # (S_vec, M)
    
    # Preconditioner terms
    G_precond_vec, F_precond = precond_inv
    F_precond_sq = F_precond ** 2
    scaled_G_vec_outer_prod = np.outer(G_precond_vec, G_precond_vec)
    scaledG_mult = scaled_off_triangles(scaled_G_vec_outer_prod, np.sqrt(2.0))
    
    # =========================================================================
    # Constraints in diffcp order: zero, nonneg, soc, psd
    # =========================================================================
    
    # --- ZERO CONE (Equality constraints) ---
    
    # 1. -B^T y1_i + Fz1_i = -b_obj
    yB1_rows = []
    yB1_rhs = []
    for i in range(N):
        curr_lhs = np.zeros((V, x_dim))
        y1_s, y1_e = y1_idx(i, 0), y1_idx(i, M)
        curr_lhs[:, y1_s:y1_e] = -Bm_T
        
        Fz1_s, Fz1_e = Fz1_idx(i, 0), Fz1_idx(i, V)
        curr_lhs[:, Fz1_s:Fz1_e] = np.eye(V)
        
        yB1_rows.append(spa.csc_matrix(curr_lhs))
        yB1_rhs.append(-b_obj)
    
    A_blocks.append(spa.vstack(yB1_rows))
    b_blocks.append(np.hstack(yB1_rhs))
    
    # 2. -B^T y2_i + Fz2_i = -(1/alpha) * b_obj
    yB2_rows = []
    yB2_rhs = []
    for i in range(N):
        curr_lhs = np.zeros((V, x_dim))
        y2_s, y2_e = y2_idx(i, 0), y2_idx(i, M)
        curr_lhs[:, y2_s:y2_e] = -Bm_T
        
        Fz2_s, Fz2_e = Fz2_idx(i, 0), Fz2_idx(i, V)
        curr_lhs[:, Fz2_s:Fz2_e] = np.eye(V)
        
        yB2_rows.append(spa.csc_matrix(curr_lhs))
        yB2_rhs.append(-alpha_inv * b_obj)
    
    A_blocks.append(spa.vstack(yB2_rows))
    b_blocks.append(np.hstack(yB2_rhs))
    cones.append(clarabel.ZeroConeT(2 * V * N))  # Combined zero cone
    
    # --- NONNEGATIVE CONE ---
    
    # 3. Epigraph for y1 branch: t - c^T y1_i - Tr(G @ Gz1_i) - F @ Fz1_i - s_i <= 0
    epi1_constr = np.zeros((N, x_dim))
    for i in range(N):
        G_sample, F_sample = G_batch[i], F_batch[i]
        epi1_constr[i, t_idx] = 1
        y1_s, y1_e = y1_idx(i, 0), y1_idx(i, M)
        epi1_constr[i, y1_s:y1_e] = -c
        epi1_constr[i, s_idx(i)] = -1
        
        Fz1_s, Fz1_e = Fz1_idx(i, 0), Fz1_idx(i, V)
        epi1_constr[i, Fz1_s:Fz1_e] = -F_sample
        
        Gz1_s, Gz1_e = Gz1_idx(i, 0), Gz1_idx(i, S_vec)
        epi1_constr[i, Gz1_s:Gz1_e] = -symm_vectorize(G_sample, 2)
    
    A_blocks.append(spa.csc_matrix(epi1_constr))
    b_blocks.append(np.zeros(N))
    
    # 4. Epigraph for y2 branch: -(1/alpha - 1)*t - c^T y2_i - Tr(G @ Gz2_i) - F @ Fz2_i - s_i <= 0
    epi2_constr = np.zeros((N, x_dim))
    for i in range(N):
        G_sample, F_sample = G_batch[i], F_batch[i]
        epi2_constr[i, t_idx] = -(alpha_inv - 1)
        y2_s, y2_e = y2_idx(i, 0), y2_idx(i, M)
        epi2_constr[i, y2_s:y2_e] = -c
        epi2_constr[i, s_idx(i)] = -1
        
        Fz2_s, Fz2_e = Fz2_idx(i, 0), Fz2_idx(i, V)
        epi2_constr[i, Fz2_s:Fz2_e] = -F_sample
        
        Gz2_s, Gz2_e = Gz2_idx(i, 0), Gz2_idx(i, S_vec)
        epi2_constr[i, Gz2_s:Gz2_e] = -symm_vectorize(G_sample, 2)
    
    A_blocks.append(spa.csc_matrix(epi2_constr))
    b_blocks.append(np.zeros(N))
    
    # 5. y1 >= 0
    y1_nonneg = np.zeros((N * M, x_dim))
    y1_s = y1_idx(0, 0)
    y1_e = y1_idx(N-1, M)
    y1_nonneg[0:N*M, y1_s:y1_e] = -np.eye(N * M)
    
    A_blocks.append(spa.csc_matrix(y1_nonneg))
    b_blocks.append(np.zeros(N * M))
    
    # 6. y2 >= 0
    y2_nonneg = np.zeros((N * M, x_dim))
    y2_s = y2_idx(0, 0)
    y2_e = y2_idx(N-1, M)
    y2_nonneg[0:N*M, y2_s:y2_e] = -np.eye(N * M)
    
    A_blocks.append(spa.csc_matrix(y2_nonneg))
    b_blocks.append(np.zeros(N * M))
    
    # Combined nonnegative cone: 2*N (epi1, epi2) + 2*N*M (y1, y2)
    cones.append(clarabel.NonnegativeConeT(2 * N + 2 * N * M))
    
    # --- SOC CONES ---
    
    # 7. SOCP for Gz1, Fz1: || [Gz1_i, Fz1_i] ||_precond <= lambda
    socp1_rows = []
    socp1_rhs = []
    for i in range(N):
        curr_lhs = np.zeros((1 + V + S_vec, x_dim))
        Fz1_s, Fz1_e = Fz1_idx(i, 0), Fz1_idx(i, V)
        Gz1_s, Gz1_e = Gz1_idx(i, 0), Gz1_idx(i, S_vec)
        
        curr_lhs[0, lambd_idx] = -1
        curr_lhs[1:V+1, Fz1_s:Fz1_e] = -np.diag(F_precond_sq)
        curr_lhs[V+1:, Gz1_s:Gz1_e] = -scaledG_mult
        
        socp1_rows.append(spa.csc_matrix(curr_lhs))
        socp1_rhs.append(np.zeros(1 + V + S_vec))
    
    A_blocks.append(spa.vstack(socp1_rows))
    b_blocks.append(np.hstack(socp1_rhs))
    
    # 8. SOCP for Gz2, Fz2: || [Gz2_i, Fz2_i] ||_precond <= lambda
    socp2_rows = []
    socp2_rhs = []
    for i in range(N):
        curr_lhs = np.zeros((1 + V + S_vec, x_dim))
        Fz2_s, Fz2_e = Fz2_idx(i, 0), Fz2_idx(i, V)
        Gz2_s, Gz2_e = Gz2_idx(i, 0), Gz2_idx(i, S_vec)
        
        curr_lhs[0, lambd_idx] = -1
        curr_lhs[1:V+1, Fz2_s:Fz2_e] = -np.diag(F_precond_sq)
        curr_lhs[V+1:, Gz2_s:Gz2_e] = -scaledG_mult
        
        socp2_rows.append(spa.csc_matrix(curr_lhs))
        socp2_rhs.append(np.zeros(1 + V + S_vec))
    
    A_blocks.append(spa.vstack(socp2_rows))
    b_blocks.append(np.hstack(socp2_rhs))
    cones.extend([clarabel.SecondOrderConeT(1 + V + S_vec) for _ in range(2 * N)])  # 2N SOC cones
    
    # --- PSD CONES ---
    
    # 9. -A^* y1_i + Gz1_i << -A_obj (PSD for each sample)
    yA1_rows = []
    yA1_rhs = []
    for i in range(N):
        curr_lhs = np.zeros((S_vec, x_dim))
        y1_s, y1_e = y1_idx(i, 0), y1_idx(i, M)
        curr_lhs[:, y1_s:y1_e] = -Am_T
        
        Gz1_s, Gz1_e = Gz1_idx(i, 0), Gz1_idx(i, S_vec)
        curr_lhs[:, Gz1_s:Gz1_e] = scaledI
        
        yA1_rows.append(spa.csc_matrix(curr_lhs))
        yA1_rhs.append(-A_obj_svec)
    
    A_blocks.append(spa.vstack(yA1_rows))
    b_blocks.append(np.hstack(yA1_rhs))
    
    # 10. -A^* y2_i + Gz2_i << -(1/alpha) * A_obj (PSD for each sample)
    yA2_rows = []
    yA2_rhs = []
    for i in range(N):
        curr_lhs = np.zeros((S_vec, x_dim))
        y2_s, y2_e = y2_idx(i, 0), y2_idx(i, M)
        curr_lhs[:, y2_s:y2_e] = -Am_T
        
        Gz2_s, Gz2_e = Gz2_idx(i, 0), Gz2_idx(i, S_vec)
        curr_lhs[:, Gz2_s:Gz2_e] = scaledI
        
        yA2_rows.append(spa.csc_matrix(curr_lhs))
        yA2_rhs.append(-alpha_inv * A_obj_svec)
    
    A_blocks.append(spa.vstack(yA2_rows))
    b_blocks.append(np.hstack(yA2_rhs))
    cones.extend([clarabel.PSDTriangleConeT(S_mat) for _ in range(2 * N)])  # 2N PSD cones
    
    # Combine all blocks
    A_csc = spa.vstack(A_blocks)
    b = np.hstack(b_blocks)
    
    return A_csc, b, c_obj, cones, x_dim


def numpy_canonicalize_pep(
    A_obj, b_obj, A_vals, b_vals, c_vals,
):
    def symm_vectorize(A, scale_factor):
        """Vectorize lower triangle of symmetric matrix with off-diagonal scaling."""
        A = np.asarray(A)  # Ensure NumPy array
        n = A.shape[0]
        rows, cols = np.tril_indices(n)
        A_vec = A[rows, cols].copy()  # Copy to avoid modifying original
        off_diag_mask = rows != cols
        A_vec[off_diag_mask] *= scale_factor
        return A_vec
    
    M = len(A_vals)
    V = b_obj.shape[0]
    S_mat = A_obj.shape[0]
    S_vec = S_mat * (S_mat + 1) // 2

    x_dim = c_vals.shape[0]

    Bm_T = np.array(b_vals).T  # (V, M)

    A_obj_svec = symm_vectorize(A_obj, np.sqrt(2.0))
    Am_svec = np.array([symm_vectorize(A_vals[m], np.sqrt(2.0)) for m in range(M)])  # (M, S_vec)
    Am_T = Am_svec.T  # (S_vec, M)

    A = spa.vstack([
        -Bm_T,
        -spa.eye(M),
        -Am_T,
    ])

    b = np.hstack([
        -b_obj,
        np.zeros(M),
        -A_obj_svec,
    ])

    c = -c_vals  # Negate for minimization (PEP primal is maximization)

    cones = [
        clarabel.ZeroConeT(V),
        clarabel.NonnegativeConeT(M),
        clarabel.PSDTriangleConeT(S_mat),
    ]

    return spa.csc_matrix(A), b, c, cones, x_dim
