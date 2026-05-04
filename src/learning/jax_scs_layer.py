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
import math

log = logging.getLogger(__name__)


# @partial(jax.jit, static_argnames=['precond_type'])
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
    
    # G path: diagonals are squared norms, mathematically >= 0.
    # Double-where idiom avoids NaN gradient at sqrt(0).
    G_diags = jax.vmap(jnp.diag)(G_batch)  # (N, dimG)
    positive_mask_G = G_diags > 0
    safe_G_diags = jnp.where(positive_mask_G, G_diags, 1.0)
    G_diag_sqrt = jnp.where(positive_mask_G, jnp.sqrt(safe_G_diags), 0.0)

    # F path: for composite problems (e.g. Lasso), F_batch entries can be
    # negative (f2(x_k) < f2(x_s) is possible even though f(x_k) >= f(x_s)).
    # Taking |F| before aggregation gives a meaningful magnitude estimate and
    # avoids sqrt-of-negative NaN traps. For positive-only F (Quad/GD), this
    # matches the previous behavior exactly.
    F_abs = jnp.abs(F_batch)

    if precond_type == 'average':
        avg_G = jnp.mean(G_diag_sqrt, axis=0)  # (dimG,)
        avg_F = jnp.mean(F_abs, axis=0)        # (dimF,)
    elif precond_type == 'max':
        avg_G = jnp.max(G_diag_sqrt, axis=0)
        avg_F = jnp.max(F_abs, axis=0)
    elif precond_type == 'min':
        avg_G = jnp.min(G_diag_sqrt, axis=0)
        avg_F = jnp.min(F_abs, axis=0)
    else:
        avg_G = jnp.ones(dimG)
        avg_F = jnp.ones(dimF)

    # Compute inverse preconditioners using double-where for safe autodiff at zero.
    # Forward semantics:
    #   precond_inv_G = avg_G / dimG       if avg_G > 0, else 1 / dimG
    #   precond_inv_F = sqrt(avg_F)/sqrt(dimF) if avg_F > 0, else 1 / sqrt(dimF)
    positive_G = avg_G > 0
    safe_avg_G = jnp.where(positive_G, avg_G, 1.0)
    precond_inv_G = jnp.where(positive_G, safe_avg_G / dimG, 1.0 / dimG)

    positive_F = avg_F > 0
    safe_avg_F = jnp.where(positive_F, avg_F, 1.0)
    precond_inv_F = jnp.where(
        positive_F,
        jnp.sqrt(safe_avg_F) / jnp.sqrt(dimF),
        1.0 / jnp.sqrt(dimF),
    )

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

        log.info('solving trivial problem to see if MKL exists')
        
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
# High-level differentiable solve function
# =============================================================================

import numpy as np
import diffcp
from jax.experimental import sparse as jsparse


# =============================================================================
# SDP solver failure tracking
# =============================================================================
# diffcp/Clarabel can return non-optimal status (Failure, MaxIterations, ...) on
# numerically tough problems; the existing fallback in `_solve_bwd` returns
# zero gradients silently, which surfaces in the trainer as raw_grad_norm=0
# — indistinguishable from a real stationary point. Track counts here so a
# stuck SGD run is easy to diagnose post-hoc, and log loudly each time the
# fallback fires.

_SOLVE_STATS = {
    'fwd_total': 0,
    'fwd_failures': 0,
    'fwd_exceptions': 0,
    'bwd_zero_grads': 0,
}

# Cone names a healthy Clarabel solve might report.
_HEALTHY_STATUSES = ("Solved", "AlmostSolved", "Solved/Inaccurate")


def _log_solve_status(prefix, status, info):
    """Log Clarabel forward status + track failures.

    Skips the per-call ``||x||, ||y||, ||s||`` norm computation that earlier
    versions emitted at INFO; those three np.linalg.norm calls run on every
    solve (~few thousand iters per training run). Norms are still computed
    if the logger is at DEBUG.
    """
    _SOLVE_STATS['fwd_total'] += 1
    log.info(f"[{prefix}] Solver status: {status}")
    if "iter" in info:
        log.info(f"[{prefix}] Iterations: {info['iter']}")
    if status not in _HEALTHY_STATUSES:
        _SOLVE_STATS['fwd_failures'] += 1
        log.warning(
            f"[{prefix}] SDP not solved optimally — status={status!r}. "
            f"Cumulative forward failures: {_SOLVE_STATS['fwd_failures']}/"
            f"{_SOLVE_STATS['fwd_total']} solves."
        )


def _record_bwd_zero_grad(prefix):
    """Note that the backward path is returning zeros (forward solve failed)."""
    _SOLVE_STATS['bwd_zero_grads'] += 1
    log.warning(
        f"[{prefix}] adjoint cache invalid — returning zero SDP gradient. "
        f"This iteration's raw_grad_norm reflects only the non-SDP part of the "
        f"loss (will appear ≈0 if the SDP is the only contributor). "
        f"Cumulative bwd zero-grad fallbacks: {_SOLVE_STATS['bwd_zero_grads']}."
    )


def get_solver_stats():
    """Return a snapshot of cumulative SDP solver health counters."""
    return dict(_SOLVE_STATS)


def reset_solver_stats():
    """Zero the SDP solver health counters (call between K's, etc.)."""
    for k in _SOLVE_STATS:
        _SOLVE_STATS[k] = 0


def _maybe_log_norms(prefix, x, y, s):
    """Log ||x||, ||y||, ||s|| only if the logger is at DEBUG."""
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"[{prefix}] ||x||={np.linalg.norm(x):.6e} "
            f"||y||={np.linalg.norm(y):.6e} ||s||={np.linalg.norm(s):.6e}"
        )


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
    
    tol_gap_abs=1e-5
    tol_gap_rel=1e-5
    tol_feas=1e-5
    reduced_tol_gap_abs=5e-5
    reduced_tol_gap_rel=5e-5
    reduced_tol_feas=1e-4

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
                # Use solve_and_derivative_internal to get solver status
                result = diffcp.solve_and_derivative_internal(
                    A_csc, b_arr, c_arr,
                    static_data.diffcp_cone_dict,
                    solve_method='CLARABEL',
                    direct_solve_method=get_direct_solve_method(),
                    verbose=False,
                    tol_gap_abs=tol_gap_abs,
                    tol_gap_rel=tol_gap_rel,
                    tol_feas=tol_feas,
                    reduced_tol_gap_abs=reduced_tol_gap_abs,
                    reduced_tol_gap_rel=reduced_tol_gap_rel,
                    reduced_tol_feas=reduced_tol_feas,
                )
                x = result["x"]
                y = result["y"]
                s = result["s"]
                adjoint_deriv = result["DT"]
                status = result["info"]["status"]

                obj = c_arr @ x
                _log_solve_status("Forward", status, result["info"])
                _maybe_log_norms("Forward", x, y, s)

                _adjoint_cache['adjoint'] = adjoint_deriv
                _adjoint_cache['valid'] = True
            except Exception as e:
                _SOLVE_STATS['fwd_exceptions'] += 1
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
            log.info('solve_impl starting (inside pure_callback)...')
            # Explicitly convert to numpy arrays
            A_arr = np.asarray(A_np)
            b_arr = np.asarray(b_np)
            c_arr = np.asarray(c_np)
            A_csc = spa.csc_matrix(A_arr)
            try:
                # Use solve_and_derivative_internal to get solver status
                result = diffcp.solve_and_derivative_internal(
                    A_csc, b_arr, c_arr,
                    static_data.diffcp_cone_dict,
                    solve_method='CLARABEL',
                    direct_solve_method=get_direct_solve_method(),
                    verbose=False,
                    tol_gap_abs=tol_gap_abs,
                    tol_gap_rel=tol_gap_rel,
                    tol_feas=tol_feas,
                    reduced_tol_gap_abs=reduced_tol_gap_abs,
                    reduced_tol_gap_rel=reduced_tol_gap_rel,
                    reduced_tol_feas=reduced_tol_feas,
                )
                x = result["x"]
                y = result["y"]
                s = result["s"]
                adjoint_deriv = result["DT"]
                status = result["info"]["status"]

                obj = c_arr @ x
                _log_solve_status("VJP Forward", status, result["info"])
                _maybe_log_norms("VJP Forward", x, y, s)

                _adjoint_cache['adjoint'] = adjoint_deriv
                _adjoint_cache['x'] = x
                _adjoint_cache['y'] = y
                _adjoint_cache['s'] = s
                _adjoint_cache['valid'] = True
            except Exception as e:
                _SOLVE_STATS['fwd_exceptions'] += 1
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
                _record_bwd_zero_grad("VJP Backward")
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


def scs_solve_wrapper_sparse(static_data, A_data, A_indices, A_shape, b, c):
    """Differentiable SDP solve with sparse A passed as (data, indices) triplets.

    Mirrors ``scs_solve_wrapper`` semantics but the constraint matrix arrives
    pre-sparsified — no host-side ``scipy.sparse.csc_matrix(dense)`` scan,
    and the dense (m, n) cotangent for ``A`` is replaced by a flat data
    cotangent of length ``A_data.shape[0]``. Indices are integer-valued and
    not differentiable; we pass them through as-is.

    Args:
        static_data: SCSSolveData with cone info (Python).
        A_data:    (nse,) JAX float64 array.
        A_indices: (nse, 2) JAX int array — (row, col) per nonzero.
        A_shape:   (m, n) Python tuple — static.
        b, c:      JAX float64 arrays.

    Returns:
        Optimal objective value (scalar, differentiable).
    """
    _adjoint_cache = {}

    tol_gap_abs = 1e-5
    tol_gap_rel = 1e-5
    tol_feas = 1e-5
    reduced_tol_gap_abs = 5e-5
    reduced_tol_gap_rel = 5e-5
    reduced_tol_feas = 1e-4

    m_rows, n_cols = A_shape

    def _build_csc_from_triplets(A_data_np, A_indices_np):
        # JAX BCOO.fromdense pads to static nse with out-of-bound (m, 0)
        # markers; filter those out before scipy.csc_matrix construction.
        idx = np.asarray(A_indices_np)
        rows = idx[:, 0]
        cols = idx[:, 1]
        valid = (rows < m_rows) & (cols < n_cols)
        return spa.csc_matrix(
            (
                np.asarray(A_data_np)[valid],
                (rows[valid].astype(np.int32), cols[valid].astype(np.int32)),
            ),
            shape=A_shape,
        )

    @jax.custom_vjp
    def _solve(A_data, A_indices, b, c):
        def solve_impl(A_data_np, A_idx_np, b_np, c_np):
            b_arr = np.asarray(b_np)
            c_arr = np.asarray(c_np)
            A_csc = _build_csc_from_triplets(A_data_np, np.asarray(A_idx_np))
            try:
                result = diffcp.solve_and_derivative_internal(
                    A_csc, b_arr, c_arr,
                    static_data.diffcp_cone_dict,
                    solve_method='CLARABEL',
                    direct_solve_method=get_direct_solve_method(),
                    verbose=False,
                    tol_gap_abs=tol_gap_abs,
                    tol_gap_rel=tol_gap_rel,
                    tol_feas=tol_feas,
                    reduced_tol_gap_abs=reduced_tol_gap_abs,
                    reduced_tol_gap_rel=reduced_tol_gap_rel,
                    reduced_tol_feas=reduced_tol_feas,
                )
                x = result["x"]
                obj = c_arr @ x
                _log_solve_status("SparseFwd", result["info"]["status"], result["info"])
                if log.isEnabledFor(logging.DEBUG):
                    _maybe_log_norms("SparseFwd", x, result["y"], result["s"])
                _adjoint_cache['adjoint'] = result["DT"]
                _adjoint_cache['valid'] = True
            except Exception as e:
                _SOLVE_STATS['fwd_exceptions'] += 1
                log.warning(f"diffcp solve failed: {e}")
                obj = np.nan
                _adjoint_cache['valid'] = False
            return np.array([obj])

        obj = jax.pure_callback(
            solve_impl,
            jax.ShapeDtypeStruct((1,), jnp.float64),
            A_data, A_indices, b, c,
        )
        return obj[0]

    def _solve_fwd(A_data, A_indices, b, c):
        def solve_impl(A_data_np, A_idx_np, b_np, c_np):
            b_arr = np.asarray(b_np)
            c_arr = np.asarray(c_np)
            A_csc = _build_csc_from_triplets(A_data_np, np.asarray(A_idx_np))
            try:
                result = diffcp.solve_and_derivative_internal(
                    A_csc, b_arr, c_arr,
                    static_data.diffcp_cone_dict,
                    solve_method='CLARABEL',
                    direct_solve_method=get_direct_solve_method(),
                    verbose=False,
                    tol_gap_abs=tol_gap_abs,
                    tol_gap_rel=tol_gap_rel,
                    tol_feas=tol_feas,
                    reduced_tol_gap_abs=reduced_tol_gap_abs,
                    reduced_tol_gap_rel=reduced_tol_gap_rel,
                    reduced_tol_feas=reduced_tol_feas,
                )
                x = result["x"]; y = result["y"]; s = result["s"]
                obj = c_arr @ x
                _log_solve_status("SparseVJPFwd", result["info"]["status"], result["info"])
                _maybe_log_norms("SparseVJPFwd", x, y, s)
                _adjoint_cache['adjoint'] = result["DT"]
                _adjoint_cache['x'] = x
                _adjoint_cache['y'] = y
                _adjoint_cache['s'] = s
                _adjoint_cache['valid'] = True
            except Exception as e:
                _SOLVE_STATS['fwd_exceptions'] += 1
                log.warning(f"diffcp solve failed in fwd: {e}")
                x = np.zeros(c_arr.shape[0])
                y = np.zeros(b_arr.shape[0])
                s = np.zeros(b_arr.shape[0])
                obj = np.nan
                _adjoint_cache['valid'] = False
            return (np.array([obj]), x, y, s)

        result_shapes = (
            jax.ShapeDtypeStruct((1,), jnp.float64),
            jax.ShapeDtypeStruct((c.shape[0],), jnp.float64),
            jax.ShapeDtypeStruct((b.shape[0],), jnp.float64),
            jax.ShapeDtypeStruct((b.shape[0],), jnp.float64),
        )
        obj, x, y, s = jax.pure_callback(
            solve_impl, result_shapes, A_data, A_indices, b, c
        )
        # residuals: keep A_data + A_indices for backward dA extraction
        return obj[0], (A_data, A_indices, b, c, x)

    def _solve_bwd(res, g):
        A_data, A_indices, b, c, x = res
        d_obj = g
        b_dim = b.shape[0]

        def compute_grads(A_data_np, A_idx_np, c_np, x_np, d_obj_np):
            A_data_arr = np.asarray(A_data_np)
            A_idx_arr = np.asarray(A_idx_np)
            c_arr = np.asarray(c_np)
            x_arr = np.asarray(x_np)
            d_obj_arr = np.asarray(d_obj_np)

            if not _adjoint_cache.get('valid', False):
                _record_bwd_zero_grad("SparseVJPBwd")
                return (
                    np.zeros_like(A_data_arr),
                    np.zeros(b_dim, dtype=np.float64),
                    np.zeros_like(c_arr),
                )

            adjoint_deriv = _adjoint_cache['adjoint']
            y_cached = _adjoint_cache['y']
            s_cached = _adjoint_cache['s']

            dx = d_obj_arr * c_arr
            dy = np.zeros_like(y_cached)
            ds = np.zeros_like(s_cached)
            dA_sol, db_sol, dc_sol = adjoint_deriv(dx, dy, ds)

            dA_csr = dA_sol.tocsr()
            rows = A_idx_arr[:, 0]
            cols = A_idx_arr[:, 1]
            valid = (rows < m_rows) & (cols < n_cols)
            dA_data_aligned = np.zeros_like(A_data_arr)
            if valid.any():
                dA_data_aligned[valid] = np.asarray(
                    dA_csr[rows[valid].astype(np.int64), cols[valid].astype(np.int64)]
                ).reshape(-1)

            dc_direct = d_obj_arr * x_arr
            return dA_data_aligned, db_sol, dc_sol + dc_direct

        result_shapes = (
            jax.ShapeDtypeStruct(A_data.shape, jnp.float64),
            jax.ShapeDtypeStruct(b.shape, jnp.float64),
            jax.ShapeDtypeStruct(c.shape, jnp.float64),
        )
        dA_data, db, dc = jax.pure_callback(
            compute_grads, result_shapes,
            A_data, A_indices, c, x, jnp.array(d_obj),
        )
        _adjoint_cache.clear()
        return dA_data, jnp.zeros_like(A_indices), db, dc

    _solve.defvjp(_solve_fwd, _solve_bwd)

    return _solve(A_data, A_indices, b, c)


# =============================================================================
# Helper Function for PSD Constraint C and d Construction
# =============================================================================

def compute_C_d_matrices(PSD_A_vals, PSD_b_vals, PSD_mat_dims):
    """
    Compute C_symvecs and d_vecs from PSD constraint data using SCS vectorization.

    This function iterates over the triangular elements of each H matrix and:
    - For each element (r,c), extracts PSD_A_vals[m_psd][r, c] (an S_mat x S_mat matrix)
    - Vectorizes it using jax_scs_symm_vectorize (handles SCS lower-triangle ordering)
    - Applies factor of 2 for off-diagonal elements of H

    Args:
        PSD_A_vals: List of (dim, dim, S_mat, S_mat) arrays - PSD constraint LHS coefficients
        PSD_b_vals: List of (dim, dim, V) arrays - PSD constraint RHS coefficients
        PSD_mat_dims: (M_psd,) - Dimensions of each H matrix

    Returns:
        C_symvecs: List of (S_vec, H_vec) matrices for PSD constraint modification
        d_vecs: List of (V, H_vec) matrices for equality constraint modification
        H_vec_dims: List of H vectorized dimensions (Python ints)
        PSD_mat_dims_list: List of PSD matrix dimensions (Python ints)
    """
    M_psd = len(PSD_A_vals)

    C_symvecs_list = []
    d_vecs_list = []
    H_vec_dims = []
    PSD_mat_dims_list = []

    for m_psd in range(M_psd):
        dim = int(PSD_mat_dims[m_psd])  # Convert to concrete int
        H_vec = dim * (dim + 1) // 2
        H_vec_dims.append(H_vec)
        PSD_mat_dims_list.append(dim)

        # Use NUMPY (not jnp) triu_indices here because `compute_C_d_matrices`
        # may be called inside an outer jax.jit (e.g. ldro-pep loss). Under jit,
        # jnp.triu_indices returns traced arrays, which break the Python
        # `for r, c in zip(...): if r == c:` loop below (needs concrete ints
        # and Python booleans). Since `dim` is a concrete Python int (the
        # trainer closes over PSD_shapes as a static tuple), np.triu_indices
        # yields concrete numpy int arrays — safe for Python control flow AND
        # for JAX array indexing `PSD_A_vals[m_psd][r, c]` (static indices).
        # The rest of the function's values stay as JAX traced arrays.
        # Note: `get_scs_lower_tri_indices` (jnp-based) is left unchanged and
        # still used by `jax_scs_symm_vectorize`, whose callers pass traced
        # indices directly into array-indexing ops — those work fine under jit.
        np_rows_upper, np_cols_upper = np.triu_indices(dim)
        rows, cols = np_cols_upper, np_rows_upper  # flip to column-major lower (same as get_scs_lower_tri_indices)

        curr_C_list = []
        curr_d_list = []

        # Iterate over triangular elements of H matrix
        # For each (r, c), extract the corresponding (S_mat, S_mat) matrix and vectorize it
        for r, c in zip(rows, cols):
            if r == c:
                # Diagonal element of H: factor of 1
                curr_C_list.append(jax_scs_symm_vectorize(PSD_A_vals[m_psd][r, c], jnp.sqrt(2.0)))
                curr_d_list.append(PSD_b_vals[m_psd][r, c])
            else:
                # Off-diagonal element of H: factor of 2
                curr_C_list.append(2.0 * jax_scs_symm_vectorize(PSD_A_vals[m_psd][r, c], jnp.sqrt(2.0)))
                curr_d_list.append(2.0 * PSD_b_vals[m_psd][r, c])

        # Stack into matrices: each column corresponds to one H element
        C_symvecs_list.append(jnp.stack(curr_C_list, axis=0).T)  # (S_vec, H_vec)
        d_vecs_list.append(jnp.stack(curr_d_list, axis=0).T)      # (V, H_vec)

    return C_symvecs_list, d_vecs_list, H_vec_dims, PSD_mat_dims_list


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
    # PSD constraint data
    PSD_A_vals=None,
    PSD_b_vals=None,
    PSD_mat_dims=None,
    C_symvecs=None,
    d_vecs=None,
    H_vec_dims=None,
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

    # Handle PSD constraint dimensions
    if PSD_A_vals is None or C_symvecs is None:
        M_psd = 0
        H_vec_sum = 0
        H_vec_dims_local = []
        PSD_mat_dims_local = []
    else:
        M_psd = len(C_symvecs)
        # Extract H_vec_dims from the shapes of C_symvecs (shape is (S_vec, H_vec))
        H_vec_dims_local = [int(C_symvecs[m].shape[1]) for m in range(M_psd)]
        H_vec_sum = sum(H_vec_dims_local)
        # PSD_mat_dims should already be a list of Python ints from compute_C_d_matrices
        PSD_mat_dims_local = PSD_mat_dims if PSD_mat_dims is not None else []

    # Decision variable dimension
    # x = [lambda, s_1..s_N, y_1..y_NM, Fz_1..Fz_NV, Gz_1..Gz_N*S_vec, H_1..H_N*H_vec_sum]
    x_dim = 1 + N * (1 + M + V + S_vec + H_vec_sum)

    # Index offsets
    lambd_idx = 0
    s_start = 1
    y_start = s_start + N
    Fz_start = y_start + N * M
    Gz_start = Fz_start + N * V
    H_start = Gz_start + N * S_vec

    # H indexing: H[m_psd][i][j] = H_start + sum(H_vec_dims[:m_psd])*N + i*H_vec_dims[m_psd] + j
    H_rel_offsets = [0]
    for m_psd_idx in range(M_psd):
        H_rel_offsets.append(H_rel_offsets[-1] + N * H_vec_dims_local[m_psd_idx])

    def H_idx(m_psd, i, j):
        return H_start + H_rel_offsets[m_psd] + i * H_vec_dims_local[m_psd] + j
    
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
    
    # 3. Equality constraints (N*V rows): -B^T y_i + Fz_i + sum_m_psd d_vecs[m_psd] @ H[m_psd][i] = -b_obj
    Bm_T = b_vals.T  # (V, M)
    eq_rows = jnp.zeros((N * V, x_dim))
    for i in range(N):
        eq_rows = eq_rows.at[i * V:(i + 1) * V, y_start + i * M:y_start + (i + 1) * M].set(-Bm_T)
        eq_rows = eq_rows.at[i * V:(i + 1) * V, Fz_start + i * V:Fz_start + (i + 1) * V].set(jnp.eye(V))

        # Add H contribution: sum_m_psd d_vecs[m_psd] @ H[m_psd][i]
        for m_psd in range(M_psd):
            H_start_idx = H_idx(m_psd, i, 0)
            H_end_idx = H_idx(m_psd, i, H_vec_dims_local[m_psd])
            eq_rows = eq_rows.at[i * V:(i + 1) * V, H_start_idx:H_end_idx].set(d_vecs[m_psd])

    eq_b = jnp.tile(-b_obj, N)
    
    # 4. PSD constraints (N*S_vec rows): -A^* y_i + Gz_i + sum_m_psd C_symvecs[m_psd] @ H[m_psd][i] << -A_obj
    Am_T = A_vals_svec.T  # (S_vec, M)
    psd_rows = jnp.zeros((N * S_vec, x_dim))
    for i in range(N):
        psd_rows = psd_rows.at[i * S_vec:(i + 1) * S_vec, y_start + i * M:y_start + (i + 1) * M].set(-Am_T)
        psd_rows = psd_rows.at[i * S_vec:(i + 1) * S_vec, Gz_start + i * S_vec:Gz_start + (i + 1) * S_vec].set(scaledI)

        # Add H contribution: sum_m_psd C_symvecs[m_psd] @ H[m_psd][i]
        for m_psd in range(M_psd):
            H_start_idx = H_idx(m_psd, i, 0)
            H_end_idx = H_idx(m_psd, i, H_vec_dims_local[m_psd])
            psd_rows = psd_rows.at[i * S_vec:(i + 1) * S_vec, H_start_idx:H_end_idx].set(C_symvecs[m_psd])

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

    # 6. H >> 0 constraints (N*H_vec_sum rows total)
    H_psd_rows_list = []
    for m_psd in range(M_psd):
        H_vec = H_vec_dims_local[m_psd]
        # Derive H_mat from H_vec using H_vec = H_mat * (H_mat + 1) / 2
        # H_mat = (-1 + sqrt(1 + 8*H_vec)) / 2
        H_mat = int((-1 + math.sqrt(1 + 8*H_vec)) / 2)
        scaledI_H = jax_scs_scaled_off_triangles(jnp.ones((H_mat, H_mat)), jnp.sqrt(2.0))

        for i in range(N):
            H_psd_row = jnp.zeros((H_vec, x_dim))
            H_start_idx = H_idx(m_psd, i, 0)
            H_end_idx = H_idx(m_psd, i, H_vec)
            H_psd_row = H_psd_row.at[:, H_start_idx:H_end_idx].set(-scaledI_H)
            H_psd_rows_list.append(H_psd_row)

    H_psd_rows = jnp.vstack(H_psd_rows_list) if H_psd_rows_list else jnp.zeros((0, x_dim))
    H_psd_b = jnp.zeros(sum([N * H_vec_dims_local[m] for m in range(M_psd)]))

    # =========================================================================
    # Combine all blocks
    # SCS cone ordering: zero (z), nonneg (l), SOC (q), PSD (s)
    # =========================================================================
    A_dense = jnp.vstack([
        eq_rows,       # N*V rows (zero cone / equality)
        epi_rows,      # N rows (nonneg)
        y_nonneg,      # N*M rows (nonneg)
        socp_rows,     # N*socp_dim rows (SOC)
        psd_rows,      # N*S_vec rows (PSD for Gz)
        H_psd_rows,    # Sum_m (N*H_vec_dims[m]) rows (PSD for H)
    ])

    b = jnp.concatenate([
        eq_b,
        epi_b,
        y_nonneg_b,
        socp_b,
        psd_b,
        H_psd_b,
    ])
    
    # Cone info for SCS
    # SCS uses: z (zero), l (nonneg), q (SOC array), s (PSD array)
    cone_info = {
        'z': N * V,           # Equality constraints (first)
        'l': N + N * M,       # Epigraph + y >= 0 (second)
        'q': [socp_dim] * N,  # SOC cones (third)
        's': [S_mat] * N if PSD_A_vals is None else [S_mat] * N + sum([[PSD_mat_dims_local[m]] * N for m in range(M_psd)], []),
        'N': N,
        'M': M,
        'V': V,
        'S_mat': S_mat,
        'S_vec': S_vec,
        'M_psd': M_psd,
        'H_vec_dims': H_vec_dims_local,
    }

    return A_dense, b, c_obj, x_dim, cone_info


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
    # PSD constraint data
    PSD_A_vals=None,
    PSD_b_vals=None,
    PSD_mat_dims=None,
    C_symvecs=None,
    d_vecs=None,
    H_vec_dims=None,
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

    # Handle PSD constraint dimensions
    if PSD_A_vals is None or C_symvecs is None:
        M_psd = 0
        H_vec_sum = 0
        H_vec_dims_local = []
        PSD_mat_dims_local = []
    else:
        M_psd = len(C_symvecs)
        # Extract H_vec_dims from the shapes of C_symvecs (shape is (S_vec, H_vec))
        H_vec_dims_local = [int(C_symvecs[m].shape[1]) for m in range(M_psd)]
        H_vec_sum = sum(H_vec_dims_local)
        # PSD_mat_dims should already be a list of Python ints from compute_C_d_matrices
        PSD_mat_dims_local = PSD_mat_dims if PSD_mat_dims is not None else []

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
    # x = [lambda, t, s_1..s_N, y1_1..y1_NM, y2_1..y2_NM, Fz1_1..Fz1_NV, Fz2_1..Fz2_NV, Gz1_1..Gz1_NS, Gz2_1..Gz2_NS, H1_1..H1_NS, H2_1..H2_NS]
    x_dim = 2 + N * (1 + 2 * (M + V + S_vec + H_vec_sum))

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
    H1_start = Gz2_start + N * S_vec
    H2_start = H1_start + N * H_vec_sum

    # H indexing: H1[m_psd][i][j] and H2[m_psd][i][j]
    H1_rel_offsets = [0]
    H2_rel_offsets = [0]
    for m_psd_idx in range(M_psd):
        H1_rel_offsets.append(H1_rel_offsets[-1] + N * H_vec_dims_local[m_psd_idx])
        H2_rel_offsets.append(H2_rel_offsets[-1] + N * H_vec_dims_local[m_psd_idx])

    def H1_idx(m_psd, i, j):
        return H1_start + H1_rel_offsets[m_psd] + i * H_vec_dims_local[m_psd] + j

    def H2_idx(m_psd, i, j):
        return H2_start + H2_rel_offsets[m_psd] + i * H_vec_dims_local[m_psd] + j
    
    # Build objective
    c_obj = jnp.zeros(x_dim)
    c_obj = c_obj.at[lambd_idx].set(eps)
    c_obj = c_obj.at[s_start:s_start + N].set(1.0 / N)
    
    Am_T = A_vals_svec.T  # (S_vec, M)
    
    # =========================================================================
    # Build constraint blocks (in SCS order: zero, nonneg, soc, psd)
    # =========================================================================
    
    # --- ZERO CONE: Equality constraints (-B^T y1 + Fz1 + sum_m_psd d_vecs[m_psd] @ H1[m_psd][i] = -b_obj)
    eq1_rows = jnp.zeros((N * V, x_dim))
    for i in range(N):
        eq1_rows = eq1_rows.at[i * V:(i + 1) * V, y1_start + i * M:y1_start + (i + 1) * M].set(-Bm_T)
        eq1_rows = eq1_rows.at[i * V:(i + 1) * V, Fz1_start + i * V:Fz1_start + (i + 1) * V].set(jnp.eye(V))

        # Add H1 contribution
        for m_psd in range(M_psd):
            H1_start_idx = H1_idx(m_psd, i, 0)
            H1_end_idx = H1_idx(m_psd, i, H_vec_dims_local[m_psd])
            eq1_rows = eq1_rows.at[i * V:(i + 1) * V, H1_start_idx:H1_end_idx].set(d_vecs[m_psd])

    eq1_b = jnp.tile(-b_obj, N)

    eq2_rows = jnp.zeros((N * V, x_dim))
    for i in range(N):
        eq2_rows = eq2_rows.at[i * V:(i + 1) * V, y2_start + i * M:y2_start + (i + 1) * M].set(-Bm_T)
        eq2_rows = eq2_rows.at[i * V:(i + 1) * V, Fz2_start + i * V:Fz2_start + (i + 1) * V].set(jnp.eye(V))

        # Add H2 contribution
        for m_psd in range(M_psd):
            H2_start_idx = H2_idx(m_psd, i, 0)
            H2_end_idx = H2_idx(m_psd, i, H_vec_dims_local[m_psd])
            eq2_rows = eq2_rows.at[i * V:(i + 1) * V, H2_start_idx:H2_end_idx].set(d_vecs[m_psd])

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

        # Add H1 contribution
        for m_psd in range(M_psd):
            H1_start_idx = H1_idx(m_psd, i, 0)
            H1_end_idx = H1_idx(m_psd, i, H_vec_dims_local[m_psd])
            psd1_rows = psd1_rows.at[i * S_vec:(i + 1) * S_vec, H1_start_idx:H1_end_idx].set(C_symvecs[m_psd])

    psd1_b = jnp.tile(-A_obj_svec, N)

    psd2_rows = jnp.zeros((N * S_vec, x_dim))
    for i in range(N):
        psd2_rows = psd2_rows.at[i * S_vec:(i + 1) * S_vec, y2_start + i * M:y2_start + (i + 1) * M].set(-Am_T)
        psd2_rows = psd2_rows.at[i * S_vec:(i + 1) * S_vec, Gz2_start + i * S_vec:Gz2_start + (i + 1) * S_vec].set(scaledI)

        # Add H2 contribution
        for m_psd in range(M_psd):
            H2_start_idx = H2_idx(m_psd, i, 0)
            H2_end_idx = H2_idx(m_psd, i, H_vec_dims_local[m_psd])
            psd2_rows = psd2_rows.at[i * S_vec:(i + 1) * S_vec, H2_start_idx:H2_end_idx].set(C_symvecs[m_psd])

    psd2_b = jnp.tile(-alpha_inv * A_obj_svec, N)

    # H1 >> 0 and H2 >> 0 constraints
    H1_psd_rows_list = []
    H2_psd_rows_list = []

    for m_psd in range(M_psd):
        H_vec = H_vec_dims_local[m_psd]
        # Derive H_mat from H_vec using H_vec = H_mat * (H_mat + 1) / 2
        # H_mat = (-1 + sqrt(1 + 8*H_vec)) / 2
        H_mat = int((-1 + math.sqrt(1 + 8*H_vec)) / 2)
        scaledI_H = jax_scs_scaled_off_triangles(jnp.ones((H_mat, H_mat)), jnp.sqrt(2.0))

        for i in range(N):
            # H1
            H1_row = jnp.zeros((H_vec, x_dim))
            H1_row = H1_row.at[:, H1_idx(m_psd, i, 0):H1_idx(m_psd, i, H_vec)].set(-scaledI_H)
            H1_psd_rows_list.append(H1_row)

            # H2
            H2_row = jnp.zeros((H_vec, x_dim))
            H2_row = H2_row.at[:, H2_idx(m_psd, i, 0):H2_idx(m_psd, i, H_vec)].set(-scaledI_H)
            H2_psd_rows_list.append(H2_row)

    H1_psd_rows = jnp.vstack(H1_psd_rows_list) if H1_psd_rows_list else jnp.zeros((0, x_dim))
    H2_psd_rows = jnp.vstack(H2_psd_rows_list) if H2_psd_rows_list else jnp.zeros((0, x_dim))
    H_psd_b = jnp.zeros(2 * sum([N * H_vec_dims_local[m] for m in range(M_psd)]))

    # Combine all blocks (SCS order: zero, nonneg, soc, psd)
    A_dense = jnp.vstack([
        eq1_rows, eq2_rows,  # zero cone (2*N*V)
        epi1_rows, epi2_rows, y1_nonneg, y2_nonneg,  # nonneg cone (2*N + 2*N*M)
        socp1_rows, socp2_rows,  # soc cones (2*N*socp_dim)
        psd1_rows, psd2_rows,  # psd cones (2*N*S_vec for Gz)
        H1_psd_rows, H2_psd_rows,  # psd cones (2*sum(N*H_vec_dims[m]) for H)
    ])

    b = jnp.concatenate([
        eq1_b, eq2_b,
        epi1_b, epi2_b, y1_nonneg_b, y2_nonneg_b,
        socp1_b, socp2_b,
        psd1_b, psd2_b,
        H_psd_b,
    ])
    
    # Cone info for SCS
    # SCS uses: z (zero), l (nonneg), q (SOC array), s (PSD array)
    cone_info = {
        'z': 2 * N * V,              # Equality constraints
        'l': 2 * N + 2 * N * M,      # Epigraph + y1 >= 0 + y2 >= 0
        'q': [socp_dim] * (2 * N),   # SOC cones (doubled)
        's': [S_mat] * (2 * N) if PSD_A_vals is None else [S_mat] * (2 * N) + sum([[PSD_mat_dims_local[m]] * (2*N) for m in range(M_psd)], []),
        'N': N,
        'M': M,
        'V': V,
        'S_mat': S_mat,
        'S_vec': S_vec,
        'M_psd': M_psd,
        'H_vec_dims': H_vec_dims_local,
    }

    return A_dense, b, c_obj, x_dim, cone_info


def jax_scs_canonicalize_wc_pep(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # PSD block data (optional)
    PSD_A_vals=None, PSD_b_vals=None, PSD_mat_dims=None,
    C_symvecs=None, d_vecs=None, H_vec_dims=None,
):
    """
    Pure JAX canonicalization of Worst-Case PEP to SCS form.

    Without PSD blocks, minimizes -c^T y subject to:
      1. -B^T y = -b_obj  (Equality / Zero Cone)
      2. y >= 0           (Non-negative Cone)
      3. A^* y >= A_obj   (PSD Cone)

    With PSD blocks (e.g., CP operator-norm constraints), adds a per-block
    auxiliary matrix H_m (vectorized length H_vec_m) that couples into the
    equality and main PSD constraints, plus H_m >> 0:
      1'. -B^T y + sum_m d_vecs[m] @ H[m] = -b_obj
      3'. -A^* y + sum_m C_symvecs[m] @ H[m] + s = -A_obj,   s >> 0 (main PSD)
      4.  -scaledI_H[m] @ H[m] + s_H[m] = 0, s_H[m] >> 0 (H PSD cones)

    Backward compatibility: PSD_A_vals=None bypasses all H-related code,
    producing the same A/b/c/cone_info as the pre-PSD implementation.
    """
    # Get dimensions
    M = A_vals.shape[0]        # Dimension of variable y
    S_mat = A_obj.shape[0]     # Size of main PSD matrix
    V = b_obj.shape[0]         # Number of equality constraints
    S_vec = S_mat * (S_mat + 1) // 2

    # Handle PSD constraint dimensions
    if PSD_A_vals is None or C_symvecs is None:
        M_psd = 0
        H_vec_dims_local = []
        H_vec_sum = 0
        PSD_mat_dims_local = []
    else:
        M_psd = len(C_symvecs)
        H_vec_dims_local = list(H_vec_dims) if H_vec_dims is not None \
            else [int(C_symvecs[m].shape[1]) for m in range(M_psd)]
        H_vec_sum = sum(H_vec_dims_local)
        PSD_mat_dims_local = list(PSD_mat_dims) if PSD_mat_dims is not None else []

    # Decision variable: x = [y (M), H_1 (H_vec_1), ..., H_M_psd (H_vec_M_psd)]
    x_dim = M + H_vec_sum
    H_start = M
    H_rel_offsets = [0]
    for m in range(M_psd):
        H_rel_offsets.append(H_rel_offsets[-1] + H_vec_dims_local[m])

    def H_idx(m_psd, j):
        return H_start + H_rel_offsets[m_psd] + j

    # Objective: minimize -c_vals^T y (H components have zero coefficient)
    c_out = jnp.zeros(x_dim).at[:M].set(-c_vals)

    # Precompute SCS-format symmetric vectorizations
    A_obj_svec = jax_scs_symm_vectorize(A_obj, jnp.sqrt(2.0))                              # (S_vec,)
    A_vals_svec = jax.vmap(lambda A: jax_scs_symm_vectorize(A, jnp.sqrt(2.0)))(A_vals)     # (M, S_vec)

    # 1. Zero cone (V rows): -B^T y + sum_m d_vecs[m] @ H[m] = -b_obj
    Bm_T = b_vals.T  # (V, M)
    eq_rows = jnp.zeros((V, x_dim)).at[:, :M].set(-Bm_T)
    for m in range(M_psd):
        h0, h1 = H_idx(m, 0), H_idx(m, H_vec_dims_local[m])
        eq_rows = eq_rows.at[:, h0:h1].set(d_vecs[m])
    eq_b = -b_obj

    # 2. Non-negative cone (M rows): y >= 0
    nonneg_rows = jnp.zeros((M, x_dim)).at[:, :M].set(-jnp.eye(M))
    nonneg_b = jnp.zeros(M)

    # 3. Main PSD cone (S_vec rows): -A^* y + sum_m C_symvecs[m] @ H[m] + s = -A_obj
    Am_T = A_vals_svec.T  # (S_vec, M)
    psd_rows = jnp.zeros((S_vec, x_dim)).at[:, :M].set(-Am_T)
    for m in range(M_psd):
        h0, h1 = H_idx(m, 0), H_idx(m, H_vec_dims_local[m])
        psd_rows = psd_rows.at[:, h0:h1].set(C_symvecs[m])
    psd_b = -A_obj_svec

    # 4. H PSD cones (H_vec_m rows each): -scaledI_H @ H[m] + s = 0, s >> 0
    H_psd_rows_list = []
    for m in range(M_psd):
        H_vec = H_vec_dims_local[m]
        H_mat = int((-1 + math.sqrt(1 + 8 * H_vec)) / 2)
        scaledI_H = jax_scs_scaled_off_triangles(jnp.ones((H_mat, H_mat)), jnp.sqrt(2.0))
        block = jnp.zeros((H_vec, x_dim))
        h0, h1 = H_idx(m, 0), H_idx(m, H_vec)
        block = block.at[:, h0:h1].set(-scaledI_H)
        H_psd_rows_list.append(block)
    H_psd_rows = jnp.vstack(H_psd_rows_list) if H_psd_rows_list else jnp.zeros((0, x_dim))
    H_psd_b = jnp.zeros(H_vec_sum)

    # Combine: SCS cone ordering z -> l -> s (no SOC in WC-PEP)
    A_dense = jnp.vstack([eq_rows, nonneg_rows, psd_rows, H_psd_rows])
    b = jnp.concatenate([eq_b, nonneg_b, psd_b, H_psd_b])

    cone_info = {
        'z': V,
        'l': M,
        'q': [],
        's': [S_mat] + PSD_mat_dims_local,
        'M': M,
        'V': V,
        'S_mat': S_mat,
        'S_vec': S_vec,
    }

    return A_dense, b, c_out, x_dim, cone_info


# =============================================================================
# Hoisted DRO expectation: splits the pipeline into a Python-static setup, a
# fully JIT-compatible "build SDP inputs" step, and the un-cached SDP solve.
# Lets callers (e.g. ``unified_trainer._build_ldro_pep_loss``) keep the SDP
# pure_callback OUTSIDE any all-encompassing jit so the JIT-cacheable parts
# (canonicalization, vmap'd trajectory, BCOO sparsification) hit the JAX
# persistent compilation cache across processes.
# =============================================================================

def dro_expectation_setup_static(
    N, M, V, S_mat, psd_mat_dims_static, h_vec_dims_static,
):
    """Compute SCSSolveData, A_shape, nse upper bound from STATIC ints.

    All inputs are Python ints (no JAX arrays). Call once at training-build
    time; the returned SCSSolveData is closed over by the SDP-solve wrapper.

    Args:
        N: minibatch size.
        M, V, S_mat: PEP dimensions (from probing pep_data_fn at build).
        psd_mat_dims_static: tuple of Python ints — H matrix dims per PSD block.
        h_vec_dims_static: list of Python ints — H_vec = dim*(dim+1)/2 per block.

    Returns:
        (static_data: SCSSolveData,
         A_shape: (m, n) tuple,
         nse_upper_bound: int  — generous structural-nnz bound for BCOO.fromdense)
    """
    S_vec = S_mat * (S_mat + 1) // 2
    socp_dim = 1 + V + S_vec
    M_psd = len(psd_mat_dims_static)
    sum_H = int(sum(h_vec_dims_static))

    cone_info = {
        'z': N * V,
        'l': N + N * M,
        'q': [socp_dim] * N,
        's': (
            [S_mat] * N
            if M_psd == 0
            else [S_mat] * N
            + sum([[psd_mat_dims_static[m]] * N for m in range(M_psd)], [])
        ),
    }

    m_total = (
        cone_info['z']
        + cone_info['l']
        + sum(cone_info['q'])
        + sum([s * (s + 1) // 2 for s in cone_info['s']])
    )
    x_dim = 1 + N * (1 + M + V + S_vec + sum_H)
    A_shape = (m_total, x_dim)

    static_data = SCSSolveData(cone_info, A_shape)

    nse_upper_bound = (
        N * V * (M + 1 + sum_H)
        + N * (1 + M + V + S_vec)
        + N * M
        + N * (1 + V + S_vec)
        + N * S_vec * (M + 1 + sum_H)
        + N * sum_H
    )
    return static_data, A_shape, int(nse_upper_bound)


def dro_expectation_canon_to_bcoo(
    A_obj, b_obj, A_vals, b_vals, c_vals,
    G_batch, F_batch,
    eps, precond_inv,
    PSD_A_vals, PSD_b_vals,
    psd_mat_dims_static,         # static Python tuple
    nse_upper_bound,             # static Python int
):
    """JIT-compatible: canonicalize then sparsify to BCOO triplets.

    No SDP solve here — ends at the JAX↔host boundary so the caller can
    invoke ``scs_solve_wrapper_sparse`` *outside* any enclosing jit.

    Returns:
        (A_data, A_indices, b, c) — all JAX arrays, ready for the SDP wrapper.
    """
    if PSD_A_vals is not None and len(PSD_A_vals) > 0:
        C_symvecs, d_vecs, _, _ = compute_C_d_matrices(
            PSD_A_vals, PSD_b_vals, psd_mat_dims_static
        )
    else:
        C_symvecs = None
        d_vecs = None
    PSD_mat_dims_to_pass = (
        list(psd_mat_dims_static) if PSD_A_vals is not None and len(PSD_A_vals) > 0 else None
    )
    A_dense, b, c, _, _ = jax_scs_canonicalize_dro_expectation(
        A_obj, b_obj, A_vals, b_vals, c_vals,
        G_batch, F_batch,
        eps, precond_inv,
        PSD_A_vals, PSD_b_vals, PSD_mat_dims_to_pass,
        C_symvecs, d_vecs, None,
    )
    A_bcoo = jsparse.BCOO.fromdense(A_dense, nse=int(nse_upper_bound))
    return A_bcoo.data, A_bcoo.indices, b, c


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
    # PSD constraint data
    PSD_A_vals=None,
    PSD_b_vals=None,
    PSD_c_vals=None,
    PSD_mat_dims=None,
):
    """
    Full differentiable DRO expectation solve using SCS + diffcp.

    Args:
        A_obj, b_obj, A_vals, b_vals, c_vals: PEP constraint data
        G_batch: (N, S_mat, S_mat) Gram matrices
        F_batch: (N, V) function values
        eps: Wasserstein radius
        precond_inv: (G_precond_inv, F_precond_inv) preconditioner
        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_mat_dims: PSD constraint data

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

    # Handle PSD constraint dimensions
    if PSD_A_vals is not None:
        M_psd = len(PSD_A_vals)
        log.info(f"[DRO Expectation] PSD constraints present: M_psd={M_psd}")
        for m in range(M_psd):
            log.info(f"  PSD[{m}]: shape={PSD_A_vals[m].shape}, mat_dim={PSD_mat_dims[m]}")
        C_symvecs, d_vecs, H_vec_dims, PSD_mat_dims_list = compute_C_d_matrices(PSD_A_vals, PSD_b_vals, PSD_mat_dims)
        H_vec_sum = sum(H_vec_dims)
    else:
        M_psd = 0
        H_vec_sum = 0
        H_vec_dims = []
        PSD_mat_dims_list = []
        C_symvecs = None
        d_vecs = None
        log.info("[DRO Expectation] No PSD constraints")

    # Cone info computed from static dimensions
    cone_info = {
        'z': N * V,           # Equality constraints
        'l': N + N * M,       # Epigraph + y >= 0
        'q': [socp_dim] * N,  # SOC cones
        's': [S_mat] * N if PSD_A_vals is None else [S_mat] * N + sum([[PSD_mat_dims_list[m]] * N for m in range(M_psd)], []),
    }

    # Calculate A shape from cone dimensions
    m = cone_info['z'] + cone_info['l'] + sum(cone_info['q']) + sum([s*(s+1)//2 for s in cone_info['s']])
    x_dim = 1 + N * (1 + M + V + S_vec + H_vec_sum)
    A_shape = (m, x_dim)

    # Log cone structure
    log.info(f"[DRO Expectation] Cone structure: z={cone_info['z']}, l={cone_info['l']}, "
             f"q={len(cone_info['q'])} cones, s={len(cone_info['s'])} PSD cones")
    log.info(f"[DRO Expectation] PSD cone dimensions: {cone_info['s']}")
    log.info(f"[DRO Expectation] Problem size: A.shape={A_shape}, x_dim={x_dim}")

    # Create static data for cones BEFORE canonicalization
    static_data = SCSSolveData(cone_info, A_shape)

    # Canonicalize to SCS form (this can be traced)
    # Pass PSD_mat_dims_list (Python ints) instead of PSD_mat_dims (JAX array)
    PSD_mat_dims_to_pass = PSD_mat_dims_list if PSD_A_vals is not None else None
    A_dense, b, c, _, _ = jax_scs_canonicalize_dro_expectation(
        A_obj, b_obj, A_vals, b_vals, c_vals,
        G_batch, F_batch,
        eps, precond_inv,
        PSD_A_vals, PSD_b_vals, PSD_mat_dims_to_pass,
        C_symvecs, d_vecs, None,  # H_vec_dims computed from C_symvecs inside
    )

    log.info('canon done')

    # Sparsify A at the JAX↔host boundary so the callback can build CSC
    # directly from triplets (skipping the O(m·n) scipy.csc_matrix scan)
    # and the backward returns a flat data cotangent of length nse rather
    # than a dense (m, n) matrix.
    sum_H = H_vec_sum
    nse_upper_bound = (
        N * V * (M + 1 + sum_H)        # eq_rows: -Bm_T + I_V + d_vecs
        + N * (1 + M + V + S_vec)      # epi_rows
        + N * M                        # y_nonneg
        + N * (1 + V + S_vec)          # socp_rows: -1 + diag(F_pre^2) + diag(scaledG)
        + N * S_vec * (M + 1 + sum_H)  # psd_rows: -Am_T + diag(scaledI) + C_symvecs
        + N * sum_H                    # H_psd_rows: diag(scaledI_H)
    )
    A_bcoo = jsparse.BCOO.fromdense(A_dense, nse=int(nse_upper_bound))

    obj_val = scs_solve_wrapper_sparse(
        static_data, A_bcoo.data, A_bcoo.indices, A_shape, b, c
    )

    return obj_val


def dro_cvar_scs_solve(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps, alpha, precond_inv,
    # PSD constraint data
    PSD_A_vals=None,
    PSD_b_vals=None,
    PSD_c_vals=None,
    PSD_mat_dims=None,
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
        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_mat_dims: PSD constraint data

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

    # Handle PSD constraint dimensions
    if PSD_A_vals is not None:
        M_psd = len(PSD_A_vals)
        log.info(f"[DRO CVaR] PSD constraints present: M_psd={M_psd}")
        for m in range(M_psd):
            log.info(f"  PSD[{m}]: shape={PSD_A_vals[m].shape}, mat_dim={PSD_mat_dims[m]}")
        C_symvecs, d_vecs, H_vec_dims, PSD_mat_dims_list = compute_C_d_matrices(PSD_A_vals, PSD_b_vals, PSD_mat_dims)
        H_vec_sum = sum(H_vec_dims)
    else:
        M_psd = 0
        H_vec_sum = 0
        H_vec_dims = []
        PSD_mat_dims_list = []
        C_symvecs = None
        d_vecs = None
        log.info("[DRO CVaR] No PSD constraints")

    # Cone info computed from static dimensions (doubled for CVaR)
    cone_info = {
        'z': 2 * N * V,              # Equality constraints (doubled)
        'l': 2 * N + 2 * N * M,      # Epigraph + y1 >= 0 + y2 >= 0
        'q': [socp_dim] * (2 * N),   # SOC cones (doubled)
        's': [S_mat] * (2 * N) if PSD_A_vals is None else [S_mat] * (2 * N) + sum([[PSD_mat_dims_list[m]] * (2*N) for m in range(M_psd)], []),
    }

    # Calculate A shape from cone dimensions
    m = cone_info['z'] + cone_info['l'] + sum(cone_info['q']) + sum([s*(s+1)//2 for s in cone_info['s']])
    x_dim = 2 + N * (1 + 2 * (M + V + S_vec + H_vec_sum))
    A_shape = (m, x_dim)

    # Log cone structure
    log.info(f"[DRO CVaR] Cone structure: z={cone_info['z']}, l={cone_info['l']}, "
             f"q={len(cone_info['q'])} cones, s={len(cone_info['s'])} PSD cones")
    log.info(f"[DRO CVaR] PSD cone dimensions: {cone_info['s']}")
    log.info(f"[DRO CVaR] Problem size: A.shape={A_shape}, x_dim={x_dim}")

    # Create static data for cones BEFORE canonicalization
    static_data = SCSSolveData(cone_info, A_shape)

    # Canonicalize to SCS form (this can be traced)
    # Pass PSD_mat_dims_list (Python ints) instead of PSD_mat_dims (JAX array)
    PSD_mat_dims_to_pass = PSD_mat_dims_list if PSD_A_vals is not None else None
    A_dense, b, c, _, _ = jax_scs_canonicalize_dro_cvar(
        A_obj, b_obj, A_vals, b_vals, c_vals,
        G_batch, F_batch,
        eps, alpha, precond_inv,
        PSD_A_vals, PSD_b_vals, PSD_mat_dims_to_pass,
        C_symvecs, d_vecs, None,  # H_vec_dims computed from C_symvecs inside
    )

    # log.info('canon done')
    
    # Solve with custom VJP
    obj_val = scs_solve_wrapper(static_data, A_dense, b, c)
    
    return obj_val


def wc_pep_scs_solve(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # PSD block data (optional; required for CP/PDLP PEPs)
    PSD_A_vals=None,
    PSD_b_vals=None,
    PSD_c_vals=None,
    PSD_mat_dims=None,
):
    """
    Full differentiable WC-PEP solve using SCS + diffcp.

    Args:
        A_obj, b_obj, A_vals, b_vals, c_vals: PEP scalar constraint data
        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_mat_dims: PSD block data
            (per-block coefficient tensors). When None or empty, behaves
            identically to the pre-PSD implementation — Quad / Lasso /
            LogReg / Huber / ISTA / FISTA all have empty PSD lists and
            take the fast path.

    Returns:
        obj_val: Optimal WC-PEP objective value (scalar, differentiable)
    """
    # Static dimensions
    M = A_vals.shape[0]
    S_mat = A_obj.shape[0]
    V = b_obj.shape[0]

    # Handle PSD constraint dimensions
    if PSD_A_vals is not None and len(PSD_A_vals) > 0:
        M_psd = len(PSD_A_vals)
        log.info(f"[WC PEP] PSD constraints present: M_psd={M_psd}")
        for m in range(M_psd):
            log.info(f"  PSD[{m}]: shape={PSD_A_vals[m].shape}, mat_dim={PSD_mat_dims[m]}")
        C_symvecs, d_vecs, H_vec_dims, PSD_mat_dims_list = compute_C_d_matrices(
            PSD_A_vals, PSD_b_vals, PSD_mat_dims
        )
        H_vec_sum = sum(H_vec_dims)
    else:
        M_psd = 0
        H_vec_sum = 0
        H_vec_dims = []
        PSD_mat_dims_list = []
        C_symvecs = None
        d_vecs = None
        log.info("[WC PEP] No PSD constraints")

    # Cone structure: z -> l -> s = [S_mat] + H block sizes
    cone_info = {
        'z': V,
        'l': M,
        'q': [],
        's': [S_mat] + PSD_mat_dims_list,
    }

    m_rows = cone_info['z'] + cone_info['l'] + sum([s * (s + 1) // 2 for s in cone_info['s']])
    x_dim = M + H_vec_sum
    A_shape = (m_rows, x_dim)

    log.info(f"[WC PEP] Cone structure: z={cone_info['z']}, l={cone_info['l']}, "
             f"s={cone_info['s']}")
    log.info(f"[WC PEP] Problem size: A.shape={A_shape}, x_dim={x_dim}")

    static_data = SCSSolveData(cone_info, A_shape)

    PSD_mat_dims_to_pass = PSD_mat_dims_list if M_psd else None
    A_dense, b, c, _, _ = jax_scs_canonicalize_wc_pep(
        A_obj, b_obj, A_vals, b_vals, c_vals,
        PSD_A_vals, PSD_b_vals, PSD_mat_dims_to_pass,
        C_symvecs, d_vecs, H_vec_dims,
    )

    obj_val = scs_solve_wrapper(static_data, A_dense, b, c)

    return obj_val


def dro_scs_solve(
    # PEP data
    A_obj, b_obj, A_vals, b_vals, c_vals,
    # Sample data
    G_batch, F_batch,
    # Parameters
    eps,
    # Precomputed preconditioner (compute once before SGD starts)
    precond_inv,
    # Risk measure selection
    risk_type='expectation',
    alpha=0.1,
    # PSD constraint data
    PSD_A_vals=None,
    PSD_b_vals=None,
    PSD_c_vals=None,
    PSD_mat_dims=None,
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
        precond_inv: (G_precond_inv, F_precond_inv) precomputed preconditioner
        risk_type: 'expectation' or 'cvar'
        alpha: CVaR confidence level (only used when risk_type='cvar')
        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_mat_dims: PSD constraint data

    Returns:
        obj_val: Optimal DRO objective value (scalar, differentiable)
    """
    if risk_type == 'expectation':
        return dro_expectation_scs_solve(
            A_obj, b_obj, A_vals, b_vals, c_vals,
            G_batch, F_batch,
            eps, precond_inv,
            PSD_A_vals=PSD_A_vals,
            PSD_b_vals=PSD_b_vals,
            PSD_c_vals=PSD_c_vals,
            PSD_mat_dims=PSD_mat_dims,
        )
    elif risk_type == 'cvar':
        return dro_cvar_scs_solve(
            A_obj, b_obj, A_vals, b_vals, c_vals,
            G_batch, F_batch,
            eps, alpha, precond_inv,
            PSD_A_vals=PSD_A_vals,
            PSD_b_vals=PSD_b_vals,
            PSD_c_vals=PSD_c_vals,
            PSD_mat_dims=PSD_mat_dims,
        )
    else:
        raise ValueError(f"Unknown risk_type: {risk_type}. Must be 'expectation' or 'cvar'.")
