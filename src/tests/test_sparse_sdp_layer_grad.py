"""Gradient checks for ``scs_solve_wrapper_sparse`` on tiny cone programs.

The wrapper returns the optimal value p*(A, b, c) of
    min c^T x  s.t.  A x + s = b,  s in K        (SCS form)
so its exact gradient is the envelope formula
    dp*/dA_ij = y_i x_j,   dp*/db = -y,   dp*/dc = x.
For each tiny problem three gradients are compared on the A_data / b slots:
the wrapper's custom VJP, the envelope formula at the solver's (x, y), and
central finite differences of tightly-solved optimal values.

Usage (from src/):
    pytest tests/test_sparse_sdp_layer_grad.py -v
"""
import diffcp
import diffcp_patch  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as spa
from jax.experimental import sparse as jsparse

jax.config.update("jax_enable_x64", True)

from learning.jax_scs_layer import SCSSolveData, scs_solve_wrapper_sparse  # noqa: E402

TIGHT = dict(tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11,
             reduced_tol_gap_abs=1e-9, reduced_tol_gap_rel=1e-9, reduced_tol_feas=1e-9)


def solve_tight(A, b, c, cone_dict):
    """(x, y, s, p*) solved to tight tolerance (forward only)."""
    r = diffcp.solve_and_derivative_internal(
        spa.csc_matrix(A), b, c, cone_dict, solve_method="CLARABEL", verbose=False, **TIGHT)
    return r["x"], r["y"], r["s"], float(c @ r["x"])


def three_gradients(A, b, c, cone_info, h=1e-6, n_dirs=4, pad=7, seed=0, value_grad="diffcp"):
    """Return dict of directional derivatives along random (dA, db) directions.

    Directions are supported on A's nonzero pattern (the only entries the
    wrapper can differentiate). Keys: 'vjp', 'env', 'fd' -> arrays (n_dirs,).
    """
    A = np.asarray(A, float)
    static = SCSSolveData(cone_info, A.shape)
    bcoo = jsparse.BCOO.fromdense(jnp.array(A), nse=int(np.count_nonzero(A)) + pad)
    A_data, A_idx = bcoo.data, bcoo.indices
    valid = np.asarray((A_idx[:, 0] < A.shape[0]) & (A_idx[:, 1] < A.shape[1]))
    rows, cols = np.asarray(A_idx[:, 0])[valid], np.asarray(A_idx[:, 1])[valid]

    f = lambda Ad, bb: scs_solve_wrapper_sparse(static, Ad, A_idx, A.shape, bb, jnp.array(c),
                                                value_grad=value_grad)
    gA, gb = jax.grad(f, argnums=(0, 1))(A_data, jnp.array(b))
    gA, gb = np.asarray(gA)[valid], np.asarray(gb)

    x, y, _, _ = solve_tight(A, b, c, static.diffcp_cone_dict)
    eA, eb = y[rows] * x[cols], -y

    rng = np.random.default_rng(seed)
    out = {"vjp": [], "env": [], "fd": []}
    for _ in range(n_dirs):
        dA = rng.standard_normal(rows.size)
        db = rng.standard_normal(b.size)
        nrm = np.sqrt(dA @ dA + db @ db)
        dA, db = dA / nrm, db / nrm
        D = np.zeros_like(A)
        D[rows, cols] = dA
        fp = solve_tight(A + h * D, b + h * db, c, static.diffcp_cone_dict)[3]
        fm = solve_tight(A - h * D, b - h * db, c, static.diffcp_cone_dict)[3]
        out["fd"].append((fp - fm) / (2 * h))
        out["vjp"].append(gA @ dA + gb @ db)
        out["env"].append(eA @ dA + eb @ db)
    return {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------- problems
def lp_simplex():
    """min c^T x s.t. 1^T x = 1, x >= 0 (unique vertex)."""
    n = 3
    A = np.vstack([np.ones((1, n)), -np.eye(n)])
    b = np.r_[1.0, np.zeros(n)]
    c = np.array([0.3, -0.7, 0.5])
    return A, b, c, {"z": 1, "l": n, "q": [], "s": []}


def lp_general():
    """Nondegenerate LP with inequality rows: min c^T x s.t. Gx <= h, x >= -1."""
    rng = np.random.default_rng(3)
    n, m = 3, 5
    G = rng.standard_normal((m, n))
    h = G @ rng.standard_normal(n) + rng.uniform(0.5, 1.5, m)
    A = np.vstack([G, -np.eye(n)])
    b = np.r_[h, np.ones(n)]
    c = rng.standard_normal(n)
    return A, b, c, {"z": 0, "l": m + n, "q": [], "s": []}


def soc_ball():
    """min c^T x s.t. ||x||_2 <= 1."""
    n = 3
    A = np.vstack([np.zeros((1, n)), -np.eye(n)])
    b = np.r_[1.0, np.zeros(n)]
    c = np.array([0.4, -1.0, 0.2])
    return A, b, c, {"z": 0, "l": 0, "q": [n + 1], "s": []}


def _svec_lower_colmajor(M):
    """SCS svec: lower triangle, column-major, off-diagonals times sqrt(2)."""
    n = M.shape[0]
    return np.array([M[i, j] * (1.0 if i == j else np.sqrt(2)) for j in range(n) for i in range(j, n)])


def sdp_small():
    """min <C, X> s.t. tr(X) = 1, <A2, X> = tr(A2)/3, X PSD (3x3); x = svec(X)."""
    rng = np.random.default_rng(5)
    B = rng.standard_normal((3, 3)); C = B + B.T
    B2 = rng.standard_normal((3, 3)); A2 = B2 + B2.T
    nv = 6
    A = np.vstack([_svec_lower_colmajor(np.eye(3)), _svec_lower_colmajor(A2), -np.eye(nv)])
    b = np.r_[1.0, np.trace(A2) / 3, np.zeros(nv)]
    c = _svec_lower_colmajor(C)
    return A, b, c, {"z": 2, "l": 0, "q": [], "s": [3]}


def sdp_blocks(dims=(4, 3, 5), seed=7):
    """Block-diagonal SDP with several PSD cones, sizes >= 4 included.

    min sum_k <C_k, X_k> s.t. tr(X_k) = 1 per block, one coupling equality,
    X_k PSD. Cones of size >= 4 exercise diffcp's Clarabel<->SCS PSD row
    permutation, which is not an involution there (the n <= 3 cases hide bugs).
    """
    rng = np.random.default_rng(seed)
    nv = sum(d * (d + 1) // 2 for d in dims)
    rows, b, cs, coupling, rhs, off = [], [], [], [], 0.0, 0
    for d in dims:
        B = rng.standard_normal((d, d)); cs.append(_svec_lower_colmajor(B + B.T))
        I = _svec_lower_colmajor(np.eye(d))
        r = np.zeros(nv); r[off:off + I.size] = I; rows.append(r); b.append(1.0); off += I.size
        B2 = rng.standard_normal((d, d)); coupling.append(_svec_lower_colmajor(B2 + B2.T))
        rhs += np.trace(B2 + B2.T) / d
    rows.append(np.concatenate(coupling)); b.append(rhs)
    A = np.vstack(rows + [-np.eye(nv)])
    return A, np.r_[b, np.zeros(nv)], np.concatenate(cs), {"z": len(dims) + 1, "l": 0, "q": [], "s": list(dims)}


PROBLEMS = {"lp_simplex": lp_simplex, "lp_general": lp_general,
            "soc_ball": soc_ball, "sdp_small": sdp_small, "sdp_blocks": sdp_blocks}


@pytest.mark.parametrize("dims", [(4,), (5,), (4, 3, 2), (8, 7, 8)])
def test_diffcp_clarabel_psd_solution_satisfies_kkt(dims):
    """diffcp's returned (x, y, s) must satisfy Ax + s = b and A^T y + c = 0."""
    A, b, c, cones = sdp_blocks(dims)
    x, y, s, _ = solve_tight(A, b, c, SCSSolveData(cones, A.shape).diffcp_cone_dict)
    assert np.linalg.norm(A @ x + s - b) < 1e-6
    assert np.linalg.norm(A.T @ y + c) < 1e-6


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_envelope_matches_fd(name):
    A, b, c, cones = PROBLEMS[name]()
    g = three_gradients(A, b, c, cones)
    np.testing.assert_allclose(g["env"], g["fd"], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("value_grad", ["diffcp", "envelope"])
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_vjp_matches_fd(name, value_grad):
    A, b, c, cones = PROBLEMS[name]()
    g = three_gradients(A, b, c, cones, value_grad=value_grad)
    # The wrapper solves at tol 1e-5, so its (x, y) carry ~1e-3 error here;
    # the formula itself is checked at tight tolerance in test_envelope_matches_fd.
    np.testing.assert_allclose(g["vjp"], g["fd"], rtol=5e-3, atol=2e-3)


if __name__ == "__main__":
    for name, fn in PROBLEMS.items():
        g = three_gradients(*fn())
        print(name, {k: np.round(v, 6) for k, v in g.items()})
