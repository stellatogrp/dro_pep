"""
Interpolation-condition diagnostic for construct_chambolle_pock_pep_data on
standard-form LPs.

For a small random LP
        min c^T x   s.t.   A x = b,  x >= 0,
cast as the saddle
        min_x max_y  f1(x) + <K x, y> - h(y),
with f1(x) = c^T x + ind_{x>=0},  K = -A,  h(y) = -b^T y,
we:
  1. Solve the LP (scipy.optimize.linprog) to get the saddle (x_s, y_s).
  2. Run Chambolle-Pock for K_max steps on the LP with the prox operators
     induced by f1, h.
  3. Recover subgradients at each iterate (via Moreau identity on the prox
     step), operator applications K x_bar and K^T y, and the analysis-specific
     slots (K x_K, K^T y_K, K dx0).
  4. Build the Gram matrix G (dimG x dimG) and function-value vector F (dimF,)
     that correspond to this real trajectory in the basis of
     `chambolle_pock.py:54-86`.
  5. Evaluate every scalar and PSD constraint from
     `construct_chambolle_pock_pep_data` at (G, F) and report the maximum
     violation magnitude.

What to expect:
  - BEFORE Phase 2 fixes: the (s, j) / (i, s) convex-interpolation rows for
    f1 and h should be violated because the construction's default
    gs = 0 in `convex_interp` contradicts the non-zero saddle subgradients
    gf1_s = A^T y_s,  gh_s = -b  that the real trajectory carries.
  - AFTER the Phase 2a fix (gs= passed): the (s, j) / (i, s) rows should hold;
    the K / K^T PSD blocks may still carry saddle-pair violations because the
    vacuous (xs, 0) / (ys, 0) pairs at lines 193-194 are misrepresenting
    K x_s = -b and K^T y_s = A^T y_s.
  - AFTER the Phase 2b fix (saddle operator pairs replaced): everything holds.
"""

import pytest
import numpy as np
from scipy.optimize import linprog

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning.pep_constructions import construct_chambolle_pock_pep_data


# ---------------------------------------------------------------------------
# LP generation
# ---------------------------------------------------------------------------

def generate_lp(m, n, seed):
    """Generate a feasible, bounded standard-form LP with ||A||_op = 1.

    Returns (A, b, c, x_feas) where:
      A ∈ R^{m x n} is normalized to operator norm 1,
      b = A @ x_feas for a chosen interior x_feas > 0,
      c is drawn so the LP has a bounded optimum.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    # Normalize A so ||A||_op = 1 exactly — matches L_M = 1 in the PEP.
    A = A / np.linalg.svd(A, compute_uv=False)[0]

    # Feasible interior point.
    x_feas = rng.uniform(0.5, 1.5, size=n)
    b = A @ x_feas

    # Objective. Pick c with a well-defined optimum by taking
    # c = A^T lam + s  with s >= 0, lam arbitrary — this is a dual feasible
    # solution and guarantees complementary slackness for some primal-dual
    # optimal pair.
    lam = rng.standard_normal(m)
    s = rng.uniform(0.1, 1.0, size=n)
    c = A.T @ lam + s

    return A, b, c, x_feas


def solve_lp(A, b, c):
    """Solve standard-form LP; return (x_s, y_s) primal-dual optimum.

    Uses scipy.optimize.linprog with HiGHS (default). Returns the primal
    x_s ∈ R^n and the dual y_s ∈ R^m corresponding to the equality constraints
    A x = b (with the sign convention that A^T y_s - c ∈ N_{>=0}(x_s)).
    """
    res = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * len(c),
                  method='highs')
    assert res.success, f"LP solve failed: {res.message}"
    x_s = res.x
    # scipy's eqlin.marginals for min c^T x s.t. Ax=b, x>=0 equals the dual
    # y_s in the convention max y^T b s.t. A^T y <= c. Verified empirically:
    # (i) marginals @ b == c @ x_s  (strong duality),
    # (ii) A^T marginals - c <= 0 componentwise, with equality on the support of x_s
    # (complementary slackness). So no sign flip.
    y_s = np.asarray(res.eqlin.marginals)
    return x_s, y_s


# ---------------------------------------------------------------------------
# Chambolle-Pock trajectory on the LP
# ---------------------------------------------------------------------------

def run_cp_on_lp(A, b, c, x0, y0, tau, sigma, theta, K_max):
    """Run CP for K_max steps; return iterate & auxiliary arrays.

    CP template used here:
      f1(x) = c^T x + ind_{x>=0},   K = -A,   h(y) = -b^T y.
    Prox:
      prox_{tau f1}(v) = max(v - tau c, 0)
      prox_{sigma h}(w) = w + sigma b
    """
    xs_iters = [x0.copy()]
    ys_iters = [y0.copy()]

    # gf1 at x_0: for x_0 > 0 (interior), ∂f1(x_0) = {c}. Choose c.
    gf1_iters = [c.copy()]
    # gh at y_0: ∂h(y_0) = {-b} for all y_0 (h is linear).
    gh_iters = [-b.copy()]

    # w_k = K^T y_k = -A^T y_k   for k = 0 .. K_max-1
    w_iters = []
    # z_{k+1} = K x_bar_k = -A x_bar_k   for k = 0 .. K_max-1 (stored at idx k+1)
    z_iters = [None] * (K_max + 1)  # z_iters[k] holds z_k for k = 1..K_max

    x_curr = x0
    y_curr = y0
    for k in range(K_max):
        # w_k = K^T y_k = -A^T y_k
        w_k = -(A.T @ y_curr)
        w_iters.append(w_k)

        # Primal step: x_{k+1} = prox_{tau f1}(x_curr - tau w_k)
        v_prox_primal = x_curr - tau * w_k
        x_new = np.maximum(v_prox_primal - tau * c, 0.0)

        # gf1_{k+1} from Moreau: tau * gf1_{k+1} = (x_curr - tau*w_k) - x_new - tau*??
        # Actually prox identity: prox_{tau f}(v) = argmin ... so
        #   v - prox_{tau f}(v) ∈ tau * ∂f(prox_{tau f}(v)).
        # For f = c^T x + ind, ∂f(x) = c + N_{>=0}(x), so we pick the subgradient
        # implied by the prox step:  gf1_{k+1} = (v_prox_primal - x_new) / tau.
        gf1_new = (v_prox_primal - x_new) / tau
        gf1_iters.append(gf1_new)

        # Extrapolation
        x_bar = x_new + theta * (x_new - x_curr)

        # z_{k+1} = K x_bar = -A x_bar
        z_kp1 = -(A @ x_bar)
        z_iters[k + 1] = z_kp1

        # Dual step: y_{k+1} = prox_{sigma h}(y_curr + sigma z_{k+1})
        v_prox_dual = y_curr + sigma * z_kp1
        y_new = v_prox_dual + sigma * b

        gh_new = -b.copy()  # ∂h = {-b} (verified also via Moreau below)
        gh_iters.append(gh_new)

        x_curr = x_new
        y_curr = y_new
        xs_iters.append(x_curr.copy())
        ys_iters.append(y_curr.copy())

    xs_iters = np.array(xs_iters)       # shape (K+1, n)
    ys_iters = np.array(ys_iters)       # shape (K+1, m)
    gf1_iters = np.array(gf1_iters)     # shape (K+1, n)
    gh_iters = np.array(gh_iters)       # shape (K+1, m)
    w_iters = np.array(w_iters)         # shape (K, n)
    # z_iters is indexed 1..K; drop slot 0 (None) and re-index.
    z_stack = np.array(z_iters[1:])     # shape (K, m)

    return xs_iters, ys_iters, gf1_iters, gh_iters, w_iters, z_stack


# ---------------------------------------------------------------------------
# Gram / F construction
# ---------------------------------------------------------------------------

def build_gram_and_F(A, b, c, xs, ys, xs_iters, ys_iters, gf1_iters, gh_iters,
                     w_iters, z_iters, tau, sigma, theta, K_max):
    """Build G (dimG x dimG) and F (dimF,) matching chambolle_pock.py's basis.

    Embedding: x-space (R^n) and y-space (R^m) stacked into R^(n+m).
      x-space basis vectors live in the first n coords.
      y-space basis vectors live in the last m coords.
    """
    m, n = A.shape

    # --- Index layout (mirrors chambolle_pock.py:65-81) -----------------
    dimG = 4 + 2 * (K_max + 2) + 2 * K_max + 3
    dimF1 = K_max + 2
    dimF_h = K_max + 2
    dimF = dimF1 + dimF_h

    idx_dx0 = 0
    idx_dy0 = 1
    idx_xs = 2
    idx_ys = 3
    idx_gf1_start = 4                   # gf1_0 .. gf1_K, gf1_s  (K+2 slots)
    idx_gh_start = idx_gf1_start + (K_max + 2)  # gh_0 .. gh_K, gh_s  (K+2)
    idx_w_start = idx_gh_start + (K_max + 2)    # w_0 .. w_{K-1}      (K)
    idx_z_start = idx_w_start + K_max           # z_1 .. z_K          (K)
    idx_K_xK = idx_z_start + K_max
    idx_Kt_yK = idx_K_xK + 1
    idx_K_dx0 = idx_Kt_yK + 1
    assert idx_K_dx0 == dimG - 1

    # --- Basis vectors in the stacked (R^n, R^m) space ------------------
    D = n + m  # ambient dim of the Gram-embedding space
    basis = np.zeros((dimG, D))

    def emb_x(vec):
        out = np.zeros(D); out[:n] = vec; return out

    def emb_y(vec):
        out = np.zeros(D); out[n:] = vec; return out

    x0 = xs_iters[0]
    y0 = ys_iters[0]

    # Shifted positions
    basis[idx_dx0] = emb_x(x0 - xs)
    basis[idx_dy0] = emb_y(y0 - ys)
    basis[idx_xs] = emb_x(xs)
    basis[idx_ys] = emb_y(ys)

    # f1 subgradients: gf1_0, gf1_1, ..., gf1_K at indices 4..4+K
    # gf1_s at index 4+K+1 = idx_gf1_start + (K+1)
    for k in range(K_max + 1):
        basis[idx_gf1_start + k] = emb_x(gf1_iters[k])
    gf1_s = A.T @ ys  # = -K^T y_s under K = -A
    basis[idx_gf1_start + (K_max + 1)] = emb_x(gf1_s)

    # h subgradients: gh_0, ..., gh_K at indices idx_gh_start..idx_gh_start+K
    # gh_s at index idx_gh_start + (K+1)
    for k in range(K_max + 1):
        basis[idx_gh_start + k] = emb_y(gh_iters[k])
    gh_s = -b  # = K x_s under K = -A and A x_s = b
    basis[idx_gh_start + (K_max + 1)] = emb_y(gh_s)

    # w_k = K^T y_k for k = 0..K-1
    for k in range(K_max):
        basis[idx_w_start + k] = emb_x(w_iters[k])

    # z_{k+1} = K x_bar_k for k = 0..K-1, stored at idx_z_start + k
    for k in range(K_max):
        basis[idx_z_start + k] = emb_y(z_iters[k])

    # Analysis slots
    K_xK = -(A @ xs_iters[K_max])
    Kt_yK = -(A.T @ ys_iters[K_max])
    K_dx0 = -(A @ (x0 - xs))
    basis[idx_K_xK] = emb_y(K_xK)
    basis[idx_Kt_yK] = emb_x(Kt_yK)
    basis[idx_K_dx0] = emb_y(K_dx0)

    # Gram
    G = basis @ basis.T  # (dimG, dimG)

    # --- F: absolute function values, with saddle slot = 0 (pinned) -----
    # f1 absolute values, shifted so f1(x_s) = 0  (i.e., use f1'(x) = c^T (x - x_s))
    F_f1 = np.zeros(dimF1)
    for k in range(K_max + 1):
        F_f1[k] = c @ (xs_iters[k] - xs)  # shifted-value convention
    F_f1[K_max + 1] = 0.0  # saddle, pinned

    # h absolute values, shifted so h(y_s) = 0  (h(y) = -b^T y)
    F_h = np.zeros(dimF_h)
    for k in range(K_max + 1):
        F_h[k] = -b @ (ys_iters[k] - ys)
    F_h[K_max + 1] = 0.0

    F = np.concatenate([F_f1, F_h])
    return G, F, dimG, dimF


# ---------------------------------------------------------------------------
# Constraint evaluation
# ---------------------------------------------------------------------------

def eval_scalar_constraint(A_vals_i, b_vals_i, c_vals_i, G, F):
    return float(np.trace(A_vals_i @ G) + b_vals_i @ F + c_vals_i)


def eval_psd_block(A_psd, b_psd, c_psd, G, F):
    """Reconstruct the H(G, F) matrix for a PSD block."""
    size_H = A_psd.shape[0]
    H = c_psd.copy().astype(float)
    for row in range(size_H):
        for col in range(size_H):
            H[row, col] = (np.trace(A_psd[row, col] @ G)
                           + b_psd[row, col] @ F
                           + c_psd[row, col])
    return H


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

CANONICAL_TAU = 0.5
CANONICAL_SIGMA = 0.5
CANONICAL_THETA = 1.0
CANONICAL_L_M = 1.0   # A is normalized to operator norm 1
CANONICAL_R = 1.0     # not directly enforced here — we just need a finite trajectory


@pytest.mark.parametrize('K_max', [1, 3])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_lp_trajectory_satisfies_cp_interpolation(K_max, seed):
    """Real LP trajectory should satisfy every CP PEP constraint.

    Reports violation magnitudes by category so that Phase 2 decisions can be
    made from the output.
    """
    m, n = 3, 5
    A, b, c, x_feas = generate_lp(m=m, n=n, seed=seed)
    xs, ys = solve_lp(A, b, c)

    # Initial point: start strictly interior so gf1_0 = c is valid.
    x0 = x_feas + 0.1  # component-wise > 0
    y0 = np.zeros(m)   # any real y_0 is valid

    xs_iters, ys_iters, gf1_iters, gh_iters, w_iters, z_iters = run_cp_on_lp(
        A, b, c, x0, y0,
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        K_max=K_max,
    )

    G, F, dimG, dimF = build_gram_and_F(
        A, b, c, xs, ys, xs_iters, ys_iters, gf1_iters, gh_iters,
        w_iters, z_iters,
        CANONICAL_TAU, CANONICAL_SIGMA, CANONICAL_THETA, K_max,
    )

    # Build PEP data via our construction.
    pep_data = construct_chambolle_pock_pep_data(
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        M=CANONICAL_L_M, R=CANONICAL_R, K_max=K_max,
    )
    (A_obj, b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = [
        np.asarray(x) if not isinstance(x, list) else [np.asarray(a) for a in x]
        for x in pep_data
    ]

    # --- Scalar constraint violations, grouped for diagnosis --------------
    num_scalar = A_vals.shape[0]
    violations = np.zeros(num_scalar)
    for i in range(num_scalar):
        violations[i] = eval_scalar_constraint(
            A_vals[i], b_vals[i], c_vals[i], G, F,
        )

    # Group the scalar constraints.
    # Per chambolle_pock.py:
    #   [0 .. 2 * n_algo_points * (n_algo_points + 1) - 1]  = f1 + h interp
    #   next 4 rows = value pinning
    #   next block = adjoint consistency
    #   last row = Euclidean IC
    n_algo = K_max + 1
    n_interp_per_fn = n_algo * (n_algo + 1)  # convex_interp output count
    n_f1_interp = n_interp_per_fn
    n_h_interp = n_interp_per_fn
    n_value_pin = 4
    # Adjoint has a variable count — figure it out by subtraction.
    n_IC = 1

    f1_interp_viols = violations[:n_f1_interp]
    h_interp_viols = violations[n_f1_interp:n_f1_interp + n_h_interp]
    value_pin_viols = violations[n_f1_interp + n_h_interp
                                  :n_f1_interp + n_h_interp + n_value_pin]
    remaining = violations[n_f1_interp + n_h_interp + n_value_pin:]
    adj_viols = remaining[:-n_IC]
    IC_viol = remaining[-n_IC:]

    # --- PSD block min-eigenvalues ---------------------------------------
    psd_min_eigs = []
    for idx in range(len(PSD_A_vals)):
        H = eval_psd_block(PSD_A_vals[idx], PSD_b_vals[idx], PSD_c_vals[idx], G, F)
        eigvals = np.linalg.eigvalsh(H)
        psd_min_eigs.append(float(np.min(eigvals)))

    # --- Report ---
    print(f"\n=== LP interpolation diagnostic (K_max={K_max}, seed={seed}) ===")
    print(f"  dimG={dimG}, dimF={dimF}, num_scalar={num_scalar}")
    print(f"  f1 interp: max violation = {np.max(f1_interp_viols):.3e}")
    print(f"  h  interp: max violation = {np.max(h_interp_viols):.3e}")
    print(f"  value pinning: max violation = {np.max(value_pin_viols):.3e}")
    print(f"  adjoint: max violation = {np.max(adj_viols):.3e}")
    print(f"  IC: violation = {IC_viol[0]:.3e}")
    print(f"  PSD block min-eigs = {psd_min_eigs}")

    # Locate top-5 worst f1/h interp violations for clarity.
    if np.max(f1_interp_viols) > 1e-6:
        idx_sorted = np.argsort(-f1_interp_viols)[:5]
        print(f"  top-5 f1 interp viols: {[(int(i), float(f1_interp_viols[i])) for i in idx_sorted]}")
    if np.max(h_interp_viols) > 1e-6:
        idx_sorted = np.argsort(-h_interp_viols)[:5]
        print(f"  top-5 h  interp viols: {[(int(i), float(h_interp_viols[i])) for i in idx_sorted]}")

    # --- Assertions ---
    eps = 1e-6
    assert np.max(f1_interp_viols) <= eps, \
        f"f1 interpolation violated (max {np.max(f1_interp_viols):.3e})"
    assert np.max(h_interp_viols) <= eps, \
        f"h interpolation violated (max {np.max(h_interp_viols):.3e})"
    assert np.max(value_pin_viols) <= eps, \
        f"value pinning violated (max {np.max(value_pin_viols):.3e})"
    assert np.max(adj_viols) <= eps, \
        f"adjoint violated (max {np.max(adj_viols):.3e})"
    # IC is a scalar radius constraint; a trajectory can freely exceed R^2.
    # We only report its value, not assert.
    for idx, me in enumerate(psd_min_eigs):
        assert me >= -eps, f"PSD block {idx} not PSD, min eig = {me:.3e}"


@pytest.mark.parametrize('K_max', [1, 3])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_lp_trajectory_objective_matches_gap(K_max, seed):
    """trace(A_obj @ G) + b_obj @ F should equal the real duality gap at x_K, y_K.

    The duality gap in shifted-function-value convention (f1(x_s) = h(y_s) = 0)
    reduces to L(x_K, y_s) - L(x_s, y_K) = f1(x_K) - h(y_K)
    (where the cross-terms evaluate via the substitutions built into the
    construction at lines 279-300).
    """
    m, n = 3, 5
    A, b, c, x_feas = generate_lp(m=m, n=n, seed=seed)
    xs, ys = solve_lp(A, b, c)

    x0 = x_feas + 0.1
    y0 = np.zeros(m)
    xs_iters, ys_iters, gf1_iters, gh_iters, w_iters, z_iters = run_cp_on_lp(
        A, b, c, x0, y0,
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        K_max=K_max,
    )

    G, F, _, _ = build_gram_and_F(
        A, b, c, xs, ys, xs_iters, ys_iters, gf1_iters, gh_iters,
        w_iters, z_iters,
        CANONICAL_TAU, CANONICAL_SIGMA, CANONICAL_THETA, K_max,
    )
    pep_data = construct_chambolle_pock_pep_data(
        tau=CANONICAL_TAU, sigma=CANONICAL_SIGMA, theta=CANONICAL_THETA,
        M=CANONICAL_L_M, R=CANONICAL_R, K_max=K_max,
    )
    A_obj = np.asarray(pep_data[0])
    b_obj = np.asarray(pep_data[1])

    pep_obj_value = float(np.trace(A_obj @ G) + b_obj @ F)

    # Real gap via the actual Lagrangian.
    x_K = xs_iters[K_max]
    y_K = ys_iters[K_max]

    def L(x, y):
        # L(x, y) = c^T x + y^T (b - A x)  (with x >= 0 implicit)
        return float(c @ x + y @ (b - A @ x))

    real_gap = L(x_K, ys) - L(xs, y_K)

    print(f"\n=== Objective match (K_max={K_max}, seed={seed}) ===")
    print(f"  PEP A_obj @ G + b_obj @ F = {pep_obj_value:.6e}")
    print(f"  Real gap L(x_K, y_s) - L(x_s, y_K) = {real_gap:.6e}")

    assert np.isclose(pep_obj_value, real_gap, rtol=1e-6, atol=1e-9), \
        f"Objective mismatch: PEP {pep_obj_value} vs real {real_gap}"
