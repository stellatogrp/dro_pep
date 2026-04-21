"""
Interpolation-condition diagnostic for construct_chambolle_pock_pep_data on
capacitated facility-location LP instances.

Facility location is a more realistic PDLP-style LP than the standard-form
LP tested in test_chambolle_pock_interpolation.py:
  - Mixed equality (demand) + inequality (capacity + linking) constraints.
  - Box-constrained primal: 0 <= v <= 1 (both y-open and x-assign components).
  - Two-cone dual: y[:m1] >= 0 (inequalities), y[m1:] in R (equalities).
  - Nontrivial operator norm ||K_mat||_op > 1.

We map into the CP template
    L(v, y) = f1(v) + <K v, y> - h(y)
with
    f1(v) = c^T v + ind_{[0,1]}(v),
    K     = -K_mat       (K_mat = [-A_ineq; A_eq]),
    h(y)  = -q^T y + ind_{R^{m1}_+ x R^{m2}}(y)
    q     = [-b_ineq; b_eq].

Saddle-point stationarity (verified numerically in build_facility_lp_instance):
    gf1_s = -K^T y_s =  K_mat^T y_s   (valid subgrad of f1 at v_s)
    gh_s  =  K   v_s = -K_mat v_s     (valid subgrad of h  at y_s)

The test:
  1. Generates a facility-location instance and solves the LP for (v_s, y_s).
  2. Runs CP/PDHG for K_max steps, recovering subgradients via Moreau.
  3. Builds G, F in our CP PEP basis.
  4. Evaluates every scalar and PSD constraint and asserts they hold.
  5. Separately asserts the PEP objective equals the real Lagrangian gap.

As with test_chambolle_pock_interpolation.py, the P-norm IC is reported but
not asserted — the trajectory's IC value is an instance-dependent radius, not
an interpolation condition.
"""
import io
import contextlib

import pytest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning.facility_location_test import (
    generate_facility_location_problem,
    make_matrix_extractor,
    solve_relaxed_lp,
)
from learning.pep_constructions import construct_chambolle_pock_pep_data

# Reuse helpers from the simple-LP interpolation test.
from tests.test_chambolle_pock_interpolation import (
    eval_scalar_constraint,
    eval_psd_block,
)


# ---------------------------------------------------------------------------
# Facility-location instance setup
# ---------------------------------------------------------------------------

def build_facility_lp_instance(n_facilities, n_customers, seed):
    """Generate a facility-location LP and solve for the saddle (v_s, y_s).

    Returns a dict with:
      K_mat  (float64, (m1+m2, n_vars))  : stacked operator [-A_ineq; A_eq]
      q      (float64, (m1+m2,))         : stacked RHS      [-b_ineq;  b_eq]
      c      (float64, (n_vars,))        : objective coefficients
      l, u   (float64, (n_vars,))        : box bounds (zeros and ones here)
      v_s    (float64, (n_vars,))        : primal optimum
      y_s    (float64, (m1+m2,))         : dual optimum (y[:m1]>=0, y[m1:] free)
      m1, m2 (ints)                      : # inequality / equality rows
    """
    problem = generate_facility_location_problem(
        n_facilities=n_facilities, random_seed=seed, n_customers=n_customers,
    )
    extractor = make_matrix_extractor(n_facilities, n_customers)
    matrices = extractor(
        problem['fixed_costs'], problem['capacities'],
        problem['demands'], problem['transportation_costs'],
    )

    # Suppress the scripty prints inside solve_relaxed_lp.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sol = solve_relaxed_lp(matrices, n_facilities, n_customers)

    # Cast to float64 for JAX x64.
    A_ineq = np.asarray(matrices.A_ineq, dtype=np.float64)
    b_ineq = np.asarray(matrices.b_ineq, dtype=np.float64)
    A_eq = np.asarray(matrices.A_eq, dtype=np.float64)
    b_eq = np.asarray(matrices.b_eq, dtype=np.float64)
    c = np.asarray(matrices.c, dtype=np.float64)
    l = np.asarray(matrices.lb, dtype=np.float64)
    u = np.asarray(matrices.ub, dtype=np.float64)

    # Facility-location PDHG convention: G = -A_ineq, h = -b_ineq.
    K_mat = np.vstack([-A_ineq, A_eq])
    q = np.concatenate([-b_ineq, b_eq])

    m1 = A_ineq.shape[0]
    m2 = A_eq.shape[0]

    v_s = np.asarray(sol['raw_x'], dtype=np.float64)
    y_s = np.asarray(sol['raw_y'], dtype=np.float64)

    # Quick KKT sanity:
    #   c - K_mat^T y_s  should lie in -N_{[l,u]}(v_s):
    #     =0 on interior, >=0 at lb, <=0 at ub.
    residual = c - K_mat.T @ y_s
    at_lb = np.isclose(v_s, l, atol=1e-6)
    at_ub = np.isclose(v_s, u, atol=1e-6)
    interior = ~(at_lb | at_ub)
    assert np.all(residual[at_lb] >= -1e-5), \
        f"KKT violated at lb: residual={residual[at_lb]}"
    assert np.all(residual[at_ub] <= 1e-5), \
        f"KKT violated at ub: residual={residual[at_ub]}"
    if np.any(interior):
        assert np.all(np.abs(residual[interior]) <= 1e-5), \
            f"KKT violated on interior: residual={residual[interior]}"

    return {
        'K_mat': K_mat, 'q': q, 'c': c, 'l': l, 'u': u,
        'v_s': v_s, 'y_s': y_s, 'm1': m1, 'm2': m2,
    }


# ---------------------------------------------------------------------------
# CP/PDHG trajectory on facility-location
# ---------------------------------------------------------------------------

def run_cp_on_facility(K_mat, q, c, l, u, m1, v0, y0, tau, sigma, theta, K_max):
    """Run CP/PDHG for K_max steps on the facility-location LP.

    Primal step:  v_{k+1} = proj_{[l,u]}(v_k - tau(c - K_mat^T y_k))
                        (= prox_{tau f1} with f1 = c^T v + ind_{[l,u]})
    Dual step:    y_{k+1} = proj_dual(y_k + sigma(q - K_mat x_bar))
                        (partial-ReLU on first m1 coords)

    Subgradient recovery (Moreau identities, under K_ours = -K_mat):
        gf1_{k+1} = (v_k + tau K_mat^T y_k - v_{k+1}) / tau
        gh_{k+1}  = (y_k - sigma K_mat x_bar_k - y_{k+1}) / sigma

    At the initial iterate (no prox identity): choose v_0, y_0 in the STRICT
    interior of the primal box / dual cone so gf1_0 = c and gh_0 = -q are valid.
    """
    def satlin(v):
        return np.minimum(u, np.maximum(v, l))

    def partial_relu(y):
        out = y.copy()
        out[:m1] = np.maximum(out[:m1], 0.0)
        return out

    vs_iters = [v0.copy()]
    ys_iters = [y0.copy()]
    gf1_iters = [c.copy()]     # interior → ∂f1(v_0) = {c}
    gh_iters = [-q.copy()]     # interior → ∂h(y_0) = {-q}

    # w_k = K_ours^T y_k = -K_mat^T y_k   (for k = 0 .. K_max - 1)
    w_iters = []
    # z_{k+1} = K_ours x_bar_k = -K_mat x_bar_k   (stored at z_iters[k] for k=0..K-1)
    z_iters = []

    v_curr = v0
    y_curr = y0
    for k in range(K_max):
        w_k = -(K_mat.T @ y_curr)
        w_iters.append(w_k)

        # Primal: v_{k+1} = prox_{tau f1}(v_k - tau K_ours^T y_k)
        prox_in_primal = v_curr - tau * w_k             # = v_curr + tau K_mat^T y_curr
        v_new = satlin(prox_in_primal - tau * c)

        gf1_new = (prox_in_primal - v_new) / tau
        gf1_iters.append(gf1_new)

        x_bar = v_new + theta * (v_new - v_curr)

        z_kp1 = -(K_mat @ x_bar)
        z_iters.append(z_kp1)

        # Dual: y_{k+1} = prox_{sigma h}(y_k + sigma K_ours x_bar)
        prox_in_dual = y_curr + sigma * z_kp1           # = y_curr - sigma K_mat x_bar
        y_new = partial_relu(prox_in_dual + sigma * q)

        gh_new = (prox_in_dual - y_new) / sigma
        gh_iters.append(gh_new)

        v_curr = v_new
        y_curr = y_new
        vs_iters.append(v_curr.copy())
        ys_iters.append(y_curr.copy())

    return (
        np.array(vs_iters), np.array(ys_iters),
        np.array(gf1_iters), np.array(gh_iters),
        np.array(w_iters), np.array(z_iters),
    )


# ---------------------------------------------------------------------------
# Gram / F construction (facility-location analogue of the simple-LP helper)
# ---------------------------------------------------------------------------

def build_gram_and_F_facility(K_mat, q, c, v_s, y_s,
                               vs_iters, ys_iters, gf1_iters, gh_iters,
                               w_iters, z_iters, tau, sigma, theta, K_max):
    """Build G (dimG x dimG) and F (dimF,) for the facility-location trajectory.

    Embedding: primal in R^{n_vars}, dual in R^{m1+m2}, stacked.
      primal basis vectors live in the first n_vars coords,
      dual basis vectors live in the last m1+m2 coords.
    """
    n_vars = K_mat.shape[1]
    n_dual = K_mat.shape[0]

    dimG = 4 + 2 * (K_max + 2) + 2 * K_max + 3
    dimF1 = K_max + 2
    dimF_h = K_max + 2
    dimF = dimF1 + dimF_h

    idx_dx0 = 0
    idx_dy0 = 1
    idx_xs = 2
    idx_ys = 3
    idx_gf1_start = 4
    idx_gh_start = idx_gf1_start + (K_max + 2)
    idx_w_start = idx_gh_start + (K_max + 2)
    idx_z_start = idx_w_start + K_max
    idx_K_xK = idx_z_start + K_max
    idx_Kt_yK = idx_K_xK + 1
    idx_K_dx0 = idx_Kt_yK + 1
    assert idx_K_dx0 == dimG - 1

    D = n_vars + n_dual
    basis = np.zeros((dimG, D))

    def emb_x(vec):
        out = np.zeros(D); out[:n_vars] = vec; return out

    def emb_y(vec):
        out = np.zeros(D); out[n_vars:] = vec; return out

    v0 = vs_iters[0]
    y0 = ys_iters[0]

    basis[idx_dx0] = emb_x(v0 - v_s)
    basis[idx_dy0] = emb_y(y0 - y_s)
    basis[idx_xs] = emb_x(v_s)
    basis[idx_ys] = emb_y(y_s)

    for k in range(K_max + 1):
        basis[idx_gf1_start + k] = emb_x(gf1_iters[k])
    gf1_s = K_mat.T @ y_s
    basis[idx_gf1_start + (K_max + 1)] = emb_x(gf1_s)

    for k in range(K_max + 1):
        basis[idx_gh_start + k] = emb_y(gh_iters[k])
    gh_s = -K_mat @ v_s
    basis[idx_gh_start + (K_max + 1)] = emb_y(gh_s)

    for k in range(K_max):
        basis[idx_w_start + k] = emb_x(w_iters[k])      # K_ours^T y_k = -K_mat^T y_k
        basis[idx_z_start + k] = emb_y(z_iters[k])       # K_ours x_bar_k = -K_mat x_bar_k

    # Analysis slots
    K_xK = -(K_mat @ vs_iters[K_max])
    Kt_yK = -(K_mat.T @ ys_iters[K_max])
    K_dx0 = -(K_mat @ (v0 - v_s))
    basis[idx_K_xK] = emb_y(K_xK)
    basis[idx_Kt_yK] = emb_x(Kt_yK)
    basis[idx_K_dx0] = emb_y(K_dx0)

    G = basis @ basis.T

    # F: absolute function values shifted so f1(v_s) = 0, h(y_s) = 0.
    # f1(v) = c^T v + ind_{[l,u]}(v). On-feasible trajectory: f1(v_k) = c^T v_k.
    # h(y)  = -q^T y + ind_dual(y).  On-feasible trajectory: h(y_k)   = -q^T y_k.
    F_f1 = np.zeros(dimF1)
    F_h = np.zeros(dimF_h)
    for k in range(K_max + 1):
        F_f1[k] = c @ (vs_iters[k] - v_s)
        F_h[k] = -q @ (ys_iters[k] - y_s)
    F_f1[K_max + 1] = 0.0
    F_h[K_max + 1] = 0.0

    F = np.concatenate([F_f1, F_h])
    return G, F, dimG, dimF


def _choose_initial_iterate(v_s, y_s, l, u, m1, seed):
    """Pick (v_0, y_0) strictly interior so gf1_0 = c and gh_0 = -q are valid."""
    rng = np.random.default_rng(seed + 100)
    # Pull v_s away from the box boundaries, then add noise, then clamp.
    v_mid = 0.5 * (l + u)
    v0 = 0.5 * v_s + 0.5 * v_mid + 0.05 * rng.standard_normal(v_s.shape)
    v0 = np.clip(v0, l + 1e-3, u - 1e-3)
    # Pull y_s away from the nonneg cone on y[:m1], then add noise, then clamp.
    y0 = y_s + 0.05 * rng.standard_normal(y_s.shape)
    y0[:m1] = np.maximum(y0[:m1], 0.05)
    return v0, y0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

N_FACILITIES = 2
N_CUSTOMERS = 3


@pytest.mark.parametrize('K_max', [1, 3])
@pytest.mark.parametrize('seed', [0, 1])
def test_facility_trajectory_satisfies_cp_interpolation(K_max, seed):
    """Facility-location CP trajectory should satisfy every CP PEP constraint.

    Reports violations per group (f1 interp / h interp / value pin / adjoint /
    IC / PSD) and asserts bounded violation on all except IC (which is a
    scalar radius, not a per-instance constraint).
    """
    inst = build_facility_lp_instance(N_FACILITIES, N_CUSTOMERS, seed)
    K_mat = inst['K_mat']; q = inst['q']; c = inst['c']
    l = inst['l']; u = inst['u']
    v_s = inst['v_s']; y_s = inst['y_s']; m1 = inst['m1']

    L_M = float(np.linalg.norm(K_mat, ord=2))
    tau = sigma = 0.9 / L_M
    theta = 1.0

    v0, y0 = _choose_initial_iterate(v_s, y_s, l, u, m1, seed)

    vs_iters, ys_iters, gf1_iters, gh_iters, w_iters, z_iters = run_cp_on_facility(
        K_mat, q, c, l, u, m1, v0, y0, tau, sigma, theta, K_max,
    )

    G, F, dimG, dimF = build_gram_and_F_facility(
        K_mat, q, c, v_s, y_s, vs_iters, ys_iters,
        gf1_iters, gh_iters, w_iters, z_iters, tau, sigma, theta, K_max,
    )

    pep_data = construct_chambolle_pock_pep_data(
        tau=tau, sigma=sigma, theta=theta, M=L_M, R=1.0, K_max=K_max,
    )
    (A_obj, b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = [
        np.asarray(x) if not isinstance(x, list) else [np.asarray(a) for a in x]
        for x in pep_data
    ]

    # --- Scalar constraints ---
    num_scalar = A_vals.shape[0]
    violations = np.zeros(num_scalar)
    for i in range(num_scalar):
        violations[i] = eval_scalar_constraint(
            A_vals[i], b_vals[i], c_vals[i], G, F,
        )

    n_algo = K_max + 1
    n_interp = n_algo * (n_algo + 1)
    n_f1 = n_interp
    n_h = n_interp
    n_value_pin = 4
    n_IC = 1

    f1_viols = violations[:n_f1]
    h_viols = violations[n_f1:n_f1 + n_h]
    value_pin_viols = violations[n_f1 + n_h:n_f1 + n_h + n_value_pin]
    remaining = violations[n_f1 + n_h + n_value_pin:]
    adj_viols = remaining[:-n_IC]
    IC_viol = remaining[-n_IC:]

    # --- PSD blocks ---
    psd_min_eigs = []
    for idx in range(len(PSD_A_vals)):
        H = eval_psd_block(PSD_A_vals[idx], PSD_b_vals[idx], PSD_c_vals[idx], G, F)
        psd_min_eigs.append(float(np.min(np.linalg.eigvalsh(H))))

    print(f"\n=== Facility LP CP interpolation (K_max={K_max}, seed={seed}) ===")
    print(f"  n_facilities={N_FACILITIES}, n_customers={N_CUSTOMERS}")
    print(f"  n_vars={K_mat.shape[1]}, m1+m2={K_mat.shape[0]}, L_M={L_M:.4f}")
    print(f"  tau={tau:.4f}, sigma={sigma:.4f}, theta={theta}")
    print(f"  dimG={dimG}, dimF={dimF}, num_scalar={num_scalar}")
    print(f"  f1 interp:     max viol = {np.max(f1_viols):.3e}")
    print(f"  h  interp:     max viol = {np.max(h_viols):.3e}")
    print(f"  value pinning: max viol = {np.max(value_pin_viols):.3e}")
    print(f"  adjoint:       max viol = {np.max(adj_viols):.3e}")
    print(f"  IC:            value    = {IC_viol[0]:.3e}  (not asserted)")
    print(f"  PSD blocks:    min eigs = {psd_min_eigs}")

    eps = 1e-6
    assert np.max(f1_viols) <= eps, \
        f"f1 interpolation violated (max {np.max(f1_viols):.3e})"
    assert np.max(h_viols) <= eps, \
        f"h interpolation violated (max {np.max(h_viols):.3e})"
    assert np.max(value_pin_viols) <= eps, \
        f"value pinning violated (max {np.max(value_pin_viols):.3e})"
    assert np.max(adj_viols) <= eps, \
        f"adjoint violated (max {np.max(adj_viols):.3e})"
    for idx, me in enumerate(psd_min_eigs):
        assert me >= -eps, \
            f"PSD block {idx} not PSD, min eig = {me:.3e}"


@pytest.mark.parametrize('K_max', [1, 3])
@pytest.mark.parametrize('seed', [0, 1])
def test_facility_objective_matches_lagrangian_gap(K_max, seed):
    """trace(A_obj @ G) + b_obj @ F  ==  L(v_K, y_s) - L(v_s, y_K).

    L(v, y) = c^T v - y^T K_mat v + q^T y  (facility-location convention).
    """
    inst = build_facility_lp_instance(N_FACILITIES, N_CUSTOMERS, seed)
    K_mat = inst['K_mat']; q = inst['q']; c = inst['c']
    l = inst['l']; u = inst['u']
    v_s = inst['v_s']; y_s = inst['y_s']; m1 = inst['m1']

    L_M = float(np.linalg.norm(K_mat, ord=2))
    tau = sigma = 0.9 / L_M
    theta = 1.0

    v0, y0 = _choose_initial_iterate(v_s, y_s, l, u, m1, seed)

    vs_iters, ys_iters, gf1_iters, gh_iters, w_iters, z_iters = run_cp_on_facility(
        K_mat, q, c, l, u, m1, v0, y0, tau, sigma, theta, K_max,
    )

    G, F, _, _ = build_gram_and_F_facility(
        K_mat, q, c, v_s, y_s, vs_iters, ys_iters,
        gf1_iters, gh_iters, w_iters, z_iters, tau, sigma, theta, K_max,
    )

    pep_data = construct_chambolle_pock_pep_data(
        tau=tau, sigma=sigma, theta=theta, M=L_M, R=1.0, K_max=K_max,
    )
    A_obj = np.asarray(pep_data[0])
    b_obj = np.asarray(pep_data[1])

    pep_obj_value = float(np.trace(A_obj @ G) + b_obj @ F)

    v_K = vs_iters[K_max]; y_K = ys_iters[K_max]

    def L(vv, yy):
        return float(c @ vv - yy @ K_mat @ vv + q @ yy)

    real_gap = L(v_K, y_s) - L(v_s, y_K)

    print(f"\n=== Facility LP objective match (K_max={K_max}, seed={seed}) ===")
    print(f"  PEP A_obj@G + b_obj@F = {pep_obj_value:.6e}")
    print(f"  Real gap L(vK,ys)-L(vs,yK) = {real_gap:.6e}")

    assert np.isclose(pep_obj_value, real_gap, rtol=1e-6, atol=1e-8), \
        f"Objective mismatch: PEP {pep_obj_value} vs real {real_gap}"
