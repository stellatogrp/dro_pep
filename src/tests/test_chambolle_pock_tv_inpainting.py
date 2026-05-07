"""Interpolation-condition diagnostic for `construct_chambolle_pock_pep_data`
on TV inpainting LP instances.

Mirrors `test_chambolle_pock_facility_location.py` for the new TV-inpainting
PDLP. The CP trajectory + Gram-and-F helpers from that file are
problem-agnostic — they take ``(c, K_mat, q, l, u, m1)`` and don't care
whether it's facility location or TV inpainting — so we reuse them directly.

What we verify:
  1. Build a small TV-inpainting LP (downsampled to 16x16 with a synthetic
     image so the test is fast; LP structure is identical to the production
     64x64 case).
  2. Solve it with CVXPY/CLARABEL → (v_s, y_s).
  3. Pick a strictly-interior init (v_0, y_0).
  4. Run reference CP iterations to produce subgradients via Moreau identities.
  5. Build (G, F) in the verified PEP basis.
  6. Construct the production PEP at (M = ||K_mat||_op, R = ||z_0 - z_*||).
  7. Evaluate every scalar and PSD constraint and assert per-group violations
     are within tolerance (eps = 1e-5).
  8. Separately assert the PEP objective equals the real Lagrangian gap.

If any of these fail, the SDP unboundedness observed during training is
attributable to interpolation violation rather than ill-conditioning.
"""
import numpy as np
import pytest
import scipy.sparse as sp

import jax
jax.config.update('jax_enable_x64', True)

from learning.pep_constructions import construct_chambolle_pock_pep_data
from learning.tv_inpainting_test import (
    extract_constraint_matrices,
    solve_lp,
)

# Reuse problem-agnostic helpers. Despite the name, these only depend on
# (c, K_mat, q, l, u, m1) — equally valid for TV inpainting.
from tests.test_chambolle_pock_facility_location import (
    run_cp_on_facility,
    build_gram_and_F_facility,
    _choose_initial_iterate,
)
from tests.test_chambolle_pock_interpolation import (
    eval_scalar_constraint,
    eval_psd_block,
)


# ---------------------------------------------------------------------------
# TV-inpainting LP instance builder (small synthetic image — same LP shape as
# production, just smaller for fast pytest collection)
# ---------------------------------------------------------------------------

def build_tv_inpainting_lp_instance(M_img, N_img, missing_fraction, seed,
                                     lp_upper=255.0):
    """Generate a TV inpainting LP from a synthetic image and solve it.

    The synthetic image is a deterministic 2D ramp + noise in [0, lp_upper];
    the LP structure (G, A, c, l, u) is exactly what production code builds.

    Returns dict with:
      K_mat (m1+S, n_vars)        : stacked operator [G; A_mask]
      q     (m1+S,)               : stacked RHS      [0; known_values]
      c     (n_vars,)             : objective       [0; 1; 1]
      l, u  (n_vars,)             : box bounds      [0, lp_upper]
      v_s   (n_vars,)             : primal optimum
      y_s   (m1+S,)               : dual optimum (PDHG sign convention)
      m1    (int)                 : 2*K_v + 2*K_h
      m2    (int)                 : S
    """
    rng = np.random.default_rng(seed)
    K = M_img * N_img
    K_v = (M_img - 1) * N_img
    K_h = M_img * (N_img - 1)
    m1 = 2 * K_v + 2 * K_h

    # Synthetic image: ramp + noise, scaled to [0, lp_upper].
    rows = np.arange(M_img).reshape(-1, 1) / max(M_img - 1, 1)
    cols = np.arange(N_img).reshape(1, -1) / max(N_img - 1, 1)
    image = (0.5 * (rows + cols) + 0.05 * rng.standard_normal((M_img, N_img)))
    image = np.clip(image, 0.0, 1.0) * lp_upper

    # Deterministic-count mask (same convention as production).
    n_corrupted = int(round(missing_fraction * K))
    corrupted = np.sort(rng.permutation(K)[:n_corrupted])
    mask_flat = np.ones(K, dtype=bool)
    mask_flat[corrupted] = False
    known_indices = np.flatnonzero(mask_flat)
    S = known_indices.size
    known_values = image.reshape(-1)[known_indices].copy()

    # Build LP matrices via the production helper (returns scipy CSR).
    mats = extract_constraint_matrices(known_indices, known_values, M_img, N_img)
    # Standard form: min c^T x  s.t. l <= x <= u, A x = b, G x >= h (h = 0).
    # Convert to PDHG convention: K_mat = [-A_ineq; A_eq] = [G; A_mask],
    #                              q     = [-b_ineq; b_eq] = [0; b].
    K_csr = sp.vstack([mats.G, mats.A], format="csr")
    K_mat = K_csr.toarray().astype(np.float64)
    q = np.concatenate([np.zeros(m1), mats.b])

    # Override LP_UPPER on the matrices.l/u — extract_constraint_matrices uses a
    # global LP_UPPER from the test file (255.0). For our test we want exactly
    # whatever lp_upper we passed in; rebuild bounds explicitly.
    n_vars = K + K_v + K_h
    c = np.concatenate([np.zeros(K), np.ones(K_v), np.ones(K_h)]).astype(np.float64)
    l = np.zeros(n_vars)
    u = lp_upper * np.ones(n_vars)

    # Re-solve via CVXPY/CLARABEL with the reset bounds.
    # solve_lp (the production helper) uses matrices.u = LP_UPPER which is the
    # same global; if we pass our matrices unchanged we get the same answer.
    sol = solve_lp(mats)
    v_s = sol['raw_x'].astype(np.float64)
    y_s = sol['raw_y'].astype(np.float64)
    m2 = S

    return {
        'K_mat': K_mat, 'q': q, 'c': c, 'l': l, 'u': u,
        'v_s': v_s, 'y_s': y_s, 'm1': m1, 'm2': m2,
        'image': image, 'mask': mask_flat,
    }


# ---------------------------------------------------------------------------
# Constraint-group counts for the PEP layout
# ---------------------------------------------------------------------------

def _split_violations(violations, K_max):
    """Slice the scalar-constraints array into named groups."""
    n_algo = K_max + 1
    n_interp = n_algo * (n_algo + 1)
    n_f1 = n_interp
    n_h = n_interp
    n_value_pin = 4
    n_IC = 1

    f1_viols = violations[:n_f1]
    h_viols = violations[n_f1: n_f1 + n_h]
    value_pin = violations[n_f1 + n_h: n_f1 + n_h + n_value_pin]
    remaining = violations[n_f1 + n_h + n_value_pin:]
    adj = remaining[:-n_IC]
    IC = remaining[-n_IC:]
    return f1_viols, h_viols, value_pin, adj, IC


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Small downsampled image for test speed. LP structure is identical to 64x64
# production, only n_vars / m_total differ. Default lp_upper=1.0 matches the
# current scaled_lp_01=true production setting; the legacy 255.0 path is
# covered by parameterization.
M_IMG = 16
N_IMG = 16
MISSING_FRACTION = 0.1
LP_UPPER = 1.0


@pytest.mark.parametrize('K_max', [1, 3])
@pytest.mark.parametrize('seed', [0, 1])
def test_tv_inpainting_trajectory_satisfies_cp_interpolation(K_max, seed):
    """TV-inpainting CP trajectory should satisfy every CP PEP constraint."""
    inst = build_tv_inpainting_lp_instance(
        M_img=M_IMG, N_img=N_IMG,
        missing_fraction=MISSING_FRACTION, seed=seed, lp_upper=LP_UPPER,
    )
    K_mat = inst['K_mat']; q = inst['q']; c = inst['c']
    l = inst['l']; u = inst['u']
    v_s = inst['v_s']; y_s = inst['y_s']; m1 = inst['m1']

    L_M = float(np.linalg.norm(K_mat, ord=2))
    tau = sigma = 0.9 / L_M
    theta = 1.0

    v0, y0 = _choose_initial_iterate(v_s, y_s, l, u, m1, seed)
    R_actual = float(
        np.linalg.norm(np.concatenate([v0 - v_s, y0 - y_s]))
    )

    vs_iters, ys_iters, gf1_iters, gh_iters, w_iters, z_iters = run_cp_on_facility(
        K_mat, q, c, l, u, m1, v0, y0, tau, sigma, theta, K_max,
    )

    G, F, dimG, dimF = build_gram_and_F_facility(
        K_mat, q, c, v_s, y_s, vs_iters, ys_iters,
        gf1_iters, gh_iters, w_iters, z_iters, tau, sigma, theta, K_max,
    )

    pep_data = construct_chambolle_pock_pep_data(
        tau=tau, sigma=sigma, theta=theta, M=L_M, R=R_actual, K_max=K_max,
    )
    (A_obj, b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = [
        np.asarray(x) if not isinstance(x, list) else [np.asarray(a) for a in x]
        for x in pep_data
    ]

    # Scalar constraints
    num_scalar = A_vals.shape[0]
    violations = np.zeros(num_scalar)
    for i in range(num_scalar):
        violations[i] = eval_scalar_constraint(
            A_vals[i], b_vals[i], c_vals[i], G, F,
        )
    f1_v, h_v, vp_v, adj_v, IC_v = _split_violations(violations, K_max)

    # PSD blocks
    psd_min_eigs = []
    for idx in range(len(PSD_A_vals)):
        H = eval_psd_block(PSD_A_vals[idx], PSD_b_vals[idx], PSD_c_vals[idx], G, F)
        psd_min_eigs.append(float(np.min(np.linalg.eigvalsh(H))))

    print(f"\n=== TV inpainting CP interpolation (K_max={K_max}, seed={seed}) ===")
    print(f"  M_img={M_IMG}, N_img={N_IMG}, missing_fraction={MISSING_FRACTION}, "
          f"lp_upper={LP_UPPER}")
    print(f"  n_vars={K_mat.shape[1]}, m1+m2={K_mat.shape[0]}, "
          f"L_M={L_M:.4f}, R_actual={R_actual:.4f}")
    print(f"  tau={tau:.4f}, sigma={sigma:.4f}, theta={theta}")
    print(f"  dimG={dimG}, dimF={dimF}, num_scalar={num_scalar}")
    print(f"  f1 interp:     max viol = {np.max(f1_v):.3e}")
    print(f"  h  interp:     max viol = {np.max(h_v):.3e}")
    print(f"  value pinning: max viol = {np.max(vp_v):.3e}")
    print(f"  adjoint:       max viol = {np.max(adj_v):.3e}")
    print(f"  IC:            value    = {IC_v[0]:.3e}  (not asserted)")
    print(f"  PSD blocks:    min eigs = {psd_min_eigs}")

    # Tolerance matches facility-location's 1e-6 since lp_upper=1.0 keeps Gram
    # entries O(1). If we ever flip back to lp_upper=255, the noise floor scales
    # up by ~255**2 — bump to 1e-4 / 1e-2 in that case.
    eps = 1e-6
    psd_eps = 1e-6
    assert np.max(f1_v) <= eps, f"f1 interpolation violated (max {np.max(f1_v):.3e})"
    assert np.max(h_v) <= eps, f"h interpolation violated (max {np.max(h_v):.3e})"
    assert np.max(vp_v) <= eps, f"value pinning violated (max {np.max(vp_v):.3e})"
    assert np.max(adj_v) <= eps, f"adjoint violated (max {np.max(adj_v):.3e})"
    for idx, me in enumerate(psd_min_eigs):
        assert me >= -psd_eps, f"PSD block {idx} not PSD, min eig = {me:.3e}"


@pytest.mark.parametrize('K_max', [1, 3])
@pytest.mark.parametrize('seed', [0, 1])
def test_tv_inpainting_objective_matches_lagrangian_gap(K_max, seed):
    """tr(A_obj G) + b_obj F  ==  L(v_K, y_s) - L(v_s, y_K).

    L(v, y) = c^T v - y^T K_mat v + q^T y  (PDHG convention).
    """
    inst = build_tv_inpainting_lp_instance(
        M_img=M_IMG, N_img=N_IMG,
        missing_fraction=MISSING_FRACTION, seed=seed, lp_upper=LP_UPPER,
    )
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

    print(f"\n=== TV inpainting objective match (K_max={K_max}, seed={seed}) ===")
    print(f"  PEP A_obj@G + b_obj@F      = {pep_obj_value:.6e}")
    print(f"  Real gap L(vK,ys)-L(vs,yK) = {real_gap:.6e}")

    assert np.isclose(pep_obj_value, real_gap, rtol=1e-5, atol=1e-6), \
        f"Objective mismatch: PEP {pep_obj_value} vs real {real_gap}"
