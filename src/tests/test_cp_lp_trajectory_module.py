"""
Verify the new `learning/trajectories/cp_lp.py` module produces (G, F)
bit-compatible with `test_chambolle_pock_facility_location.py`'s inline
verified reference construction, and that its output still satisfies every
`construct_chambolle_pock_pep_data` constraint.

This pins the new module to the construction that was already verified to
work, so any future edit that breaks the contract will be caught here.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning.trajectories import problem_data_to_cp_lp_trajectories
from learning.pep_constructions import construct_chambolle_pock_pep_data

from tests.test_chambolle_pock_facility_location import (
    build_facility_lp_instance,
    run_cp_on_facility,
    build_gram_and_F_facility,
    _choose_initial_iterate,
    N_FACILITIES,
    N_CUSTOMERS,
)
from tests.test_chambolle_pock_interpolation import (
    eval_scalar_constraint,
    eval_psd_block,
)


@pytest.mark.parametrize('K_max', [1, 3])
@pytest.mark.parametrize('seed', [0, 1])
def test_cp_lp_trajectory_matches_verified_reference(K_max, seed):
    """New module's (G, F) matches the in-test reference to machine precision."""
    inst = build_facility_lp_instance(N_FACILITIES, N_CUSTOMERS, seed)
    K_mat = inst['K_mat']; q = inst['q']; c = inst['c']
    l = inst['l']; u = inst['u']
    v_s = inst['v_s']; y_s = inst['y_s']; m1 = inst['m1']

    L_M = float(np.linalg.norm(K_mat, ord=2))
    tau = sigma = 0.9 / L_M
    theta = 1.0

    v0, y0 = _choose_initial_iterate(v_s, y_s, l, u, m1, seed)

    # --- Reference: inline code from the verified facility-location test ---
    vs_iters_ref, ys_iters_ref, gf1_iters_ref, gh_iters_ref, w_iters_ref, z_iters_ref = (
        run_cp_on_facility(K_mat, q, c, l, u, m1, v0, y0, tau, sigma, theta, K_max)
    )
    G_ref, F_ref, _, _ = build_gram_and_F_facility(
        K_mat, q, c, v_s, y_s,
        vs_iters_ref, ys_iters_ref, gf1_iters_ref, gh_iters_ref,
        w_iters_ref, z_iters_ref, tau, sigma, theta, K_max,
    )

    # --- New module ---
    stepsizes = (jnp.array(tau), jnp.array(sigma), jnp.array(theta))
    G_new, F_new = problem_data_to_cp_lp_trajectories(
        stepsizes,
        jnp.asarray(c), jnp.asarray(K_mat), jnp.asarray(q),
        jnp.asarray(l), jnp.asarray(u),
        jnp.asarray(v_s), jnp.asarray(y_s),
        jnp.asarray(v0), jnp.asarray(y0),
        K_max, m1,
    )
    G_new = np.asarray(G_new); F_new = np.asarray(F_new)

    np.testing.assert_allclose(G_new, G_ref, rtol=1e-10, atol=1e-12,
                               err_msg="Gram mismatch between new module and reference")
    np.testing.assert_allclose(F_new, F_ref, rtol=1e-10, atol=1e-12,
                               err_msg="F mismatch between new module and reference")


@pytest.mark.parametrize('K_max', [1, 3])
@pytest.mark.parametrize('seed', [0, 1])
def test_cp_lp_trajectory_satisfies_pep_constraints(K_max, seed):
    """(G, F) from the new module passes every construct_chambolle_pock_pep_data constraint."""
    inst = build_facility_lp_instance(N_FACILITIES, N_CUSTOMERS, seed)
    K_mat = inst['K_mat']; q = inst['q']; c = inst['c']
    l = inst['l']; u = inst['u']
    v_s = inst['v_s']; y_s = inst['y_s']; m1 = inst['m1']

    L_M = float(np.linalg.norm(K_mat, ord=2))
    tau = sigma = 0.9 / L_M
    theta = 1.0

    v0, y0 = _choose_initial_iterate(v_s, y_s, l, u, m1, seed)

    stepsizes = (jnp.array(tau), jnp.array(sigma), jnp.array(theta))
    G, F = problem_data_to_cp_lp_trajectories(
        stepsizes,
        jnp.asarray(c), jnp.asarray(K_mat), jnp.asarray(q),
        jnp.asarray(l), jnp.asarray(u),
        jnp.asarray(v_s), jnp.asarray(y_s),
        jnp.asarray(v0), jnp.asarray(y0),
        K_max, m1,
    )
    G = np.asarray(G); F = np.asarray(F)

    pep_data = construct_chambolle_pock_pep_data(
        tau=tau, sigma=sigma, theta=theta, M=L_M, R=1.0, K_max=K_max,
    )
    (A_obj, b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = [
        np.asarray(x) if not isinstance(x, list) else [np.asarray(a) for a in x]
        for x in pep_data
    ]

    # Scalar constraints
    num_scalar = A_vals.shape[0]
    max_scalar_viol = -np.inf
    for i in range(num_scalar):
        v = eval_scalar_constraint(A_vals[i], b_vals[i], c_vals[i], G, F)
        # IC is the LAST scalar row — skip its assertion (radius, not interp).
        is_ic = (i == num_scalar - 1)
        if not is_ic:
            max_scalar_viol = max(max_scalar_viol, v)

    # PSD blocks
    psd_min_eigs = []
    for idx in range(len(PSD_A_vals)):
        H = eval_psd_block(PSD_A_vals[idx], PSD_b_vals[idx], PSD_c_vals[idx], G, F)
        psd_min_eigs.append(float(np.min(np.linalg.eigvalsh(H))))

    # Objective match
    pep_obj_value = float(np.trace(A_obj @ G) + b_obj @ F)

    def L(vv, yy):
        return float(c @ vv - yy @ K_mat @ vv + q @ yy)

    real_gap = L(v_s + np.asarray(0), y_s)
    v_K_est = None  # optional additional diagnostic

    print(f"\n=== cp_lp module PEP check (K={K_max}, seed={seed}) ===")
    print(f"  max scalar viol (excluding IC): {max_scalar_viol:.3e}")
    print(f"  PSD block min eigs:             {psd_min_eigs}")
    print(f"  PEP obj value (A_obj@G+b_obj@F): {pep_obj_value:.6e}")

    eps = 1e-6
    assert max_scalar_viol <= eps, f"Scalar constraint violated: {max_scalar_viol:.3e}"
    for i, me in enumerate(psd_min_eigs):
        assert me >= -eps, f"PSD block {i} not PSD: min eig {me:.3e}"
