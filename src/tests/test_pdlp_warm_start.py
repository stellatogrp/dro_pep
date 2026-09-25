"""Tests for the PDLP (TV-inpainting) PDHG initializations.

The PEP basis assumes (x0, y0) is in the strict interior of the box and the
nonnegative cone, so the warm start must (1) keep every coordinate strictly
inside (0, lp_upper) and (2) satisfy every G x0 >= 0 row strictly. The numpy
builder (sample creation / evaluation) and the JAX builder (trainer) must
agree, and 'center' must reproduce the original init exactly.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from learning_experiment_classes.pdlp import (  # noqa: E402
    _build_G_sparse,
    _jax_warm_start_primal,
    _warm_start_primal_single,
    build_pdhg_init,
    build_strict_interior_init,
)

M, N = 16, 12
K, K_v, K_h = M * N, (M - 1) * N, M * (N - 1)
N_VARS = K + K_v + K_h
M1 = 2 * K_v + 2 * K_h


def _instance(seed, lp_upper=1.0, missing=0.2):
    rng = np.random.default_rng(seed)
    pix = rng.random(K) * lp_upper
    pix[:5] = 0.0            # exact box boundary values must still end up interior
    pix[5:10] = lp_upper
    mask = rng.random(K) >= missing
    return pix, mask


@pytest.mark.parametrize('lp_upper', [1.0, 255.0])
@pytest.mark.parametrize('delta', [1e-3, 1e-2])
def test_warm_start_strictly_interior(lp_upper, delta):
    pix, mask = _instance(0, lp_upper)
    S = int(mask.sum())
    x0, y0 = build_pdhg_init('warm', N_VARS, M1, S, lp_upper,
                             pix_channels=[pix], mask=mask, M=M, N=N, delta=delta)
    assert x0.shape == (N_VARS,) and y0.shape == (M1 + S,)
    assert np.all(x0 > 0) and np.all(x0 < lp_upper)
    assert np.all(_build_G_sparse(M, N) @ x0 > 0)
    # Missing pixels start at the lower clip, known pixels at their clipped value.
    p0 = x0[:K]
    np.testing.assert_allclose(p0[~mask], delta * lp_upper)
    np.testing.assert_allclose(
        p0[mask], np.clip(pix[mask], delta * lp_upper, (1 - delta) * lp_upper))


def test_center_matches_original_init():
    x0, y0 = build_pdhg_init('center', N_VARS, M1, 50, 1.0)
    x0_ref, y0_ref = build_strict_interior_init(N_VARS, M1, 50, 1.0)
    np.testing.assert_array_equal(x0, x0_ref)
    np.testing.assert_array_equal(y0, y0_ref)


def test_dual_ineq_init():
    _, y0 = build_pdhg_init('center', N_VARS, M1, 50, 1.0, dual_ineq_init=0.5)
    np.testing.assert_array_equal(y0[:M1], 0.5)
    np.testing.assert_array_equal(y0[M1:], 0.0)


def test_dual_init_sets_all_duals():
    from omegaconf import OmegaConf
    from learning_experiment_classes.pdlp import init_settings_from_cfg
    _, _, d_ineq, d_eq = init_settings_from_cfg(OmegaConf.create({"dual_init": 1.0}))
    _, y0 = build_pdhg_init('center', N_VARS, M1, 50, 1.0, dual_ineq_init=d_ineq, dual_eq_init=d_eq)
    np.testing.assert_array_equal(y0, 1.0)
    assert init_settings_from_cfg(OmegaConf.create({}))[2:] == (0.1, 0.0)
    with pytest.raises(ValueError):
        init_settings_from_cfg(OmegaConf.create({"dual_init": 1.0, "warm_dual": True}))


def test_numpy_and_jax_warm_start_agree():
    pix, mask = _instance(1)
    x_np = _warm_start_primal_single(pix, mask, M, N, 0.01, 1.0)
    x_jax = _jax_warm_start_primal(jnp.asarray(pix), jnp.asarray(mask), M, N, 0.01, 1.0)
    np.testing.assert_allclose(np.asarray(x_jax), x_np, rtol=0, atol=1e-15)


def test_color_layout_is_per_channel_blocks():
    """Color x0 = [p_R, v_R, w_R, p_G, ...], matching build_color_lp's block-diagonal G."""
    import scipy.sparse as sp
    chans = [_instance(s)[0] for s in (2, 3, 4)]
    mask = _instance(5)[1]
    S = 3 * int(mask.sum())
    x0, _ = build_pdhg_init('warm', 3 * N_VARS, 3 * M1, S, 1.0,
                            pix_channels=chans, mask=mask, M=M, N=N)
    for c, pix in enumerate(chans):
        np.testing.assert_array_equal(
            x0[c * N_VARS:(c + 1) * N_VARS],
            _warm_start_primal_single(pix, mask, M, N, 0.01, 1.0))
    G = sp.block_diag([_build_G_sparse(M, N)] * 3)
    assert np.all(G @ x0 > 0)


def test_unknown_init_type_raises():
    with pytest.raises(ValueError):
        build_pdhg_init('zeros', N_VARS, M1, 10, 1.0)


# ---------------------------------------------------------------------------
# Blurred forward model (A = [E H, 0], b = (H p)[known])
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sigma', [0.5, 1.0, 2.0])
def test_blur_matrix_is_normalized(sigma):
    from scipy.sparse.linalg import svds
    from learning_experiment_classes.pdlp import gaussian_blur_matrix
    H = gaussian_blur_matrix(M, N, sigma)
    np.testing.assert_allclose(np.asarray(H.sum(axis=1)).ravel(), 1.0)
    assert H.min() >= 0
    assert svds(H, k=1, return_singular_vectors=False)[0] <= 1.01  # border renormalization


def test_zero_blur_recovers_plain_inpainting():
    from learning.tv_inpainting_test import extract_constraint_matrices
    from learning_experiment_classes.pdlp import blurred_data_block, gaussian_blur_matrix
    pix, mask = _instance(6)
    known = np.flatnonzero(mask)
    mats = extract_constraint_matrices(known, pix[known], M, N)
    A, b = blurred_data_block(known, [pix], [gaussian_blur_matrix(M, N, 0.0)], N_VARS)
    assert (A != mats.A).nnz == 0
    np.testing.assert_array_equal(b, mats.b)


def test_blurred_lp_is_feasible_at_original_image():
    """The original image (with exact TV slacks) satisfies A x = b, so the LP is feasible."""
    from learning_experiment_classes.pdlp import blurred_data_block, gaussian_blur_matrix
    chans = [_instance(s)[0] for s in (7, 8, 9)]
    mask = _instance(10)[1]
    known = np.flatnonzero(mask)
    Hs = [gaussian_blur_matrix(M, N, s) for s in (0.5, 1.0, 2.0)]
    A, b = blurred_data_block(known, chans, Hs, N_VARS)
    x = np.concatenate([_warm_start_primal_single(p, np.ones(K, bool), M, N, 0.0, 1.0) for p in chans])
    np.testing.assert_allclose(A @ x, b, atol=1e-12)


# ---------------------------------------------------------------- consistent init
def _consistent(seed, lp_upper=1.0, init_type='consistent'):
    import scipy.sparse as sp
    from learning.tv_inpainting_test import extract_constraint_matrices
    pix, mask = _instance(seed, lp_upper)
    known = np.flatnonzero(mask)
    mats = extract_constraint_matrices(known, pix[known], M, N)
    x0, y0 = build_pdhg_init(init_type, N_VARS, M1, known.size, lp_upper,
                             pix_channels=[pix], mask=mask, M=M, N=N, delta=0.01)
    return pix, mask, known, mats, x0, y0, sp.vstack([mats.G, mats.A]).tocsr()


@pytest.mark.parametrize("init_type", ["consistent", "consistent_holes"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_consistent_init_interior_feasible_and_stationary(seed, init_type):
    pix, mask, known, mats, x0, y0, Km = _consistent(seed, init_type=init_type)
    if init_type == "consistent_holes":
        np.testing.assert_allclose(x0[:M * N][~mask], 0.5)   # holes start at 0.5
    assert np.all(x0 > mats.l) and np.all(x0 < mats.u)
    assert np.all(mats.G @ x0 - mats.h > 0)                  # strictly feasible TV rows
    # known pixels at their values, up to the delta clip that keeps x0 strictly interior
    assert np.abs(mats.A @ x0 - mats.b).max() <= 0.01 + 1e-12
    assert np.all(y0[:M1] > 0)                                # TV duals strictly positive
    Kv, Kh = (M - 1) * N, M * (N - 1)
    np.testing.assert_allclose(y0[:Kv] + y0[Kv:2 * Kv], 1.0)  # each TV pair sums to its cost
    np.testing.assert_allclose(y0[2 * Kv:2 * Kv + Kh] + y0[2 * Kv + Kh:M1], 1.0)
    rc = (mats.c - Km.T @ y0)[:M * N]
    np.testing.assert_allclose(rc[known], 0.0, atol=1e-12)    # known pixels stationary


def test_consistent_init_numpy_matches_jax():
    from learning_experiment_classes.pdlp import _consistent_init_single
    pix, mask, known, mats, x0, y0, _ = _consistent(3)
    xj, yGj, yAj = _consistent_init_single(jnp, jnp.asarray(pix), jnp.asarray(mask), M, N, 0.01, 1.0, known.size)
    np.testing.assert_allclose(np.asarray(xj), x0, atol=1e-14)
    np.testing.assert_allclose(np.r_[np.asarray(yGj), np.asarray(yAj)], y0, atol=1e-14)


def test_consistent_init_color_layout():
    pix, mask = _instance(4)
    chans = [pix, np.clip(pix * 0.8, 0, 1), np.clip(1 - pix, 0, 1)]
    S = int(mask.sum())
    x0, y0 = build_pdhg_init('consistent', 3 * N_VARS, 3 * M1, 3 * S, 1.0,
                             pix_channels=chans, mask=mask, M=M, N=N, delta=0.01)
    for c, pc in enumerate(chans):
        x1, y1 = build_pdhg_init('consistent', N_VARS, M1, S, 1.0, pix_channels=[pc], mask=mask, M=M, N=N)
        np.testing.assert_allclose(x0[c * N_VARS:(c + 1) * N_VARS], x1)
        np.testing.assert_allclose(y0[c * M1:(c + 1) * M1], y1[:M1])
        np.testing.assert_allclose(y0[3 * M1 + c * S:3 * M1 + (c + 1) * S], y1[M1:])


def test_consistent_holes_numpy_matches_jax():
    from learning_experiment_classes.pdlp import _consistent_init_single
    pix, mask, known, mats, x0, y0, _ = _consistent(5, init_type="consistent_holes")
    xj, yGj, yAj = _consistent_init_single(jnp, jnp.asarray(pix), jnp.asarray(mask), M, N, 0.01, 1.0,
                                           known.size, hole_value=0.5)
    np.testing.assert_allclose(np.asarray(xj), x0, atol=1e-14)
    np.testing.assert_allclose(np.r_[np.asarray(yGj), np.asarray(yAj)], y0, atol=1e-14)
