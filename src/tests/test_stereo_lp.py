"""Tests for the stereo-disparity LP and the JAX gap trajectory of tools/stereo_l2o_check.py.

Usage (from src/):
    pytest tests/test_stereo_lp.py -v
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from learning.stereo_lp import grad_x, make_pair, stereo_lp, true_disparity  # noqa: E402
from learning.tv_averages import run_pdhg_capture_gaps  # noqa: E402
from learning.tv_inpainting_reduced import reduced_opnorm, solve_reduced  # noqa: E402

M = N = 12


def pair(seed, C=1):
    rng = np.random.default_rng(seed)
    img = np.clip(np.cumsum(rng.normal(0, 0.1, (M, N, C)), axis=1) + 0.5, 0, 1)
    img[:, :3] = 0.5                                   # flat region: zero gradients
    d = true_disparity(M, N, rng, dmax=0.5)
    return img, d


@pytest.mark.parametrize("place", ["inside", "cost"])
def test_opnorm_formula(place):
    img, d = pair(0)
    L, R = make_pair(img, d)
    m = stereo_lp(L, R, 10.0, place)
    w = 10.0 if place == "inside" else 1.0
    from learning.tv_inpainting_reduced import diff_matrix
    Dm = diff_matrix(M, N)
    gram = 2 * (np.diag(w ** 2 * grad_x(L[0]).ravel() ** 2) + (Dm.T @ Dm).toarray())
    assert reduced_opnorm(m) ** 2 == pytest.approx(max(np.linalg.eigvalsh(gram).max(), 2.0), rel=1e-6)


def test_solution_satisfies_constraints_and_zero_shift_gives_zero_disparity_cost():
    img, _ = pair(1, C=3)
    L, R = make_pair(img, np.zeros((M, N)))          # identical views
    m = stereo_lp(L, R, 10.0)
    s = solve_reduced(m)
    assert s["objective_value"] == pytest.approx(0.0, abs=1e-6)
    assert np.all(m.G @ s["raw_x"] - m.h >= -1e-6)


def test_jax_gap_traj_matches_numpy():
    from stereo_l2o_check import make_gap_traj, pattern_template
    args = SimpleNamespace(lam=10.0, place="inside")
    tmpl = pattern_template(M, N, args)
    gap_traj = make_gap_traj(tmpl, M * N, 1.0)
    rng = np.random.default_rng(3)
    K = 4
    tau, sigma, theta = rng.uniform(0.01, 0.05, K), rng.uniform(0.01, 0.05, K), rng.uniform(0.3, 1.0, K)
    for seed in (0, 1):
        img, d = pair(seed)
        L, R = make_pair(img, d)
        m = stereo_lp(L, R, 10.0)
        s = solve_reduced(m)
        x0, y0 = 0.5 * (m.l + m.u), np.ones(m.G.shape[0])
        g_np = run_pdhg_capture_gaps(m.c, m.G, m.h, m.A, m.b, m.l, m.u, s["raw_x"], s["raw_y"],
                                     x0, y0, tau, sigma, theta)
        g_jax = gap_traj((jnp.array(tau), jnp.array(sigma), jnp.array(theta)),
                         jnp.array(10.0 * grad_x(L[0]).ravel()), jnp.array(m.h), jnp.array(m.u),
                         jnp.array(s["raw_x"]), jnp.array(s["raw_y"]), K)
        np.testing.assert_allclose(np.asarray(g_jax), g_np, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("C", [1, 3])
def test_feasible_init_is_feasible_and_interior(C):
    from learning.stereo_lp import stereo_x0
    img, d = pair(4, C=C)
    L, R = make_pair(img, d)
    m = stereo_lp(L, R, 10.0)
    x0 = stereo_x0(m.l, m.u, M * N, feasible_h=m.h, n_channels=C)
    assert np.all(x0 > m.l) and np.all(x0 < m.u)
    assert np.all(m.G @ x0 - m.h > 0)                     # strictly feasible
    assert np.allclose(x0[:M * N], 0.0)


def test_jax_gap_traj_matches_numpy_feasible_init():
    from learning.stereo_lp import stereo_x0
    from stereo_l2o_check import make_gap_traj, pattern_template
    tmpl = pattern_template(M, N, SimpleNamespace(lam=10.0, place="inside"))
    gap_traj = make_gap_traj(tmpl, M * N, 1.0, feasible=True)
    rng = np.random.default_rng(5)
    K = 4
    tau, sigma, theta = rng.uniform(0.01, 0.05, K), rng.uniform(0.01, 0.05, K), rng.uniform(0.3, 1.0, K)
    img, d = pair(2)
    L, R = make_pair(img, d)
    m = stereo_lp(L, R, 10.0)
    s = solve_reduced(m)
    x0 = stereo_x0(m.l, m.u, M * N, feasible_h=m.h)
    g_np = run_pdhg_capture_gaps(m.c, m.G, m.h, m.A, m.b, m.l, m.u, s["raw_x"], s["raw_y"],
                                 x0, np.ones(m.G.shape[0]), tau, sigma, theta)
    g_jax = gap_traj((jnp.array(tau), jnp.array(sigma), jnp.array(theta)),
                     jnp.array(10.0 * grad_x(L[0]).ravel()), jnp.array(m.h), jnp.array(m.u),
                     jnp.array(s["raw_x"]), jnp.array(s["raw_y"]), K)
    np.testing.assert_allclose(np.asarray(g_jax), g_np, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("C", [1, 3])
def test_warm_dual_is_positive_and_sums_to_cost(C):
    from learning.stereo_lp import stereo_y0
    img, d = pair(6, C=C)
    L, R = make_pair(img, d)
    m = stereo_lp(L, R, 10.0)
    y0 = stereo_y0(m.h, M * N, C)
    assert np.all(y0 > 0)
    K = M * N
    for c in range(C):
        np.testing.assert_allclose(y0[2 * c * K:(2 * c + 1) * K] + y0[(2 * c + 1) * K:(2 * c + 2) * K], 1.0)
    # active row at d = 0 gets the large weight: top active iff h_top > 0
    top = m.h[:K]
    assert np.all(y0[:K][top > 0] > 0.5) and np.all(y0[:K][top < 0] < 0.5)


def test_jax_gap_traj_matches_numpy_warm_start():
    from learning.stereo_lp import stereo_x0, stereo_y0
    from stereo_l2o_check import make_gap_traj, pattern_template
    tmpl = pattern_template(M, N, SimpleNamespace(lam=10.0, place="inside"))
    gap_traj = make_gap_traj(tmpl, M * N, 1.0, feasible=True, dual_warm=True)
    rng = np.random.default_rng(8)
    K = 4
    tau, sigma, theta = rng.uniform(0.01, 0.05, K), rng.uniform(0.01, 0.05, K), rng.uniform(0.3, 1.0, K)
    img, d = pair(7)
    L, R = make_pair(img, d)
    m = stereo_lp(L, R, 10.0)
    s = solve_reduced(m)
    g_np = run_pdhg_capture_gaps(m.c, m.G, m.h, m.A, m.b, m.l, m.u, s["raw_x"], s["raw_y"],
                                 stereo_x0(m.l, m.u, M * N, feasible_h=m.h), stereo_y0(m.h, M * N),
                                 tau, sigma, theta)
    g_jax = gap_traj((jnp.array(tau), jnp.array(sigma), jnp.array(theta)),
                     jnp.array(10.0 * grad_x(L[0]).ravel()), jnp.array(m.h), jnp.array(m.u),
                     jnp.array(s["raw_x"]), jnp.array(s["raw_y"]), K)
    np.testing.assert_allclose(np.asarray(g_jax), g_np, rtol=1e-9, atol=1e-9)
