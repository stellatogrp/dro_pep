"""Tests for the reduced TV-inpainting LP and the padded JAX PDHG of the L2O gate.

Usage (from src/):
    pytest tests/test_tv_inpainting_reduced.py -v
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as sla

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from learning.tv_averages import run_pdhg_capture_gaps  # noqa: E402
from learning.tv_inpainting_reduced import (  # noqa: E402
    build_reduced_lp,
    known_known_tv,
    reduced_edges,
    reduced_opnorm,
    solve_reduced,
)
from learning.tv_inpainting_test import extract_constraint_matrices, solve_lp  # noqa: E402

M = N = 8


def instance(seed, frac=0.3):
    rng = np.random.default_rng(seed)
    img = np.clip(np.add.outer(np.linspace(0, 1, M), np.linspace(0, 0.5, N)) + 0.2 * rng.random((M, N)), 0, 1)
    known = rng.random((M, N)) >= frac
    return img.reshape(-1), known.reshape(-1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reduced_optimum_matches_full_lp(seed):
    pix, known = instance(seed)
    idx = np.flatnonzero(known)
    full = solve_lp(extract_constraint_matrices(idx, pix[idx], M, N))["objective_value"]
    red = solve_reduced(build_reduced_lp(pix, known, M, N))["objective_value"]
    assert red + known_known_tv(pix, known, M, N) == pytest.approx(full, abs=1e-5)


@pytest.mark.parametrize("seed", [0, 1])
def test_reduced_kkt(seed):
    pix, known = instance(seed)
    mats = build_reduced_lp(pix, known, M, N)
    s = solve_reduced(mats)
    x, y = s["raw_x"], s["raw_y"]
    assert np.all(y >= -1e-7)
    assert np.all(mats.G @ x - mats.h >= -1e-6)
    # c - G^T y must be a valid box-normal at x: >= 0 where x = l, <= 0 where x = u, 0 inside
    r = mats.c - mats.G.T @ y
    inside = (x > 1e-6) & (x < 1 - 1e-6)
    assert np.abs(r[inside]).max(initial=0) < 1e-5
    # strong duality with the box kept in the dual: h^T y + sum_i min(l_i r_i, u_i r_i)
    dual = mats.h @ y + np.minimum(mats.l * r, mats.u * r).sum()
    assert dual == pytest.approx(s["objective_value"], abs=1e-5)


@pytest.mark.parametrize("seed", [0, 3])
def test_opnorm_formula(seed):
    pix, known = instance(seed)
    mats = build_reduced_lp(pix, known, M, N)
    D_U, _ = reduced_edges(known, M, N)
    s2 = sla.svds(D_U.tocsc().astype(float), k=1, return_singular_vectors=False)[0] ** 2
    assert reduced_opnorm(mats) ** 2 == pytest.approx(2 * max(s2, 1.0), rel=1e-8)


def test_padded_jax_pdhg_matches_numpy():
    from pdlp_reduced_l2o_check import Y0, make_instance_fn, pad_instances

    insts = [instance(s) for s in (0, 1)]
    # masks must share the unknown count for padding; force it
    U = int((~insts[0][1]).sum())
    fixed = []
    for pix, known in insts:
        unk = np.flatnonzero(~known)[:U]
        k2 = np.ones(M * N, bool); k2[unk] = False
        if (~k2).sum() < U:
            extra = np.flatnonzero(k2)[: U - (~k2).sum()]; k2[extra] = False
        fixed.append((pix, k2))
    sols, mats_list = [], []
    for pix, known in fixed:
        m = build_reduced_lp(pix, known, M, N)
        s = solve_reduced(m)
        sols.append((s["raw_x"], s["raw_y"], s["objective_value"], reduced_opnorm(m)))
        mats_list.append(m)
    E_max = max(reduced_edges(k, M, N)[0].shape[0] for _, k in fixed) + 3   # force real padding
    Upad, data = pad_instances([p for p, _ in fixed], [k for _, k in fixed], sols, M, N, E_max, 2 * E_max)
    rng = np.random.default_rng(9)
    K = 4
    tau, sigma, theta = rng.uniform(0.05, 0.3, K), rng.uniform(0.05, 0.3, K), rng.uniform(0.3, 1.0, K)
    gap_traj = make_instance_fn(Upad, E_max)
    keys = ("rows", "cols", "vals", "h", "c", "u", "x_opt", "y_opt")
    for i, (m, (rx, ry, _, _)) in enumerate(zip(mats_list, sols)):
        g_jax = np.asarray(gap_traj((jnp.array(tau), jnp.array(sigma), jnp.array(theta)),
                                    *(data[k][i] for k in keys), K))
        x0, y0 = 0.5 * (m.l + m.u), Y0 * np.ones(m.G.shape[0])
        g_np = run_pdhg_capture_gaps(m.c, m.G, m.h, m.A, m.b, m.l, m.u, rx, ry, x0, y0, tau, sigma, theta)
        np.testing.assert_allclose(g_jax, g_np, rtol=1e-9, atol=1e-9)
