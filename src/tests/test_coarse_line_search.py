"""Tests for the coarse line-search baselines (learning/baselines).

Reference implementations: the JAX trajectory functions used by the paper's
evaluation scripts (Quad GD, Lasso ISTA), the numpy LogReg simulators in
logreg_rebuttal/build_logreg_table.py, and learning.tv_averages.
run_pdhg_capture_gaps for PDHG. With ``growth = 1`` the baseline must
reproduce the fixed-default-step method exactly.
"""
import os
import sys

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import scipy.sparse as sp
import scipy.sparse.linalg as spla

jax.config.update('jax_enable_x64', True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'logreg_rebuttal'))

from learning.baselines.coarse_line_search import (  # noqa: E402
    fgm_logreg, gd_logreg, gd_quad, ista_lasso, pdhg_lp,
)
from learning.acceleration_stepsizes import get_nesterov_fgm_beta_sequence  # noqa: E402
from learning.trajectories.gd_fgm import problem_data_to_gd_trajectories  # noqa: E402
from learning.trajectories.ista_fista import problem_data_to_ista_trajectories  # noqa: E402
from learning.tv_averages import run_pdhg_capture_gaps  # noqa: E402
from learning.tv_inpainting_test import LP_UPPER, extract_constraint_matrices  # noqa: E402
import build_logreg_table as blt  # noqa: E402

K_MAX = 12
RTOL = 1e-10
ATOL = 1e-12


# ---------------------------------------------------------------------------
# Synthetic problem batches
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def quad():
    rng = np.random.default_rng(0)
    N, d, mu, L = 6, 10, 1.0, 10.0
    Q = np.empty((N, d, d))
    for i in range(N):
        U, _ = np.linalg.qr(rng.standard_normal((d, d)))
        eig = rng.uniform(mu, L, size=d)
        eig[0], eig[1] = mu, L
        Q[i] = U @ np.diag(eig) @ U.T
    z0 = rng.standard_normal((N, d)) * 3.0
    return dict(Q=Q, z0=z0, mu=mu, L=L, t_default=1.5 / (mu + L))


@pytest.fixture(scope='module')
def logreg():
    rng = np.random.default_rng(1)
    N, m, n = 5, 40, 6
    A = rng.standard_normal((N, m, n))
    b = (rng.random((N, m)) < 0.5).astype(np.float64)
    L = max(np.linalg.eigvalsh(A[i].T @ A[i]).max() / (4 * m) for i in range(N))
    f_opt = rng.uniform(0.3, 0.6, size=N)   # arbitrary offset; only differences matter
    beta = get_nesterov_fgm_beta_sequence(0.0, L, K_MAX)
    return dict(A=A, b=b, f_opt=f_opt, L=L, t_default=1.0 / L, beta=beta)


@pytest.fixture(scope='module')
def lasso():
    rng = np.random.default_rng(2)
    N, m, n, lambd = 5, 20, 30, 0.1
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    B = rng.standard_normal((N, m))
    L = float(np.linalg.eigvalsh(A.T @ A).max())
    f_opt = rng.uniform(-1.0, 1.0, size=N)
    return dict(A=A, B=B, lambd=lambd, L=L, f_opt=f_opt, t_default=1.5 / L)


@pytest.fixture(scope='module')
def pdhg():
    rng = np.random.default_rng(3)
    M = Nn = 8
    N = 3
    mask = rng.random((M, Nn)) >= 0.2
    known = np.flatnonzero(mask)
    S = known.size
    mats = [extract_constraint_matrices(known, rng.uniform(0, LP_UPPER, S), M, Nn)
            for _ in range(N)]
    m1 = mats[0].G.shape[0]
    K = sp.vstack([mats[0].G, mats[0].A], format='csr')
    n = K.shape[1]
    Qmat = np.stack([np.concatenate([mt.h, mt.b]) for mt in mats], axis=1)
    Xstar = rng.uniform(0, LP_UPPER, (n, N))
    Ystar = rng.standard_normal((K.shape[0], N))
    Ystar[:m1] = np.abs(Ystar[:m1])
    x0 = 0.5 * LP_UPPER * np.ones(n)
    y0 = np.concatenate([0.1 * np.ones(m1), np.zeros(S)])
    normK = float(spla.svds(K, k=1, which='LM', return_singular_vectors=False)[0])
    return dict(mats=mats, K=K, c=mats[0].c, Qmat=Qmat, l=mats[0].l, u=mats[0].u,
                m1=m1, x0=x0, y0=y0, Xstar=Xstar, Ystar=Ystar,
                eta_default=0.5 / (1.2 * normK), normK=normK, N=N)


def _run(name, p, **kw):
    if name == 'quad':
        return gd_quad(p['Q'], p['z0'], t_default=p['t_default'], K_max=K_MAX, **kw)
    if name == 'logreg_gd':
        return gd_logreg(p['A'], p['b'], p['f_opt'], t_default=p['t_default'],
                         K_max=K_MAX, **kw)
    if name == 'logreg_fgm':
        return fgm_logreg(p['A'], p['b'], p['f_opt'], p['beta'],
                          t_default=p['t_default'], K_max=K_MAX, **kw)
    if name == 'lasso':
        return ista_lasso(p['A'], p['B'], p['f_opt'], p['lambd'],
                          t_default=p['t_default'], K_max=K_MAX, **kw)
    if name == 'pdhg':
        kw.pop('c', None)
        X0 = np.tile(p['x0'][:, None], (1, p['N']))
        Y0 = np.tile(p['y0'][:, None], (1, p['N']))
        return pdhg_lp(p['K'], p['c'], p['Qmat'], p['l'], p['u'], p['m1'], X0, Y0,
                       p['Xstar'], p['Ystar'], eta_default=p['eta_default'],
                       K_max=K_MAX, **kw)
    raise ValueError(name)


ALL = ['quad', 'logreg_gd', 'logreg_fgm', 'lasso', 'pdhg']


@pytest.fixture
def problems(quad, logreg, lasso, pdhg):
    return {'quad': quad, 'logreg_gd': logreg, 'logreg_fgm': logreg,
            'lasso': lasso, 'pdhg': pdhg}


# ---------------------------------------------------------------------------
# (a) growth = 1 reproduces the fixed-default-step method
# ---------------------------------------------------------------------------

def test_quad_growth1_matches_jax(quad):
    res = _run('quad', quad, growth=1.0)
    t = jnp.full(K_MAX, quad['t_default'])
    for i in range(quad['Q'].shape[0]):
        _, _, f_stack = problem_data_to_gd_trajectories(
            (t,), jnp.asarray(quad['Q'][i]), jnp.asarray(quad['z0'][i]),
            jnp.zeros(quad['Q'].shape[1]), 0.0, K_MAX, return_Gram_representation=False)
        np.testing.assert_allclose(res.losses[i], np.asarray(f_stack), rtol=RTOL, atol=ATOL)


def test_lasso_growth1_matches_jax(lasso):
    res = _run('lasso', lasso, growth=1.0)
    n = lasso['A'].shape[1]
    rng = np.random.default_rng(7)
    gamma = jnp.full(K_MAX, lasso['t_default'])
    for i in range(lasso['B'].shape[0]):
        x_opt = rng.standard_normal(n)        # arbitrary shift; ISTA is shift-equivariant
        out = problem_data_to_ista_trajectories(
            gamma, jnp.asarray(lasso['A']), jnp.asarray(lasso['B'][i]), jnp.zeros(n),
            jnp.asarray(x_opt), lasso['f_opt'][i], lasso['lambd'], K_MAX,
            return_Gram_representation=False)
        f1_iter, f2_iter = np.asarray(out[3]), np.asarray(out[4])
        np.testing.assert_allclose(res.losses[i], f1_iter + f2_iter, rtol=RTOL, atol=1e-10)


def test_logreg_growth1_matches_reference(logreg):
    data = {'A_batch': logreg['A'], 'b_batch': logreg['b'], 'f_opt_batch': logreg['f_opt']}
    res_gd = _run('logreg_gd', logreg, growth=1.0)
    res_fgm = _run('logreg_fgm', logreg, growth=1.0)
    for K in range(1, K_MAX + 1):
        t_vec = np.full(K, logreg['t_default'])
        np.testing.assert_allclose(res_gd.losses[:, K], blt.gd_losses(t_vec, data),
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(res_fgm.losses[:, K],
                                   blt.fgm_losses(t_vec, logreg['beta'][:K], data),
                                   rtol=RTOL, atol=ATOL)


def test_pdhg_growth1_matches_reference(pdhg):
    res = _run('pdhg', pdhg, growth=1.0)
    eta = np.full(K_MAX, pdhg['eta_default'])
    for i, mt in enumerate(pdhg['mats']):
        gaps = run_pdhg_capture_gaps(mt.c, mt.G, mt.h, mt.A, mt.b, mt.l, mt.u,
                                     pdhg['Xstar'][:, i], pdhg['Ystar'][:, i],
                                     pdhg['x0'], pdhg['y0'], eta, eta, np.ones(K_MAX))
        np.testing.assert_allclose(res.losses[i], gaps, rtol=RTOL, atol=1e-10)


# ---------------------------------------------------------------------------
# (b) an absurd candidate always falls back to the default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['quad', 'logreg_gd', 'logreg_fgm', 'lasso'])
def test_huge_growth_is_fixed_step(name, problems):
    p = problems[name]
    ref = _run(name, p, growth=1.0)
    res = _run(name, p, growth=1e6)
    assert not res.accepted.any()
    np.testing.assert_array_equal(res.steps, np.full_like(res.steps, res.t_default))
    np.testing.assert_allclose(res.losses, ref.losses, rtol=RTOL, atol=ATOL)


def test_pdhg_default_step_always_accepted(pdhg):
    """eta_default <= 0.5/||K|| < 1/||K|| <= eta_hat, so growth = 1 never falls back.

    The converse ("an absurd candidate is always rejected") is NOT a property
    of the PDLP rule: when the primal iterate is pinned at its box bounds,
    dx = 0 and the local condition holds for any eta. That is why PDHG is
    left out of test_huge_growth_is_fixed_step.
    """
    res = _run('pdhg', pdhg, growth=1.0)
    assert res.accepted.all()
    assert res.fallback_would_fail.sum() == 0
    assert 0.5 / pdhg['normK'] >= pdhg['eta_default']


# ---------------------------------------------------------------------------
# (c) monotone decrease for the descent methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['quad', 'logreg_gd', 'lasso'])
def test_monotone_decrease(name, problems):
    p = dict(problems[name])
    if name == 'lasso':
        p['t_default'] = 1.0 / p['L']   # the composite condition certifies t <= 1/L
    res = _run(name, p, growth=2.0)
    diff = np.diff(res.losses, axis=1)
    assert np.all(diff <= 1e-12 * (1.0 + np.abs(res.losses[:, :-1])))
    assert res.fallback_would_fail.sum() == 0
    assert 0.0 < res.accepted.mean() < 1.0, 'candidate never/always accepted; test is vacuous'


# ---------------------------------------------------------------------------
# (d) one K_max run contains every shorter horizon
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ALL)
def test_k_independence(name, problems):
    p = problems[name]
    long = _run(name, p, growth=2.0)
    if name == 'logreg_fgm':
        short = fgm_logreg(p['A'], p['b'], p['f_opt'], p['beta'][:5],
                           t_default=p['t_default'], K_max=5, growth=2.0)
    elif name == 'pdhg':
        X0 = np.tile(p['x0'][:, None], (1, p['N']))
        Y0 = np.tile(p['y0'][:, None], (1, p['N']))
        short = pdhg_lp(p['K'], p['c'], p['Qmat'], p['l'], p['u'], p['m1'], X0, Y0,
                        p['Xstar'], p['Ystar'], eta_default=p['eta_default'],
                        K_max=5, growth=2.0)
    else:
        fn = {'quad': lambda: gd_quad(p['Q'], p['z0'], t_default=p['t_default'], K_max=5, growth=2.0),
              'logreg_gd': lambda: gd_logreg(p['A'], p['b'], p['f_opt'], t_default=p['t_default'], K_max=5, growth=2.0),
              'lasso': lambda: ista_lasso(p['A'], p['B'], p['f_opt'], p['lambd'], t_default=p['t_default'], K_max=5, growth=2.0)}[name]
        short = fn()
    np.testing.assert_array_equal(short.losses, long.losses[:, :6])
    np.testing.assert_array_equal(short.steps, long.steps[:, :5])


def test_fgm_beta_prefix_property(logreg):
    b5 = get_nesterov_fgm_beta_sequence(0.0, logreg['L'], 5)
    b15 = get_nesterov_fgm_beta_sequence(0.0, logreg['L'], 15)
    np.testing.assert_allclose(b5, b15[:5], rtol=1e-14)


# ---------------------------------------------------------------------------
# (e) oracle cost accounting
# ---------------------------------------------------------------------------

def test_cost_quad(quad):
    ctr = {}
    res = _run('quad', quad, growth=2.0, counter=ctr)
    assert ctr['matvec'] == K_MAX + 1
    assert res.n_extra_matvec == 0


@pytest.mark.parametrize('name', ['logreg_gd', 'logreg_fgm'])
def test_cost_logreg(name, problems):
    ctr = {}
    res = _run(name, problems[name], growth=2.0, counter=ctr)
    assert ctr['matvec'] == 2 * K_MAX
    assert res.n_extra_matvec == 0


def test_cost_lasso(lasso):
    ctr = {}
    res = _run('lasso', lasso, growth=2.0, counter=ctr)
    iters_with_rejection = int(np.sum(~res.accepted.all(axis=0)))
    assert ctr['matvec'] == 2 * K_MAX + iters_with_rejection
    assert res.n_extra_matvec == int((~res.accepted).sum())


def test_cost_pdhg(pdhg):
    ctr = {}
    res = _run('pdhg', pdhg, growth=2.0, counter=ctr)
    iters_with_rejection = int(np.sum(~res.accepted.all(axis=0)))
    assert ctr['matvec'] == 1 + 2 * K_MAX + iters_with_rejection
    assert res.n_extra_matvec == int((~res.accepted).sum())
    assert res.fallback_would_fail.sum() == 0      # eta_default <= 0.5/||K|| always passes
