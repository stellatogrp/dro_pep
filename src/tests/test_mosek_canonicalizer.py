"""Tests that the direct MOSEK backend matches the Clarabel backend.

Builds small logreg DRO reformulations (both algorithms x both measures)
and checks that DROReformulator with wrapper='mosek' returns the same
optimal value as wrapper='clarabel'. Skips if MOSEK or its license is
unavailable.

Run: pytest tests/test_mosek_canonicalizer.py -v
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from hydra import initialize, compose

jax.config.update('jax_enable_x64', True)

mosek = pytest.importorskip('mosek')

from experiment_classes import logreg_data as lrd
from learning.pep_constructions import (
    construct_gd_pep_data, construct_fgm_pep_data, pep_data_to_numpy,
)
from learning.trajectories import logreg_gd_trajectories, logreg_fgm_trajectories
from learning.acceleration_stepsizes import jax_get_nesterov_fgm_beta_sequence
from reformulator.dro_reformulator import DROReformulator
from reformulator.canonicalizers.clarabel_canonicalizer import symm_vectorize
from reformulator.canonicalizers.mosek_canonicalizer import (
    svec_clarabel_to_mosek_perm,
)

N_TINY = 3
K_TEST = 4
M_SUB = 300
EPS = 1e-2
ALPHA = 0.1


def _load_cfg(**overrides):
    override_list = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(version_base='1.2', config_path='../configs'):
        cfg = compose(config_name='logreg', overrides=override_list)
    return cfg


@pytest.fixture(scope='module')
def tiny_problem():
    cfg = _load_cfg(m_sub=M_SUB, N=N_TINY)
    A_full, b_full = lrd.load_dataset('a9a', intercept=True)
    rng = np.random.default_rng(0)
    instances, _ = lrd.sample_instance_batch(rng, A_full, b_full, cfg, N_TINY)
    return cfg, instances


def _build(cfg, instances, alg):
    mu = float(cfg.delta)
    L = float(max(inst[4] for inst in instances))
    R = float(max(np.linalg.norm(inst[2]) for inst in instances))
    t = cfg.eta / L
    t_vec = jnp.full(K_TEST, t)

    if alg == 'grad_desc':
        stp = (t_vec,)
        pep_data = pep_data_to_numpy(construct_gd_pep_data(
            t_vec, mu, L, R, K_TEST, cfg.pep_obj, composition_type='final'))
        traj_fn = logreg_gd_trajectories
    else:
        beta = jax_get_nesterov_fgm_beta_sequence(mu, L, K_TEST)
        stp = (t_vec, beta)
        pep_data = pep_data_to_numpy(construct_fgm_pep_data(
            t_vec, beta, mu, L, R, K_TEST, cfg.pep_obj,
            composition_type='final'))
        traj_fn = logreg_fgm_trajectories

    samples = []
    for A, b, x_opt, f_opt, _L_i in instances:
        z0 = -jnp.asarray(x_opt)
        G, F = traj_fn(stp, jnp.asarray(A), jnp.asarray(b), z0,
                       jnp.asarray(x_opt), f_opt, cfg.delta, K_TEST,
                       return_Gram_representation=True)
        samples.append((np.asarray(G), np.asarray(F)))
    return pep_data, samples


def _solve(pep_data, samples, measure, wrapper, cfg):
    DR = DROReformulator(pep_data, samples, measure, wrapper,
                         precond=cfg.precond, precond_type=cfg.precond_type,
                         mro_clusters=None)
    DR.set_params(eps=EPS, alpha=ALPHA)
    try:
        out = DR.solve()
    except mosek.Error as e:
        if 'license' in str(e).lower():
            pytest.skip(f'MOSEK license unavailable: {e}')
        raise
    return out['obj'], DR


def test_svec_permutation_roundtrip():
    rng = np.random.default_rng(1)
    for n in [2, 3, 7, 26]:
        S = rng.standard_normal((n, n))
        S = S + S.T
        cl = symm_vectorize(S.copy(), np.sqrt(2.))
        mo = np.array([S[r, c] if r == c else np.sqrt(2.) * S[r, c]
                       for c in range(n) for r in range(c, n)])
        perm = svec_clarabel_to_mosek_perm(n)
        assert np.allclose(cl[perm], mo)


@pytest.mark.parametrize('measure', ['expectation', 'cvar'])
@pytest.mark.parametrize('alg', ['grad_desc', 'nesterov_fgm'])
def test_mosek_matches_clarabel_objective(tiny_problem, alg, measure):
    cfg, instances = tiny_problem
    pep_data, samples = _build(cfg, instances, alg)

    obj_cl, _ = _solve(pep_data, samples, measure, 'clarabel', cfg)
    obj_mo, _ = _solve(pep_data, samples, measure, 'mosek', cfg)

    # two different interior-point cores at default tolerances agree to
    # ~1e-5 relative on these problems; 1e-4 leaves headroom without
    # masking structural errors (those show up at O(1))
    assert obj_mo == pytest.approx(obj_cl, rel=1e-4, abs=1e-8), (
        f'{alg}/{measure}: mosek {obj_mo} != clarabel {obj_cl}')


@pytest.mark.parametrize('alg', ['grad_desc', 'nesterov_fgm'])
def test_mosek_matches_clarabel_solution_vector(tiny_problem, alg):
    # same variable layout: the certified bound recomputed from the primal
    # solution must agree, not just the solver-reported objective
    cfg, instances = tiny_problem
    pep_data, samples = _build(cfg, instances, alg)

    _, DR_cl = _solve(pep_data, samples, 'expectation', 'clarabel', cfg)
    _, DR_mo = _solve(pep_data, samples, 'expectation', 'mosek', cfg)

    sol_cl = DR_cl.extract_solution()
    sol_mo = DR_mo.extract_solution()
    assert sol_mo['lambda'] == pytest.approx(sol_cl['lambda'],
                                             rel=1e-3, abs=1e-6)
    assert np.mean(sol_mo['s']) == pytest.approx(np.mean(sol_cl['s']),
                                                 rel=1e-4, abs=1e-8)
