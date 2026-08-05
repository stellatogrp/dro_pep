"""Regression tests for the sparse conic assembly.

The golden .npz fixtures in tests/data/ were captured from the original
dense-scratch ClarabelCanonicalizer (before the _SparseRows rewrite) at
K=3, N=2, eps=1e-2, alpha=0.1 on a9a instances (rng seed 0). The sparse
assembly must reproduce A, b, q, and the cone list exactly.

Run: pytest tests/test_canonicalizer_assembly.py -v
"""
import os

import numpy as np
import scipy.sparse as spa
import jax
import jax.numpy as jnp
import pytest
from hydra import initialize, compose

jax.config.update('jax_enable_x64', True)

from experiment_classes import logreg_data as lrd
from learning.pep_constructions import (
    construct_gd_pep_data, construct_fgm_pep_data, pep_data_to_numpy,
)
from learning.trajectories import logreg_gd_trajectories, logreg_fgm_trajectories
from learning.acceleration_stepsizes import jax_get_nesterov_fgm_beta_sequence
from reformulator.dro_reformulator import DROReformulator

K_GOLD = 3
N_GOLD = 2
EPS_GOLD = 1e-2
ALPHA_GOLD = 0.1
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='module')
def gold_instances():
    with initialize(version_base='1.2', config_path='../configs'):
        cfg = compose(config_name='logreg',
                      overrides=['m_sub=300', f'N={N_GOLD}'])
    A_full, b_full = lrd.load_dataset('a9a', intercept=True)
    rng = np.random.default_rng(0)
    instances, _ = lrd.sample_instance_batch(rng, A_full, b_full, cfg, N_GOLD)
    return cfg, instances


def _build_canon(cfg, instances, alg, measure):
    mu = float(cfg.delta)
    L = float(max(i[4] for i in instances))
    R = float(max(np.linalg.norm(i[2]) for i in instances))
    t_vec = jnp.full(K_GOLD, cfg.eta / L)

    if alg == 'grad_desc':
        stp = (t_vec,)
        pep_data = pep_data_to_numpy(construct_gd_pep_data(
            t_vec, mu, L, R, K_GOLD, cfg.pep_obj, composition_type='final'))
        traj_fn = logreg_gd_trajectories
    else:
        beta = jax_get_nesterov_fgm_beta_sequence(mu, L, K_GOLD)
        stp = (t_vec, beta)
        pep_data = pep_data_to_numpy(construct_fgm_pep_data(
            t_vec, beta, mu, L, R, K_GOLD, cfg.pep_obj,
            composition_type='final'))
        traj_fn = logreg_fgm_trajectories

    samples = []
    for A, b, x_opt, f_opt, _ in instances:
        z0 = -jnp.asarray(x_opt)
        G, F = traj_fn(stp, jnp.asarray(A), jnp.asarray(b), z0,
                       jnp.asarray(x_opt), f_opt, cfg.delta, K_GOLD,
                       return_Gram_representation=True)
        samples.append((np.asarray(G), np.asarray(F)))

    DR = DROReformulator(pep_data, samples, measure, 'clarabel',
                         precond=cfg.precond, precond_type=cfg.precond_type)
    DR.set_params(eps=EPS_GOLD, alpha=ALPHA_GOLD)
    return DR.canon


@pytest.mark.parametrize('measure', ['expectation', 'cvar'])
@pytest.mark.parametrize('alg', ['grad_desc', 'nesterov_fgm'])
def test_assembly_matches_golden(gold_instances, alg, measure):
    cfg, instances = gold_instances
    canon = _build_canon(cfg, instances, alg, measure)
    if measure == 'expectation':
        A_new, b_new, cones = canon.A, canon.b, canon.cones
    else:
        A_new, b_new, cones = canon.A_full, canon.b_full, canon.cones_full
    A_new = spa.csc_matrix(A_new)

    gold = np.load(os.path.join(
        DATA_DIR, f'canon_golden_{alg}_{measure}.npz'), allow_pickle=False)
    A_gold = spa.csc_matrix(
        (gold['A_data'], gold['A_indices'], gold['A_indptr']),
        shape=tuple(gold['A_shape']))

    assert A_new.shape == A_gold.shape
    diff = abs(A_new - A_gold)
    max_diff = diff.max() if diff.nnz else 0.0
    assert max_diff == 0.0, f'{alg}/{measure}: A differs by {max_diff}'
    np.testing.assert_array_equal(np.asarray(b_new), gold['b'])
    np.testing.assert_array_equal(np.asarray(canon.q), gold['q'])
    assert [repr(c) for c in cones] == list(gold['cones'])
