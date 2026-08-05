"""Tests for the real-data logistic regression certification experiment.

Verifies:
  1. LIBSVM loading invariants (binary labels, intercept column).
  2. Instance sampling with the separability guard yields solvable instances.
  3. Per-instance (G, F) Gram trajectories satisfy every interpolation
     constraint of construct_gd_pep_data / construct_fgm_pep_data at the
     family constants (mu = 0, L = max instance L, R = max ||x_opt||).
  4. The Gram-side PEP objective equals the numpy simulate_alg gap at K
     (samples stage and DRO stage bound the same quantity).

Requires the a9a dataset (downloaded on first run, ~2MB, cached under
$DRO_PEP_DATA or <repo>/data/libsvm).

Run: pytest tests/test_logreg_certification.py -v
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from hydra import initialize, compose

jax.config.update('jax_enable_x64', True)

from experiment_classes import logreg as lr
from experiment_classes import logreg_data as lrd
from learning.pep_constructions import (
    construct_gd_pep_data, construct_fgm_pep_data,
)
from learning.trajectories import logreg_gd_trajectories, logreg_fgm_trajectories
from learning.acceleration_stepsizes import jax_get_nesterov_fgm_beta_sequence

N_TINY = 3
K_TEST = 4
M_SUB = 300


def _load_cfg(**overrides):
    override_list = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(version_base='1.2', config_path='../configs'):
        cfg = compose(config_name='logreg', overrides=override_list)
    return cfg


@pytest.fixture(scope='module')
def dataset():
    return lrd.load_dataset('a9a', intercept=True)


@pytest.fixture(scope='module')
def tiny_instances(dataset):
    cfg = _load_cfg(m_sub=M_SUB, N=N_TINY)
    A_full, b_full = dataset
    rng = np.random.default_rng(0)
    instances, _ = lrd.sample_instance_batch(rng, A_full, b_full, cfg, N_TINY)
    return cfg, instances


def test_dataset_invariants(dataset):
    A, b = dataset
    assert A.shape[1] == 124  # 123 features + intercept
    assert np.allclose(A[:, -1], 1.0)
    assert set(np.unique(b)) <= {0.0, 1.0}
    assert A.shape[0] > 30000


def test_instances_solvable(tiny_instances):
    cfg, instances = tiny_instances
    for A, b, x_opt, f_opt, L in instances:
        assert A.shape == (M_SUB, 124)
        assert np.isfinite(f_opt)
        assert np.linalg.norm(x_opt) <= cfg.xopt_norm_max
        assert L > 0
        # first-order optimality of the unregularized solve
        g = lr.logreg_grad(A, b, x_opt, cfg.delta)
        assert np.linalg.norm(g) <= 1e-5


@pytest.mark.parametrize('alg', ['grad_desc', 'nesterov_fgm'])
def test_gram_satisfies_pep_constraints(tiny_instances, alg):
    cfg, instances = tiny_instances
    mu = float(cfg.delta)
    L = float(max(inst[4] for inst in instances))
    R = float(max(np.linalg.norm(inst[2]) for inst in instances))
    t = cfg.eta / L
    t_vec = jnp.full(K_TEST, t)

    if alg == 'grad_desc':
        stp = (t_vec,)
        pep_data = construct_gd_pep_data(
            t_vec, mu, L, R, K_TEST, 'obj_val', composition_type='final')
        traj_fn = logreg_gd_trajectories
    else:
        beta = jax_get_nesterov_fgm_beta_sequence(mu, L, K_TEST)
        stp = (t_vec, beta)
        pep_data = construct_fgm_pep_data(
            t_vec, beta, mu, L, R, K_TEST, 'obj_val', composition_type='final')
        traj_fn = logreg_fgm_trajectories

    _, _, A_vals, b_vals, c_vals = pep_data[:5]
    A_vals, b_vals, c_vals = np.array(A_vals), np.array(b_vals), np.array(c_vals)

    for i, (A, b, x_opt, f_opt, _L_i) in enumerate(instances):
        z0 = -jnp.asarray(x_opt)
        G, F = traj_fn(stp, jnp.asarray(A), jnp.asarray(b), z0,
                       jnp.asarray(x_opt), f_opt, cfg.delta, K_TEST,
                       return_Gram_representation=True)
        G, F = np.array(G), np.array(F)
        assert G.shape == (K_TEST + 2, K_TEST + 2)
        assert F.shape == (K_TEST + 1,)
        vals = np.einsum('mjk,kj->m', A_vals, G) + b_vals @ F + c_vals
        max_violation = float(np.max(vals))
        assert max_violation <= 1e-6, (
            f"{alg} instance {i}: PEP constraint violated by {max_violation:.3e}")


@pytest.mark.parametrize('pep_obj', ['obj_val', 'grad_sq_norm'])
@pytest.mark.parametrize('alg', ['grad_desc', 'nesterov_fgm'])
def test_gram_objective_matches_simulation(tiny_instances, alg, pep_obj):
    cfg, instances = tiny_instances
    cfg.alg = alg
    cfg.pep_obj = pep_obj
    cfg.K_max = K_TEST
    mu = float(cfg.delta)
    cfg.L = float(max(inst[4] for inst in instances))
    R = float(max(np.linalg.norm(inst[2]) for inst in instances))
    t = cfg.eta / cfg.L
    t_vec = jnp.full(K_TEST, t)

    if alg == 'grad_desc':
        stp = (t_vec,)
        pep_data = construct_gd_pep_data(
            t_vec, mu, cfg.L, R, K_TEST, pep_obj, composition_type='final')
        traj_fn = logreg_gd_trajectories
    else:
        beta = jax_get_nesterov_fgm_beta_sequence(mu, cfg.L, K_TEST)
        stp = (t_vec, beta)
        pep_data = construct_fgm_pep_data(
            t_vec, beta, mu, cfg.L, R, K_TEST, pep_obj, composition_type='final')
        traj_fn = logreg_fgm_trajectories

    A_obj = np.array(pep_data[0])
    b_obj = np.array(pep_data[1])

    for A, b, x_opt, f_opt, _L_i in instances:
        z0 = -jnp.asarray(x_opt)
        G, F = traj_fn(stp, jnp.asarray(A), jnp.asarray(b), z0,
                       jnp.asarray(x_opt), f_opt, cfg.delta, K_TEST,
                       return_Gram_representation=True)
        emp = float(np.trace(A_obj @ np.array(G)) + b_obj @ np.array(F))
        gap_sim = lr.simulate_alg(cfg, A, b, x_opt, f_opt, t)[-1]
        assert emp == pytest.approx(gap_sim, rel=1e-6, abs=1e-10), (
            f"{alg}: Gram objective {emp} != simulated gap {gap_sim}")
