"""Integration / smoke tests for LogRegProblemModule (GD + FGM).

Every test runs for BOTH instance distributions -- the synthetic Gaussian
generator and real standardized german.numer subsamples (cfg.data) -- because
the point of the `data` switch is that everything downstream is shared.

Verifies:
  1. The module instantiates from logreg.yaml and samples consistent batches
     (z0 = -x_opt, finite f_opt) with fresh tiny sampling (no data files).
  2. Per-sample (G, F) from `compute_batched_trajectories` satisfies every
     constraint of `construct_gd_pep_data` / `construct_fgm_pep_data` at
     L = max per-sample smoothness, mu = 0, R = max ||x_opt||.
  3. `create_metric_fn` with obj_val indexes the shifted f-stack correctly,
     including metric_fn(K) = f(x_K) - f_opt for FGM.
  4. The real-data OOD axis is the exact L-shift it claims to be.

The real-data cases need the LIBSVM file in the cache ($DRO_PEP_DATA or
<repo>/data/libsvm); they skip when it is absent, since compute nodes and CI
have no network.

Run: pytest tests/test_logreg_problem_module.py -v
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from hydra import initialize, compose

jax.config.update('jax_enable_x64', True)

from learning_experiment_classes.logreg import (
    LogRegProblemModule,
    compute_logreg_L_single,
)
from learning.pep_constructions import construct_gd_pep_data, construct_fgm_pep_data
from experiment_classes.logreg_data import DATASETS, data_cache_dir

N_TINY = 3
K_TEST = 4
REAL_DATASET = 'german.numer'
DATA_MODES = ['synth', REAL_DATASET]


def _skip_if_uncached(data):
    if data == 'synth':
        return
    path = data_cache_dir() / DATASETS[data][0]
    if not path.is_file():
        pytest.skip(f"{data} not cached at {path}; "
                    f"run `python -m experiment_classes.logreg_data {data}`")


def _load_logreg_cfg(**overrides):
    override_list = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(version_base='1.2', config_path='../configs_learning'):
        cfg = compose(config_name='logreg', overrides=override_list)
    return cfg


def _tiny_module_and_batch(alg, data='synth'):
    _skip_if_uncached(data)
    cfg = _load_logreg_cfg(
        alg=alg,
        data=data,
        data_source_dir='null',
        training_sample_N=N_TINY,
        N=N_TINY,
        out_of_sample_val_N=N_TINY,
    )
    module = LogRegProblemModule(cfg)
    problem_data, ground_truth = module._sample_fresh_batch(
        jax.random.PRNGKey(123), N_TINY, 'in'
    )
    return cfg, module, problem_data, ground_truth


# ---------------------------------------------------------------------------
# Test 1: instantiation + sampling invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('data', DATA_MODES)
def test_module_instantiates_and_samples(data):
    _cfg, module, problem_data, ground_truth = _tiny_module_and_batch(
        'vanilla_gd', data)

    A = problem_data['A_batch']
    b = problem_data['b_batch']
    z0 = problem_data['z0_batch']
    x_opt = ground_truth['x_opt_batch']
    f_opt = ground_truth['f_opt_batch']

    # Dimensions come from the module, not the config: for real data they are
    # the dataset's (m_sub rows, n_features + intercept columns).
    assert A.shape == (N_TINY, module.N_data_val, module.n_val)
    assert b.shape == (N_TINY, module.N_data_val)
    assert z0.shape == (N_TINY, module.n_val)
    assert x_opt.shape == (N_TINY, module.n_val)
    assert f_opt.shape == (N_TINY,)

    # Intercept column and binary labels
    assert jnp.allclose(A[:, :, -1], 1.0)
    assert jnp.all((b == 0.0) | (b == 1.0))

    # x0 = 0 in original coordinates
    assert jnp.allclose(z0, -x_opt)
    assert jnp.all(jnp.isfinite(f_opt))

    # delta guard: this benchmark is unregularized
    module.validate_config()
    assert module.delta_val == 0.0


# ---------------------------------------------------------------------------
# Test 2: per-sample (G, F) satisfies all PEP constraints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('data', DATA_MODES)
@pytest.mark.parametrize('alg', ['vanilla_gd', 'nesterov_fgm'])
def test_gram_satisfies_pep_constraints(alg, data):
    _cfg, module, problem_data, ground_truth = _tiny_module_and_batch(alg, data)

    A_batch = problem_data['A_batch']
    x_opt_batch = ground_truth['x_opt_batch']

    L_vals = jax.vmap(lambda A: compute_logreg_L_single(A, 0.0))(A_batch)
    L = float(jnp.max(L_vals))
    R = float(jnp.max(jnp.linalg.norm(x_opt_batch, axis=1)))
    mu = 0.0

    stepsizes = module.get_initial_stepsizes(alg, K_TEST, L, mu)
    traj_fn = module.get_trajectory_fn(alg)

    batched_data = {
        'A': A_batch,
        'b': problem_data['b_batch'],
        'z0': problem_data['z0_batch'],
        'x_opt': x_opt_batch,
        'f_opt': ground_truth['f_opt_batch'],
    }
    G_batch, F_batch = module.compute_batched_trajectories(
        stepsizes, batched_data, {}, traj_fn, K_TEST
    )
    assert G_batch.shape == (N_TINY, K_TEST + 2, K_TEST + 2)
    assert F_batch.shape == (N_TINY, K_TEST + 1)

    if alg == 'vanilla_gd':
        pep_data = construct_gd_pep_data(
            stepsizes[0], mu, L, R, K_TEST, 'obj_val', composition_type='final'
        )
    else:
        pep_data = construct_fgm_pep_data(
            stepsizes[0], stepsizes[1], mu, L, R, K_TEST, 'obj_val',
            composition_type='final'
        )
    _, _, A_vals, b_vals, c_vals = pep_data[:5]
    A_vals = np.array(A_vals)
    b_vals = np.array(b_vals)
    c_vals = np.array(c_vals)

    for i in range(N_TINY):
        G = np.array(G_batch[i])
        F = np.array(F_batch[i])
        vals = np.einsum('mjk,kj->m', A_vals, G) + b_vals @ F + c_vals
        max_violation = float(np.max(vals))
        assert max_violation <= 1e-6, (
            f"{alg} sample {i}: PEP constraint violated by {max_violation:.3e}"
        )


# ---------------------------------------------------------------------------
# Test 3: obj_val metric fn indexes the shifted f-stack
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('data', DATA_MODES)
@pytest.mark.parametrize('alg', ['vanilla_gd', 'nesterov_fgm'])
def test_metric_fn_obj_val(alg, data):
    _cfg, module, problem_data, ground_truth = _tiny_module_and_batch(alg, data)

    L_vals = jax.vmap(
        lambda A: compute_logreg_L_single(A, 0.0)
    )(problem_data['A_batch'])
    L = float(jnp.max(L_vals))
    stepsizes = module.get_initial_stepsizes(alg, K_TEST, L, 0.0)
    traj_fn = module.get_trajectory_fn(alg)

    i = 0
    trajectories = traj_fn(
        stepsizes,
        problem_data['A_batch'][i],
        problem_data['b_batch'][i],
        problem_data['z0_batch'][i],
        ground_truth['x_opt_batch'][i],
        ground_truth['f_opt_batch'][i],
        K_TEST,
        return_Gram_representation=False,
    )
    f_stack = trajectories[2]
    assert f_stack.shape == (K_TEST + 1,)

    metric_fn = module.create_metric_fn(trajectories, {}, {}, 'obj_val')
    for k in range(K_TEST + 1):
        assert float(metric_fn(k)) == pytest.approx(float(f_stack[k]))

    # f-gap at the start is f(x0) - f_opt >= 0, and the algorithms should
    # not end above where they started with a 1/L-based initialization.
    assert float(metric_fn(0)) >= 0.0
    assert float(metric_fn(K_TEST)) <= float(metric_fn(0))
    # Final metric must be the x_K point (not y_{K-1}) for FGM: it is the
    # last f-stack entry by construction of the trajectory function.
    assert float(metric_fn(K_TEST)) == pytest.approx(float(f_stack[-1]))


# ---------------------------------------------------------------------------
# Test 4: the real-data OOD axis is an exact L-shift
# ---------------------------------------------------------------------------

def test_real_ood_is_exact_L_shift():
    """Scaling the design matrix by `mult` multiplies L by mult^2 and leaves f*
    alone, which is the whole justification for reusing `ood_std_multiplier` on
    real data: it reproduces the synthetic OOD design's L-shift exactly rather
    than approximately.

    L and A are asserted tightly because they are algebraic. ||x*|| is NOT,
    even though x -> x/mult is exact in theory: some german.numer subsamples
    are near-separable, so the objective has a very flat valley and CLARABEL
    stops at different points along it in the two coordinate systems. Observed
    at seed 12345: one instance of three had f* agreeing to 2.7e-9 and both
    gradient norms below 3e-6, yet ||x*|| differing by 9%. That is a property
    of the instance distribution, not of the scaling -- and it is benign for
    PEP, where R = max ||x*|| only has to be an upper bound.
    """
    _skip_if_uncached(REAL_DATASET)
    cfg = _load_logreg_cfg(
        data=REAL_DATASET, data_source_dir='null',
        training_sample_N=N_TINY, N=N_TINY, out_of_sample_val_N=N_TINY,
    )
    module = LogRegProblemModule(cfg)
    mult = module._ood_multiplier
    assert mult != 1.0, 'ood_std_multiplier must actually shift the distribution'

    seed = 12345
    in_data, in_gt = module._sample_fresh_batch(None, N_TINY, 'in', seed=seed)
    ood_data, ood_gt = module._sample_fresh_batch(None, N_TINY, 'ood', seed=seed)

    # Same seed and an unchanged rejection path => the same row subsets, so the
    # two batches are the same instances in different coordinates.
    assert jnp.allclose(ood_data['A_batch'], mult * in_data['A_batch'])
    assert jnp.array_equal(ood_data['b_batch'], in_data['b_batch'])

    L_in = jax.vmap(lambda A: compute_logreg_L_single(A, 0.0))(in_data['A_batch'])
    L_ood = jax.vmap(lambda A: compute_logreg_L_single(A, 0.0))(ood_data['A_batch'])
    assert jnp.allclose(L_ood, (mult ** 2) * L_in, rtol=1e-10)

    # f* is invariant: the OOD set is genuinely the same optimization problems,
    # so the shift is in the algorithm's geometry, not in the objective values.
    # This holds to solver accuracy even where x* itself does not.
    assert jnp.allclose(ood_gt['f_opt_batch'], in_gt['f_opt_batch'], atol=1e-7)

    # ||x*|| shrinks by mult in the well-determined majority; assert it loosely
    # so a single flat-valley instance does not fail the suite (see docstring).
    r_in = jnp.linalg.norm(in_gt['x_opt_batch'], axis=1)
    r_ood = jnp.linalg.norm(ood_gt['x_opt_batch'], axis=1)
    assert float(jnp.median(r_ood / (r_in / mult))) == pytest.approx(1.0, abs=0.02)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
