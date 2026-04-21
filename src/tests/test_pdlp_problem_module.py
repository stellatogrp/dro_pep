"""
Integration / smoke tests for PDLPProblemModule.

Verifies:
  1. The module instantiates from pdlp.yaml without errors.
  2. `sample_training_batch` produces the expected dict keys and shapes.
  3. Per-sample (G, F) from `compute_batched_trajectories` satisfies every
     `construct_chambolle_pock_pep_data` constraint — same structural
     assertions exercised by `test_chambolle_pock_facility_location.py`.

These tests pin the PDLP problem-module wiring to the VERIFIED-CORRECT CP
trajectory + PEP construction. Any future regression in either path will
surface here.
"""
import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from hydra import initialize, compose

jax.config.update('jax_enable_x64', True)

from learning_experiment_classes.pdlp import PDLPProblemModule
from learning.pep_constructions import construct_chambolle_pock_pep_data
from tests.test_chambolle_pock_interpolation import (
    eval_scalar_constraint,
    eval_psd_block,
)


# ---------------------------------------------------------------------------
# Config helper: load pdlp.yaml via Hydra
# ---------------------------------------------------------------------------

def _load_pdlp_cfg(**overrides):
    """Load pdlp.yaml from configs_learning with optional overrides.

    Uses a small instance (2x3) by default to keep tests fast.
    """
    override_list = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(version_base='1.2', config_path='../configs_learning'):
        cfg = compose(config_name='pdlp', overrides=override_list)
    return cfg


# ---------------------------------------------------------------------------
# Test 1: module instantiates
# ---------------------------------------------------------------------------

def test_pdlp_module_instantiates():
    cfg = _load_pdlp_cfg(
        n_facilities=2, n_customers=3,
        precond_sample_size=5,
    )
    module = PDLPProblemModule(cfg)
    assert module.n_vars == 2 + 2 * 3
    assert module.m1 == 2 + 2 * 3
    assert module.m2 == 3
    assert module.M_val > 0
    assert module.R_val > 0
    # Validation shouldn't raise
    module.validate_config()


# ---------------------------------------------------------------------------
# Test 2: sample_training_batch shapes
# ---------------------------------------------------------------------------

def test_pdlp_sample_training_batch_shapes():
    cfg = _load_pdlp_cfg(
        n_facilities=2, n_customers=3,
        precond_sample_size=5,
    )
    module = PDLPProblemModule(cfg)

    key = jax.random.PRNGKey(0)
    N = 3
    problem_data, ground_truth = module.sample_training_batch(key, N)

    assert set(problem_data.keys()) == {'c_batch', 'K_mat_batch', 'q_batch'}
    assert set(ground_truth.keys()) == {'x_opt_batch', 'y_opt_batch'}

    n_vars = module.n_vars
    m = module.m_total

    assert problem_data['c_batch'].shape == (N, n_vars)
    assert problem_data['K_mat_batch'].shape == (N, m, n_vars)
    assert problem_data['q_batch'].shape == (N, m)
    assert ground_truth['x_opt_batch'].shape == (N, n_vars)
    assert ground_truth['y_opt_batch'].shape == (N, m)


# ---------------------------------------------------------------------------
# Test 3: Per-sample (G, F) passes CP PEP constraints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('K_max', [1, 3])
def test_pdlp_batched_trajectory_satisfies_pep(K_max):
    """Every sample's (G_i, F_i) satisfies every CP PEP scalar + PSD constraint."""
    cfg = _load_pdlp_cfg(
        n_facilities=2, n_customers=3,
        precond_sample_size=5,
        K_max=[K_max],
    )
    module = PDLPProblemModule(cfg)

    key = jax.random.PRNGKey(1)
    N = 2
    problem_data, ground_truth = module.sample_training_batch(key, N)

    traj_fn = module.get_trajectory_fn('cp')
    L_val, mu_val, R_val = module.compute_L_mu_R()
    stepsizes = module.get_initial_stepsizes('cp', K_max, L_val, mu_val)

    batched_data = {
        'c': problem_data['c_batch'],
        'K_mat': problem_data['K_mat_batch'],
        'q': problem_data['q_batch'],
        'x_opt': ground_truth['x_opt_batch'],
        'y_opt': ground_truth['y_opt_batch'],
    }
    G_batch, F_batch = module.compute_batched_trajectories(
        stepsizes, batched_data, {}, traj_fn, K_max,
    )
    G_batch = np.asarray(G_batch)
    F_batch = np.asarray(F_batch)

    # Build PEP data at the module-level (L_val, R_val)
    # Note: stepsizes may be jnp vectors; unpack cleanly.
    tau, sigma, theta = stepsizes
    pep_data = construct_chambolle_pock_pep_data(
        tau=tau, sigma=sigma, theta=theta,
        M=L_val, R=R_val, K_max=K_max,
    )
    (A_obj, b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = [
        np.asarray(x) if not isinstance(x, list) else [np.asarray(a) for a in x]
        for x in pep_data
    ]

    num_scalar = A_vals.shape[0]
    eps = 1e-5  # slightly looser than the pure-facility test due to cross-sample scale diffs

    for i in range(N):
        G = G_batch[i]
        F = F_batch[i]

        # Scalar constraints, excluding IC (last row = trajectory-dependent radius)
        max_viol = -np.inf
        for j in range(num_scalar - 1):
            v = eval_scalar_constraint(A_vals[j], b_vals[j], c_vals[j], G, F)
            max_viol = max(max_viol, v)
        assert max_viol <= eps, \
            f"Sample {i}: scalar constraint violation {max_viol:.3e}"

        # PSD blocks
        for idx in range(len(PSD_A_vals)):
            H = eval_psd_block(PSD_A_vals[idx], PSD_b_vals[idx],
                               PSD_c_vals[idx], G, F)
            min_eig = float(np.min(np.linalg.eigvalsh(H)))
            assert min_eig >= -eps, \
                f"Sample {i}: PSD block {idx} not PSD, min eig {min_eig:.3e}"


# ---------------------------------------------------------------------------
# Test 4: Metric function returns sensible gap values
# ---------------------------------------------------------------------------

def test_pdlp_metric_fn_duality_gap():
    cfg = _load_pdlp_cfg(
        n_facilities=2, n_customers=3,
        precond_sample_size=5,
    )
    module = PDLPProblemModule(cfg)

    key = jax.random.PRNGKey(2)
    problem_data, ground_truth = module.sample_training_batch(key, 1)

    # Use 1 sample, K_max=3
    K_max = 3
    traj_fn = module.get_trajectory_fn('cp')
    L_val, mu_val, R_val = module.compute_L_mu_R()
    stepsizes = module.get_initial_stepsizes('cp', K_max, L_val, mu_val)

    # Call traj_fn on single sample (extract batch dim).
    c = problem_data['c_batch'][0]
    K_mat = problem_data['K_mat_batch'][0]
    q = problem_data['q_batch'][0]
    x_opt = ground_truth['x_opt_batch'][0]
    y_opt = ground_truth['y_opt_batch'][0]

    trajectories = traj_fn(stepsizes, c, K_mat, q, x_opt, y_opt, K_max,
                           return_Gram_representation=False)

    prob_data = {'c': c, 'K_mat': K_mat, 'q': q}
    gt = {'x_opt': x_opt, 'y_opt': y_opt}
    metric_fn = module.create_metric_fn(trajectories, prob_data, gt, pep_obj='obj_val')

    # gap at K_max should be a finite scalar, ideally >= 0 if we're near saddle
    gap_K = float(metric_fn(K_max))
    print(f"\nDuality gap at K={K_max}: {gap_K:.6e}")
    assert np.isfinite(gap_K), "Duality gap is not finite"

    # At the saddle itself, the gap should be 0 (sanity check).
    v_iter, y_iter = trajectories[0], trajectories[1]
    print(f"  v_0 - x_opt norm: {float(jnp.linalg.norm(v_iter[0] - x_opt)):.4f}")
    print(f"  y_0 - y_opt norm: {float(jnp.linalg.norm(y_iter[0] - y_opt)):.4f}")
