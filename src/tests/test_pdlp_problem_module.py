"""Integration / smoke tests for the TV-inpainting PDLPProblemModule.

Verifies:
  1. The module instantiates from pdlp.yaml without errors.
  2. `sample_training_batch` produces the expected dict keys and shapes
     under the new lazy-K_mat schema.
  3. Per-sample (G, F) from `compute_batched_trajectories` satisfies every
     `construct_chambolle_pock_pep_data` constraint at the cached pooled
     `M_val` / `R_val`.

These tests pin the production training path to the verified CP PEP
construction. Any future regression in either path will surface here.
The pdlp.yaml file's `data_source_dir` is expected to point at a previous
sample-creation Hydra run; if not present, tests are skipped (rather than
failing) so that a fresh checkout doesn't break CI before sample creation.
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
# Hydra config helper
# ---------------------------------------------------------------------------

def _load_pdlp_cfg(**overrides):
    """Load pdlp.yaml from configs_learning with optional overrides.

    Skips tests cleanly if the configured `data_source_dir` doesn't exist on
    this machine (fresh checkout, no sample creation yet).
    """
    override_list = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(version_base='1.2', config_path='../configs_learning'):
        cfg = compose(config_name='pdlp', overrides=override_list)
    if cfg.get('data_source_dir', None) is None:
        pytest.skip("pdlp.yaml has no data_source_dir set; run sample creation first.")
    if not os.path.isdir(cfg.data_source_dir):
        pytest.skip(
            f"data_source_dir {cfg.data_source_dir} not found on this machine; "
            "run sample creation first."
        )
    return cfg


# ---------------------------------------------------------------------------
# Test 1: module instantiates
# ---------------------------------------------------------------------------

def test_pdlp_module_instantiates():
    cfg = _load_pdlp_cfg()
    module = PDLPProblemModule(cfg)

    # Basic shape assertions consistent with the metadata loaded from disk.
    assert module.M_img > 0 and module.N_img > 0
    assert module.K == module.M_img * module.N_img
    assert module.n_vars == module.K + module.K_v + module.K_h
    assert module.m1 == 2 * module.K_v + 2 * module.K_h
    assert module.S_in_dist > 0 and module.S_out_of_dist > 0
    assert module.M_val > 0 and module.R_val > 0
    # JAX device fixtures
    assert module.G_bcoo.shape == (module.m1, module.n_vars)
    assert module.c_jnp.shape == (module.n_vars,)
    assert module.images_jnp.shape[0] == 400  # Olivetti dataset


# ---------------------------------------------------------------------------
# Test 2: sample_training_batch shapes (light pool schema)
# ---------------------------------------------------------------------------

def test_pdlp_sample_training_batch_shapes():
    cfg = _load_pdlp_cfg()
    module = PDLPProblemModule(cfg)
    N = 2

    pd, gt = module.sample_training_batch(jax.random.PRNGKey(0), N)

    # Light pool: only image_index + mask in problem_data.
    assert set(pd.keys()) == {'image_index_batch', 'mask_batch'}
    assert set(gt.keys()) == {'x_opt_batch', 'y_opt_batch'}

    assert pd['image_index_batch'].shape == (N,)
    assert pd['image_index_batch'].dtype == jnp.int32
    assert pd['mask_batch'].shape == (N, module.K)
    assert pd['mask_batch'].dtype == jnp.bool_

    assert gt['x_opt_batch'].shape == (N, module.n_vars)
    assert gt['y_opt_batch'].shape == (N, module.m1 + module.S_in_dist)

    # Each cached mask must have exactly S_in_dist True entries (deterministic count).
    for i in range(N):
        n_known = int(jnp.sum(pd['mask_batch'][i]))
        assert n_known == module.S_in_dist, \
            f"instance {i}: mask has {n_known} known pixels, expected {module.S_in_dist}"


# ---------------------------------------------------------------------------
# Test 3: every constraint of the production PEP is satisfied by the
#         module-produced Gram on real cached instances.
# ---------------------------------------------------------------------------

def _split_violations(violations, K_max):
    """Slice the scalar-constraints array into named groups."""
    n_algo = K_max + 1
    n_interp = n_algo * (n_algo + 1)
    n_value_pin = 4
    n_IC = 1
    f1 = violations[:n_interp]
    h = violations[n_interp:2 * n_interp]
    vp = violations[2 * n_interp:2 * n_interp + n_value_pin]
    rest = violations[2 * n_interp + n_value_pin:]
    adj = rest[:-n_IC]
    IC = rest[-n_IC:]
    return f1, h, vp, adj, IC


@pytest.mark.parametrize('K_max', [1, 3])
@pytest.mark.parametrize('N', [2])
def test_pdlp_compute_batched_trajectories_passes_constraints(K_max, N):
    """Per-sample (G, F) from the module satisfies every CP PEP constraint
    at the pooled (M_val, R_val) loaded from metadata.
    """
    cfg = _load_pdlp_cfg()
    module = PDLPProblemModule(cfg)

    L_val, mu_val, R_val = module.compute_L_mu_R()
    stepsizes = module.get_initial_stepsizes('cp', K_max, L_val, mu_val)
    traj_fn = module.get_trajectory_fn('cp')

    pd, gt = module.sample_training_batch(jax.random.PRNGKey(42), N)
    full = {**pd, **gt}
    batched_keys = module.get_batched_parameters()
    batched_data = {k: full[f'{k}_batch'] for k in batched_keys}

    G_batch, F_batch = module.compute_batched_trajectories(
        stepsizes, batched_data, {}, traj_fn, K_max,
    )
    assert G_batch.shape[0] == N
    assert F_batch.shape[0] == N

    pep_data = construct_chambolle_pock_pep_data(
        tau=stepsizes[0], sigma=stepsizes[1], theta=stepsizes[2],
        M=L_val, R=R_val, K_max=K_max,
    )
    (A_obj, b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = [
        np.asarray(x) if not isinstance(x, list) else [np.asarray(a) for a in x]
        for x in pep_data
    ]
    num_scalar = A_vals.shape[0]

    # Tolerance scales with lp_upper. At lp_upper=1.0 (production) Gram entries
    # are O(n_vars) ~ O(12160), so 1e-6 abs is meaningful. At lp_upper=255
    # (legacy) Gram entries are ~65000x larger; bump tolerance accordingly.
    lp_upper = float(module.lp_upper)
    scale = lp_upper * lp_upper
    eps = 1e-6 * max(1.0, scale)
    psd_eps = 1e-6 * max(1.0, scale)

    print(f"\n=== PDLP module CP interpolation (K_max={K_max}, N={N}) ===")
    print(f"  M_val={L_val:.4f} R_val={R_val:.4f} S_in={module.S_in_dist}")
    for i in range(N):
        G_i = np.asarray(G_batch[i])
        F_i = np.asarray(F_batch[i])

        violations = np.zeros(num_scalar)
        for j in range(num_scalar):
            violations[j] = eval_scalar_constraint(
                A_vals[j], b_vals[j], c_vals[j], G_i, F_i,
            )
        f1_v, h_v, vp_v, adj_v, IC_v = _split_violations(violations, K_max)

        psd_min_eigs = []
        for idx in range(len(PSD_A_vals)):
            H = eval_psd_block(
                PSD_A_vals[idx], PSD_b_vals[idx], PSD_c_vals[idx], G_i, F_i,
            )
            psd_min_eigs.append(float(np.min(np.linalg.eigvalsh(H))))

        print(f"  sample {i}: f1={np.max(f1_v):.2e} h={np.max(h_v):.2e} "
              f"vp={np.max(vp_v):.2e} adj={np.max(adj_v):.2e} "
              f"IC={IC_v[0]:.2e} PSD_min={psd_min_eigs}")

        assert np.max(f1_v) <= eps, \
            f"sample {i}: f1 interpolation violated (max {np.max(f1_v):.3e})"
        assert np.max(h_v) <= eps, \
            f"sample {i}: h interpolation violated (max {np.max(h_v):.3e})"
        assert np.max(vp_v) <= eps, \
            f"sample {i}: value pinning violated (max {np.max(vp_v):.3e})"
        assert np.max(adj_v) <= eps, \
            f"sample {i}: adjoint violated (max {np.max(adj_v):.3e})"
        # IC is asserted here (unlike Tier 1) because pooled R is supposed to
        # be a strict upper bound for *every* in-distribution instance.
        assert IC_v[0] <= eps, \
            f"sample {i}: IC violated (value {IC_v[0]:.3e}); pooled R may be too small"
        for idx, me in enumerate(psd_min_eigs):
            assert me >= -psd_eps, \
                f"sample {i}: PSD block {idx} not PSD, min eig = {me:.3e}"
