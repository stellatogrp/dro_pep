"""Tests for the per-leaf signed-momentum reparameterization in UnifiedTrainer.

The legacy behavior sqrt-reparameterizes every parameter leaf, which (a)
clamps FGM momentum beta >= 0 and (b) pins any beta_k initialized at 0
(zero gradient through sqrt at 0). With cfg.signed_momentum=True the beta
leaf of a 2-leaf (t, beta) family stays raw, so momentum coefficients can
be learned with either sign and beta_0 = 0 receives a gradient.

Run: pytest tests/test_signed_momentum_reparam.py -v
"""
import jax
import jax.numpy as jnp
import pytest
from hydra import initialize, compose

jax.config.update('jax_enable_x64', True)

from learning.unified_trainer import (
    UnifiedTrainer,
    _reparam_modes,
    to_raw_params,
    to_actual_params,
)
from learning_experiment_classes.logreg import LogRegProblemModule


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------

def test_default_modes_all_sqrt():
    assert _reparam_modes(1, False) == ('sqrt',)
    assert _reparam_modes(2, False) == ('sqrt', 'sqrt')
    assert _reparam_modes(3, False) == ('sqrt', 'sqrt', 'sqrt')
    # Flag only changes 2-leaf (t, beta) families
    assert _reparam_modes(1, True) == ('sqrt',)
    assert _reparam_modes(3, True) == ('sqrt', 'sqrt', 'sqrt')


def test_signed_modes_beta_identity():
    assert _reparam_modes(2, True) == ('sqrt', 'identity')


def test_round_trip_default():
    stepsizes = (jnp.array([0.1, 0.2]), jnp.array([0.0, 0.5]))
    modes = _reparam_modes(2, False)
    raw = to_raw_params(stepsizes, modes)
    back = to_actual_params(raw, modes)
    for orig, rec in zip(stepsizes, back):
        assert jnp.allclose(orig, rec)
    # None modes = legacy all-sqrt behavior
    back_none = to_actual_params(to_raw_params(stepsizes), None)
    for orig, rec in zip(stepsizes, back_none):
        assert jnp.allclose(orig, rec)


def test_round_trip_signed_negative_beta():
    stepsizes = (jnp.array([0.1, 0.2]), jnp.array([-0.3, 0.5]))
    modes = _reparam_modes(2, True)
    raw = to_raw_params(stepsizes, modes)
    back = to_actual_params(raw, modes)
    assert jnp.allclose(back[0], stepsizes[0])
    assert jnp.allclose(back[1], stepsizes[1])  # sign survives
    # Legacy modes would destroy the sign (sqrt of negative -> nan)
    raw_legacy = to_raw_params(stepsizes, _reparam_modes(2, False))
    assert jnp.isnan(raw_legacy[1][0])


# ---------------------------------------------------------------------------
# End-to-end gradient test: beta_0 pinning
# ---------------------------------------------------------------------------

def _load_logreg_cfg(**overrides):
    override_list = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(version_base='1.2', config_path='../configs_learning'):
        cfg = compose(config_name='logreg', overrides=override_list)
    return cfg


def _beta0_grad(signed_momentum: bool) -> float:
    """Gradient of the L2O loss w.r.t. the raw beta_0 parameter at init."""
    K = 3
    cfg = _load_logreg_cfg(
        alg='nesterov_fgm',
        learning_framework='l2o',
        signed_momentum=str(signed_momentum).lower(),
        data_source_dir='null',
        training_sample_N=4,
        N=4,
        out_of_sample_val_N=4,
    )
    module = LogRegProblemModule(cfg)
    key = jax.random.PRNGKey(0)
    trainer = UnifiedTrainer(module, cfg, key)

    L, mu, R = module.compute_L_mu_R()
    stepsizes = module.get_initial_stepsizes('nesterov_fgm', K, L, mu)
    assert float(stepsizes[1][0]) == 0.0  # Nesterov sequence starts at beta_0 = 0

    trainer._modes = _reparam_modes(len(stepsizes), trainer.signed_momentum)
    raw = to_raw_params(stepsizes, trainer._modes)

    loss_fn = trainer._build_loss_function(K, L, mu, R, stepsizes)

    problem_data, ground_truth = module._sample_fresh_batch(
        jax.random.PRNGKey(1), 4, cfg.A_std
    )
    minibatch = {k[:-6]: v for k, v in {**problem_data, **ground_truth}.items()}

    grads = jax.grad(loss_fn)(raw, minibatch)
    return float(grads[1][0])


def test_beta0_gradient_unpinned_with_signed_momentum():
    g = _beta0_grad(signed_momentum=True)
    assert jnp.isfinite(g)
    assert g != 0.0, "beta_0 gradient should be nonzero with identity reparam"


def test_beta0_gradient_pinned_without_signed_momentum():
    # Documents the legacy bug this feature fixes: sqrt reparam at 0 kills
    # the gradient, so beta_0 could never be learned.
    g = _beta0_grad(signed_momentum=False)
    assert g == pytest.approx(0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
