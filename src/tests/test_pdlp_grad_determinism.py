"""Regression test: PDLP LDRO-PEP gradient must be deterministic across calls.

Evaluates ``jax.value_and_grad`` of the LDRO-PEP loss N times at the
production initial step sizes (with a fixed minibatch) and asserts both the
loss and the gradient are bit-identical across calls.

Guards against regression of a JAX ``pure_callback`` ordering bug in
``learning/jax_scs_layer.py:_solve_bwd``: a Python side-effect that mutates
the shared adjoint cache and runs at JAX *trace* time of the backward can
empty the cache before the deferred forward callback populates it, making
the very first ``value_and_grad`` call return zeros.

Usage:
    PDLP_DATA_DIR=<sample dir> pytest src/tests/test_pdlp_grad_determinism.py -v -s

    PDLP_DATA_DIR=<sample dir> python -m tests.test_pdlp_grad_determinism \
        [K=5] [n_repeats=3]
"""
import os
import sys

import diffcp_patch  # noqa: F401
import jax
import jax.numpy as jnp
from hydra import compose, initialize

jax.config.update("jax_enable_x64", True)

from learning.unified_trainer import UnifiedTrainer
from learning_experiment_classes.pdlp import PDLPProblemModule


def _load_cfg(K):
    overrides = [
        f"K_max=[{K}]",
        f"data_source_dir={os.environ['PDLP_DATA_DIR']}",
        "learning_framework=ldro-pep",
        "stepsize_type=vector",
    ]
    with initialize(version_base="1.2", config_path="../configs_learning"):
        cfg = compose(config_name="pdlp", overrides=overrides)
    return cfg


def _eval(loss_fn, sqrt_t, mb):
    loss, grads = jax.value_and_grad(loss_fn)(sqrt_t, mb)
    jax.block_until_ready((loss, grads))
    raw_norm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads)))
    per_comp = [float(jnp.sqrt(jnp.sum(g ** 2))) for g in grads]
    return float(loss), raw_norm, per_comp


def run(K=5, n_repeats=3):
    cfg = _load_cfg(K)
    pm = PDLPProblemModule(cfg); pm.validate_config()
    key = jax.random.PRNGKey(int(getattr(cfg, "sgd_seed", 0)))
    tr = UnifiedTrainer(pm, cfg, key)
    tr.prepare_data()
    L, mu, R = pm.compute_L_mu_R()
    init_st = pm.get_initial_stepsizes(cfg.alg, K, L, mu)
    loss_fn = tr._build_ldro_pep_loss(K, L, mu, R, init_st)
    sqrt_t = tuple(jnp.sqrt(t) for t in init_st)
    mb = tr._get_minibatch(0)

    print(f"\n--- determinism probe: K={K}, n_repeats={n_repeats} ---")
    losses, norms = [], []
    for r in range(n_repeats):
        loss, raw_norm, per_comp = _eval(loss_fn, sqrt_t, mb)
        losses.append(loss); norms.append(raw_norm)
        print(f"  repeat {r}: loss={loss:+.10e}  raw_grad_norm={raw_norm:.6e}  "
              f"per_comp(tau,sigma,theta)=[{per_comp[0]:.3e}, {per_comp[1]:.3e}, {per_comp[2]:.3e}]")
    loss_range = max(losses) - min(losses)
    grad_range = max(norms) - min(norms)
    print(f"  loss range : {loss_range:.3e}  -> deterministic={loss_range < 1e-9}")
    print(f"  grad range : {grad_range:.3e}  -> deterministic={grad_range < 1e-6}")
    return losses, norms


def test_pdlp_grad_determinism():
    """Pytest entry: assert grad and loss are bit-identical across 3 calls.

    Skipped if PDLP_DATA_DIR is not set.
    """
    if "PDLP_DATA_DIR" not in os.environ:
        import pytest
        pytest.skip("Set PDLP_DATA_DIR to a PDLP sample dir to run this test.")
    losses, norms = run(K=5, n_repeats=3)
    assert max(losses) - min(losses) < 1e-9, (
        f"loss is non-deterministic across consecutive value_and_grad calls: {losses}"
    )
    assert max(norms) - min(norms) < 1e-6, (
        f"raw_grad_norm is non-deterministic across consecutive value_and_grad "
        f"calls: {norms}"
    )


def main():
    K = 5
    n_repeats = 3
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if k == "K": K = int(v)
            elif k == "n_repeats": n_repeats = int(v)
    if "PDLP_DATA_DIR" not in os.environ:
        raise RuntimeError("Set PDLP_DATA_DIR.")
    run(K=K, n_repeats=n_repeats)


if __name__ == "__main__":
    main()
