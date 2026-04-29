"""Diagnostic for LDRO-PEP gradient norms at initialization.

Probes the hypothesis that the LDRO-PEP SDP's KKT system is ill-conditioned
near the theoretically-optimal stepsize 2/(mu+L), inflating the raw gradient
norm even when the validation loss is the same as the L2O baseline.

For Quad with K=3 and the default `configs_learning/quad.yaml` setup, this
solves the LDRO-PEP SDP at two stepsize initializations and reports the L2
norm of the gradient w.r.t. the sqrt-reparameterized parameters (the same
quantity logged as `raw_grad_norm` during training):

  1. t = 2/(mu+L)    (theoretically best initialization; vector_init='fixed')
  2. t = 1.0/(mu+L)  (perturbed away from the optimum)

Run as a standalone script (prints diagnostics):
    python -m tests.test_ldro_pep_grad_init

Or under pytest (assertions only check the run completes; diagnostics are
emitted as captured stdout — use `pytest -s` to see them):
    pytest src/tests/test_ldro_pep_grad_init.py -v -s
"""
import os
import sys

import diffcp_patch  # noqa: F401  # COO -> CSC fix for diffcp/Clarabel
import jax
import jax.numpy as jnp
from hydra import compose, initialize

jax.config.update("jax_enable_x64", True)

from learning.unified_trainer import UnifiedTrainer
from learning_experiment_classes.quad import QuadProblemModule


def _load_quad_cfg(**overrides):
    """Compose configs_learning/quad.yaml with overrides."""
    override_list = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(version_base="1.2", config_path="../configs_learning"):
        cfg = compose(config_name="quad", overrides=override_list)
    return cfg


def _eval_grad_at_t(loss_fn, minibatch, t_value, K):
    """Evaluate value_and_grad at a vector stepsize t_value (length K)."""
    sqrt_stepsizes = (jnp.full(K, jnp.sqrt(t_value)),)
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = value_and_grad_fn(sqrt_stepsizes, minibatch)
    jax.block_until_ready((loss, grads))

    # raw_grad_norm: global L2 across the gradient tuple (matches CSV)
    raw_grad_norm = float(jnp.sqrt(sum(jnp.sum(g**2) for g in grads)))

    # Convert d/d(sqrt_t) -> d/dt via chain rule: g_t = g_sqrt / (2 * sqrt(t))
    grads_t = tuple(g / (2.0 * jnp.sqrt(t_value)) for g in grads)
    grad_t_norm = float(jnp.sqrt(sum(jnp.sum(g**2) for g in grads_t)))

    return {
        "loss": float(loss),
        "raw_grad_norm": raw_grad_norm,
        "grad_t_norm": grad_t_norm,
        "grad_sqrt_per_elem": [float(x) for x in grads[0].tolist()],
        "grad_t_per_elem": [float(x) for x in grads_t[0].tolist()],
    }


def run_diagnostic(K=3, eta_t=None, eps=None, seed=None):
    """Solve the LDRO-PEP SDP at two stepsize inits and report grad norms.

    All overrides default to the values in `configs_learning/quad.yaml`. The
    K_max list is forced to [K] so the trainer materializes for that K only.
    """
    overrides = {
        "learning_framework": "ldro-pep",
        "K_max": [K],
        "stepsize_type": "vector",
        "vector_init": "fixed",
        "alg": "vanilla_gd",
    }
    if eta_t is not None:
        overrides["eta_t"] = eta_t
    if eps is not None:
        overrides["eps"] = eps
    if seed is not None:
        overrides["seed"] = seed

    cfg = _load_quad_cfg(**overrides)

    print("=" * 70)
    print("LDRO-PEP gradient norm diagnostic at initialization")
    print("=" * 70)
    print(
        f"K={K}, mu={cfg.mu}, L={cfg.L}, R={cfg.R}, "
        f"eps={cfg.eps}, dro_obj={cfg.dro_obj}, alpha={cfg.alpha}, "
        f"precond_type={cfg.precond_type}, N={cfg.N}, "
        f"sdp_backend={cfg.sdp_backend}, dro_canon_backend={cfg.dro_canon_backend}"
    )

    problem_module = QuadProblemModule(cfg)
    problem_module.validate_config()

    key = jax.random.PRNGKey(cfg.seed)
    trainer = UnifiedTrainer(problem_module, cfg, key)
    trainer.prepare_data()  # uses data_source_dir if available, else fresh sample

    L, mu, R = problem_module.compute_L_mu_R()

    # Probe init (matches the trainer's static-PSD-shape capture pattern).
    initial_stepsizes = problem_module.get_initial_stepsizes(cfg.alg, K, L, mu)
    loss_fn = trainer._build_ldro_pep_loss(K, L, mu, R, initial_stepsizes)

    # Hold the minibatch fixed across both evaluations so the only thing
    # varying is the stepsize parameter.
    minibatch = trainer._get_minibatch(0)

    t_opt = 2.0 / (mu + L)
    t_perturbed = 1.0 / (mu + L)

    print()
    print(f"--- Init A: t = 2/(mu+L) = {t_opt:.6f} (theoretically optimal) ---")
    res_a = _eval_grad_at_t(loss_fn, minibatch, t_opt, K)
    print(f"  loss                  : {res_a['loss']:.8f}")
    print(f"  raw_grad_norm (sqrt)  : {res_a['raw_grad_norm']:.8f}")
    print(f"  grad_t_norm (chain)   : {res_a['grad_t_norm']:.8f}")
    print(f"  grad sqrt per-elem    : {res_a['grad_sqrt_per_elem']}")
    print(f"  grad t per-elem       : {res_a['grad_t_per_elem']}")

    print()
    print(f"--- Init B: t = 1.0/(mu+L) = {t_perturbed:.6f} (perturbed) ---")
    res_b = _eval_grad_at_t(loss_fn, minibatch, t_perturbed, K)
    print(f"  loss                  : {res_b['loss']:.8f}")
    print(f"  raw_grad_norm (sqrt)  : {res_b['raw_grad_norm']:.8f}")
    print(f"  grad_t_norm (chain)   : {res_b['grad_t_norm']:.8f}")
    print(f"  grad sqrt per-elem    : {res_b['grad_sqrt_per_elem']}")
    print(f"  grad t per-elem       : {res_b['grad_t_per_elem']}")

    print()
    print("--- Comparison ---")
    ratio_sqrt = res_a["raw_grad_norm"] / max(res_b["raw_grad_norm"], 1e-30)
    ratio_t = res_a["grad_t_norm"] / max(res_b["grad_t_norm"], 1e-30)
    print(f"  raw_grad_norm(A) / raw_grad_norm(B) = {ratio_sqrt:.4f}")
    print(f"  grad_t_norm(A)   / grad_t_norm(B)   = {ratio_t:.4f}")
    print(
        "  Interpretation: a large ratio (>>1) suggests the SDP gradient is "
        "inflated near the theoretically-optimal stepsize, which is "
        "consistent with KKT ill-conditioning at the optimum."
    )

    return res_a, res_b


def test_ldro_pep_grad_norm_at_initialization():
    """Pytest entry: runs the diagnostic and checks both evaluations finish."""
    res_a, res_b = run_diagnostic(K=3)
    assert res_a["raw_grad_norm"] >= 0.0
    assert res_b["raw_grad_norm"] >= 0.0


if __name__ == "__main__":
    # Allow simple CLI overrides: K, eta_t, eps, seed
    kwargs = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        if k == "K":
            kwargs[k] = int(v)
        elif k in ("eta_t", "eps"):
            kwargs[k] = float(v)
        elif k == "seed":
            kwargs[k] = int(v)
    run_diagnostic(**kwargs)
