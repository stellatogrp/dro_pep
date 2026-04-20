"""Numerical comparison test between old and new Lasso implementations.

This script sets up a small Lasso problem, computes G_batch/F_batch and
PEP data through both pathways, and verifies they match exactly.

Old pathway: direct calls to problem_data_to_ista_trajectories + construct_ista_pep_data
New pathway: via LassoProblemModule
"""

import diffcp_patch  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from learning.trajectories import problem_data_to_ista_trajectories
from learning.pep_constructions import construct_ista_pep_data
from learning.jax_scs_layer import compute_preconditioner_from_samples, dro_scs_solve
from learning_experiment_classes.lasso import (
    LassoProblemModule,
    generate_A,
    generate_batch_b_jax,
    solve_batch_lasso_cvxpy,
    LassoProblemDPP,
)

jax.config.update("jax_enable_x64", True)


def setup_test_problem():
    """Create a small Lasso problem for testing."""
    # Match dimensions from the working January run (m=250, n=500, N=15, K=5)
    m, n = 250, 500
    lambd = 0.4
    A_seed = 1000
    p_xsamp_nonzero = 0.1
    b_noise_std = 1e-2
    N = 15
    K_max = 5
    R = 7.5958
    eps = 0.1

    cfg = OmegaConf.create({
        'm': m,
        'n': n,
        'lambd': lambd,
        'A_seed': A_seed,
        'A_out_of_dist_seed': 4000,
        'p_xsamp_nonzero': p_xsamp_nonzero,
        'b_noise_std': b_noise_std,
        'R_nonstrongcvx': R,
        'R_strongcvx': R,
        'R_sample_size': 3,
        'R_seed': 3000,
        'stepsize_type': 'vector',
        'vector_init': 'fixed',
        'alg': 'ista',
        'optimizer_type': 'adamw',
        'learning_framework': 'ldro-pep',
        'sgd_iters': 2,
        'sgd_seed': 50,
        'eta_t': 1e-3,
        'N': N,
        'training_sample_N': 30,  # Full training set (larger than N for preconditioner)
        'K_max': [K_max],
        'pep_obj': 'obj_val',
        'dro_obj': 'expectation',
        'eps': eps,
        'alpha': 0.1,
        'precond_type': 'average',
        'dro_canon_backend': 'manual_jax',
        'weight_decay': 1e-3,
        'learn_beta': True,
        'training_loss_type_composition': 'final',
        'validation_loss_type_composition': 'final',
        'decay_rate': 0.9,
        'output_dir': 'learn_dro_outputs',
        'out_of_sample_val_N': 5,
        'out_of_sample_test_N': 5,
        'out_of_sample_val_seed': 10000,
        'out_of_sample_test_seed': 20000,
        'out_of_dist_N': 5,
        'out_of_dist_seed': 30000,
    })
    return cfg, K_max, N


def sample_lasso_batch_old_style(A_jax, A_np, key, N, cfg, lasso_dpp):
    """Replicate the old code's sample_lasso_batch."""
    b_batch = generate_batch_b_jax(
        key, A_jax, N, cfg.p_xsamp_nonzero, cfg.b_noise_std
    )
    b_batch_np = np.array(b_batch)
    x_opt_batch_np, f_opt_batch_np, _ = solve_batch_lasso_cvxpy(
        A_np, b_batch_np, cfg.lambd, lasso_dpp=lasso_dpp
    )
    return b_batch, jnp.array(x_opt_batch_np), jnp.array(f_opt_batch_np)


def run_old_pipeline(cfg, K_max, N, gamma, mu, L, R):
    """Replicate old lasso_dro_pipeline exactly."""
    A_np = generate_A(cfg.A_seed, cfg.m, cfg.n)
    A_jax = jnp.array(A_np)
    lasso_dpp = LassoProblemDPP(A_np, cfg.lambd)

    # Sample training data using same seed as new pipeline will use
    key = jax.random.PRNGKey(cfg.sgd_seed)
    key, train_key = jax.random.split(key)

    b_batch, x_opt_batch, f_opt_batch = sample_lasso_batch_old_style(
        A_jax, A_np, train_key, N, cfg, lasso_dpp
    )

    # x0 is always zero (in original coordinates, old code convention)
    x0_batch = jnp.zeros((N, cfg.n))

    # Stepsizes: old code uses (gamma,) tuple for ISTA
    stepsizes_tuple = (gamma,)
    traj_stepsizes = stepsizes_tuple[0]  # Just gamma for ISTA

    # Compute G_batch, F_batch via old-style direct vmap
    batch_GF_fn = jax.vmap(
        lambda b, x0, x_opt, f_opt: problem_data_to_ista_trajectories(
            traj_stepsizes, A_jax, b, x0, x_opt, f_opt, cfg.lambd, K_max,
            return_Gram_representation=True
        ),
        in_axes=(0, 0, 0, 0)
    )
    G_batch, F_batch = batch_GF_fn(b_batch, x0_batch, x_opt_batch, f_opt_batch)

    # PEP data (old code calls with default composition_type='final')
    pep_data = construct_ista_pep_data(
        stepsizes_tuple[0], mu, L, R, K_max, cfg.pep_obj
    )
    A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]

    # Preconditioner (old code computes from full training set once; here we use this N)
    precond_inv = compute_preconditioner_from_samples(
        G_batch, F_batch, precond_type=cfg.precond_type
    )

    return {
        'b_batch': b_batch,
        'x_opt_batch': x_opt_batch,
        'f_opt_batch': f_opt_batch,
        'x0_batch': x0_batch,
        'G_batch': G_batch,
        'F_batch': F_batch,
        'A_obj': A_obj,
        'b_obj': b_obj,
        'A_vals': A_vals,
        'b_vals': b_vals,
        'c_vals': c_vals,
        'precond_inv': precond_inv,
        'A_jax': A_jax,
    }


def run_new_pipeline(cfg, K_max, N, gamma, mu, L, R):
    """Run through LassoProblemModule."""
    pm = LassoProblemModule(cfg)

    # Use SAME training key as old pipeline
    key = jax.random.PRNGKey(cfg.sgd_seed)
    key, train_key = jax.random.split(key)

    problem_data, ground_truth = pm.sample_training_batch(train_key, N)

    # Build minibatch dict (strip _batch suffix, matching unified_trainer._get_minibatch)
    minibatch = {}
    for k, v in {**problem_data, **ground_truth}.items():
        key_name = k[:-6] if k.endswith('_batch') else k
        minibatch[key_name] = v

    # Stepsizes as tuple (gamma,) for ISTA
    stepsizes = (gamma,)

    # Get trajectory function (wrapped with A, lambd, x0 bound)
    traj_fn = pm.get_trajectory_fn('ista')
    pep_data_fn = pm.get_pep_data_fn('ista')

    # Compute batched trajectories
    batched_params = pm.get_batched_parameters()
    fixed_params = pm.get_fixed_parameters()
    batched_data = {k: minibatch[k] for k in batched_params if k in minibatch}
    fixed_data = {k: minibatch[k] for k in fixed_params if k in minibatch}

    G_batch, F_batch = pm.compute_batched_trajectories(
        stepsizes, batched_data, fixed_data, traj_fn, K_max
    )

    # PEP data (same composition as old default)
    pep_data = pep_data_fn(
        stepsizes, mu, L, R, K_max, cfg.pep_obj,
        composition_type='final', decay_rate=0.9,
    )
    A_obj, b_obj, A_vals, b_vals, c_vals = pep_data[:5]

    precond_inv = compute_preconditioner_from_samples(
        G_batch, F_batch, precond_type=cfg.precond_type
    )

    return {
        'b_batch': minibatch['b'],
        'x_opt_batch': minibatch['x_opt'],
        'f_opt_batch': minibatch['f_opt'],
        'G_batch': G_batch,
        'F_batch': F_batch,
        'A_obj': A_obj,
        'b_obj': b_obj,
        'A_vals': A_vals,
        'b_vals': b_vals,
        'c_vals': c_vals,
        'precond_inv': precond_inv,
        'A_jax': pm.A_jax,
    }


def diff_arr(name, old, new, atol=1e-10):
    """Print max abs diff between two arrays."""
    old = np.array(old)
    new = np.array(new)
    if old.shape != new.shape:
        print(f"  {name}: SHAPE MISMATCH old={old.shape}, new={new.shape}")
        return False
    diff = np.max(np.abs(old - new))
    passed = diff <= atol
    status = "OK " if passed else "FAIL"
    print(f"  [{status}] {name}: shape={old.shape}, max_abs_diff={diff:.2e}")
    return passed


def main():
    cfg, K_max, N = setup_test_problem()

    # Problem parameters (same as what LassoProblemModule would compute)
    A_np = generate_A(cfg.A_seed, cfg.m, cfg.n)
    A_jax = jnp.array(A_np)
    ATA = A_jax.T @ A_jax
    eigvals = jnp.linalg.eigvalsh(ATA)
    L = float(jnp.max(eigvals))
    mu = float(jnp.min(eigvals)) if cfg.m >= cfg.n else 0.0
    R = float(cfg.R_nonstrongcvx)

    print(f"Problem: m={cfg.m}, n={cfg.n}, K_max={K_max}, N={N}")
    print(f"L={L:.4f}, mu={mu:.4f}, R={R}, lambd={cfg.lambd}")

    # Initial stepsize (fixed vector)
    gamma = jnp.full(K_max, 1.5 / L)
    print(f"Initial gamma: {gamma}")

    print("\n--- Running OLD pipeline ---")
    old = run_old_pipeline(cfg, K_max, N, gamma, mu, L, R)
    print("\n--- Running NEW pipeline ---")
    new = run_new_pipeline(cfg, K_max, N, gamma, mu, L, R)

    print("\n=== SAMPLED DATA DIFF ===")
    all_ok = True
    all_ok &= diff_arr("b_batch", old['b_batch'], new['b_batch'])
    all_ok &= diff_arr("x_opt_batch", old['x_opt_batch'], new['x_opt_batch'], atol=1e-6)
    all_ok &= diff_arr("f_opt_batch", old['f_opt_batch'], new['f_opt_batch'], atol=1e-6)
    all_ok &= diff_arr("A_jax", old['A_jax'], new['A_jax'])

    print("\n=== TRAJECTORY (G, F) DIFF ===")
    all_ok &= diff_arr("G_batch", old['G_batch'], new['G_batch'])
    all_ok &= diff_arr("F_batch", old['F_batch'], new['F_batch'])

    print("\n=== PEP DATA DIFF ===")
    all_ok &= diff_arr("A_obj", old['A_obj'], new['A_obj'])
    all_ok &= diff_arr("b_obj", old['b_obj'], new['b_obj'])
    all_ok &= diff_arr("A_vals", old['A_vals'], new['A_vals'])
    all_ok &= diff_arr("b_vals", old['b_vals'], new['b_vals'])
    all_ok &= diff_arr("c_vals", old['c_vals'], new['c_vals'])

    print("\n=== PRECONDITIONER DIFF ===")
    all_ok &= diff_arr("precond_inv_G", old['precond_inv'][0], new['precond_inv'][0])
    all_ok &= diff_arr("precond_inv_F", old['precond_inv'][1], new['precond_inv'][1])

    print("\n=== SDP SOLVES (sanity check) ===")
    print("Calling dro_scs_solve on OLD data...")
    try:
        old_loss = dro_scs_solve(
            old['A_obj'], old['b_obj'], old['A_vals'], old['b_vals'], old['c_vals'],
            old['G_batch'], old['F_batch'],
            cfg.eps, old['precond_inv'],
            risk_type='expectation',
        )
        print(f"  OLD loss: {float(old_loss):.6f}")
    except Exception as e:
        print(f"  OLD solve FAILED: {e}")

    print("Calling dro_scs_solve on NEW data...")
    try:
        new_loss = dro_scs_solve(
            new['A_obj'], new['b_obj'], new['A_vals'], new['b_vals'], new['c_vals'],
            new['G_batch'], new['F_batch'],
            cfg.eps, new['precond_inv'],
            risk_type='expectation',
        )
        print(f"  NEW loss: {float(new_loss):.6f}")
    except Exception as e:
        print(f"  NEW solve FAILED: {e}")

    # =========================================================================
    # PRECONDITIONER HYPOTHESIS: old code uses a STABLE preconditioner computed
    # once from the full training set with INITIAL stepsizes. The new code
    # recomputes per-minibatch from CURRENT stepsizes (N samples only).
    # Test: does using the old-style stable preconditioner fix the unbounded SDP?
    # =========================================================================
    print("\n=== PRECONDITIONER HYPOTHESIS TEST ===")

    # Compute preconditioner from a LARGER set of samples with initial stepsizes
    # (mimicking old code's behavior)
    N_precond = cfg.training_sample_N
    print(f"Computing stable preconditioner from {N_precond} samples (full training set)...")

    A_np = generate_A(cfg.A_seed, cfg.m, cfg.n)
    A_jax = jnp.array(A_np)
    lasso_dpp_full = LassoProblemDPP(A_np, cfg.lambd)

    # Use a different key for the precond set (matching old style)
    key_full = jax.random.PRNGKey(cfg.sgd_seed)
    _, train_key_full = jax.random.split(key_full)
    b_full, x_opt_full, f_opt_full = sample_lasso_batch_old_style(
        A_jax, A_np, train_key_full, N_precond, cfg, lasso_dpp_full
    )
    x0_full = jnp.zeros((N_precond, cfg.n))

    # Compute G, F using INITIAL stepsizes on FULL set
    initial_gamma = jnp.full(K_max, 1.5 / L)
    batch_GF_fn_full = jax.vmap(
        lambda b, x0, x_opt, f_opt: problem_data_to_ista_trajectories(
            initial_gamma, A_jax, b, x0, x_opt, f_opt, cfg.lambd, K_max,
            return_Gram_representation=True
        ),
        in_axes=(0, 0, 0, 0)
    )
    G_full, F_full = batch_GF_fn_full(b_full, x0_full, x_opt_full, f_opt_full)
    stable_precond = compute_preconditioner_from_samples(
        G_full, F_full, precond_type=cfg.precond_type
    )

    print(f"Per-minibatch precond_inv_G: {np.array(new['precond_inv'][0])}")
    print(f"Stable (full-set) precond_inv_G: {np.array(stable_precond[0])}")
    print(f"Per-minibatch precond_inv_F: {np.array(new['precond_inv'][1])}")
    print(f"Stable (full-set) precond_inv_F: {np.array(stable_precond[1])}")

    print("\nSolving with STABLE preconditioner (old-style)...")
    try:
        stable_loss = dro_scs_solve(
            new['A_obj'], new['b_obj'], new['A_vals'], new['b_vals'], new['c_vals'],
            new['G_batch'], new['F_batch'],
            cfg.eps, stable_precond,
            risk_type='expectation',
        )
        print(f"  Loss with STABLE precond: {float(stable_loss):.6f}")
    except Exception as e:
        print(f"  STABLE solve FAILED: {e}")

    print("\nSolving with PER-MINIBATCH preconditioner (new-style)...")
    try:
        minibatch_loss = dro_scs_solve(
            new['A_obj'], new['b_obj'], new['A_vals'], new['b_vals'], new['c_vals'],
            new['G_batch'], new['F_batch'],
            cfg.eps, new['precond_inv'],
            risk_type='expectation',
        )
        print(f"  Loss with PER-MINIBATCH precond: {float(minibatch_loss):.6f}")
    except Exception as e:
        print(f"  PER-MINIBATCH solve FAILED: {e}")

    print()
    if all_ok:
        print("ALL DATA CHECKS PASSED")
    else:
        print("SOME DATA CHECKS FAILED - see above")


if __name__ == '__main__':
    main()
