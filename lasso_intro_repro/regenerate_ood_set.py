"""Regenerate the out-of-distribution LASSO test set used by lasso_intro.pdf.

The OOD set actually plotted in the paper figure is NOT any of the ood_set.npz
files archived on della: it was produced by the newer sample-creation code
path (x_ood_dist='normal', x_samp_ood_normal_std=3.0 — matching the paper's
"sigma_x = 3.0 out-of-distribution"), apparently run locally and never synced.

Generation is fully deterministic, mirroring
src/learning_experiment_classes/lasso.py (branch `neurips`,
stellatogrp/dro_pep):
  - A: in-distribution dictionary (A_seed=1000), reused for OOD,
  - key = jax.random.PRNGKey(out_of_dist_seed=30000), split into N=250 keys,
  - per key: split 3 -> x ~ 3.0*N(0,1) (n=500), mask ~ Bern(0.1),
    noise ~ 0.01*N(0,1) (m=250), b = A @ (x*mask) + noise,
  - f* via CVXPY with solver CLARABEL (same as LassoProblemDPP.solve).

Writes data/test_sets/ood_set_normal3.npz  (b_batch, x_opt_batch, f_opt_batch).

Usage: .venv/bin/python regenerate_ood_set.py
"""

import functools

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import cvxpy as cp
from tqdm import trange

M, N_DIM = 250, 500
N_SAMPLES = 250
LAMBD = 0.4
P_NONZERO = 0.1
B_NOISE_STD = 0.01
X_STD_OOD = 3.0
OOD_SEED = 30000


def generate_single_b(key, A):
    key1, key2, key3 = jax.random.split(key, 3)
    x_samp = X_STD_OOD * jax.random.normal(key1, (N_DIM,))
    x_mask = jax.random.bernoulli(key2, p=P_NONZERO, shape=(N_DIM,)).astype(jnp.float64)
    x_samp = x_samp * x_mask
    noise = B_NOISE_STD * jax.random.normal(key3, (M,))
    return A @ x_samp + noise


def main():
    A = np.load("data/test_sets/A_in_dist.npz")["A"]
    A_jax = jnp.array(A)

    keys = jax.random.split(jax.random.PRNGKey(OOD_SEED), N_SAMPLES)
    gen = functools.partial(generate_single_b, A=A_jax)
    b_batch = np.array(jax.vmap(gen)(keys))

    # Solve the 250 LASSO problems with CVXPY/CLARABEL (DPP-parametrized,
    # mirroring LassoProblemDPP).
    b_param = cp.Parameter(M)
    x = cp.Variable(N_DIM)
    obj = 0.5 * cp.sum_squares(A @ x - b_param) + LAMBD * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(obj))

    x_opt = np.zeros((N_SAMPLES, N_DIM))
    f_opt = np.zeros(N_SAMPLES)
    for i in trange(N_SAMPLES):
        b_param.value = b_batch[i]
        prob.solve(solver="CLARABEL")
        x_opt[i] = x.value
        f_opt[i] = prob.value

    np.savez_compressed(
        "data/test_sets/ood_set_normal3.npz",
        b_batch=b_batch, x_opt_batch=x_opt, f_opt_batch=f_opt,
    )
    print("saved data/test_sets/ood_set_normal3.npz")
    print("b std:", b_batch.std())


if __name__ == "__main__":
    main()
