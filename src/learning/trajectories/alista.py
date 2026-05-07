"""ALISTA trajectory computation for L2O Lasso experiments.

Mirrors `ista_fista.problem_data_to_ista_trajectories` but uses a fixed,
precomputed W^T instead of A^T in the gradient-step direction. The W matrix
comes from the offline convex problem (paper eq. 16) — see
`learning_experiment_classes/alista_w.py`.

Update (in shifted coords, x_opt mapped to 0):
    y_k     = x_k - t_k * W^T (A (x_k + x_opt) - b)
    x_{k+1} = soft_threshold(y_k + x_opt, t_k * lambd) - x_opt

Only `t_k` is learnable; W and lambd are fixed. The L2O metric (true Lasso
objective) is computed in `LassoProblemModule.create_metric_fn` from x_iter,
using the true A — so the metric is honest even though the step direction
is approximate.
"""
import logging
from functools import partial

import jax
import jax.numpy as jnp

from .ista_fista import soft_threshold_jax

log = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def problem_data_to_alista_trajectories(stepsizes, A, W, b, x0, x_opt, f_opt,
                                        lambd, K_max,
                                        return_Gram_representation=False):
    """Compute ALISTA-ISTA trajectories on the shifted Lasso problem.

    Args:
        stepsizes: (K_max,) array of step sizes (gamma_0, ..., gamma_{K-1}).
        A: (m, n) measurement matrix (true).
        W: (m, n) precomputed ALISTA matrix; W^T replaces A^T in the step.
        b: (m,) observation vector.
        x0: (n,) initial point in ORIGINAL coords (shifted by x_opt internally).
        x_opt: (n,) optimal point.
        f_opt: scalar optimal objective value.
        lambd: L1 regularization parameter.
        K_max: number of iterations.
        return_Gram_representation: must be False for the L2O ALISTA path
            (no SDP / PEP wired up for ALISTA).

    Returns:
        Tuple (x_iter, g_iter, h_iter, f1_iter, f2_iter) of raw trajectory
        arrays in shifted coords. `x_iter[:, 0]` is x0 shifted; `x_iter[:, k]`
        is the k-th iterate. `g_iter` stores the **true** smooth gradient
        A^T (A x - b) at each iterate (kept for parity with ISTA; unused by
        the L2O metric path).
    """
    n = A.shape[1]

    def f1_shifted(x):
        residual = A @ (x + x_opt) - b
        return 0.5 * jnp.sum(residual ** 2) - f_opt

    def f2_shifted(x):
        return lambd * jnp.sum(jnp.abs(x + x_opt))

    def grad_f1_shifted(x):
        # True smooth gradient (used for diagnostics / parity).
        return A.T @ (A @ (x + x_opt) - b)

    def alista_dir_shifted(x):
        # ALISTA step direction: W^T (A(x + x_opt) - b).
        return W.T @ (A @ (x + x_opt) - b)

    def subgrad_f2_shifted(x):
        return lambd * jnp.sign(x + x_opt)

    x0_shifted = x0 - x_opt

    x_iter = jnp.zeros((n, K_max + 1))
    g_iter = jnp.zeros((n, K_max + 1))
    h_iter = jnp.zeros((n, K_max + 1))
    f1_iter = jnp.zeros(K_max + 1)
    f2_iter = jnp.zeros(K_max + 1)

    x_iter = x_iter.at[:, 0].set(x0_shifted)
    g_iter = g_iter.at[:, 0].set(grad_f1_shifted(x0_shifted))
    h_iter = h_iter.at[:, 0].set(subgrad_f2_shifted(x0_shifted))
    f1_iter = f1_iter.at[0].set(f1_shifted(x0_shifted))
    f2_iter = f2_iter.at[0].set(f2_shifted(x0_shifted))

    x_curr = x0_shifted

    def alista_step(k, val):
        x_iter, g_iter, h_iter, f1_iter, f2_iter, x_curr = val
        gamma = stepsizes[k]

        # Step direction uses W^T instead of A^T
        y_k = x_curr - gamma * alista_dir_shifted(x_curr)

        # Proximal step on shifted f2 (same as ISTA)
        x_new_plus_xopt = soft_threshold_jax(y_k + x_opt, gamma * lambd)
        x_new = x_new_plus_xopt - x_opt

        # Subgradient from proximal optimality
        h_new = (y_k - x_new) / gamma

        x_iter = x_iter.at[:, k + 1].set(x_new)
        g_iter = g_iter.at[:, k + 1].set(grad_f1_shifted(x_new))
        h_iter = h_iter.at[:, k + 1].set(h_new)
        f1_iter = f1_iter.at[k + 1].set(f1_shifted(x_new))
        f2_iter = f2_iter.at[k + 1].set(f2_shifted(x_new))

        return (x_iter, g_iter, h_iter, f1_iter, f2_iter, x_new)

    x_iter, g_iter, h_iter, f1_iter, f2_iter, _ = jax.lax.fori_loop(
        0, K_max, alista_step, (x_iter, g_iter, h_iter, f1_iter, f2_iter, x_curr)
    )

    if return_Gram_representation:
        raise NotImplementedError(
            "ALISTA trajectories do not support Gram representation; "
            "l2o-alista is L2O-only and never asks for it."
        )

    return x_iter, g_iter, h_iter, f1_iter, f2_iter


def _make_alista_traj_fn(A_jax, W_jax, lambd):
    """Closure factory for the L2O training pipeline.

    Mirrors `_make_ista_traj_fn` in lasso.py but binds an additional
    precomputed W matrix.
    """
    def wrapped_traj_fn(stepsizes, b, x_opt, f_opt, K_max,
                        return_Gram_representation=False):
        gamma = stepsizes[0]  # trainer always passes a tuple
        x0 = jnp.zeros_like(x_opt)
        return problem_data_to_alista_trajectories(
            gamma, A_jax, W_jax, b, x0, x_opt, f_opt, lambd, K_max,
            return_Gram_representation=return_Gram_representation,
        )
    return wrapped_traj_fn
