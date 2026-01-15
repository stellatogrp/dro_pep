"""
Acceleration step size and parameter schedules.

Contains functions for computing parameter sequences for accelerated gradient methods.
"""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


def get_nesterov_fgm_beta_sequence(mu, L, K_max):
    """
    Compute the beta sequence for Nesterov's Fast Gradient Method (FGM).
    
    Based on Algorithm 28 from https://arxiv.org/pdf/2101.09545
    
    Args:
        mu: Strong convexity parameter
        L: Lipschitz constant of the gradient
        K_max: Number of iterations
        
    Returns:
        beta: Array of length K_max containing beta values for each iteration
    """
    q = mu / L
    
    # Initialize A sequence: A[0] = 0, A[1] = 1/(1-q)
    A = [0, 1 / (1 - q)]
    
    beta = []
    
    for k in range(K_max):
        # Compute A[k+2]
        A_kplus2_numerator = 2 * A[k+1] + 1 + np.sqrt(1 + 4 * A[k+1] + 4 * q * A[k+1] ** 2)
        A_kplus2 = A_kplus2_numerator / (2 * (1 - q))
        A.append(A_kplus2)
        
        # Compute beta_k
        beta_k_numerator = (A[k+2] - A[k+1]) * (A[k+1] * (1 - q) - A[k] - 1)
        beta_k_denominator = A[k+2] * (2 * q * A[k+1] + 1) - q * A[k+1] ** 2
        beta_k = beta_k_numerator / beta_k_denominator
        
        beta.append(beta_k)
    
    return np.array(beta)


@partial(jax.jit, static_argnames=['K_max'])
def jax_get_nesterov_fgm_beta_sequence(mu, L, K_max):
    q = mu / L

    A = jnp.zeros(K_max + 2)
    A = A.at[1].set(1 / (1 - q))

    beta = jnp.zeros(K_max)

    def body_fun(k, val):
        A, beta = val

        A_kplus2_numerator = 2 * A[k+1] + 1 + jnp.sqrt(1 + 4 * A[k+1] + 4 * q * A[k+1] ** 2)
        A_kplus2 = A_kplus2_numerator / (2 * (1 - q))
        A = A.at[k+2].set(A_kplus2)

        beta_k_numerator = (A[k+2] - A[k+1]) * (A[k+1] * (1 - q) - A[k] - 1)
        beta_k_denominator = A[k+2] * (2 * q * A[k+1] + 1) - q * A[k+1] ** 2
        beta_k = beta_k_numerator / beta_k_denominator
        beta = beta.at[k].set(beta_k)

        return (A, beta)
    
    _, beta = jax.lax.fori_loop(0, K_max, body_fun, (A, beta))

    return beta
