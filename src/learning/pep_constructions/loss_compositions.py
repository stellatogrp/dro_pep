"""Loss composition functions for PEP objectives."""
import jax
import jax.numpy as jnp
from typing import Callable, Tuple

ObjBuilderFn = Callable[[int], Tuple[jnp.ndarray, jnp.ndarray]]


def compose_final(obj_builder: ObjBuilderFn, K_max: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Use only final iterate (k=K_max)."""
    return obj_builder(K_max)


def compose_weighted(
    obj_builder: ObjBuilderFn,
    K_max: int,
    decay_rate: float = 0.9
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Weighted sum: w_k = decay_rate^(K-k), normalized."""
    weights = jnp.array([decay_rate ** (K_max - k) for k in range(K_max + 1)])
    weights = weights / jnp.sum(weights)

    A_0, b_0 = obj_builder(0)

    def accumulate(k, carry):
        A_accum, b_accum = carry
        A_k, b_k = obj_builder(k)
        return (A_accum + weights[k] * A_k, b_accum + weights[k] * b_k)

    A_init = jnp.zeros_like(A_0)
    b_init = jnp.zeros_like(b_0)
    return jax.lax.fori_loop(0, K_max + 1, accumulate, (A_init, b_init))


def compose_objective(
    obj_builder: ObjBuilderFn,
    K_max: int,
    composition_type: str = 'final',
    decay_rate: float = 0.9,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Main entry point for composing PEP objectives.

    Args:
        obj_builder: Function that takes an iteration index k and returns (A_obj_k, b_obj_k)
        K_max: Number of algorithm iterations
        composition_type: 'final' (use only final iterate) or 'weighted' (weighted sum)
        decay_rate: Decay rate for weighted composition (w_k = decay_rate^(K-k))

    Returns:
        (A_obj, b_obj): Composed objective matrices
    """
    if composition_type == 'final':
        return compose_final(obj_builder, K_max)
    elif composition_type == 'weighted':
        return compose_weighted(obj_builder, K_max, decay_rate)
    else:
        raise ValueError(f"Unknown composition_type: {composition_type}. Must be 'final' or 'weighted'.")
