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
    """Weighted sum over k = 1..K_max with w_k = decay_rate^(K_max - k), normalized.

    Why k=0 is excluded
    -------------------
    x_0 is fixed input, not chosen by the algorithm, so f(x_0) is stepsize-
    independent. Beyond that, for COMPOSITE problems (e.g. Lasso: f = f1 + f2
    with f2 convex nonsmooth), the free subgradient h_0 of f2 at x_0 is
    unconstrained by the algorithm dynamics — ISTA uses h_1, ..., h_K in its
    updates but never h_0. In the pure WC-PEP, the convex-f2 interpolation
    constraint at (0, s),
            f2(x_0) - f2(x_s) + <h_0, x_s - x_0> <= 0,
    simplifies (with f2(x_s)=0, x_s=0) to f2(x_0) <= <h_0, x_0>, which with
    h_0 free is a VACUOUS bound. So f(x_0) is unbounded in the worst-case
    composite PEP. Verified empirically: PEPit itself returns NumericalError
    on CLARABEL / spurious large values on SCS, MOSEK when f(x_0) is in the
    weighted sum. Skipping k=0 makes the weighted PEP bounded for single-
    function problems (Quad) AND for ISTA on composite problems.

    Caveat: FISTA with K >= 3 is still unbounded after this fix. See
    `construct_fista_pep_data` docstring in `ista_fista.py` for the separate
    (documented, not yet fixed) FISTA-specific limitation.
    """
    weights = jnp.array([decay_rate ** (K_max - k) for k in range(1, K_max + 1)])
    weights = weights / jnp.sum(weights)

    A_0, b_0 = obj_builder(0)

    def accumulate(k, carry):
        A_accum, b_accum = carry
        A_k, b_k = obj_builder(k)
        return (A_accum + weights[k - 1] * A_k, b_accum + weights[k - 1] * b_k)

    A_init = jnp.zeros_like(A_0)
    b_init = jnp.zeros_like(b_0)
    return jax.lax.fori_loop(1, K_max + 1, accumulate, (A_init, b_init))


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
