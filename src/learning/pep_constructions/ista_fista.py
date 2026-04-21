"""
JAX-compatible PEP data construction for ISTA/FISTA (Lasso/proximal problems).

Constructs PEP constraint matrices (A_vals, b_vals, A_obj, b_obj) directly
using JAX, enabling autodifferentiation through step sizes.

ISTA/FISTA optimize composite problems: min f1(x) + f2(x)
- f1: smooth (L-smooth, μ-strongly convex)
- f2: convex (nonsmooth, e.g., L1 regularization)

Key difference from GD: requires TWO sets of interpolation conditions:
1. f1 interpolation: smooth strongly convex conditions (gradients g)
2. f2 interpolation: convex conditions (subgradients h)

NOTE on composite WC-PEP boundedness (see individual docstrings for detail):
- ISTA weighted composition: bounded (after `compose_weighted` skips k=0 to
  avoid the free h_0 subgradient unbounding f(x_0)).
- FISTA weighted composition: bounded for K <= 2, UNBOUNDED for K >= 3 due
  to a missing set of f1 interpolation constraints at the intermediate
  algorithm iterates x_2, ..., x_{K-1}. See `construct_fista_pep_data`
  docstring for the full analysis, diagnosis, and a sketch of the fix.
  Workaround: use `composition_type='final'` for FISTA LPEP, or use the
  `ldro-pep` framework (sample constraints bound the SDP).

NOTE on PEP convention (x_s = 0, F_s = 0):
The composite obj builder implicitly assumes the stationary point is at
the origin with zero objective. This is standard PEP practice and is
valid without loss of generality:
  - Coordinates are shifted so x_s = 0. The basis uses x_0 - x_s as its
    first vector, and `xs_rep = jnp.zeros(dimG)`. All interpolation
    constraints are translation-invariant, so this preserves the problem.
  - Function values are shifted so F_s = 0. The f1/f2 representation
    vectors at x_s are set to zeros (no basis slot for f1(x_s), f2(x_s)),
    making F_s ≡ 0 in the SDP. Equivalent to subtracting F_s from every
    function value, which doesn't affect gradients/subgradients.
Under this convention:
  - `obj_val` returns f1(x_k) + f2(x_k) = F(x_k) - F_s.
  - `opt_dist_sq_norm` returns outer(x_k, x_k) which is ||x_k - x_s||^2
    (since x_s = 0 in the shifted basis).
  - `grad_sq_norm` returns ||g_k + h_k||^2, the composite subgradient norm
    at x_k (vanishes at x_s via the stationarity constraint g_s + h_s = 0).

NOTE on FISTA non-obj_val metrics with weighted composition (KNOWN ISSUE):
The composite obj builder indexes into `repX_f1[k]` / `repG_f1[k]`, which
for FISTA contain y_k / g(y_k) for k < K_max (they only equal x_K / g(x_K)
at k = K_max). Consequences for FISTA + weighted composition:
  - `opt_dist_sq_norm` measures ||y_k - x_s||^2 instead of ||x_k - x_s||^2
    at intermediate iterates — almost certainly not the intended metric.
  - `grad_sq_norm` computes ||g(y_k) + h_k||^2, mixing a gradient at y_k
    with a subgradient at x_k — semantically ill-defined.
  - `obj_val` is unaffected: the objective uses f1(y_k) + f2(x_k), and the
    PEPit reference for FISTA's weighted performance metric is also
    defined on (y_k, x_k) pairs in the same way.
FISTA `final` composition is fine for all three metrics because at
k = K_max we have repY_f1[K_max] = x_K and repG_f1[K_max] = g(x_K)
(see lines that set `x_final` at the K_max slot).
ISTA is unaffected — repX_f1[k] = x_k and repG_f1[k] = g(x_k) throughout.
Fix (not implemented): pass a separate repX_at_xk / repG_at_xk into the
obj builder for FISTA so non-obj_val weighted metrics can use x_k, g(x_k)
at intermediate iterates. Requires additional basis slots for g(x_k) at
k = 1..K-1 (overlap with the intermediate-iterate interpolation fix
described in `construct_fista_pep_data`'s KNOWN LIMITATION section).
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from .interpolation_conditions import smooth_strongly_convex_interp, convex_interp
from .loss_compositions import compose_objective


def _create_composite_obj_builder(repX_f1, repG_f1, repF_f1,
                                   repG_f2, repF_f2,
                                   dimG, dimF1, dimF2, pep_obj):
    """Create objective builder for composite (f1 + f2) problems.

    Assumes the PEP is set up with x_s = 0 (shifted coords) and F_s = 0
    (shifted function values); see the module docstring for why this is
    w.l.o.g. Under that convention:
        obj_val           -> f1(x_k) + f2(x_k)      (== F(x_k) - F_s)
        opt_dist_sq_norm  -> <x_k, x_k> via Gram     (== ||x_k - x_s||^2)
        grad_sq_norm      -> ||g_k + h_k||^2        (composite subgrad norm)

    Args:
        repX_f1: Array of point representations for f1
        repG_f1: Array of gradient representations for f1 (gradients g)
        repF_f1: Array of function value representations for f1
        repG_f2: Array of subgradient representations for f2 (subgradients h)
        repF_f2: Array of function value representations for f2
        dimG: Dimension of Gram basis
        dimF1: Dimension of f1 function value basis
        dimF2: Dimension of f2 function value basis
        pep_obj: Performance metric type

    Returns:
        obj_builder: Function that takes iteration index k and returns (A_obj_k, b_obj_k)

    KNOWN ISSUE — FISTA non-obj_val metrics with weighted composition:
        For FISTA, this builder is called with repX_f1 = repY_f1 and
        repG_f1 = g(y) gradients (+ g(x_K) in the last slot). So at
        intermediate k < K_max:
          - `opt_dist_sq_norm` measures ||y_k - x_s||^2, NOT ||x_k - x_s||^2.
          - `grad_sq_norm` uses g(y_k) + h_k, mixing a gradient at y_k
            with a subgradient at x_k (semantically wrong).
        `obj_val` is unaffected (matches PEPit's weighted-metric semantics).
        `final` composition is unaffected (repY_f1[K_max] = x_K).
        ISTA is unaffected (repX_f1[k] = x_k throughout).
        See the module docstring for a fix sketch.
    """
    dimF = dimF1 + dimF2

    def obj_builder(k: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        xk = repX_f1[k]
        gk = repG_f1[k]  # grad f1
        hk = repG_f2[k]  # subgrad f2
        fk_f1, fk_f2 = repF_f1[k], repF_f2[k]

        if pep_obj == 'obj_val':
            return jnp.zeros((dimG, dimG)), jnp.concatenate([fk_f1, fk_f2])
        elif pep_obj == 'grad_sq_norm':
            gk_plus_hk = gk + hk
            return jnp.outer(gk_plus_hk, gk_plus_hk), jnp.zeros(dimF)
        elif pep_obj == 'opt_dist_sq_norm':
            return jnp.outer(xk, xk), jnp.zeros(dimF)
        else:
            raise ValueError(f"Unknown pep_obj: {pep_obj}")
    return obj_builder


@partial(jax.jit, static_argnames=['K_max', 'pep_obj', 'composition_type'])
def construct_ista_pep_data(t, mu, L, R, K_max, pep_obj,
                            composition_type='final', decay_rate=0.9):
    """
    Construct PEP constraint matrices for ISTA using step sizes t.

    ISTA dynamics: x_{k+1} = prox_{t_k * f2}(x_k - t_k * g_k)
                           = x_k - t_k * g_k - t_k * h_{k+1}
    where g_k = grad f1(x_k) and h_{k+1} is a subgradient of f2 at x_{k+1}.

    Args:
        t: Step sizes - scalar (same for all iterations) or vector of length K_max
        mu: Strong convexity parameter of f1
        L: Lipschitz constant of gradient of f1
        R: Initial radius bound (||x0 - xs|| <= R)
        K_max: Number of ISTA iterations
        pep_obj: Performance metric type:
            'obj_val': f(xK) - f(xs) (requires both F1 and F2)
            'grad_sq_norm': ||gK + hK||^2 (composite gradient)
            'opt_dist_sq_norm': ||xK - xs||^2
        composition_type: 'final' (use only final iterate) or 'weighted' (weighted sum)
        decay_rate: Decay rate for weighted composition (w_k = decay_rate^(K-k))

    Returns:
        pep_data: Tuple (A_obj, b_obj, A_vals, b_vals, c_vals,
                        PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)

    Representation structure (from test_ista_interpolation.py):
        - dimG = 2K + 5: [x_0-x_s, g_0, h_0, h_1, g_1, h_2, g_2, ..., h_K, g_K, g_s, h_s]
        - dimF1 = K + 2: f1 values at [x_0, x_1, ..., x_K, x_s]
        - dimF2 = K + 2: f2 values at [x_0, x_1, ..., x_K, x_s]
        - Points: x_0, x_1, ..., x_K, x_s (optimal point last)
        - Stationarity at x_s: g_s + h_s = 0
    """
    # Broadcast t to vector if scalar
    t_vec = jnp.broadcast_to(t, (K_max,))

    # Dimensions for Gram representation
    # dimG = 2K + 5: [x_0-x_s, g_0, h_0, h_1, g_1, ..., h_K, g_K, g_s, h_s]
    dimG = 2 * K_max + 5
    dimF1 = K_max + 2  # K+1 algorithm points + x_s
    dimF2 = K_max + 2  # K+1 algorithm points + x_s

    # Identity matrices for symbolic representation
    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)
    eyeF2 = jnp.eye(dimF2)

    # Precompute indices for Gram basis (avoids traced conditionals)
    # idx_g[k] = index of gradient g_k: g_0 at 1, g_k at 2k+2 for k >= 1
    # idx_h[k] = index of subgradient h_k: h_0 at 2, h_k at 2k+1 for k >= 1
    idx_g = jnp.array([1] + [2 * k + 2 for k in range(1, K_max + 1)])  # Shape (K_max+1,)
    idx_h = jnp.array([2] + [2 * k + 1 for k in range(1, K_max + 1)])  # Shape (K_max+1,)

    idx_gs = 2 * K_max + 3  # Index of g_s
    idx_hs = 2 * K_max + 4  # Index of h_s

    # Number of algorithm points: K+1 (x_0, x_1, ..., x_K)
    n_points = K_max + 1

    # Initialize representations
    repX_f1 = jnp.zeros((n_points + 1, dimG))  # +1 for x_s
    repG_f1 = jnp.zeros((n_points + 1, dimG))
    repF_f1 = jnp.zeros((n_points + 1, dimF1))

    repX_f2 = jnp.zeros((n_points + 1, dimG))
    repG_f2 = jnp.zeros((n_points + 1, dimG))  # Subgradients h
    repF_f2 = jnp.zeros((n_points + 1, dimF2))

    # Initial point x_0
    x_rep = eyeG[0, :]  # x_0 - x_s

    repX_f1 = repX_f1.at[0].set(x_rep)
    repG_f1 = repG_f1.at[0].set(eyeG[idx_g[0], :])  # g_0
    repF_f1 = repF_f1.at[0].set(eyeF1[0, :])

    repX_f2 = repX_f2.at[0].set(x_rep)
    repG_f2 = repG_f2.at[0].set(eyeG[idx_h[0], :])  # h_0
    repF_f2 = repF_f2.at[0].set(eyeF2[0, :])

    # ISTA dynamics: x_{k+1} = x_k - t_k * g_k - t_k * h_{k+1}
    def ista_step(k, carry):
        repX_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_prev = carry
        t_k = t_vec[k]

        # x_{k+1} = x_k - t_k * g_k - t_k * h_{k+1}
        g_k = eyeG[idx_g[k], :]
        h_kp1 = eyeG[idx_h[k + 1], :]

        x_new = x_prev - t_k * g_k - t_k * h_kp1

        # Store x_{k+1}
        repX_f1 = repX_f1.at[k + 1].set(x_new)
        repG_f1 = repG_f1.at[k + 1].set(eyeG[idx_g[k + 1], :])
        repF_f1 = repF_f1.at[k + 1].set(eyeF1[k + 1, :])

        repX_f2 = repX_f2.at[k + 1].set(x_new)
        repG_f2 = repG_f2.at[k + 1].set(eyeG[idx_h[k + 1], :])
        repF_f2 = repF_f2.at[k + 1].set(eyeF2[k + 1, :])

        return (repX_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_new)

    init_carry = (repX_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_rep)
    repX_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_final = jax.lax.fori_loop(
        0, K_max, ista_step, init_carry
    )

    # Optimal point x_s: all zeros in relative representation
    xs_rep = jnp.zeros(dimG)

    repX_f1 = repX_f1.at[n_points].set(xs_rep)
    repG_f1 = repG_f1.at[n_points].set(eyeG[idx_gs, :])  # g_s
    repF_f1 = repF_f1.at[n_points].set(jnp.zeros(dimF1))

    repX_f2 = repX_f2.at[n_points].set(xs_rep)
    repG_f2 = repG_f2.at[n_points].set(eyeG[idx_hs, :])  # h_s
    repF_f2 = repF_f2.at[n_points].set(jnp.zeros(dimF2))

    # Compute interpolation conditions.
    # For the composite problem (f1 smooth SC + f2 convex), the stationary point
    # of f1 + f2 does NOT have zero (sub)gradients for f1 or f2 individually.
    # Stationarity: g_s + h_s = 0. Pass correct Gram-basis representations.
    A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
        repX_f1, repG_f1, repF_f1, mu, L, n_points, gs=eyeG[idx_gs, :]
    )

    A_vals_f2, b_vals_f2 = convex_interp(
        repX_f2, repG_f2, repF_f2, n_points, gs=eyeG[idx_hs, :]
    )

    # Combine constraints with F = [F1, F2]
    dimF = dimF1 + dimF2

    b_vals_f1_combined = jnp.concatenate([b_vals_f1, jnp.zeros((b_vals_f1.shape[0], dimF2))], axis=1)
    b_vals_f2_combined = jnp.concatenate([jnp.zeros((b_vals_f2.shape[0], dimF1)), b_vals_f2], axis=1)

    A_vals = jnp.concatenate([A_vals_f1, A_vals_f2], axis=0)
    b_vals = jnp.concatenate([b_vals_f1_combined, b_vals_f2_combined], axis=0)

    num_constraints = A_vals.shape[0]
    c_vals = jnp.zeros(num_constraints)

    # Initial condition: ||x0 - xs||^2 <= R^2
    A_init = jnp.outer(repX_f1[0], repX_f1[0])
    b_init = jnp.zeros(dimF)
    c_init = -R ** 2

    A_vals = jnp.concatenate([A_vals, A_init[None, :, :]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None, :]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)

    # Stationarity constraint: ||g_s + h_s||^2 = 0
    gs_plus_hs = eyeG[idx_gs, :] + eyeG[idx_hs, :]
    A_stationarity = jnp.outer(gs_plus_hs, gs_plus_hs)
    b_stationarity = jnp.zeros(dimF)
    c_stationarity = 0.0

    A_vals = jnp.concatenate([A_vals, A_stationarity[None, :, :]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_stationarity[None, :]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_stationarity])], axis=0)

    # Objective: use composition
    obj_builder = _create_composite_obj_builder(
        repX_f1, repG_f1, repF_f1, repG_f2, repF_f2, dimG, dimF1, dimF2, pep_obj
    )
    A_obj, b_obj = compose_objective(obj_builder, K_max, composition_type, decay_rate)

    PSD_A_vals = []
    PSD_b_vals = []
    PSD_c_vals = []
    PSD_shapes = []

    pep_data = (A_obj, b_obj, A_vals, b_vals, c_vals,
                PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)

    return pep_data


@partial(jax.jit, static_argnames=['K_max', 'pep_obj', 'composition_type'])
def construct_fista_pep_data(t, beta, mu, L, R, K_max, pep_obj,
                             composition_type='final', decay_rate=0.9):
    """
    Construct PEP constraint matrices for FISTA using step sizes t and momentum beta.

    FISTA dynamics:
        y_0 = x_0
        For k = 0, ..., K-1:
            x_{k+1} = prox_{t_k * f2}(y_k - t_k * grad_f1(y_k))
                    = y_k - t_k * g(y_k) - t_k * h_{k+1}
            y_{k+1} = x_{k+1} + beta_k * (x_{k+1} - x_k)

    Key: gradients g are evaluated at y points, subgradients h at x points.

    Representation structure:
        - Gram basis: [x_0-x_s, g(y_0), g(y_1), ..., g(y_{K-1}), g(x_K), h_0, h_1, ..., h_K, g_s]
        - dimG = 2K + 4
        - f1 interpolation points: y_0, y_1, ..., y_{K-1}, x_K, x_s
        - f2 interpolation points: x_0, x_1, ..., x_K, x_s

    Args:
        t: Step sizes - scalar or vector of length K_max
        beta: Raw momentum sequence (length K_max + 1) where beta[0]=1.0.
              The momentum coefficient at step k is computed as (beta[k] - 1) / beta[k+1].
        mu: Strong convexity parameter of f1
        L: Lipschitz constant of gradient of f1
        R: Initial radius bound
        K_max: Number of FISTA iterations
        pep_obj: Performance metric type
        composition_type: 'final' (use only final iterate) or 'weighted' (weighted sum)
        decay_rate: Decay rate for weighted composition (w_k = decay_rate^(K-k))

    Returns:
        pep_data tuple

    KNOWN LIMITATION (weighted composition, K >= 3)
    -----------------------------------------------
    The weighted-composition LPEP solve is UNBOUNDED for FISTA with K >= 3
    because the PEP is missing f1 interpolation at the intermediate
    algorithm iterates x_2, ..., x_{K-1} (where y_k != x_k starts at k=2:
    beta_0 = 0 is built in by lam_0=1, but beta_k > 0 for k >= 1).

    Why this doesn't affect `final` composition or K <= 2
    -----------------------------------------------------
    - `final` only uses f1(x_K), which IS in the PEP (x_K is an f1 interp
      point), so `final` composition is bounded for all K.
    - K=1: weighted sum is over k=1 only, which is x_K. Same as final.
    - K=2: beta_0 = 0 so y_1 = x_1, and the weighted sum uses f1(y_1) +
      f2(x_1) = f1(x_1) + f2(x_1) as a mathematical identity. Bounded.
    - K >= 3: y_k != x_k for k >= 2, and our PEP's f1 interpolation points
      are only {y_0, ..., y_{K-1}, x_K, x_s}. The weighted objective term
      at k in {2,...,K-1} pulls on f1(y_k) (which IS in the PEP) plus
      f2(x_k) (also in the PEP via f2's own interp set). But PEPit's
      equivalent performance metric uses f1(x_k) + f2(x_k), and PEPit's
      internal f1 interpolation set DOES include x_2, ..., x_{K-1}
      (verified empirically: for K=5 weighted, PEPit holds f1 at 10 points
      vs our 7). Those additional pairwise smooth-strongly-convex f1
      constraints between the y-set and x-set are what bound the PEP in
      PEPit. Without them, our SDP is under-constrained and the WC
      objective is unbounded.

    Why not `compose_weighted`'s skip-k=0 fix (in loss_compositions.py)
    ------------------------------------------------------------------
    The k=0 skip addresses a separate issue: the free subgradient h_0 at
    x_0 makes f(x_0) unbounded in any composite PEP (affects ISTA too, and
    the k=0 fix handles it there). FISTA K >= 3 is unbounded even after
    skipping k=0 because of the intermediate-iterate interpolation gap
    described above, which is specific to FISTA's momentum-induced
    divergence between y_k and x_k.

    Fix sketch (not yet implemented; requires moderate Gram-basis growth)
    --------------------------------------------------------------------
    Add f1 interpolation at x_k for k = 2, ..., K-1. This requires:
      - K-2 new Gram basis slots for g(x_2), ..., g(x_{K-1}):
            dimG: 2K+4  ->  3K+2
      - K-2 new dimF1 slots for f1(x_2), ..., f1(x_{K-1}):
            dimF1: K+2  ->  2K
      - Extend `smooth_strongly_convex_interp` over the combined point set
        {y_0, ..., y_{K-1}, x_1, ..., x_K, x_s}. Note y_0=x_0 and y_1=x_1
        always (beta_0=0 by convention); those pairs can be elided to save
        slots.
      - Extend the objective builder so the weighted sum at iteration k
        (k < K) pulls on f1(x_k) instead of f1(y_k). This matches PEPit's
        semantic definition of the composite performance metric
        sum_k w_k * (f1(x_k) + f2(x_k) - Fs).
    For K=5 the Gram basis grows from 14 to 17 slots (~20%). The final
    composition path is unaffected (f1 at x_K is already present).

    Workarounds for now
    -------------------
    - Use `composition_type='final'` for FISTA LPEP. This is bounded and
      well-tested (see tests/test_pep_lasso_debug.py).
    - Use `learning_framework='ldro-pep'` (not 'lpep'): DRO's sample-based
      constraints bound the SDP even when the pure WC-PEP is unbounded.
    """
    # Broadcast t to vector if scalar
    t_vec = jnp.broadcast_to(t, (K_max,))

    # Dimensions
    # f1 evaluated at: y_0, ..., y_{K-1}, x_K, x_s (K+2 points)
    # f2 evaluated at: x_0, ..., x_K, x_s (K+2 points)
    # Gram basis: [x_0-x_s, g(y_0), g(y_1), ..., g(y_{K-1}), g(x_K), h_0, h_1, ..., h_K, g_s]
    # Total: 1 + K + 1 + (K+1) + 1 = 2K + 4
    dimG = 2 * K_max + 4
    dimF1 = K_max + 2  # f1 at y_0, ..., y_{K-1}, x_K, x_s
    dimF2 = K_max + 2  # f2 at x_0, ..., x_K, x_s

    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)
    eyeF2 = jnp.eye(dimF2)

    # Index arrays for Gram basis
    # g(y_k) at index 1 + k for k = 0, ..., K-1
    # g(x_K) at index 1 + K = K + 1
    # h_k at index K + 2 + k for k = 0, ..., K
    # g_s at index K + 2 + K + 1 = 2K + 3
    idx_g_y = jnp.arange(1, K_max + 1)  # indices 1 to K for g(y_0) to g(y_{K-1})
    idx_g_xK = K_max + 1  # index for g(x_K)
    idx_h = jnp.arange(K_max + 2, 2 * K_max + 3)  # indices K+2 to 2K+2 for h_0 to h_K
    idx_gs = 2 * K_max + 3  # g_s (and h_s = -g_s)

    # Number of points
    n_f1_points = K_max + 1  # y_0, ..., y_{K-1}, x_K (not counting x_s)
    n_f2_points = K_max + 1  # x_0, ..., x_K (not counting x_s)

    # f1 representations: y_0, ..., y_{K-1}, x_K, x_s
    repY_f1 = jnp.zeros((n_f1_points + 1, dimG))
    repG_f1 = jnp.zeros((n_f1_points + 1, dimG))
    repF_f1 = jnp.zeros((n_f1_points + 1, dimF1))

    # f2 representations: x_0, ..., x_K, x_s
    repX_f2 = jnp.zeros((n_f2_points + 1, dimG))
    repG_f2 = jnp.zeros((n_f2_points + 1, dimG))
    repF_f2 = jnp.zeros((n_f2_points + 1, dimF2))

    # Initial: y_0 = x_0
    x_rep = eyeG[0, :]  # x_0 - x_s
    y_rep = x_rep  # y_0 = x_0

    # y_0 for f1
    repY_f1 = repY_f1.at[0].set(y_rep)
    repG_f1 = repG_f1.at[0].set(eyeG[idx_g_y[0], :])  # g(y_0)
    repF_f1 = repF_f1.at[0].set(eyeF1[0, :])

    # x_0 for f2
    repX_f2 = repX_f2.at[0].set(x_rep)
    repG_f2 = repG_f2.at[0].set(eyeG[idx_h[0], :])  # h_0
    repF_f2 = repF_f2.at[0].set(eyeF2[0, :])

    # FISTA dynamics
    def fista_step(k, carry):
        repY_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_prev, y_prev = carry
        t_k = t_vec[k]

        # Momentum coefficient: (beta_k - 1) / beta_{k+1}
        mom_coef = (beta[k] - 1.0) / beta[k + 1]

        # x_{k+1} = y_k - t_k * g(y_k) - t_k * h_{k+1}
        g_yk = eyeG[idx_g_y[k], :]
        h_kp1 = eyeG[idx_h[k + 1], :]

        x_new = y_prev - t_k * g_yk - t_k * h_kp1

        # Store x_{k+1} for f2
        repX_f2 = repX_f2.at[k + 1].set(x_new)
        repG_f2 = repG_f2.at[k + 1].set(h_kp1)
        repF_f2 = repF_f2.at[k + 1].set(eyeF2[k + 1, :])

        # y_{k+1} = x_{k+1} + mom_coef * (x_{k+1} - x_k)
        y_new = x_new + mom_coef * (x_new - x_prev)

        # Store y_{k+1} for f1 (only for k < K-1, since y_K is not used)
        # For f1 interpolation, we store y_1, ..., y_{K-1} at indices 1, ..., K-1
        repY_f1 = jax.lax.cond(
            k < K_max - 1,
            lambda args: args[0].at[args[1] + 1].set(args[2]),
            lambda args: args[0],
            (repY_f1, k, y_new)
        )
        repG_f1 = jax.lax.cond(
            k < K_max - 1,
            lambda args: args[0].at[args[1] + 1].set(eyeG[idx_g_y[args[1] + 1], :]),
            lambda args: args[0],
            (repG_f1, k, None)
        )
        repF_f1 = jax.lax.cond(
            k < K_max - 1,
            lambda args: args[0].at[args[1] + 1].set(eyeF1[args[1] + 1, :]),
            lambda args: args[0],
            (repF_f1, k, None)
        )

        return (repY_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_new, y_new)

    init_carry = (repY_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_rep, y_rep)
    repY_f1, repX_f2, repG_f1, repG_f2, repF_f1, repF_f2, x_final, y_final = jax.lax.fori_loop(
        0, K_max, fista_step, init_carry
    )

    # x_K is already stored in repX_f2[K_max]
    # Now add x_K to f1 interpolation (for full obj_val)
    # x_K is at position K in repY_f1 (index K_max for f1 points: y_0, ..., y_{K-1}, x_K)
    repY_f1 = repY_f1.at[K_max].set(x_final)  # x_K position
    repG_f1 = repG_f1.at[K_max].set(eyeG[idx_g_xK, :])  # g(x_K)
    repF_f1 = repF_f1.at[K_max].set(eyeF1[K_max, :])  # f1(x_K)

    # Optimal point x_s = y_s
    xs_rep = jnp.zeros(dimG)
    repY_f1 = repY_f1.at[n_f1_points].set(xs_rep)
    repG_f1 = repG_f1.at[n_f1_points].set(eyeG[idx_gs, :])  # g_s
    repF_f1 = repF_f1.at[n_f1_points].set(jnp.zeros(dimF1))

    repX_f2 = repX_f2.at[n_f2_points].set(xs_rep)
    repG_f2 = repG_f2.at[n_f2_points].set(-eyeG[idx_gs, :])  # h_s = -g_s
    repF_f2 = repF_f2.at[n_f2_points].set(jnp.zeros(dimF2))

    # Compute interpolation conditions.
    # Stationarity: g_s + h_s = 0, so h_s = -g_s. No separate h_s basis vector
    # in FISTA's Gram basis; we represent h_s as -eyeG[idx_gs, :].
    A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
        repY_f1, repG_f1, repF_f1, mu, L, n_f1_points, gs=eyeG[idx_gs, :]
    )

    A_vals_f2, b_vals_f2 = convex_interp(
        repX_f2, repG_f2, repF_f2, n_f2_points, gs=-eyeG[idx_gs, :]
    )

    # Combine constraints
    dimF = dimF1 + dimF2

    b_vals_f1_combined = jnp.concatenate([b_vals_f1, jnp.zeros((b_vals_f1.shape[0], dimF2))], axis=1)
    b_vals_f2_combined = jnp.concatenate([jnp.zeros((b_vals_f2.shape[0], dimF1)), b_vals_f2], axis=1)

    A_vals = jnp.concatenate([A_vals_f1, A_vals_f2], axis=0)
    b_vals = jnp.concatenate([b_vals_f1_combined, b_vals_f2_combined], axis=0)

    num_constraints = A_vals.shape[0]
    c_vals = jnp.zeros(num_constraints)

    # Initial condition
    A_init = jnp.outer(repX_f2[0], repX_f2[0])
    b_init = jnp.zeros(dimF)
    c_init = -R ** 2

    A_vals = jnp.concatenate([A_vals, A_init[None, :, :]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None, :]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)

    # Objective: use composition
    # For FISTA, repY_f1 contains y points and x_K at position K_max
    # repG_f1 contains g(y) gradients and g(x_K)
    # repG_f2 contains h subgradients
    obj_builder = _create_composite_obj_builder(
        repY_f1, repG_f1, repF_f1, repG_f2, repF_f2, dimG, dimF1, dimF2, pep_obj
    )
    A_obj, b_obj = compose_objective(obj_builder, K_max, composition_type, decay_rate)

    PSD_A_vals = []
    PSD_b_vals = []
    PSD_c_vals = []
    PSD_shapes = []

    pep_data = (A_obj, b_obj, A_vals, b_vals, c_vals,
                PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)

    return pep_data


def ista_pep_data_to_numpy(pep_data):
    """
    Convert JAX arrays in ISTA/FISTA pep_data to numpy arrays.

    Args:
        pep_data: Tuple from construct_ista_pep_data or construct_fista_pep_data

    Returns:
        pep_data_np: Same tuple structure with numpy arrays
    """
    import numpy as np

    (A_obj, b_obj, A_vals, b_vals, c_vals,
     PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes) = pep_data

    return (
        np.asarray(A_obj),
        np.asarray(b_obj),
        np.asarray(A_vals),
        np.asarray(b_vals),
        np.asarray(c_vals),
        [np.asarray(a) for a in PSD_A_vals],
        [np.asarray(b) for b in PSD_b_vals],
        [np.asarray(c) for c in PSD_c_vals],
        PSD_shapes
    )
