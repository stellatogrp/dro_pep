"""
JAX-compatible interpolation conditions for PEP.

Provides functions to compute interpolation constraint matrices (A, b) for
smooth strongly convex functions, designed to work with JAX autodiff.
"""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['n_points'])
def smooth_strongly_convex_interp(repX, repG, repF, mu, L, n_points):
    """
    Compute interpolation conditions for smooth strongly convex functions.

    JAX-compatible version that uses fori_loop for nested iteration.

    Args:
        repX: Array of point representations (n_points+1, dimG)
              Includes algorithm iterates and optimal point as last row.
        repG: Array of gradient representations (n_points+1, dimG)
        repF: Array of function value representations (n_points+1, dimF)
        mu: Strong convexity parameter
        L: Lipschitz constant of gradient
        n_points: Number of algorithm points (excluding optimal point)

    Returns:
        A_vals: Array of constraint matrices ((n_points+1)*(n_points), dimG, dimG)
        b_vals: Array of constraint vectors ((n_points+1)*(n_points), dimF)

    Note: The number of constraints is (n_points+1)*(n_points) since we have
    all (i,j) pairs with i != j.
    """
    dimG = repX.shape[1]
    dimF = repF.shape[1]
    total_points = n_points + 1  # Includes optimal point

    # Total number of (i,j) pairs with i != j
    num_constraints = total_points * (total_points - 1)

    # Pre-allocate output arrays
    A_vals = jnp.zeros((num_constraints, dimG, dimG))
    b_vals = jnp.zeros((num_constraints, dimF))

    # Precompute coefficient for interpolation
    coeff = 1.0 / (2.0 * (1.0 - mu / L))

    def compute_single_constraint(i, j, repX, repG, repF, mu, L, coeff):
        """Compute A and b for a single (i, j) interpolation constraint."""
        xi, xj = repX[i, :], repX[j, :]
        gi, gj = repG[i, :], repG[j, :]
        fi, fj = repF[i, :], repF[j, :]

        diff_x = xi - xj
        diff_g = gi - gj

        # A_ij = (1/2) * (gj ⊗ (xi-xj) + (xi-xj) ⊗ gj)
        #      + coeff * ((1/L)(gi-gj)⊗(gi-gj) + μ(xi-xj)⊗(xi-xj)
        #                 - (μ/L)(gi-gj)⊗(xi-xj) - (μ/L)(xi-xj)⊗(gi-gj))
        A_ij = 0.5 * jnp.outer(gj, diff_x) + 0.5 * jnp.outer(diff_x, gj)
        A_ij = A_ij + coeff * (
            (1.0 / L) * jnp.outer(diff_g, diff_g)
            + mu * jnp.outer(diff_x, diff_x)
            - (mu / L) * jnp.outer(diff_g, diff_x)
            - (mu / L) * jnp.outer(diff_x, diff_g)
        )

        # b_ij = fj - fi
        b_ij = fj - fi

        return A_ij, b_ij

    def outer_body(i, carry):
        """Outer loop body over i."""
        A_vals, b_vals, constraint_idx = carry

        def inner_body(j, inner_carry):
            """Inner loop body over j, skipping i==j."""
            A_vals, b_vals, constraint_idx = inner_carry

            # Compute constraint for (i, j)
            A_ij, b_ij = compute_single_constraint(
                i, j, repX, repG, repF, mu, L, coeff
            )

            # Only update if i != j (use jax.lax.cond for conditional)
            def update_fn(_):
                return (
                    A_vals.at[constraint_idx].set(A_ij),
                    b_vals.at[constraint_idx].set(b_ij),
                    constraint_idx + 1
                )

            def skip_fn(_):
                return (A_vals, b_vals, constraint_idx)

            return jax.lax.cond(i != j, update_fn, skip_fn, operand=None)

        # Run inner loop over j
        A_vals, b_vals, constraint_idx = jax.lax.fori_loop(
            0, total_points, inner_body, (A_vals, b_vals, constraint_idx)
        )

        return (A_vals, b_vals, constraint_idx)

    # Run outer loop over i
    A_vals, b_vals, _ = jax.lax.fori_loop(
        0, total_points, outer_body, (A_vals, b_vals, 0)
    )

    return A_vals, b_vals


@partial(jax.jit, static_argnames=['n_points'])
def smooth_strongly_convex_interp_consecutive(repX, repG, repF, mu, L, n_points):
    """
    Compute interpolation conditions for consecutive points only.

    This is a reduced version that only considers (k, k+1) interpolation
    conditions plus all (s, j) conditions where s is the optimal point index.
    Useful for gradient descent analysis where only consecutive relationships matter.

    Args:
        repX: Array of point representations (n_points+1, dimG)
        repG: Array of gradient representations (n_points+1, dimG)
        repF: Array of function value representations (n_points+1, dimF)
        mu: Strong convexity parameter
        L: Lipschitz constant of gradient
        n_points: Number of algorithm points (excluding optimal point)

    Returns:
        A_vals: Array of constraint matrices (2*n_points - 1, dimG, dimG)
        b_vals: Array of constraint vectors (2*n_points - 1, dimF)
    """
    dimG = repX.shape[1]
    dimF = repF.shape[1]
    s = n_points  # Index of optimal point

    # Number of constraints: (n_points-1) for consecutive + n_points for (s, j)
    num_constraints = 2 * n_points - 1

    A_vals = jnp.zeros((num_constraints, dimG, dimG))
    b_vals = jnp.zeros((num_constraints, dimF))

    coeff = 1.0 / (2.0 * (1.0 - mu / L))

    def compute_constraint(i, j):
        """Compute A and b for constraint (i, j)."""
        xi, xj = repX[i, :], repX[j, :]
        gi, gj = repG[i, :], repG[j, :]
        fi, fj = repF[i, :], repF[j, :]

        diff_x = xi - xj
        diff_g = gi - gj

        A_ij = 0.5 * jnp.outer(gj, diff_x) + 0.5 * jnp.outer(diff_x, gj)
        A_ij = A_ij + coeff * (
            (1.0 / L) * jnp.outer(diff_g, diff_g)
            + mu * jnp.outer(diff_x, diff_x)
            - (mu / L) * jnp.outer(diff_g, diff_x)
            - (mu / L) * jnp.outer(diff_x, diff_g)
        )
        b_ij = fj - fi

        return A_ij, b_ij

    # Add (k, k+1) constraints for k = 0, ..., n_points-2
    def consecutive_body(k, carry):
        A_vals, b_vals = carry
        A_k, b_k = compute_constraint(k, k + 1)
        A_vals = A_vals.at[k].set(A_k)
        b_vals = b_vals.at[k].set(b_k)
        return A_vals, b_vals

    A_vals, b_vals = jax.lax.fori_loop(
        0, n_points - 1, consecutive_body, (A_vals, b_vals)
    )

    # Add (s, j) constraints for j = 0, ..., n_points-1
    offset = n_points - 1

    def optimal_body(j, carry):
        A_vals, b_vals = carry
        A_sj, b_sj = compute_constraint(s, j)
        A_vals = A_vals.at[offset + j].set(A_sj)
        b_vals = b_vals.at[offset + j].set(b_sj)
        return A_vals, b_vals

    A_vals, b_vals = jax.lax.fori_loop(
        0, n_points, optimal_body, (A_vals, b_vals)
    )

    return A_vals, b_vals


@partial(jax.jit, static_argnames=['n_points'])
def convex_interp(repX, repG, repF, n_points):
    """
    Compute interpolation conditions for convex functions.

    For a convex function h, the interpolation condition for points (i, j) is:
        h(x_i) - h(x_j) - <g_j, x_i - x_j> >= 0

    In matrix form for SDP (with <= constraint):
        trace(A_ij @ G) + b_ij^T @ F <= 0
    where:
        A_ij = (1/2)(g_j ⊗ (x_i - x_j) + (x_i - x_j) ⊗ g_j)
        b_ij = f_j - f_i

    Args:
        repX: Array of point representations (n_points+1, dimG)
              Includes algorithm iterates and optimal point as last row.
        repG: Array of (sub)gradient representations (n_points+1, dimG)
        repF: Array of function value representations (n_points+1, dimF)
        n_points: Number of algorithm points (excluding optimal point)

    Returns:
        A_vals: Array of constraint matrices ((n_points+1)*(n_points), dimG, dimG)
        b_vals: Array of constraint vectors ((n_points+1)*(n_points), dimF)
    """
    dimG = repX.shape[1]
    dimF = repF.shape[1]
    total_points = n_points + 1  # Includes optimal point

    # Total number of (i,j) pairs with i != j
    num_constraints = total_points * (total_points - 1)

    # Pre-allocate output arrays
    A_vals = jnp.zeros((num_constraints, dimG, dimG))
    b_vals = jnp.zeros((num_constraints, dimF))

    def compute_single_constraint(i, j, repX, repG, repF):
        """Compute A and b for a single (i, j) interpolation constraint."""
        xi, xj = repX[i, :], repX[j, :]
        gj = repG[j, :]
        fi, fj = repF[i, :], repF[j, :]

        diff_x = xi - xj

        # A_ij = (1/2) * (gj ⊗ (xi-xj) + (xi-xj) ⊗ gj)
        A_ij = 0.5 * jnp.outer(gj, diff_x) + 0.5 * jnp.outer(diff_x, gj)

        # b_ij = fj - fi
        b_ij = fj - fi

        return A_ij, b_ij

    def outer_body(i, carry):
        """Outer loop body over i."""
        A_vals, b_vals, constraint_idx = carry

        def inner_body(j, inner_carry):
            """Inner loop body over j, skipping i==j."""
            A_vals, b_vals, constraint_idx = inner_carry

            # Compute constraint for (i, j)
            A_ij, b_ij = compute_single_constraint(i, j, repX, repG, repF)

            # Only update if i != j
            def update_fn(_):
                return (
                    A_vals.at[constraint_idx].set(A_ij),
                    b_vals.at[constraint_idx].set(b_ij),
                    constraint_idx + 1
                )

            def skip_fn(_):
                return (A_vals, b_vals, constraint_idx)

            return jax.lax.cond(i != j, update_fn, skip_fn, operand=None)

        # Run inner loop over j
        A_vals, b_vals, constraint_idx = jax.lax.fori_loop(
            0, total_points, inner_body, (A_vals, b_vals, constraint_idx)
        )

        return (A_vals, b_vals, constraint_idx)

    # Run outer loop over i
    A_vals, b_vals, _ = jax.lax.fori_loop(
        0, total_points, outer_body, (A_vals, b_vals, 0)
    )

    return A_vals, b_vals


@partial(jax.jit, static_argnames=['K'])
def smooth_strongly_convex_proximal_gradient_interp(
    repX_f1, repG_f1, repF_f1,
    repX_f2, repG_f2, repF_f2,
    mu, L, K
):
    """
    Compute interpolation conditions for proximal gradient descent.

    This function handles the composite minimization problem:
        min F(x) = f1(x) + f2(x)
    where f1 is L-smooth and mu-strongly convex, f2 is closed convex proper.

    The proximal gradient algorithm:
        y_k = x_k - gamma * grad_f1(x_k)
        x_{k+1} = prox_{gamma * f2}(y_k) = y_k - gamma * h_{k+1}
    where h_{k+1} is a subgradient of f2 at x_{k+1}.

    Representation structure for K iterations:
        - f1 points: x_0, x_1, ..., x_K, x_s (K+2 points total)
        - f2 points: x_0, x_1, x_2, ..., x_K, x_s (K+2 points total)

    Gram basis (dimG = 2K + 3):
        [x_0 - x_s, g_0, h_0, h_1, g_1, h_2, g_2, ..., h_K, g_K]
        where g_k = grad_f1(x_k), h_k = subgrad_f2(x_k)

        Note: h_0 is a free subgradient at x_0 (not used in algorithm dynamics,
        but needed for complete interpolation conditions).

    Function values:
        - dimF1 = K + 1: [f1(x_0)-f1_s, f1(x_1)-f1_s, ..., f1(x_K)-f1_s]
        - dimF2 = K + 1: [f2(x_0)-f2_s, f2(x_1)-f2_s, ..., f2(x_K)-f2_s]

    Args:
        repX_f1: Point representations for f1 (K+2, dimG), includes x_s at end
        repG_f1: Gradient representations for f1 (K+2, dimG)
        repF_f1: Function value representations for f1 (K+2, dimF1)
        repX_f2: Point representations for f2 (K+2, dimG), includes x_s at end
        repG_f2: Subgradient representations for f2 (K+2, dimG), includes h_0
        repF_f2: Function value representations for f2 (K+2, dimF2)
        mu: Strong convexity parameter for f1
        L: Lipschitz constant for grad f1
        K: Number of proximal gradient iterations

    Returns:
        A_vals_f1: Constraint matrices for f1 interpolation
        b_vals_f1: Constraint vectors for f1 interpolation
        A_vals_f2: Constraint matrices for f2 interpolation
        b_vals_f2: Constraint vectors for f2 interpolation
    """
    # Number of points for f1: x_0, x_1, ..., x_K = K+1 algorithm points + x_s
    n_points_f1 = K + 1

    # Number of points for f2: x_0, x_1, x_2, ..., x_K = K+1 algorithm points + x_s
    n_points_f2 = K + 1

    # Compute smooth strongly convex interpolation for f1
    A_vals_f1, b_vals_f1 = smooth_strongly_convex_interp(
        repX_f1, repG_f1, repF_f1, mu, L, n_points_f1
    )

    # Compute convex interpolation for f2
    A_vals_f2, b_vals_f2 = convex_interp(
        repX_f2, repG_f2, repF_f2, n_points_f2
    )

    return A_vals_f1, b_vals_f1, A_vals_f2, b_vals_f2
