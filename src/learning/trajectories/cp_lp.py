"""
Chambolle-Pock / PDHG trajectory for box-constrained LPs with mixed
inequality + equality constraints.

This module is the VERIFIED-CORRECT trajectory function for use with
`construct_chambolle_pock_pep_data`. It is a JAX-jitted port of
`run_cp_on_facility` + `build_gram_and_F_facility` from
`tests/test_chambolle_pock_facility_location.py`, where the Gram / F
representation was confirmed to satisfy every CP PEP constraint on real
facility-location LP trajectories.

The legacy `trajectories/pdhg.py` uses a *different* saddle-subgradient
convention (gf1_s = c, gh_s = q) and a non-Euclidean W-metric for Gram
construction — those do not match what the verified CP PEP construction
requires. Use THIS module, not `pdhg.py`, when hooking up a PEP-based
learning pipeline.

LP template handled:
    min_x max_y   L(x, y) = (c^T x + ind_{[l,u]}(x))
                            + <K x, y>
                            - (-q^T y + ind_{R^{m1}_+ × R^{m2}}(y))

where K = K_mat is the stacked constraint operator (in the
facility-location convention K_mat = [-A_ineq; A_eq], q = [-b_ineq; b_eq]).
Box constraints [l, u] apply to the primal; the first m1 dual coords are
constrained to be non-negative (inequality multipliers) while the last m2
are free (equality multipliers).

Saddle stationarity (structurally enforced in `construct_chambolle_pock_pep_data`
via operator-pair identities and used here to build valid saddle subgradients):
    gf1_s =  K_mat^T y_s       (= -K_ours^T y_s, since K_ours = -K_mat)
    gh_s  = -K_mat   v_s       (=  K_ours   v_s)
"""
import jax
import jax.numpy as jnp
from functools import partial


def proj_box(v, l, u):
    return jnp.clip(v, l, u)


def proj_nonneg_first_m1(v, m1):
    n_dim = v.shape[0]
    mask = jnp.arange(n_dim) < m1
    return jnp.where(mask, jnp.maximum(v, 0.0), v)


@partial(jax.jit, static_argnames=['K_max', 'm1', 'return_Gram_representation'])
def problem_data_to_cp_lp_trajectories(
    stepsizes,
    c, K_mat, q, l, u,
    x_opt, y_opt,
    x0, y0,
    K_max, m1,
    return_Gram_representation=True,
):
    """Compute CP/PDHG trajectory + Gram representation for a box-constrained LP.

    Args:
        stepsizes:          tuple (tau, sigma, theta). Each may be a scalar or a
                            (K_max,)-vector.
        c:                  (n_vars,) primal cost.
        K_mat:              (m1 + m2, n_vars) stacked operator [-A_ineq; A_eq].
        q:                  (m1 + m2,) stacked RHS [-b_ineq; b_eq].
        l, u:               (n_vars,) primal box bounds.
        x_opt, y_opt:       saddle primal / dual from the LP solve.
        x0, y0:             initial iterate. Choose (x0, y0) in the STRICT
                            interior of {x: l<x<u} and {y: y[:m1]>0} so that
                            the implied gf1_0 = c and gh_0 = -q are valid
                            subgradients.
        K_max:              number of CP iterations (static).
        m1:                 number of inequality-multiplier dual coords (static).
        return_Gram_representation:
                            if True, returns (G, F) matching the PEP basis.
                            if False, returns raw iterate + subgradient arrays.

    Returns:
        If return_Gram_representation:
            G: (dimG, dimG)  with  dimG = 4*K_max + 11.
            F: (dimF,)       with  dimF = 2*(K_max + 2).
        Else:
            (v_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter)
        where v_iter, y_iter are the iterate trajectories (shape (K_max+1, ·)),
        gf1_iter, gh_iter are Moreau-recovered subgradients (same shape),
        and w_iter = -K_mat^T y_k, z_iter = -K_mat x_bar_k (both (K_max, ·)).
    """
    tau_raw, sigma_raw, theta_raw = stepsizes
    tau = jnp.broadcast_to(tau_raw, (K_max,))
    sigma = jnp.broadcast_to(sigma_raw, (K_max,))
    theta = jnp.broadcast_to(theta_raw, (K_max,))

    n = c.shape[0]
    m = K_mat.shape[0]

    # --- Initialize iterate + subgradient arrays ---
    v_iter = jnp.zeros((K_max + 1, n))
    y_iter = jnp.zeros((K_max + 1, m))
    gf1_iter = jnp.zeros((K_max + 1, n))
    gh_iter = jnp.zeros((K_max + 1, m))
    w_iter = jnp.zeros((K_max, n))  # w_k = K_ours^T y_k = -K_mat^T y_k
    z_iter = jnp.zeros((K_max, m))  # z_k = K_ours x_bar_k = -K_mat x_bar_k

    v_iter = v_iter.at[0].set(x0)
    y_iter = y_iter.at[0].set(y0)
    gf1_iter = gf1_iter.at[0].set(c)     # interior → ∂f1(x0) = {c}
    gh_iter = gh_iter.at[0].set(-q)      # interior → ∂h(y0) = {-q}

    # --- CP loop ---
    def cp_step(k, carry):
        v_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter, v_curr, y_curr = carry
        tau_k, sigma_k, theta_k = tau[k], sigma[k], theta[k]

        # w_k = K_ours^T y_curr = -K_mat^T y_curr
        w_k = -(K_mat.T @ y_curr)
        w_iter = w_iter.at[k].set(w_k)

        # Primal: v_{k+1} = prox_{tau_k f1}(v_curr - tau_k K_ours^T y_curr)
        #                 = proj_{[l,u]}((v_curr - tau_k w_k) - tau_k c)
        prox_in_primal = v_curr - tau_k * w_k  # = v_curr + tau_k K_mat^T y_curr
        v_new = proj_box(prox_in_primal - tau_k * c, l, u)
        gf1_new = (prox_in_primal - v_new) / tau_k

        x_bar = v_new + theta_k * (v_new - v_curr)

        # z_{k+1} = K_ours x_bar = -K_mat x_bar
        z_k = -(K_mat @ x_bar)
        z_iter = z_iter.at[k].set(z_k)

        # Dual: y_{k+1} = prox_{sigma_k h}(y_curr + sigma_k K_ours x_bar)
        #              = partial_relu((y_curr + sigma_k z_k) + sigma_k q)
        prox_in_dual = y_curr + sigma_k * z_k  # = y_curr - sigma_k K_mat x_bar
        y_new = proj_nonneg_first_m1(prox_in_dual + sigma_k * q, m1)
        gh_new = (prox_in_dual - y_new) / sigma_k

        v_iter = v_iter.at[k + 1].set(v_new)
        y_iter = y_iter.at[k + 1].set(y_new)
        gf1_iter = gf1_iter.at[k + 1].set(gf1_new)
        gh_iter = gh_iter.at[k + 1].set(gh_new)

        return v_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter, v_new, y_new

    init_carry = (v_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter, x0, y0)
    v_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter, _, _ = jax.lax.fori_loop(
        0, K_max, cp_step, init_carry,
    )

    if not return_Gram_representation:
        return v_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter

    # --- Build Gram / F representation exactly as build_gram_and_F_facility ---
    dimG = 4 + 2 * (K_max + 2) + 2 * K_max + 3
    dimF1 = K_max + 2
    dimF_h = K_max + 2

    idx_dx0 = 0
    idx_dy0 = 1
    idx_xs = 2
    idx_ys = 3
    idx_gf1_start = 4
    idx_gh_start = idx_gf1_start + (K_max + 2)
    idx_w_start = idx_gh_start + (K_max + 2)
    idx_z_start = idx_w_start + K_max
    idx_K_xK = idx_z_start + K_max
    idx_Kt_yK = idx_K_xK + 1
    idx_K_dx0 = idx_Kt_yK + 1

    D = n + m
    # Build basis as a (dimG, D) matrix of stacked (x-space, y-space) vectors.
    basis = jnp.zeros((dimG, D))

    def emb_x(vec):
        return jnp.concatenate([vec, jnp.zeros(m)])

    def emb_y(vec):
        return jnp.concatenate([jnp.zeros(n), vec])

    v0 = v_iter[0]
    y0_arr = y_iter[0]

    basis = basis.at[idx_dx0].set(emb_x(v0 - x_opt))
    basis = basis.at[idx_dy0].set(emb_y(y0_arr - y_opt))
    basis = basis.at[idx_xs].set(emb_x(x_opt))
    basis = basis.at[idx_ys].set(emb_y(y_opt))

    # gf1_0, ..., gf1_K  then gf1_s
    for k in range(K_max + 1):
        basis = basis.at[idx_gf1_start + k].set(emb_x(gf1_iter[k]))
    gf1_s = K_mat.T @ y_opt
    basis = basis.at[idx_gf1_start + (K_max + 1)].set(emb_x(gf1_s))

    # gh_0, ..., gh_K  then gh_s
    for k in range(K_max + 1):
        basis = basis.at[idx_gh_start + k].set(emb_y(gh_iter[k]))
    gh_s = -K_mat @ x_opt
    basis = basis.at[idx_gh_start + (K_max + 1)].set(emb_y(gh_s))

    for k in range(K_max):
        basis = basis.at[idx_w_start + k].set(emb_x(w_iter[k]))
        basis = basis.at[idx_z_start + k].set(emb_y(z_iter[k]))

    # Analysis slots
    K_xK_vec = -(K_mat @ v_iter[K_max])
    Kt_yK_vec = -(K_mat.T @ y_iter[K_max])
    K_dx0_vec = -(K_mat @ (v0 - x_opt))
    basis = basis.at[idx_K_xK].set(emb_y(K_xK_vec))
    basis = basis.at[idx_Kt_yK].set(emb_x(Kt_yK_vec))
    basis = basis.at[idx_K_dx0].set(emb_y(K_dx0_vec))

    # Euclidean Gram
    G = basis @ basis.T

    # F: shifted function values with f1(x_s) = 0, h(y_s) = 0. Saddle slot = 0.
    F_f1 = jnp.zeros(dimF1)
    F_h = jnp.zeros(dimF_h)
    for k in range(K_max + 1):
        F_f1 = F_f1.at[k].set(jnp.dot(c, v_iter[k] - x_opt))
        F_h = F_h.at[k].set(-jnp.dot(q, y_iter[k] - y_opt))

    F = jnp.concatenate([F_f1, F_h])
    return G, F
