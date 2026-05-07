"""
JAX-compatible PEP data construction for Chambolle-Pock algorithm.

Constructs PEP constraint matrices for primal-dual optimization with
linear operators.
"""

import jax
import jax.numpy as jnp
from functools import partial

from .interpolation_conditions import convex_interp
from .loss_compositions import compose_objective


def _create_cp_gap_obj_builder(repX_f1, repY_h, repF_f1, repF_h,
                               idx_xs, idx_ys,
                               gf1_s_vec, gh_s_vec,
                               dimG, dimF1, dimF_h, eyeG):
    """Build obj_builder(k) encoding the duality gap at iterate k.

    Gap_k = f1(x_k) + h(y_k) + <K x_k, y_s> - <K x_s, y_k>
          = F_f1[k] + F_h[k] + <x_k, -gf1_s> - <y_k, gh_s>
    using the saddle-stationarity identities gf1_s = -K^T y_s, gh_s = K x_s
    (enforced as structural operator-pair constraints in the SDP).

    x_k, y_k are used in ORIGINAL (unshifted) coords — repX_f1 stores
    (x_k - x_s), so we add eyeG[idx_xs] to recover x_k. Same for y_k.
    """
    neg_gf1_s = -gf1_s_vec

    def obj_builder(k):
        x_k = repX_f1[k] + eyeG[idx_xs]
        y_k = repY_h[k] + eyeG[idx_ys]

        A1 = 0.5 * (jnp.outer(x_k, neg_gf1_s) + jnp.outer(neg_gf1_s, x_k))
        A2 = -0.5 * (jnp.outer(gh_s_vec, y_k) + jnp.outer(y_k, gh_s_vec))
        A_k = A1 + A2

        b_k = jnp.concatenate([repF_f1[k], repF_h[k]])

        return A_k, b_k

    return obj_builder


@partial(jax.jit, static_argnames=['K_max', 'composition_type'])
def construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K_max,
                                       composition_type='final',
                                       decay_rate=0.9):
    """
    Construct PEP for Chambolle-Pock with a Euclidean-ball initial condition:
        ||x0 - xs||^2 + ||u0 - us||^2 <= R^2.

    Prior form was the P-norm (Lyapunov) IC
        ||dx||^2/tau + ||du||^2/sigma - 2<K dx, du> <= R^2
    which baked stepsizes and K into the IC. The Euclidean form is
    decoupled and was confirmed to keep the PEP bounded via the PEPit
    probe in tests/test_cp_ic_boundedness_probe.py.

    Supports two composition types for the duality-gap performance metric:
      - 'final':    Gap at the last iterate K_max (single-term objective).
      - 'weighted': Normalized decay-weighted sum sum_{k=1}^{K_max} w_k * Gap_k
                    with w_k = decay_rate^(K_max - k) (k=0 is skipped — both
                    iterate-0 subgradients gf1_0, gh_0 are free and would
                    unbound the objective; see loss_compositions.py).

    Structural constraints:
    - Adjoint Consistency: Enforces <Ku, p> = <u, K.T p> for ALL operator pairs.
    - Saddle stationarity: gf1_s = -K^T y_s and gh_s = K x_s enforced via
      operator pairs (x_s, gh_s) in pairs_K and (y_s, -gf1_s) in pairs_Kt.
    - Objective Construction: Uses stationarity substitutions <K xK, y*> = <xK, -df(x*)>.

    Args:
        tau: Primal step sizes - scalar or vector of length K_max
        sigma: Dual step sizes - scalar or vector of length K_max
        theta: Extrapolation parameters - scalar or vector of length K_max
        M: Operator norm bound (||K|| <= M)
        R: Euclidean initial radius bound
        K_max: Number of iterations
        composition_type: 'final' or 'weighted'
        decay_rate: Decay for weighted composition (w_k = decay_rate^(K_max - k))

    Returns:
        pep_data tuple
    """

    # 1. Setup Parameters
    tau_vec = jnp.broadcast_to(tau, (K_max,))
    sigma_vec = jnp.broadcast_to(sigma, (K_max,))
    theta_vec = jnp.broadcast_to(theta, (K_max,))

    # 2. Define Basis Dimensions
    # Init(4) + Grads(2*(K+2)) + Trace(2*K) + Analysis(3)
    dimG = 4 + 2*(K_max + 2) + 2*K_max + 3
    dimF1 = K_max + 2
    dimF_h = K_max + 2
    dimF = dimF1 + dimF_h

    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)
    eyeF_h = jnp.eye(dimF_h)

    # --- Index Management ---
    idx_c = 0
    idx_dx0 = idx_c; idx_c+=1
    idx_dy0 = idx_c; idx_c+=1
    idx_xs  = idx_c; idx_c+=1
    idx_ys  = idx_c; idx_c+=1

    idx_gf1_start = idx_c; idx_c += (K_max + 2)
    idx_gh_start  = idx_c; idx_c += (K_max + 2)

    idx_w_start = idx_c; idx_c += K_max
    idx_z_start = idx_c; idx_c += K_max

    # Analysis Vectors
    idx_K_xK  = idx_c; idx_c+=1  # K * x_K
    idx_Kt_yK = idx_c; idx_c+=1  # K^T * y_K
    idx_K_dx0 = idx_c; idx_c+=1  # K * dx0 (declared via pairs_K for operator-norm coherence; unused in the Euclidean IC)

    def gf1_vec(k): return eyeG[idx_gf1_start + k]
    def gh_vec(k): return eyeG[idx_gh_start + k]
    def w_vec(k): return eyeG[idx_w_start + k]
    def z_vec(k): return eyeG[idx_z_start + (k-1)]

    # 3. Algorithm Trace
    n_points = K_max + 2
    idx_saddle = K_max + 1

    repX_f1 = jnp.zeros((n_points, dimG))
    repG_f1 = jnp.zeros((n_points, dimG))
    repF_f1 = jnp.zeros((n_points, dimF1))

    repY_h = jnp.zeros((n_points, dimG))
    repG_h = jnp.zeros((n_points, dimG))
    repF_h = jnp.zeros((n_points, dimF_h))

    pairs_K = []   # (u, v) -> v = K u
    pairs_Kt = []  # (p, q) -> q = K^T p

    # -- Init --
    x_curr = eyeG[idx_xs] + eyeG[idx_dx0]
    y_curr = eyeG[idx_ys] + eyeG[idx_dy0]

    repX_f1 = repX_f1.at[0].set(x_curr - eyeG[idx_xs])
    repG_f1 = repG_f1.at[0].set(gf1_vec(0))
    repF_f1 = repF_f1.at[0].set(eyeF1[0])

    repY_h = repY_h.at[0].set(y_curr - eyeG[idx_ys])
    repG_h = repG_h.at[0].set(gh_vec(0))
    repF_h = repF_h.at[0].set(eyeF_h[0])

    # -- Loop --
    for k in range(K_max):
        t, s, th = tau_vec[k], sigma_vec[k], theta_vec[k]

        # Primal
        w_k = w_vec(k)
        pairs_Kt.append((y_curr, w_k))
        x_next = x_curr - t * w_k - t * gf1_vec(k+1)

        repX_f1 = repX_f1.at[k+1].set(x_next - eyeG[idx_xs])
        repG_f1 = repG_f1.at[k+1].set(gf1_vec(k+1))
        repF_f1 = repF_f1.at[k+1].set(eyeF1[k+1])

        # Extrapolation
        x_bar = x_next + th * (x_next - x_curr)

        # Dual
        z_kp1 = z_vec(k+1)
        pairs_K.append((x_bar, z_kp1))
        y_next = y_curr + s * z_kp1 - s * gh_vec(k+1)

        repY_h = repY_h.at[k+1].set(y_next - eyeG[idx_ys])
        repG_h = repG_h.at[k+1].set(gh_vec(k+1))
        repF_h = repF_h.at[k+1].set(eyeF_h[k+1])

        x_curr = x_next
        y_curr = y_next

    # -- Saddle Point --
    repX_f1 = repX_f1.at[idx_saddle].set(jnp.zeros(dimG))
    repG_f1 = repG_f1.at[idx_saddle].set(gf1_vec(idx_saddle))
    repF_f1 = repF_f1.at[idx_saddle].set(jnp.zeros(dimF1))

    repY_h = repY_h.at[idx_saddle].set(jnp.zeros(dimG))
    repG_h = repG_h.at[idx_saddle].set(gh_vec(idx_saddle))
    repF_h = repF_h.at[idx_saddle].set(jnp.zeros(dimF_h))

    # 4. Interpolation Constraints
    # convex_interp expects n_points = number of algorithm points (excluding optimal)
    # We have K_max + 1 iterates (x_0 to x_K_max) + 1 saddle point = K_max + 2 total
    # So we pass K_max + 1 as the number of algorithm points.
    #
    # Pass the saddle-point subgradient basis slots as gs= — analogous to the
    # ISTA/FISTA fix. Without this, convex_interp defaults to gs=0, which
    # contradicts the generically-nonzero saddle subgradients gf1_s = -K^T u_s
    # and gh_s = K x_s required by saddle-point stationarity.
    n_algo_points = K_max + 1
    A_f1, b_f1 = convex_interp(repX_f1, repG_f1, repF_f1, n_algo_points,
                               gs=gf1_vec(idx_saddle))
    A_h, b_h = convex_interp(repY_h, repG_h, repF_h, n_algo_points,
                             gs=gh_vec(idx_saddle))

    b_f1_pad = jnp.concatenate([b_f1, jnp.zeros((b_f1.shape[0], dimF_h))], axis=1)
    b_h_pad  = jnp.concatenate([jnp.zeros((b_h.shape[0], dimF1)), b_h], axis=1)

    A_vals = jnp.concatenate([A_f1, A_h], axis=0)
    b_vals = jnp.concatenate([b_f1_pad, b_h_pad], axis=0)
    c_vals = jnp.zeros(A_vals.shape[0])

    # 5. Value Pinning (f(x_*)=0, h(y_*)=0)
    row_f = jnp.zeros(dimF); row_f = row_f.at[idx_saddle].set(1.0)
    row_h = jnp.zeros(dimF); row_h = row_h.at[dimF1 + idx_saddle].set(1.0)

    A_zero = jnp.zeros((4, dimG, dimG))
    b_zero = jnp.stack([row_f, -row_f, row_h, -row_h])
    c_zero = jnp.zeros(4)

    A_vals = jnp.concatenate([A_vals, A_zero], axis=0)
    b_vals = jnp.concatenate([b_vals, b_zero], axis=0)
    c_vals = jnp.concatenate([c_vals, c_zero], axis=0)

    # NOTE: Solution bound ||x_s||^2 + ||y_s||^2 <= 1 removed.
    # PEPit doesn't require this - the KKT operator pairs are sufficient to pin down
    # the saddle point. This constraint only works with shifted coordinates (xs=0, ys=0)
    # and fails when xs=x_opt, ys=y_opt in original coordinates.

    # 6. Gather Operator Pairs
    # Saddle-point stationarity identities (CP analog of ISTA's g_s + h_s = 0).
    # The saddle (x_s, u_s) satisfies
    #     -K^T u_s ∈ ∂f1(x_s)    i.e.,   gf1_s = -K^T u_s
    #      K   x_s ∈ ∂h(u_s)     i.e.,   gh_s  =  K   x_s
    # Encode these through operator pairs so the PSD and adjoint blocks see them:
    #     pairs_K.append((x_s, K x_s))   with K x_s = gh_s
    #     pairs_Kt.append((u_s, K^T u_s))  with K^T u_s = -gf1_s
    # This replaces the previous vacuous pairs (x_s, 0) / (u_s, 0) which both
    # misstated the identity and left gf1_s, gh_s structurally decoupled.
    pairs_K.append((eyeG[idx_xs], gh_vec(idx_saddle)))      # (x_s, K x_s = gh_s)
    pairs_Kt.append((eyeG[idx_ys], -gf1_vec(idx_saddle)))   # (u_s, K^T u_s = -gf1_s)

    # Declare (dx0, K*dx0) as a K-pair so the operator-norm PSD block sees it.
    # Vestigial from the prior P-norm IC cross-term; harmless under the
    # current Euclidean IC (the corresponding basis vector simply does not
    # appear in any other constraint).
    pairs_K.append((eyeG[idx_dx0], eyeG[idx_K_dx0]))

    # Final Iterate Observations
    pairs_K.append((x_curr, eyeG[idx_K_xK]))
    pairs_Kt.append((y_curr, eyeG[idx_Kt_yK]))

    # 7. Build TWO SEPARATE PSD Constraints (X->Y and Y->X)
    n_K = len(pairs_K)
    A_psd_K = jnp.zeros((n_K, n_K, dimG, dimG))
    for i in range(n_K):
        u_i, v_i = pairs_K[i]
        for j in range(n_K):
            u_j, v_j = pairs_K[j]
            # M^2 <u_i, u_j> - <v_i, v_j> >= 0
            term_u = 0.5 * (jnp.outer(u_i, u_j) + jnp.outer(u_j, u_i)) * (M**2)
            term_v = 0.5 * (jnp.outer(v_i, v_j) + jnp.outer(v_j, v_i)) * (-1.0)
            A_psd_K = A_psd_K.at[i, j, :, :].add(term_u + term_v)

    n_Kt = len(pairs_Kt)
    A_psd_Kt = jnp.zeros((n_Kt, n_Kt, dimG, dimG))
    for i in range(n_Kt):
        p_i, q_i = pairs_Kt[i]
        for j in range(n_Kt):
            p_j, q_j = pairs_Kt[j]
            # M^2 <p_i, p_j> - <q_i, q_j> >= 0
            term_p = 0.5 * (jnp.outer(p_i, p_j) + jnp.outer(p_j, p_i)) * (M**2)
            term_q = 0.5 * (jnp.outer(q_i, q_j) + jnp.outer(q_j, q_i)) * (-1.0)
            A_psd_Kt = A_psd_Kt.at[i, j, :, :].add(term_p + term_q)

    PSD_A_vals = [A_psd_K, A_psd_Kt]
    PSD_b_vals = [jnp.zeros((n_K, n_K, dimF)), jnp.zeros((n_Kt, n_Kt, dimF))]
    PSD_c_vals = [jnp.zeros((n_K, n_K)), jnp.zeros((n_Kt, n_Kt))]
    PSD_shapes = [n_K, n_Kt]

    # 8. Adjoint Consistency
    # Enforces <K u, p> = <u, K^T p> for all pairs.
    # This links the K matrices in the P-norm, the Trace, and the Objective.
    adj_A_list = []
    for i in range(n_K):
        u_vec, v_vec = pairs_K[i]
        for j in range(n_Kt):
            p_vec, q_vec = pairs_Kt[j]

            # <v, p> - <u, q> = 0
            term_vp = 0.5 * (jnp.outer(v_vec, p_vec) + jnp.outer(p_vec, v_vec))
            term_uq = 0.5 * (jnp.outer(u_vec, q_vec) + jnp.outer(q_vec, u_vec))

            A_diff = term_vp - term_uq
            adj_A_list.append(A_diff)
            adj_A_list.append(-A_diff)

    if adj_A_list:
        A_adj = jnp.stack(adj_A_list)
        b_adj = jnp.zeros((len(adj_A_list), dimF))
        c_adj = jnp.zeros(len(adj_A_list))
        A_vals = jnp.concatenate([A_vals, A_adj], axis=0)
        b_vals = jnp.concatenate([b_vals, b_adj], axis=0)
        c_vals = jnp.concatenate([c_vals, c_adj], axis=0)

    # 9. Euclidean Initial Condition
    # ||dx0||^2 + ||du0||^2 <= R^2
    # (Replaces the prior P-norm Lyapunov form
    #  (1/tau)||dx0||^2 + (1/sigma)||du0||^2 - 2<K dx0, du0> <= R^2.
    #  The weaker Euclidean IC was verified to keep the CP PEP bounded via
    #  the PEPit probe in tests/test_cp_ic_boundedness_probe.py and removes
    #  the stepsize coupling that complicated PDLP LDRO-PEP training.)
    vec_dx0 = eyeG[idx_dx0]
    vec_dy0 = eyeG[idx_dy0]

    A_init = jnp.outer(vec_dx0, vec_dx0) + jnp.outer(vec_dy0, vec_dy0)
    b_init = jnp.zeros(dimF)
    c_init = -R**2

    A_vals = jnp.concatenate([A_vals, A_init[None]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)

    # 10. Objective: duality gap, dispatched by composition_type.
    # At iterate k: Gap_k = f1(x_k) + h(y_k) + <K x_k, y_s> - <K x_s, y_k>.
    # Using saddle stationarity -K^T y_s = gf1_s and K x_s = gh_s (enforced
    # via operator pairs in Section 6), the cross-terms simplify to
    #     <K x_k, y_s> = <x_k, -gf1_s>    and    -<K x_s, y_k> = -<y_k, gh_s>,
    # which avoids needing K@x_k basis slots for intermediate k.
    obj_builder = _create_cp_gap_obj_builder(
        repX_f1, repY_h, repF_f1, repF_h,
        idx_xs, idx_ys,
        gf1_vec(idx_saddle), gh_vec(idx_saddle),
        dimG, dimF1, dimF_h, eyeG,
    )
    A_obj, b_obj = compose_objective(
        obj_builder, K_max, composition_type, decay_rate,
    )

    return (A_obj, b_obj, A_vals, b_vals, c_vals,
            PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)


def chambolle_pock_pep_data_to_numpy(pep_data):
    """Convert JAX arrays in Chambolle-Pock pep_data to numpy arrays."""
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
