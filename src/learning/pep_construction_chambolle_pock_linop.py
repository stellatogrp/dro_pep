import jax
import jax.numpy as jnp
from functools import partial
from .interpolation_conditions import convex_interp

@partial(jax.jit, static_argnames=['K_max'])
def construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K_max):
    """
    Construct PEP constraint matrices for Chambolle-Pock with exact Gap objective.
    Includes Linear Operator interpolation inequalities for ||K|| <= M.
    """
    # 1. Setup Parameters
    tau_vec = jnp.broadcast_to(tau, (K_max,))
    sigma_vec = jnp.broadcast_to(sigma, (K_max,))
    theta_vec = jnp.broadcast_to(theta, (K_max,))

    # 2. Define Basis Dimensions
    # [FIX 1] We need gradients for iterates 0..K AND saddle point (K+1) -> K+2 total
    # Previous code had K_max + 1, which caused overflow.
    dimG = 4 + 2*(K_max + 2) + 2*K_max + 2
    dimF1 = K_max + 2
    dimF_h = K_max + 2
    dimF = dimF1 + dimF_h

    eyeG = jnp.eye(dimG)
    eyeF1 = jnp.eye(dimF1)
    eyeF_h = jnp.eye(dimF_h)

    # Index Management
    idx_c = 0
    idx_dx0 = idx_c; idx_c+=1
    idx_dy0 = idx_c; idx_c+=1
    idx_xs  = idx_c; idx_c+=1
    idx_ys  = idx_c; idx_c+=1
    
    # [FIX 1] Reserve K_max + 2 slots
    idx_gf1_start = idx_c; idx_c += (K_max + 2)
    idx_gh_start  = idx_c; idx_c += (K_max + 2)
    
    idx_w_start = idx_c; idx_c += K_max # w_k = K^T y_k
    idx_z_start = idx_c; idx_c += K_max # z_k = K x_bar_k

    # Final iterate operator outputs for Objective Gap
    idx_K_xK  = idx_c; idx_c+=1  # K * x_K
    idx_Kt_yK = idx_c; idx_c+=1  # K^T * y_K

    def gf1_vec(k): return eyeG[idx_gf1_start + k]
    def gh_vec(k): return eyeG[idx_gh_start + k]
    def w_vec(k): return eyeG[idx_w_start + k] 
    def z_vec(k): return eyeG[idx_z_start + (k-1)] 

    # 3. Algorithm Trace
    # [FIX 2] Total points = Iterates (K_max + 1) + Saddle Point (1) = K_max + 2
    n_points = K_max + 2
    
    repX_f1 = jnp.zeros((n_points, dimG))
    repG_f1 = jnp.zeros((n_points, dimG))
    repF_f1 = jnp.zeros((n_points, dimF1))
    
    repY_h = jnp.zeros((n_points, dimG))
    repG_h = jnp.zeros((n_points, dimG))
    repF_h = jnp.zeros((n_points, dimF_h))

    pairs_K = []   # (u, v) where v = K u
    pairs_Kt = []  # (p, q) where q = K^T p

    # Init (Step 0)
    x_curr = eyeG[idx_xs] + eyeG[idx_dx0]
    y_curr = eyeG[idx_ys] + eyeG[idx_dy0]

    repX_f1 = repX_f1.at[0].set(x_curr - eyeG[idx_xs])
    repG_f1 = repG_f1.at[0].set(gf1_vec(0))
    repF_f1 = repF_f1.at[0].set(eyeF1[0])

    repY_h = repY_h.at[0].set(y_curr - eyeG[idx_ys])
    repG_h = repG_h.at[0].set(gh_vec(0))
    repF_h = repF_h.at[0].set(eyeF_h[0])

    # Loop
    for k in range(K_max):
        t, s, th = tau_vec[k], sigma_vec[k], theta_vec[k]

        # --- Primal Update ---
        w_k = w_vec(k)
        pairs_Kt.append((y_curr, w_k))

        x_next = x_curr - t * w_k - t * gf1_vec(k+1)

        repX_f1 = repX_f1.at[k+1].set(x_next - eyeG[idx_xs])
        repG_f1 = repG_f1.at[k+1].set(gf1_vec(k+1))
        repF_f1 = repF_f1.at[k+1].set(eyeF1[k+1])

        # --- Extrapolation ---
        x_bar = x_next + th * (x_next - x_curr)

        # --- Dual Update ---
        z_kp1 = z_vec(k+1)
        pairs_K.append((x_bar, z_kp1))

        y_next = y_curr + s * z_kp1 - s * gh_vec(k+1)

        repY_h = repY_h.at[k+1].set(y_next - eyeG[idx_ys])
        repG_h = repG_h.at[k+1].set(gh_vec(k+1))
        repF_h = repF_h.at[k+1].set(eyeF_h[k+1])

        x_curr = x_next
        y_curr = y_next

    # Saddle Point Storage (Free Variables)
    # The saddle point is stored at the last index: K_max + 1
    idx_saddle = K_max + 1
    
    repX_f1 = repX_f1.at[idx_saddle].set(jnp.zeros(dimG))
    repG_f1 = repG_f1.at[idx_saddle].set(gf1_vec(idx_saddle))
    repF_f1 = repF_f1.at[idx_saddle].set(jnp.zeros(dimF1))

    repY_h = repY_h.at[idx_saddle].set(jnp.zeros(dimG))
    repG_h = repG_h.at[idx_saddle].set(gh_vec(idx_saddle))
    repF_h = repF_h.at[idx_saddle].set(jnp.zeros(dimF_h))

    # 4. Standard Convex Interpolation Constraints
    # [FIX 2] Pass 'n_points' (K_max + 2) so the loop includes the saddle point
    A_f1, b_f1 = convex_interp(repX_f1, repG_f1, repF_f1, n_points)
    A_h, b_h = convex_interp(repY_h, repG_h, repF_h, n_points)

    b_f1_pad = jnp.concatenate([b_f1, jnp.zeros((b_f1.shape[0], dimF_h))], axis=1)
    b_h_pad  = jnp.concatenate([jnp.zeros((b_h.shape[0], dimF1)), b_h], axis=1)

    A_scalar = jnp.concatenate([A_f1, A_h], axis=0)
    b_scalar = jnp.concatenate([b_f1_pad, b_h_pad], axis=0)
    c_scalar = jnp.zeros(A_scalar.shape[0])
    
    A_vals = A_scalar
    b_vals = b_scalar
    c_vals = c_scalar

    # 5. Linear Operator PSD Interpolation Constraints
    # -- Add Saddle Point Optimality Conditions --
    # K^T(ys) = -gf1_s.  ys basis is eyeG[idx_ys]. gf1_s is gf1_vec(idx_saddle)
    pairs_Kt.append((eyeG[idx_ys], -gf1_vec(idx_saddle)))
    # K(xs) = gh_s.      xs basis is eyeG[idx_xs]. gh_s is gh_vec(idx_saddle)
    pairs_K.append((eyeG[idx_xs], gh_vec(idx_saddle)))

    # -- Add Final Iterate Operator Outputs for Objective --
    vec_K_xK = eyeG[idx_K_xK]
    pairs_K.append((x_curr, vec_K_xK))
    
    vec_Kt_yK = eyeG[idx_Kt_yK]
    pairs_Kt.append((y_curr, vec_Kt_yK))

    # -- Build H Matrix --
    n_K = len(pairs_K)
    n_Kt = len(pairs_Kt)
    size_H = n_K + n_Kt
    A_psd = jnp.zeros((size_H, size_H, dimG, dimG))

    def add_gram_term(row, col, vec_a, vec_b, scale):
        term = 0.5 * (jnp.outer(vec_a, vec_b) + jnp.outer(vec_b, vec_a))
        return A_psd.at[row, col, :, :].add(scale * term)

    # Term 1: Inputs (Scaled by M^2)
    # Block 1,1 (u, u)
    for i in range(n_K):
        u_i = pairs_K[i][0]
        for j in range(n_K):
            u_j = pairs_K[j][0]
            A_psd = add_gram_term(i, j, u_i, u_j, M**2)

    # Block 2,2 (p, p)
    for i in range(n_Kt):
        p_i = pairs_Kt[i][0]
        for j in range(n_Kt):
            p_j = pairs_Kt[j][0]
            row, col = n_K + i, n_K + j
            A_psd = add_gram_term(row, col, p_i, p_j, M**2)

    # Off-Diagonals (u, p)
    for i in range(n_K):
        u_i = pairs_K[i][0]
        for j in range(n_Kt):
            p_j = pairs_Kt[j][0]
            row, col = i, n_K + j
            A_psd = add_gram_term(row, col, u_i, p_j, M**2)
            A_psd = add_gram_term(col, row, p_j, u_i, M**2)

    # Term 2: Outputs (Scaled by -1.0)
    # Block 1,1 (v, v)
    for i in range(n_K):
        v_i = pairs_K[i][1]
        for j in range(n_K):
            v_j = pairs_K[j][1]
            A_psd = add_gram_term(i, j, v_i, v_j, -1.0)

    # Block 2,2 (q, q)
    for i in range(n_Kt):
        q_i = pairs_Kt[i][1]
        for j in range(n_Kt):
            q_j = pairs_Kt[j][1]
            row, col = n_K + i, n_K + j
            A_psd = add_gram_term(row, col, q_i, q_j, -1.0) 

    # Off-Diagonals (v, q)
    for i in range(n_K):
        v_i = pairs_K[i][1]
        for j in range(n_Kt):
            q_j = pairs_Kt[j][1]
            row, col = i, n_K + j
            A_psd = add_gram_term(row, col, v_i, q_j, -1.0)
            A_psd = add_gram_term(col, row, q_j, v_i, -1.0)

    b_psd = jnp.zeros((size_H, size_H, dimF))
    c_psd = jnp.zeros((size_H, size_H))
    
    PSD_A_vals = [A_psd]
    PSD_b_vals = [b_psd]
    PSD_c_vals = [c_psd]
    PSD_shapes = [size_H]

    # 6. Initial Conditions
    vec_dx0 = eyeG[idx_dx0]
    vec_dy0 = eyeG[idx_dy0]
    A_init = jnp.outer(vec_dx0, vec_dx0) + jnp.outer(vec_dy0, vec_dy0)
    b_init = jnp.zeros(dimF)
    c_init = -R**2

    A_vals = jnp.concatenate([A_vals, A_init[None]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)

    # 7. Objective: Exact Primal-Dual Gap
    # Gap = f1(xK) - f1(xs) + h(yK) - h(ys) + <K xK, ys> - <K xs, yK>
    vec_ys = eyeG[idx_ys]
    vec_yK = repY_h[K_max]
    
    A_cross1 = 0.5 * (jnp.outer(vec_K_xK, vec_ys) + jnp.outer(vec_ys, vec_K_xK))
    
    # K(xs) is represented by gh_vec(idx_saddle)
    vec_K_xs = gh_vec(idx_saddle)
    A_cross2 = -0.5 * (jnp.outer(vec_K_xs, vec_yK) + jnp.outer(vec_yK, vec_K_xs))
    A_obj = A_cross1 + A_cross2

    idx_f1_K = K_max
    idx_h_K  = K_max
    
    b_obj = jnp.zeros(dimF)
    b_obj = b_obj.at[idx_f1_K].set(1.0)           # + f(x_K)
    b_obj = b_obj.at[dimF1 + idx_h_K].set(1.0)    # + h(y_K)
    b_obj = b_obj.at[idx_saddle].set(-1.0)        # - f(x_*)
    b_obj = b_obj.at[dimF1 + idx_saddle].set(-1.0)# - h(y_*)

    return (A_obj, b_obj, A_vals, b_vals, c_vals,
            PSD_A_vals, PSD_b_vals, PSD_c_vals, PSD_shapes)