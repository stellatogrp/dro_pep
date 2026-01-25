import jax
import jax.numpy as jnp
from functools import partial
from .interpolation_conditions import convex_interp

@partial(jax.jit, static_argnames=['K_max'])
def construct_chambolle_pock_pep_data(tau, sigma, theta, M, R, K_max):
    """
    Construct PEP for Chambolle-Pock with P-norm (Lyapunov) Initial Condition.
    
    Fixes applied:
    1. P-norm Initial Condition: Added explicit basis vector for K(dx0) to handle
       the cross-term -2<K(x0-xs), y0-ys> correctly.
    2. Adjoint Consistency: Enforces <Ku, p> = <u, K.T p> for ALL operator pairs.
    3. Solution Bound: Constraints ||x*||^2 + ||y*||^2 <= B^2 to prevent unboundedness.
    4. Objective Construction: Uses stationarity substitutions <K xK, y*> = <xK, -df(x*)>.
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
    idx_K_dx0 = idx_c; idx_c+=1  # K * dx0 (For P-norm IC)

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
    A_f1, b_f1 = convex_interp(repX_f1, repG_f1, repF_f1, n_points)
    A_h, b_h = convex_interp(repY_h, repG_h, repF_h, n_points)

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
    
    # [CRITICAL FIX] Bound the solution magnitude.
    # Without this, the solver can scale x* and y* to infinity to exploit cross-terms.
    # ||x_*||^2 + ||y_*||^2 <= B^2 (where B=1 matches R=1)
    vec_xs = eyeG[idx_xs]
    vec_ys = eyeG[idx_ys]
    A_sol_bound = jnp.outer(vec_xs, vec_xs) + jnp.outer(vec_ys, vec_ys)
    b_sol_bound = jnp.zeros(dimF)
    c_sol_bound = -1.0 # B^2 = 1.0
    
    A_vals = jnp.concatenate([A_vals, A_sol_bound[None]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_sol_bound[None]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_sol_bound])], axis=0)

    # 6. Gather Operator Pairs
    # Optimality Conditions
    pairs_Kt.append((eyeG[idx_ys], -gf1_vec(idx_saddle))) 
    pairs_K.append((eyeG[idx_xs], gh_vec(idx_saddle)))    

    # [CRITICAL FIX] Initial Condition Observations (K * dx0)
    # Allows us to form the cross term <K dx0, dy0> in the P-norm
    pairs_K.append((eyeG[idx_dx0], eyeG[idx_K_dx0]))

    # Objective Observations
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

    # 9. P-Norm Initial Condition (Lyapunov)
    # 1/tau ||dx0||^2 + 1/sigma ||dy0||^2 - 2 <K dx0, dy0> <= R^2
    vec_dx0 = eyeG[idx_dx0]
    vec_dy0 = eyeG[idx_dy0]
    vec_K_dx0 = eyeG[idx_K_dx0] 
    
    term1 = (1.0 / tau_vec[0]) * jnp.outer(vec_dx0, vec_dx0)
    term2 = (1.0 / sigma_vec[0]) * jnp.outer(vec_dy0, vec_dy0)
    
    # Cross term: -2 <K dx0, dy0>
    # Note: jnp.outer(u,v) + jnp.outer(v,u) represents 2*<u,v> in the trace.
    # So multiplying by -1.0 gives -2*<u,v>.
    term3 = -1.0 * (jnp.outer(vec_K_dx0, vec_dy0) + jnp.outer(vec_dy0, vec_K_dx0))
    
    A_init = term1 + term2 + term3
    b_init = jnp.zeros(dimF)
    c_init = -R**2

    A_vals = jnp.concatenate([A_vals, A_init[None]], axis=0)
    b_vals = jnp.concatenate([b_vals, b_init[None]], axis=0)
    c_vals = jnp.concatenate([c_vals, jnp.array([c_init])], axis=0)

    # 10. Objective: Gap
    # Gap = f(xK) + h(yK) + <K xK, ys> - <K xs, yK>
    vec_ys = eyeG[idx_ys]
    vec_yK = y_curr  # Use full y_K position, not y_K - y*
    
    # Subst: <K xK, ys> = <xK, -grad_f(xs)>
    vec_neg_gf1_s = -gf1_vec(idx_saddle) 
    
    # Subst: - <K xs, yK> = - <grad_h(ys), yK>
    vec_gh_s = gh_vec(idx_saddle)        
    
    A_cross1 = 0.5 * (jnp.outer(x_curr, vec_neg_gf1_s) + jnp.outer(vec_neg_gf1_s, x_curr))
    A_cross2 = -0.5 * (jnp.outer(vec_gh_s, vec_yK) + jnp.outer(vec_yK, vec_gh_s))
    
    A_obj = A_cross1 + A_cross2
    
    idx_f1_K = K_max
    idx_h_K  = K_max
    
    b_obj = jnp.zeros(dimF)
    b_obj = b_obj.at[idx_f1_K].set(1.0)
    b_obj = b_obj.at[dimF1 + idx_h_K].set(1.0)

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