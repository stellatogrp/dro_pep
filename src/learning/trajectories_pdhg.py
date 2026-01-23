import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnames=['K_max', 'return_Gram_representation'])
def problem_data_to_pdhg_trajectories(tau, sigma, theta, 
                                      c, A, b, C, d, 
                                      x0, y0, x_opt, y_opt, 
                                      K_max,
                                      return_Gram_representation=True):
    """
    Compute PDHG trajectories on a Linear Program with shifted functions.
    
    The problem is shifted so the optimal saddle point is at (0, 0).
    The main loop operates on the shifted variables (x_shifted, y_shifted).
    
    Args:
        tau, sigma, theta: Algorithm parameters
        c, A, b, C, d: LP problem data (min c'x s.t. Ax=b, Cx<=d)
        x0, y0: Initial points in ORIGINAL coordinates
        x_opt, y_opt: Optimal points (used for shifting)
        K_max: Number of iterations
        return_Gram_representation: If True, returns (G, F). Else returns raw trajectories.
    
    Returns:
        If return_Gram_representation=True:
            G: (2K+6, 2K+6) Gram matrix
            F: (2K+4,) Concatenated function values
        If return_Gram_representation=False:
            x_iter: (n, K+1) Shifted primal iterates
            y_iter: (m, K+1) Shifted dual iterates
            gf1_iter: (dim_embed, K+1) Shifted f1 gradients
            gh_iter: (dim_embed, K+1) Shifted h subgradients
            f1_iter: (K+1,) Shifted f1 values
            h_iter: (K+1,) Shifted h values
    """
    
    n = c.shape[0]
    m1 = b.shape[0] if b is not None else 0
    m2 = d.shape[0] if d is not None else 0
    m = m1 + m2
    
    # Common dimension for Gram embedding
    dim_embed = jnp.maximum(n, m)
    
    # --- 1. Shifted Function Definitions ---
    
    # Pre-compute optimal values for valid shifting
    f1_opt = jnp.dot(c, x_opt)
    
    def compute_h_raw(y):
        # h(y) = <y, rhs> for valid y
        val = 0.0
        if m1 > 0: val = val + jnp.dot(y[:m1], b)
        if m2 > 0: val = val + jnp.dot(y[m1:], d)
        return val

    h_opt = compute_h_raw(y_opt)

    # Shifted Values
    def f1_shifted(x_s):
        # f1(x_s + x_opt) - f1_opt = c'(x_s + x_opt) - c'x_opt = c'x_s
        return jnp.dot(c, x_s)
    
    def h_shifted(y_s):
        # h(y_s + y_opt) - h_opt
        return compute_h_raw(y_s + y_opt) - h_opt

    # Shifted Linear Operator Helpers
    def apply_K(x_s):
        # K(x_s) - no shift needed for linear operator K acting on difference
        parts = []
        if m1 > 0: parts.append(A @ x_s)
        if m2 > 0: parts.append(C @ x_s)
        return jnp.concatenate(parts) if parts else jnp.array([])

    def apply_K_T(y_s):
        # K^T(y_s)
        res = jnp.zeros(n)
        if m1 > 0: res = res + A.T @ y_s[:m1]
        if m2 > 0: res = res + C.T @ y_s[m1:]
        return res

    # --- 2. Proximal and Gradient Helpers (Shifted) ---

    def prox_f1_step_shifted(x_s, y_s):
        """
        Computes x_{k+1}_shifted using current shifted variables.
        Standard update: x_new = x - tau * (K'y + c)
        Shifted update:
            x_new_s + x* = x_s + x* - tau * (K'(y_s + y*) + c)
            x_new_s = x_s - tau * (K'y_s + (K'y* + c))
        """
        # Term (K'y* + c) is the gradient of Lagrangian at optimality (should be 0 for LP subspace)
        grad_L_x_opt = apply_K_T(y_opt) + c
        
        # Update
        step = apply_K_T(y_s) + grad_L_x_opt
        return x_s - tau * step

    def prox_h_step_shifted(y_s, x_bar_s):
        """
        Computes y_{k+1}_shifted using shifted variables.
        y_new = prox_{sigma h} (y + sigma K x_bar)
        """
        # 1. Recover original intermediate point
        y_prev_orig = y_s + y_opt
        x_bar_orig  = x_bar_s + x_opt
        
        # Argument to prox
        Kx_bar = apply_K(x_bar_orig) 
        v_orig = y_prev_orig + sigma * Kx_bar
        
        # 2. Apply Prox (Component-wise)
        nu_new = jnp.array([])
        lam_new = jnp.array([])
        
        if m1 > 0:
            # Prox of linear <nu, b>: shift by -sigma*b
            nu_temp = v_orig[:m1]
            nu_new = nu_temp - sigma * b
            
        if m2 > 0:
            # Prox of <lam, d> + I(lam>=0): shift by -sigma*d then project
            lam_temp = v_orig[m1:]
            lam_new = jnp.maximum(0.0, lam_temp - sigma * d)
            
        y_new_orig = jnp.concatenate([nu_new, lam_new])
        
        # 3. Shift back
        return y_new_orig - y_opt

    # --- 3. Padding Helpers ---
    def pad(v):
        return jnp.pad(v, (0, dim_embed - v.shape[0]), mode='constant')

    # --- 4. Main Loop ---
    
    # Initialize shifted variables
    x0_shifted = x0 - x_opt
    y0_shifted = y0 - y_opt
    
    # Initial Storage (Step 0)
    col_x0 = pad(x0_shifted)
    col_y0 = pad(y0_shifted)
    
    # Gradient at 0 
    gf1_0 = pad(c) 
    
    # Subgradient of h at y0: explicit [b; d]
    rhs_parts = []
    if m1 > 0: rhs_parts.append(b)
    if m2 > 0: rhs_parts.append(d)
    rhs_vec = jnp.concatenate(rhs_parts) if rhs_parts else jnp.zeros(0)
    gh_0 = pad(rhs_vec)
    
    f1_0_val = f1_shifted(x0_shifted)
    h_0_val  = h_shifted(y0_shifted)
    
    # Loop State: (x_s, y_s)
    init_state = (x0_shifted, y0_shifted)

    def pdhg_scan_step(state, _):
        x_s, y_s = state
        
        # 1. Primal Step (Shifted)
        x_new_s = prox_f1_step_shifted(x_s, y_s)
        
        # Back-calculate Implicit Gradient gf1_{k+1}
        # x_{k+1} = x_k - tau * K^T y_k - tau * gf1
        gf1_s = (x_s - x_new_s) / tau - apply_K_T(y_s)
        
        # 2. Extrapolation
        x_bar_s = x_new_s + theta * (x_new_s - x_s)
        
        # 3. Dual Step (Shifted)
        y_new_s = prox_h_step_shifted(y_s, x_bar_s)
        
        # Back-calculate Implicit Gradient gh_{k+1}
        # y_{k+1} = y_k + sigma * K x_bar - sigma * gh
        gh_s = (y_s + sigma * apply_K(x_bar_s) - y_new_s) / sigma
        
        # 4. Values
        f1_v = f1_shifted(x_new_s)
        h_v  = h_shifted(y_new_s)
        
        # 5. Output tuple (including raw trajectories for else block)
        # We output everything needed for both Gram construction AND raw return
        scan_out = (pad(gf1_s), pad(gh_s), f1_v, h_v, x_new_s, y_new_s)
        
        return (x_new_s, y_new_s), scan_out

    # Run Scan
    final_state, scan_res = lax.scan(pdhg_scan_step, init_state, None, length=K_max)
    
    # Unpack Scan Results
    gf1_seq, gh_seq, f1_seq, h_seq, x_seq, y_seq = scan_res
    
    if return_Gram_representation:
        # --- Assemble G Matrix ---
        # Columns: [x0_s, y0_s, xs(=0), ys(=0), gf1_0, gh_0, gf1_1, gh_1, ...]
        
        col_xs = jnp.zeros((dim_embed, 1))
        col_ys = jnp.zeros((dim_embed, 1))
        
        fixed_cols = [
            col_x0.reshape(-1, 1),
            col_y0.reshape(-1, 1),
            col_xs, 
            col_ys,
            gf1_0.reshape(-1, 1),
            gh_0.reshape(-1, 1)
        ]
        
        # Stack sequence columns
        seq_cols_list = []
        for k in range(K_max):
            seq_cols_list.append(gf1_seq[k].reshape(-1, 1))
            seq_cols_list.append(gh_seq[k].reshape(-1, 1))
            
        all_cols = jnp.concatenate(fixed_cols + seq_cols_list, axis=1) # (dim, 2K+6)
        G = all_cols.T @ all_cols # (2K+6, 2K+6)
        
        # --- Assemble F Vector ---
        # F1: [f1(x0), ..., f1(xK), f1(xs)=0]
        # Fh: [h(y0), ..., h(yK), h(ys)=0]
        
        F1 = jnp.concatenate([jnp.array([f1_0_val]), f1_seq, jnp.array([0.0])])
        Fh = jnp.concatenate([jnp.array([h_0_val]), h_seq, jnp.array([0.0])])
        
        F = jnp.concatenate([F1, Fh])
        
        return G, F
        
    else:
        # Return all raw trajectories (concatenating step 0 with steps 1..K)
        
        # x_iter: (dim, K+1)
        # Note: x_seq is (K, n), we transpose to (n, K) then concat
        x_iter = jnp.concatenate([x0_shifted[:, None], x_seq.T], axis=1)
        y_iter = jnp.concatenate([y0_shifted[:, None], y_seq.T], axis=1)
        
        # gf1_iter: (dim_embed, K+1) - using padded versions
        gf1_iter = jnp.concatenate([gf1_0[:, None], gf1_seq.T], axis=1)
        gh_iter  = jnp.concatenate([gh_0[:, None],  gh_seq.T],  axis=1)
        
        # f1_iter: (K+1,)
        f1_iter = jnp.concatenate([jnp.array([f1_0_val]), f1_seq])
        h_iter  = jnp.concatenate([jnp.array([h_0_val]), h_seq])
        
        return x_iter, y_iter, gf1_iter, gh_iter, f1_iter, h_iter
