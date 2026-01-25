"""
PDHG (Chambolle-Pock) trajectory computation for DRO-PEP on Linear Programs.

This module computes PDHG trajectories and constructs Gram representations
that match the PEP construction from pep_construction_chambolle_pock_linop.py.

The LP problem is:
    min_x  c^T x
    s.t.   G x >= h      (m1 inequality constraints)
           A x = b       (m2 equality constraints)
           l <= x <= u   (box constraints)

This is reformulated as a saddle-point problem:
    min_x max_y L(x, y) = f1(x) + <Kx, y> - h_dual(y)

where:
    K = [G; A] is the stacked constraint matrix
    q = [h; b] is the stacked RHS
    f1(x) = c^T x + indicator_{[l,u]}(x)
    h_dual(y) = q^T y + indicator_Y(y)  where Y = {y: y[:m1] >= 0}

PDHG Algorithm:
    For k = 0, ..., K-1:
        x_{k+1} = prox_{τ f1}(x_k - τ K^T y_k) = proj_{[l,u]}(x_k - τ K^T y_k - τ c)
        x_bar_{k+1} = x_{k+1} + θ (x_{k+1} - x_k)
        y_{k+1} = prox_{σ h_dual}(y_k + σ K x_bar_{k+1}) = proj_Y(y_k + σ K x_bar_{k+1} + σ q)

For the DRO-PEP framework, we shift coordinates so the saddle point is at origin:
    x_shifted = x - x_opt
    y_shifted = y - y_opt
"""

import jax
import jax.numpy as jnp
from functools import partial


def proj_box(v, l, u):
    """Project v onto box [l, u]."""
    return jnp.clip(v, l, u)


def proj_nonneg_first_m1(v, m1):
    """Project first m1 components to non-negative, leave rest unchanged."""
    v_ineq = jnp.maximum(v[:m1], 0.0)
    v_eq = v[m1:]
    return jnp.concatenate([v_ineq, v_eq])


@partial(jax.jit, static_argnames=['K_max', 'm1'])
def problem_data_to_pdhg_trajectories(
    stepsizes,
    c, K_mat, q, l, u,
    x0, y0,
    x_opt, y_opt,
    K_max, m1,
    M=None
):
    """
    Compute PDHG trajectories on LP and return Gram representation matching PEP construction.

    The LP data defines the saddle-point problem:
        min_x max_y  c^T x + y^T (q - K x)
        s.t.  l <= x <= u,  y[:m1] >= 0

    This function shifts coordinates so the saddle point is at the origin,
    runs PDHG, and constructs the Gram matrix matching pep_construction_chambolle_pock_linop.py.

    The Gram basis has:
        dimG = 4 + 2*(K_max + 2) + 2*K_max + 3 = 4*K_max + 11

        Indices:
        - 0: dx0 (x_0 - x_s)
        - 1: dy0 (y_0 - y_s)
        - 2: x_s (= 0 in shifted coords)
        - 3: y_s (= 0 in shifted coords)
        - 4 to 5+K: gf1_0, ..., gf1_K, gf1_s (K+2 subgradients of f1)
        - 6+K to 7+2K: gh_0, ..., gh_K, gh_s (K+2 subgradients of h)
        - 8+2K to 7+3K: w_0, ..., w_{K-1} (K^T y_k)
        - 8+3K to 7+4K: z_1, ..., z_K (K x_bar_k)
        - 8+4K: K_xK
        - 9+4K: Kt_yK
        - 10+4K: K_dx0

    Function values:
        dimF = 2*(K_max + 2)
        F1: [f1(x_0)-f1_s, ..., f1(x_K)-f1_s, 0]
        F_h: [h(y_0)-h_s, ..., h(y_K)-h_s, 0]

    Args:
        stepsizes: Tuple (tau, sigma, theta) - scalars or arrays of length K_max
        c: (n,) cost vector
        K_mat: (m, n) constraint matrix [G; A] stacked
        q: (m,) RHS vector [h; b] stacked
        l: (n,) lower bounds on x
        u: (n,) upper bounds on x
        x0: (n,) initial primal point (ORIGINAL coordinates)
        y0: (m,) initial dual point (ORIGINAL coordinates)
        x_opt: (n,) optimal primal point
        y_opt: (m,) optimal dual point
        K_max: Number of PDHG iterations
        m1: Number of inequality constraints (first m1 components of y >= 0)
        M: Operator norm ||K||. If None, computed from K_mat.

    Returns:
        G: (dimG, dimG) Gram matrix
        F: (dimF,) concatenated function values [F1, F_h]
    """
    tau_raw, sigma_raw, theta_raw = stepsizes
    n = c.shape[0]
    m = K_mat.shape[0]

    # Broadcast step sizes
    tau = jnp.broadcast_to(tau_raw, (K_max,))
    sigma = jnp.broadcast_to(sigma_raw, (K_max,))
    theta = jnp.broadcast_to(theta_raw, (K_max,))

    if M is None:
        M = jnp.linalg.norm(K_mat, ord=2)

    # ===== Shift coordinates =====
    # Shifted: x_shifted = x - x_opt, y_shifted = y - y_opt
    # Shifted box constraints: l_s = l - x_opt, u_s = u - x_opt
    l_shifted = l - x_opt
    u_shifted = u - x_opt

    x0_shifted = x0 - x_opt
    y0_shifted = y0 - y_opt

    # ===== Define shifted functions =====
    # Original: f1(x) = c^T x + indicator_{[l,u]}(x)
    # Shifted: f1_shifted(x) = c^T (x + x_opt) + indicator_{[l_s, u_s]}(x)
    #                        = c^T x + c^T x_opt + indicator
    # At x=0: f1_shifted(0) = c^T x_opt (assuming x_opt in [l, u])

    def f1_shifted(x):
        """f1(x + x_opt) = c^T (x + x_opt)"""
        return jnp.dot(c, x + x_opt)

    def prox_f1_shifted(v, tau_k):
        """
        prox_{τ f1_shifted}(v) where f1_shifted(x) = c^T(x + x_opt) + indicator_{[l_s, u_s]}(x)
        = proj_{[l_s, u_s]}(v - τ c)
        """
        return proj_box(v - tau_k * c, l_shifted, u_shifted)

    # Original: h_dual(y) = q^T y + indicator_Y(y)
    # Shifted: h_shifted(y) = q^T (y + y_opt) + indicator_Y_shifted(y)
    # where Y_shifted = {y: (y + y_opt)[:m1] >= 0} = {y: y[:m1] >= -y_opt[:m1]}

    def h_shifted(y):
        """h(y + y_opt) = q^T (y + y_opt)"""
        return jnp.dot(q, y + y_opt)

    def prox_h_shifted(v, sigma_k):
        """
        prox_{σ h_shifted}(v) where h_shifted(y) = q^T(y + y_opt) + indicator
        = proj_Y_shifted(v + σ q)

        Note: The proximal of h(y) = q^T y + indicator_Y(y) is:
        prox_{σ h}(v) = proj_Y(v - σ q)  [minus because h = q^T y, grad = q]

        Wait, let me reconsider. For h(y) = q^T y:
        prox_{σ h}(v) = argmin_y { σ q^T y + 0.5||y - v||^2 }
        Taking derivative: σ q + (y - v) = 0  =>  y = v - σ q
        Then project onto Y.

        For shifted h_shifted(y) = q^T(y + y_opt):
        prox_{σ h_shifted}(v) = prox_{σ h}(v + y_opt) - y_opt
                              = proj_Y(v + y_opt - σ q) - y_opt

        In shifted coords, Y_shifted = {y: (y + y_opt)[:m1] >= 0}
        """
        # Project in original space, then shift back
        v_orig = v + y_opt - sigma_k * q
        y_proj = proj_nonneg_first_m1(v_orig, m1)
        return y_proj - y_opt

    # ===== Run PDHG in shifted coordinates =====
    x_iter = jnp.zeros((K_max + 1, n))
    y_iter = jnp.zeros((K_max + 1, m))
    gf1_iter = jnp.zeros((K_max + 1, n))  # subgradients of f1
    gh_iter = jnp.zeros((K_max + 1, m))   # subgradients of h
    w_iter = jnp.zeros((K_max, n))        # w_k = K^T @ y_k
    z_iter = jnp.zeros((K_max, m))        # z_k = K @ x_bar_k
    f1_iter = jnp.zeros(K_max + 1)
    h_iter = jnp.zeros(K_max + 1)

    # Initial point
    x_iter = x_iter.at[0].set(x0_shifted)
    y_iter = y_iter.at[0].set(y0_shifted)
    f1_iter = f1_iter.at[0].set(f1_shifted(x0_shifted))
    h_iter = h_iter.at[0].set(h_shifted(y0_shifted))

    # Initial subgradient of f1: for f1(x) = c^T x + indicator, subgrad = c + normal_cone
    # At interior point, subgrad = c
    gf1_iter = gf1_iter.at[0].set(c)

    # Initial subgradient of h: for h(y) = q^T y + indicator, subgrad = q + normal_cone
    gh_iter = gh_iter.at[0].set(q)

    x_curr = x0_shifted
    y_curr = y0_shifted

    def pdhg_step(k, carry):
        x_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter, f1_iter, h_iter, x_curr, y_curr = carry

        tau_k = tau[k]
        sigma_k = sigma[k]
        theta_k = theta[k]

        # w_k = K^T @ y_k (note: y_curr is in shifted coords)
        # But K^T @ y_shifted = K^T @ (y - y_opt) = K^T @ y - K^T @ y_opt
        # For the algorithm update, we need K^T @ y_actual where y_actual = y_shifted + y_opt
        w_k = K_mat.T @ (y_curr + y_opt)
        w_iter = w_iter.at[k].set(w_k)

        # Primal update: x_{k+1} = prox_{τ f1}(x_k - τ K^T y_k)
        v_x = x_curr - tau_k * w_k
        x_next = prox_f1_shifted(v_x, tau_k)

        # Subgradient from prox optimality: gf1_{k+1} = (v_x - x_next) / τ
        gf1_next = (v_x - x_next) / tau_k

        # Extrapolation
        x_bar = x_next + theta_k * (x_next - x_curr)

        # z_{k+1} = K @ x_bar (in original coords: K @ (x_bar + x_opt))
        z_kp1 = K_mat @ (x_bar + x_opt)
        z_iter = z_iter.at[k].set(z_kp1)

        # Dual update: y_{k+1} = prox_{σ h}(y_k + σ K x_bar)
        v_y = y_curr + sigma_k * z_kp1
        y_next = prox_h_shifted(v_y, sigma_k)

        # Subgradient from prox optimality: gh_{k+1} = (v_y - y_next) / σ
        gh_next = (v_y - y_next) / sigma_k

        # Store
        x_iter = x_iter.at[k + 1].set(x_next)
        y_iter = y_iter.at[k + 1].set(y_next)
        gf1_iter = gf1_iter.at[k + 1].set(gf1_next)
        gh_iter = gh_iter.at[k + 1].set(gh_next)
        f1_iter = f1_iter.at[k + 1].set(f1_shifted(x_next))
        h_iter = h_iter.at[k + 1].set(h_shifted(y_next))

        return (x_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter, f1_iter, h_iter, x_next, y_next)

    carry = (x_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter, f1_iter, h_iter, x_curr, y_curr)
    result = jax.lax.fori_loop(0, K_max, pdhg_step, carry)
    x_iter, y_iter, gf1_iter, gh_iter, w_iter, z_iter, f1_iter, h_iter, x_K, y_K = result

    # ===== Build Gram Matrix =====
    dimG = 4 + 2*(K_max + 2) + 2*K_max + 3
    dimF1 = K_max + 2
    dimF_h = K_max + 2
    dimF = dimF1 + dimF_h

    # Index mapping (must match pep_construction_chambolle_pock_linop.py)
    idx_dx0 = 0
    idx_dy0 = 1
    idx_xs = 2
    idx_ys = 3
    idx_gf1_start = 4
    idx_gh_start = 4 + (K_max + 2)
    idx_w_start = idx_gh_start + (K_max + 2)
    idx_z_start = idx_w_start + K_max
    idx_K_xK = idx_z_start + K_max
    idx_Kt_yK = idx_K_xK + 1
    idx_K_dx0 = idx_Kt_yK + 1

    # Shifted coords: x_s = 0, y_s = 0
    dx0 = x0_shifted
    dy0 = y0_shifted

    # Saddle point subgradients (at origin in shifted coords)
    # At x_opt: subgrad f1 = c (if x_opt is interior to [l, u])
    # At y_opt: subgrad h = q (if y_opt is interior to Y)
    gf1_s = c
    gh_s = q

    # Analysis vectors (in original coords for K application)
    K_xK = K_mat @ (x_K + x_opt)
    Kt_yK = K_mat.T @ (y_K + y_opt)
    K_dx0 = K_mat @ dx0  # K @ (x0 - x_opt) = K @ x0 - K @ x_opt

    # Build G_half: embed primal in R^n and dual in R^m into R^{n+m}
    embedded_dim = n + m
    G_half = jnp.zeros((embedded_dim, dimG))

    def embed_primal(v):
        return jnp.concatenate([v, jnp.zeros(m)])

    def embed_dual(v):
        return jnp.concatenate([jnp.zeros(n), v])

    # Fill basis vectors
    G_half = G_half.at[:, idx_dx0].set(embed_primal(dx0))
    G_half = G_half.at[:, idx_dy0].set(embed_dual(dy0))
    G_half = G_half.at[:, idx_xs].set(embed_primal(jnp.zeros(n)))
    G_half = G_half.at[:, idx_ys].set(embed_dual(jnp.zeros(m)))

    # gf1_k for k = 0, ..., K, and gf1_s
    for k in range(K_max + 1):
        G_half = G_half.at[:, idx_gf1_start + k].set(embed_primal(gf1_iter[k]))
    G_half = G_half.at[:, idx_gf1_start + K_max + 1].set(embed_primal(gf1_s))

    # gh_k for k = 0, ..., K, and gh_s
    for k in range(K_max + 1):
        G_half = G_half.at[:, idx_gh_start + k].set(embed_dual(gh_iter[k]))
    G_half = G_half.at[:, idx_gh_start + K_max + 1].set(embed_dual(gh_s))

    # w_k = K^T @ y_k (primal space)
    for k in range(K_max):
        G_half = G_half.at[:, idx_w_start + k].set(embed_primal(w_iter[k]))

    # z_k = K @ x_bar_k (dual space)
    for k in range(K_max):
        G_half = G_half.at[:, idx_z_start + k].set(embed_dual(z_iter[k]))

    # Analysis vectors
    G_half = G_half.at[:, idx_K_xK].set(embed_dual(K_xK))
    G_half = G_half.at[:, idx_Kt_yK].set(embed_primal(Kt_yK))
    G_half = G_half.at[:, idx_K_dx0].set(embed_dual(K_dx0))

    # Metric matrix W for modified inner product
    # <embed_primal(x), embed_dual(y)>_W = <Kx, y> / M
    W = jnp.block([
        [jnp.eye(n), K_mat.T / M],
        [K_mat / M, jnp.eye(m)]
    ])

    G = G_half.T @ W @ G_half

    # ===== Build Function Values =====
    # f1_s = f1_shifted(0) = c^T x_opt
    # h_s = h_shifted(0) = q^T y_opt
    f1_s = jnp.dot(c, x_opt)
    h_s = jnp.dot(q, y_opt)

    # F1: [f1(x_0) - f1_s, ..., f1(x_K) - f1_s, 0]
    F1 = jnp.concatenate([f1_iter - f1_s, jnp.array([0.0])])

    # F_h: [h(y_0) - h_s, ..., h(y_K) - h_s, 0]
    F_h = jnp.concatenate([h_iter - h_s, jnp.array([0.0])])

    F = jnp.concatenate([F1, F_h])

    return G, F
