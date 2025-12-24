import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import time
from cvxpylayers.jax import CvxpyLayer
from frozendict import frozendict
from functools import partial
from tqdm import trange

# from .utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, nesterov_fgm, generate_trajectories, sample_x0_centered_disk, generate_P_fixed_mu_L
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexQuadraticFunction, SmoothStronglyConvexFunction
from reformulator.dro_reformulator import DROReformulator

log = logging.getLogger(__name__)


def marchenko_pastur(key, d, mu, L, M):
    sigma = (jnp.sqrt(L) + jnp.sqrt(mu)) / 2
    X = jax.random.normal(key, shape=(d, M)) * sigma
    H = X.T @ X / d
    return H


def rejection_sample_single(key, dim, mu, L, M):
    def body_fun(state):
        key, _, _ = state
        key, subkey = jax.random.split(key)
        
        H = marchenko_pastur(subkey, dim, mu, L, M)
        
        eigvals = jnp.real(jnp.linalg.eigvals(H))
        is_valid = (mu <= jnp.min(eigvals)) & (L >= jnp.max(eigvals))
        return (key, H, is_valid)

    def cond_fun(state):
        return ~state[2]

    key, subkey = jax.random.split(key)
    init_H = marchenko_pastur(subkey, dim, mu, L, M)
    init_valid = jnp.array(False) 
    
    final_state = jax.lax.while_loop(cond_fun, body_fun, (key, init_H, init_valid))
    return final_state[1]


@partial(jax.jit, static_argnames=['dim', 'mu', 'L', 'M'])
def get_Q_samples(subkeys, dim, mu, L, M):
    sampler = partial(rejection_sample_single, dim=dim, mu=mu, L=L, M=M)
    return jax.vmap(sampler)(subkeys)


def sample_z0_single(key, d, R):
    k1, k2 = jax.random.split(key)
    
    z = jax.random.normal(k1, shape=(d,))
    z = z / jnp.linalg.norm(z)
    
    u = jax.random.uniform(k2)
    dist = u ** (1.0 / d)
    
    return R * dist * z


@partial(jax.jit, static_argnames=['d'])
def get_z0_samples(subkeys, d, R):
    # Fix d and R using partial, then vmap over keys
    sampler = partial(sample_z0_single, d=d, R=R)
    return jax.vmap(sampler)(subkeys)


@partial(jax.jit, static_argnames=['K_max'])
def problem_data_to_gd_trajectories(t, Q, z0, zs, K_max):
    def f(x):
        return .5 * x.T @ Q @ x
    
    def g(x):
        return Q @ x

    dim = z0.shape[0]
    # zs = jnp.zeros(dim)
    z_stack = jnp.zeros((dim, K_max + 2))
    g_stack = jnp.zeros((dim, K_max + 2))
    f_stack = jnp.zeros(K_max + 2)

    z_stack = z_stack.at[:, 0].set(zs)
    z_stack = z_stack.at[:, 1].set(z0)
    g_stack = g_stack.at[:, 0].set(g(zs))
    g_stack = g_stack.at[:, 1].set(g(z0))
    f_stack = f_stack.at[0].set(f(zs))
    f_stack = f_stack.at[1].set(f(z0))

    def body_fun(i, val):
        z, z_stack, g_stack, f_stack = val
        z = z - t * g(z)
        z_stack = z_stack.at[:, i+1].set(z)
        g_stack = g_stack.at[:, i+1].set(g(z))
        f_stack = f_stack.at[i+1].set(f(z))
        return (z, z_stack, g_stack, f_stack)

    init_val = (z0, z_stack, g_stack, f_stack)
    _, z_stack, g_stack, f_stack = \
        jax.lax.fori_loop(1, K_max + 1, body_fun, init_val)

    G_half = jnp.hstack([z_stack[:,:2], g_stack[:,1:]])
    return G_half.T@G_half, f_stack


def create_exp_cp_layer(DR, eps, alpha, G_shape, F_shape):
    G_param = cp.Parameter(G_shape)
    F_param = cp.Parameter(F_shape)

    A_vals = DR.canon.A_vals
    b_vals = DR.canon.b_vals
    c_vals = DR.canon.c_vals
    A_obj = DR.canon.A_obj
    b_obj = DR.canon.b_obj

    N = G_shape[0]
    M = A_vals.shape[0]
    mat_shape = A_obj.shape
    vec_shape = b_obj.shape

    lambd = cp.Variable()
    s = cp.Variable(N)
    y = cp.Variable((N, M))
    Gz = [cp.Variable(mat_shape, symmetric=True) for _ in range(N)]
    Fz = [cp.Variable(vec_shape) for _ in range(N)]

    obj = lambd * eps + 1 / N * cp.sum(s)
    constraints = [y >= 0]

    G_preconditioner = np.diag(DR.canon.precond_inv[0])
    F_preconditioner = DR.canon.precond_inv[1]

    for i in range(N):
        G_sample, F_sample = G_param[i], F_param[i]
        constraints += [- c_vals.T @ y[i] - cp.trace(G_sample @ Gz[i]) - F_sample.T @ Fz[i] <= s[i]]
        constraints += [cp.SOC(lambd, cp.hstack([cp.vec( G_preconditioner@Gz[i]@G_preconditioner, order='F'), cp.multiply(F_preconditioner**2, Fz[i])]))]

        LstarG = 0
        LstarF = 0
        for m in range(M):
            Am = A_vals[m]
            bm = b_vals[m]
            LstarG = LstarG + y[i, m] * Am
            LstarF = LstarF + y[i, m] * bm
    
        constraints += [LstarG - Gz[i] - A_obj >> 0]
        constraints += [LstarF - Fz[i] - b_obj == 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    return CvxpyLayer(prob, parameters=[G_param, F_param], variables=[lambd, s])


@jax.jit
def dro_pep_obj_jax(eps, lambd_star, s_star):
    N = s_star.shape[0]
    return lambd_star * eps + 1 / N * jnp.sum(s_star)


def SCS_pipeline(t, Q_batch, z0_batch, zs_batch, K_max, eps, alpha, DR, large_sdp_layer):
    batch_GF_func = jax.vmap(problem_data_to_gd_trajectories, in_axes=(None, 0, 0, 0, None))
    G_batch, F_batch = batch_GF_func(t, Q_batch, z0_batch, zs_batch, K_max)
    (lambd_star, s_star) = large_sdp_layer(G_batch, F_batch)
    loss = dro_pep_obj_jax(eps, lambd_star, s_star)
    return loss


def quad_run(cfg):
    log.info(cfg)
    
    d_val = cfg.dim
    mu_val = cfg.mu
    L_val = cfg.L
    
    r_val = (np.sqrt(L_val) - np.sqrt(mu_val))**2 / (np.sqrt(L_val) + np.sqrt(mu_val))**2
    M_val = int(np.round(r_val * d_val))
    
    log.info(f"Precomputed matrix width M: {M_val}")
    
    seed = 42
    key = jax.random.PRNGKey(seed)
    key, batch_seed = jax.random.split(key, 2)
    subkeys = jax.random.split(batch_seed, cfg.N)
    
    log.info(f"Compiling and sampling {cfg.N} matrices...")
    start = time.time()
    
    # Pass the precomputed M_val into the JIT function
    Q_values = get_Q_samples(subkeys, d_val, mu_val, L_val, M_val)
    
    Q_values.block_until_ready()
    end = time.time()
    
    log.info(f"Done. Generated shape: {Q_values.shape}")
    log.info(f"Time: {end - start:.4f}s")

    key, batch_seed = jax.random.split(key, 2)
    subkeys = jax.random.split(batch_seed, cfg.N)

    z0_values = get_z0_samples(subkeys, M_val, cfg.R)
    log.info(f"z0 values shape: {z0_values.shape}")

    zs_values = jnp.zeros(z0_values.shape)

    batch_GF_func = jax.vmap(problem_data_to_gd_trajectories, in_axes=(None, 0, 0, 0, None))
    grad_fn = jax.grad(SCS_pipeline, argnums=(0, 1, 2))
