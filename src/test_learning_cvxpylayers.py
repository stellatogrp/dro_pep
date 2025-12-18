import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import time
from frozendict import frozendict
from functools import partial
from tqdm import trange

# from .utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, nesterov_fgm, generate_trajectories, sample_x0_centered_disk, generate_P_fixed_mu_L
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexQuadraticFunction, SmoothStronglyConvexFunction
from reformulator.dro_reformulator import DROReformulator

log = logging.getLogger(__name__)

def marchenko_pastur(d, mu, L):
    r = (np.sqrt(L) - np.sqrt(mu))**2 / (np.sqrt(L) + np.sqrt(mu))**2
    sigma = (np.sqrt(L) + np.sqrt(mu)) / 2
    X = np.random.normal(0, sigma, size=(d, np.round(r*d).astype(np.int64)))
    # there is a possibility that H has eigenvalue larger than L
    H = X.T@X/d
    return H

def rejection_sample_MP(dim, mu, L):
    Q = marchenko_pastur(dim, mu, L)
    eigvals = np.real(np.linalg.eigvals(Q))
    if mu > np.min(eigvals) or L < np.max(eigvals):
        # print('reject sample')
        return rejection_sample_MP(dim, mu, L)
    return Q

def sample_x0_centered_disk(n, R):
    x = np.random.normal(0, 1, n)
    x /= np.linalg.norm(x)
    dist = np.random.uniform(0, 1) ** (1 / n)
    return R * dist * x


class Quad(object):

    def __init__(self, dim, mu=0, L=10, R=1):
        self.dim = dim
        self.mu = mu
        self.L = L
        self.R = R

        # self.x0 = np.zeros(dim)
        # self.x0[0] = R
        self.x0 = self.sample_init_point()

        self.Q = rejection_sample_MP(dim, mu, L)

        self.f_star = 0
        self.x_star = np.zeros(self.dim)

    def f(self, x):
        return .5 * x.T @ self.Q @ x
    
    def g(self, x):
        return self.Q @ x

    def sample_init_point(self):
        return sample_x0_centered_disk(self.dim, self.R)


class JaxQuad(object):
    def __init__(self, Q, z0):
        self.Q = jnp.array(Q)
        self.z0 = jnp.array(z0)
        self.dim = Q.shape[0]
        self.f_star = 0
        self.z_star = jnp.zeros(self.dim)
    
    @partial(jax.jit, static_argnames=['self'])
    def f(self, x):
        return .5 * x.T @ self.Q @ x
    
    @partial(jax.jit, static_argnames=['self'])
    def g(self, x):
        return self.Q @ x


@partial(jax.jit, static_argnames=['f', 'g', 'params'])
def jax_gd(f, g, x0, xs, params):
    t = params['t']
    K_max = params['K_max']
    dim = x0.shape[0]
    x_stack = jnp.zeros((dim, K_max + 2))
    g_stack = jnp.zeros((dim, K_max + 2))
    f_stack = jnp.zeros(K_max + 2)

    x_stack = x_stack.at[:, 0].set(xs)
    x_stack = x_stack.at[:, 1].set(x0)
    g_stack = g_stack.at[:, 0].set(g(xs))
    g_stack = g_stack.at[:, 1].set(g(x0))
    f_stack = f_stack.at[0].set(f(xs))
    f_stack = f_stack.at[1].set(f(x0))

    def body_fun(i, val):
        x, x_stack, g_stack, f_stack = val
        x = x - t * g(x)
        x_stack = x_stack.at[:, i+1].set(x)
        g_stack = g_stack.at[:, i+1].set(g(x))
        f_stack = f_stack.at[i+1].set(f(x))
        return (x, x_stack, g_stack, f_stack)

    init_val = (x0, x_stack, g_stack, f_stack)
    _, x_stack, g_stack, f_stack = \
        jax.lax.fori_loop(1, K_max + 1, body_fun, init_val)
    
    return x_stack, g_stack, f_stack


@partial(jax.jit, static_argnames=['f', 'g', 'algorithm', 'params'])
def jax_generate_trajectories(f, g, x0, xs, algorithm, params):
    x_stack, g_stack, f_stack = algorithm(f, g, x0, xs, params)

    G_half = jnp.hstack([x_stack[:,:2], g_stack[:,1:]])
    return G_half.T@G_half, f_stack


def main():
    dim = 10
    mu = 0
    L = 1
    R = 1
    N = 8
    np.random.seed(0)
    quad_funcs = []
    jax_quad_funcs = []
    for i in range(N):
        q = Quad(dim, mu=mu, L=L, R=R)
        # q = QuadBadAccel(cfg.dim, mu=cfg.mu, L=cfg.L, R=cfg.R)
        quad_funcs.append(q)
        jax_quad_funcs.append(JaxQuad(q.Q, q.x0))
    
    j = jax_quad_funcs[0]
    params = frozendict({
        't': 1,
        'K_max': 10,
    })

    # print(jax_gd(j.f, j.g, j.z0, j.z_star, params))
    # G, F = jax_generate_trajectories(j.f, j.g, j.z0, j.z_star, jax_gd, params)
    # print(G.shape, F.shape)
    Q_stack = jnp.array([q.Q for q in jax_quad_funcs])
    z0_stack = jnp.array([q.z0 for q in jax_quad_funcs])
    print(Q_stack.shape)
    print(z0_stack.shape)


if __name__ == '__main__':
    main()
