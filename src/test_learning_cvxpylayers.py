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

# from utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, nesterov_fgm, generate_trajectories, sample_x0_centered_disk, generate_P_fixed_mu_L
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


def gradient_descent(f, g, x0, xs, params):
    t = params['t']
    # K = params['K']
    K_max = params['K_max']

    x_stack = []
    g_stack = []
    f_stack = []

    x = x0
    x_stack = [xs, x0]
    g_stack = [g(xs), g(x0)]
    f_stack = [f(xs), f(x0)]

    for k in range(1, K_max+1):
        # algorithm: GD
        x = x - t * g(x)
        x_stack.append(x)
        g_stack.append(g(x))
        f_stack.append(f(x))

    return x_stack, g_stack, f_stack


def generate_trajectories(f, g, x0, xs, fs, algorithm, params):
    x_stack, g_stack, f_stack = algorithm(f, g, x0, xs, params)
    x_stack = np.array(x_stack).T
    g_stack = np.array(g_stack).T
    f_stack = np.array(f_stack)

    G_half = np.hstack([x_stack[:,:2], g_stack[:,1:]])
    # F = np.concatenate([f_stack, [f_stack[-1]-f_stack[0]]])
    F = f_stack

    return G_half.T@G_half, F


def quad_pep_subproblem(algo, mu, L, R, t, k, obj, return_problem=False):
    problem = PEP()
    # func = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=cfg.mu, L=cfg.L)
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    xs = func.stationary_point()
    fs = func(xs)
    x0 = problem.set_initial_point()

    params = {
        't': t,
        'K_max': k,
    }
    log.info(params)

    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)
    x_stack, g_stack, f_stack = algo(func, func.gradient, x0, xs, params)

    # problem.set_performance_metric(func(x) - fs)
    if obj == 'obj_val':
        problem.set_performance_metric(f_stack[-1] - fs)
    elif obj == 'grad_sq_norm':
        problem.set_performance_metric((g_stack[-1]) ** 2)
    elif obj == 'opt_dist_sq_norm':
        problem.set_performance_metric((x_stack[-1] - xs) ** 2)
    else:
        log.info('should be unreachable code')
        exit(0)

    if return_problem:
        return problem

    # start = time.time()
    # pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')
    pepit_tau = problem.solve(wrapper='cvxpy', solver='MOSEK')
    # pepit_tau = problem.solve(wrapper='mosek')
    # solvetime = time.time() - start

    solvetime = problem.wrapper.prob.solver_stats.solve_time

    log.info(pepit_tau)
    return pepit_tau, solvetime


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
    dim = 5
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
        't': 1.,
        'K_max': 2,
    })

    # print(jax_gd(j.f, j.g, j.z0, j.z_star, params))
    # G, F = jax_generate_trajectories(j.f, j.g, j.z0, j.z_star, jax_gd, params)
    # print(G.shape, F.shape)
    Q_stack = jnp.array([q.Q for q in jax_quad_funcs])
    z0_stack = jnp.array([q.z0 for q in jax_quad_funcs])
    print(Q_stack.shape)
    print(z0_stack.shape)
    training_test(params['t'], params['K_max'], Q_stack, z0_stack)


@partial(jax.jit, static_argnames=['K_max'])
def problem_data_to_gd_trajectories(t, Q, z0, K_max):
    def f(x):
        return .5 * x.T @ Q @ x
    
    def g(x):
        return Q @ x

    dim = z0.shape[0]
    zs = jnp.zeros(dim)
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


def dro_pep_obj_jax(eps, lambd_star, s_star):
    N = s_star.shape[0]
    return lambd_star * eps + 1 / N * jnp.sum(s_star)


def SCS_pipeline(t, Q_batch, z0_batch, K_max, eps, alpha, DR, large_sdp_layer):
    batch_GF_func = jax.vmap(problem_data_to_gd_trajectories, in_axes=(None, 0, 0, None))
    G_batch, F_batch = batch_GF_func(t, Q_batch, z0_batch, K_max)
    (lambd_star, s_star) = large_sdp_layer(G_batch, F_batch)
    loss = dro_pep_obj_jax(eps, lambd_star, s_star)
    return loss


def training_test(t, K_max, Q_stack, z0_stack):
    print(t, K_max)
    batch_GF_func = jax.vmap(problem_data_to_gd_trajectories, in_axes=(None, 0, 0, None))
    G_batch, F_batch = batch_GF_func(t, Q_stack, z0_stack, K_max)

    eps = 0.1
    alpha = 0.1
    mu = 0
    L = 1
    R = 1
    k = K_max
    obj = 'obj_val'
    dro_obj = 'expectation'

    pep_problem = quad_pep_subproblem(gradient_descent, mu, L, R, t, k, obj, return_problem=True)
    mosek_params = {
            # 'intpntCoTolDfeas': 1e-7,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-3,
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-3,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-3,
        }
    pepit_tau = pep_problem.solve(
        wrapper='cvxpy',
        solver='MOSEK',
        mosek_params=mosek_params,
    )
    samples = [(np.array(G_batch[i]), np.array(F_batch[i])) for i in range(G_batch.shape[0]) ]

    DR = DROReformulator(
        pep_problem,
        samples,
        dro_obj,
        'cvxpy',
        precond=True,
        precond_type='average',
    )

    large_sdp_layer = create_exp_cp_layer(DR, eps, alpha, G_batch.shape, F_batch.shape)

    loss = SCS_pipeline(t, Q_stack, z0_stack, K_max, eps, alpha, DR, large_sdp_layer)
    print(loss)

    grad_fn = jax.grad(SCS_pipeline, argnums=(0, 1, 2))
    dt, dQ, dz0 = grad_fn(t, Q_stack, z0_stack, K_max, eps, alpha, DR, large_sdp_layer)

    print(dt)
    print(dQ)
    print(z0_stack, dz0)


if __name__ == '__main__':
    main()
