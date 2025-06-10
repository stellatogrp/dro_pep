import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sp
import logging
import time
from tqdm import trange

# from .utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, generate_trajectories
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexQuadraticFunction, ConvexLipschitzFunction, ConvexFunction
from PEPit.primitive_steps import proximal_step
from PEPit.tools.expressions_to_matrices import expression_to_matrices
from reformulator.dro_reformulator import DROReformulator

log = logging.getLogger(__name__)


def generate_A(cfg):
    np.random.seed(cfg.seed.A)
    A = np.random.normal(scale=1/cfg.m, size=(cfg.m, cfg.n))

    A_mask = np.random.binomial(1, p=cfg.p_A_nonzero, size=(cfg.m, cfg.n))

    A = np.multiply(A, A_mask)

    A = A / np.linalg.norm(A, axis=0)

    return A


def generate_single_b(cfg, A):
    x_samp = np.random.normal(size=(cfg.n,))
    x_mask = np.random.binomial(1, p=cfg.p_xsamp_nonzero, size=(cfg.n,))
    x_samp = np.multiply(x_samp, x_mask)

    return A @ x_samp + cfg.noise_eps * np.random.normal(size=(cfg.m, ))


def soft_threshold(v, delta):
    return np.sign(v) * np.maximum(np.abs(v) - delta, 0)


def solve_single_cvxpy(cfg, A, b):
    m, n = A.shape
    x = cp.Variable(n)
    obj = .5 * cp.sum_squares(A @ x - b) + cfg.lambd * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(obj))
    res = prob.solve()
    log.info(f'single x sol with lambda = {cfg.lambd}: {x.value}')
    log.info(f'opt value = {res}')

    R = np.linalg.norm(x.value)
    return x.value, R


def lasso_samples(cfg):
    log.info(cfg)


def lasso_pep(cfg):
    log.info(cfg)


def ista_pep_subproblem(cfg, K, mu, L, R, return_problem=False):
    problem = PEP()
    gamma = cfg.eta / L
    lambd = cfg.lambd

    f1 = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=mu, L=L, reuse_gradient=True)
    f2 = problem.declare_function(ConvexLipschitzFunction, M=lambd * np.sqrt(cfg.n), reuse_gradient=True)
    # f2 = problem.declare_function(ConvexFunction, reuse_gradient=True)
    func = f1 + f2

    xs = func.stationary_point()
    x0 = problem.set_initial_point()

    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)

    x = [x0 for _ in range(K+1)]
    g = [None for _ in range(K+1)]
    f = [None for _ in range(K+1)]

    for k in range(K):
        y = x[k] - gamma * f1.gradient(x[k])
        x[k+1], g[k+1], f[k+1] = proximal_step(y, f2, gamma)

    if cfg.pep_obj == 'obj_val':
        problem.set_performance_metric(func(x[-1]) - func(xs))
    elif cfg.pep_obj == 'grad_sq_norm':
        problem.set_performance_metric((g[-1]) ** 2)
    elif cfg.pep_obj == 'opt_dist_sq_norm':
        problem.set_performance_metric((x[-1] - xs) ** 2)
    else:
        log.info('should be unreachable code')
        exit(0)

    if return_problem:
        return problem

    pepit_tau = problem.solve(wrapper='cvxpy', solver='MOSEK')
    # pepit_tau = problem.solve()
    log.info(f'pepit_tau at K={K}: {pepit_tau}')

    solvetime = problem.wrapper.prob.solver_stats.solve_time
    log.info(f'solvetime: {solvetime}')

    return pepit_tau, solvetime


def single_trajectory(cfg, K, A, b, x_opt, x0, lu, piv, L, alg='ista'):
    x_ls = sp.linalg.lu_solve((lu, piv), A.T @ b)
    lambd = cfg.lambd
    gamma = cfg.eta / L

    def f1(x):
        return .5 * np.linalg.norm(A @ x - b) ** 2 - .5 * np.linalg.norm(A @ x_ls - b)

    def f2(x):
        return lambd * np.linalg.norm(x, 1)

    def grad_f1(x):
        return A.T @ (A @ x - b)

    def subgrad_f2(x):
        return lambd * np.sign(x)

    def f(x):
        return f1(x) + f2(x)

    G = [x_opt, grad_f1(x_opt), x0, grad_f1(x0)]
    F = [f(x_opt), f1(x_opt), f1(x0)]

    # # TODO: fix below, this is just for K=1
    # y1 = x0 - gamma * grad_f1(x0)
    # x1 = soft_threshold(y1, gamma * lambd)
    # g1 = (y1 - x1) / gamma
    # subgrad_f1 = grad_f1(x1) + g1
    # f2_1 = f2(x1)
    # f_1 = f(x1)

    # G += [g1, subgrad_f1, x_ls]
    # F += [f2_1, f_1]

    # # F+= [f(x1) - f(x_opt), f1(x_ls)]

    xk = x0
    for _ in range(K - 1):
        ykplus1 = xk - gamma * grad_f1(xk)
        xkplus1 = soft_threshold(ykplus1, gamma * lambd)
        gkplus1 = (ykplus1 - xkplus1) / gamma
        grad_f1_xkplus1 = grad_f1(xkplus1)

        G += [gkplus1, grad_f1_xkplus1]
        F += [f2(xkplus1), f1(xkplus1)]

        xk = xkplus1

    yk = xk - gamma * grad_f1(xk)
    xk = soft_threshold(yk, gamma * lambd)
    gk = (yk - xk) / gamma
    subgrad_f_xk = grad_f1(xk) + gk
    f2_k = f2(xk)
    f_k = f(xk)

    G += [gk, subgrad_f_xk, x_ls]
    F += [f2_k, f_k]

    # F += [f(xk) - f(x_opt), f1(x_ls)]

    G = np.array(G)
    F = np.array(F)

    return G @ G.T, F


def lasso_dro(cfg):
    log.info(cfg)
    A = generate_A(cfg)
    log.info(A)
    ATA = A.T @ A

    ATA_lu, ATA_piv = sp.linalg.lu_factor(ATA)

    ATA_eigvals = np.real(np.linalg.eigvals(ATA))
    L = np.max(ATA_eigvals)

    if cfg.m >= cfg.n:
        mu = np.min(ATA_eigvals)
    else:
        mu = 0
    log.info(f'L: {L}, mu: {mu}')
    
    b_test = generate_single_b(cfg, A)
    log.info(b_test)
    x_test_opt, R = solve_single_cvxpy(cfg, A, b_test)
    log.info(f'radius: {R}')

    K = 10
    x0 = np.zeros(cfg.n)

    problem = ista_pep_subproblem(cfg, K, mu, L, R, return_problem=True)
    tau = problem.solve(wrapper='cvxpy', solver='MOSEK')
    log.info(f'----pep problem solved at k={K} with tau={tau}----')

    G, F = single_trajectory(cfg, K, A, b_test, x_test_opt, x0, ATA_lu, ATA_piv, L)
    # print(G, F)
    # for constr in problem._list_of_constraints_sent_to_wrapper[1:]:
    #     A_cons, b_cons, c_cons = expression_to_matrices(constr.expression)
    #     print('---')
    #     print(A_cons, b_cons, c_cons)
    #     print(constr.equality_or_inequality)
    #     print(np.trace(A_cons @ G) + b_cons @ F + c_cons)

    samples = [(G, F)]
    DR = DROReformulator(
        problem,
        samples,
        cfg.dro_obj,
        'clarabel',
        precond=True,
        precond_type=cfg.precond_type,
        mro_clusters=None,
        obj_vec_cutoff=2,
    )

    eps = 1e-3
    alpha = 0.1

    DR.set_params(eps=eps, alpha=alpha)
    out = DR.solve()
    log.info(out['obj'])
