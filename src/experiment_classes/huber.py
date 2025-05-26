import numpy as np
import pandas as pd
import logging
import time
from tqdm import trange

from .utils import generate_P_bounded_mu_L, gradient_descent, nesterov_accelerated_gradient
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction

log = logging.getLogger(__name__)


class Huber(object):

    def __init__(self, dim, mu=0, L=10, delta=0.25, R=1):
        self.dim = dim
        self.mu = mu
        self.L = L
        self.delta = delta
        self.R = R

        self.x0 = np.zeros(dim)
        self.x0[0] = R

        self.f_star = 0
        self.x_star = np.zeros(dim)

        self.Q = generate_P_bounded_mu_L(dim, mu, L)

    def f(self, x):
        Q, delta = self.Q, self.delta
        bound = np.sqrt(x.T @ Q @ x)
        if bound <= delta:
            # return .5 * x.T @ Q @ x
            return .5 * bound ** 2
        else:
            return delta * (bound - .5 * delta)

    def g(self, x):
        Q, delta = self.Q, self.delta
        bound = np.sqrt(x.T @ Q @ x)
        if bound <= delta:
            return Q @ x
        else:
            return (delta / bound) * Q @ x


def huber_samples(cfg):
    log.info(cfg)
    np.random.seed(cfg.seed.full_samples)

    df = []

    params = {
        't': cfg.eta / cfg.L,
        'K_max': cfg.K_max,
    }

    if cfg.alg == 'grad_desc':
        algo = gradient_descent
    elif cfg.alg == 'nesterov_grad_desc':
        algo = nesterov_accelerated_gradient
    else:
        log.info('invalid alg in cfg')
        exit(0)

    for i in trange(cfg.sample_N):
        h = Huber(cfg.dim, mu=cfg.mu, L=cfg.L, delta=cfg.delta, R=cfg.R)
        x0 = h.x0
        xs = h.x_star
        fs = h.f_star

        x_stack, g_stack, f_stack = algo(h.f, h.g, x0, xs, params)
        # stacks: [xs, x0, ..., xK]
        for k in range(1, cfg.K_max + 1):
            df.append(pd.Series({
                'i': i,
                'K': k,
                'obj_val': f_stack[k+1] - fs,
                'grad_sq_norm': np.linalg.norm(g_stack[k+1]) ** 2,
                'opt_dist_sq_norm': np.linalg.norm(x_stack[k+1] - xs) ** 2,
            }))

        if i % 1000 == 0:
            log.info(f'saving at i={i}')
            df_to_save = pd.DataFrame(df)
            df_to_save.to_csv(cfg.sample_fname, index=False)

    df_to_save = pd.DataFrame(df)
    df_to_save.to_csv(cfg.sample_fname, index=False)


def huber_pep(cfg):
    log.info(cfg)
    if cfg.alg == 'grad_desc':
        algo = gradient_descent
    elif cfg.alg == 'nesterov_grad_desc':
        algo = nesterov_accelerated_gradient
    else:
        log.info('invalid alg in cfg')
        exit(0)

    objs = ['obj_val', 'grad_sq_norm', 'opt_dist_sq_norm']

    res = []

    huber_pep_subproblem(cfg, algo, 1, objs[0])
    for k in range(cfg.K_min, cfg.K_max + 1):
        for obj in objs:
            tau, solvetime = huber_pep_subproblem(cfg, algo, k, obj)

            res.append(pd.Series({
                'K': k,
                'obj': obj,
                'val': tau,
                'solvetime': solvetime,
            }))
            df = pd.DataFrame(res)
            df.to_csv(cfg.pep_fname, index=False)



def huber_pep_subproblem(cfg, algo, k, obj):
    problem = PEP()
    # func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=cfg.mu, L=cfg.L)

    xs = func.stationary_point()
    fs = func(xs)

    x0 = problem.set_initial_point()

    params = {
        't': cfg.eta / cfg.L,
        'K_max': k, 
    }
    log.info(params)

    problem.set_initial_condition((x0 - xs) ** 2 <= cfg.R ** 2)
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

    start = time.time()
    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')
    solvetime = time.time() - start

    log.info(pepit_tau)
    return pepit_tau, solvetime


def main():
    np.random.seed(0)
    dim = 300
    L = 10
    delta = 1
    h = Huber(dim, L=L, delta=delta, R=10)
    print(h.x0)
    print(h.Q)
    print(np.linalg.eigvals(h.Q))

    # x_test = .5 * np.random.uniform(size=(dim,))
    # print(np.linalg.norm(x_test))
    # print(h.f(x_test))
    # print(.5 * x_test.T @ h.Q @ x_test)
    # print(h.delta * (np.sqrt(x_test.T @ h.Q @ x_test) - .5 * h.delta))
    x_test = h.x0

    K = 100
    for i in range(K):
        print(f'---{i}---')
        x_new = x_test - 1 / L * h.g(x_test)
        print(f'obj: {h.f(x_new) - h.f_star}')
        print(f'grad sq norm: {np.linalg.norm(h.g(x_new)) ** 2}')
        print(f'sq dist: {np.linalg.norm(x_new - h.x_star) ** 2}')
        x_test = x_new


if __name__ == '__main__':
    main()
