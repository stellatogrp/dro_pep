import numpy as np
import pandas as pd
import logging
import time
from tqdm import trange

from .utils import generate_P_bounded_mu_L, gradient_descent, generate_trajectories
from PEPit import PEP
from PEPit.functions import SmoothConvexFunction
from reformulator.dro_reformulator import DROReformulator

log = logging.getLogger(__name__)


class Quad(object):

    def __init__(self, dim, mu=0, L=11, R=1):
        self.dim = dim
        self.mu = mu
        self.L = L
        self.R = R

        self.x0 = np.zeros(dim)
        self.x0[0] = R

        self.f_star = 0
        self.x_star = np.zeros(dim)

        self.Q = generate_P_bounded_mu_L(dim, mu, L)

    def f(self, x):
        return .5 * x.T @ self.Q @ x
    
    def g(self, x):
        return self.Q @ x


def simple_quad_dro(cfg):
    log.info(cfg)
    np.random.seed(cfg.seed)

    eps_vals = np.logspace(cfg.eps.log_min, cfg.eps.log_max, num=cfg.eps.logspace_count)
    eps_vals = [0] + list(eps_vals)

    quad_funcs = []
    for i in trange(cfg.sample_N):
        q = Quad(cfg.dim, mu=cfg.mu, L=cfg.L, R=cfg.R)
        quad_funcs.append(q)
    
    res = []
    pep_res = []

    for k in cfg.K_vals:
        samples = []
        log.info(k)
        problem = quad_pep_subproblem(cfg, gradient_descent, k, cfg.pep_obj, return_problem=True)
        tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')
        log.info(f'----pep problem solved at k={k}----')
        log.info(f'pep obj: {tau}')
        pep_res.append(pd.Series({
            'K': k,
            'pep_obj': tau,
        }))

        pep_df = pd.DataFrame(pep_res)
        pep_df.to_csv(cfg.pep_fname, index=False)

        for i in range(cfg.sample_N):
            h = quad_funcs[i]
            x0 = h.x0
            xs = h.x_star
            fs = h.f_star

            params = {
                't': cfg.eta / cfg.L,
                'K_max': k,
            }

            G, F = generate_trajectories(h.f, h.g, x0, xs, fs, gradient_descent, params)
            # log.info(F.shape)
            samples.append((G, F))

        DR = DROReformulator(
            problem,
            samples,
            cfg.dro_obj,
            'clarabel',
            precond=True,
            mro_clusters=cfg.num_clusters,
        )

        for eps_idx, eps in enumerate(eps_vals):
            log.info(eps_idx)
            log.info(eps)

            DR.set_params(eps=eps)
            out = DR.solve()
            if cfg.num_clusters is not None:
                dro_feas = DR.extract_dro_feas_sol_from_mro(eps=eps)
            else:
                dro_feas = out['obj']

            res.append(pd.Series({
                'K': k,
                'eps_idx': eps_idx,
                'eps': eps,
                'mro_sol': out['obj'],
                'solvetime': out['solvetime'],
                'dro_feas_sol': dro_feas,
            }))
        
            df = pd.DataFrame(res)
            df.to_csv(cfg.dro_fname, index=False)


def quad_pep_subproblem(cfg, algo, k, obj, return_problem=False):
    problem = PEP()
    func = problem.declare_function(SmoothConvexFunction, L=cfg.L)
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

    if return_problem:
        return problem

    start = time.time()
    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')
    # pepit_tau = problem.solve(wrapper='cvxpy', solver='MOSEK')
    solvetime = time.time() - start

    log.info(pepit_tau)
    return pepit_tau, solvetime
