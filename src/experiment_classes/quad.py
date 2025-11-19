import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import time
from tqdm import trange

from .utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, nesterov_fgm, generate_trajectories, sample_x0_centered_disk, generate_P_fixed_mu_L
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexQuadraticFunction, SmoothStronglyConvexFunction
from reformulator.dro_reformulator import DROReformulator
from .lyap_classes.gd import gd_lyap, gd_lyap_nobisect

log = logging.getLogger(__name__)


def rejection_sample_MP(dim, mu, L):
    Q = marchenko_pastur(dim, mu, L)
    eigvals = np.real(np.linalg.eigvals(Q))
    if mu > np.min(eigvals) or L < np.max(eigvals):
        # print('reject sample')
        return rejection_sample_MP(dim, mu, L)
    return Q


class Quad(object):

    def __init__(self, dim, mu=0, L=10, R=1):
        # self.dim = dim
        self.mu = mu
        self.L = L
        self.R = R

        self.x0 = np.zeros(dim)
        self.x0[0] = R

        # self.Q = marchenko_pastur(dim, mu, L)
        self.Q = rejection_sample_MP(dim, mu, L)
        # eigvals = np.real(np.linalg.eigvals(self.Q))
        # print(np.min(eigvals), np.max(eigvals), self.Q.shape)
        # # print(eigvals)

        self.dim = self.Q.shape[0]
        # self.Q = generate_P_fixed_mu_L(dim, mu, L)

        self.f_star = 0
        self.x_star = np.zeros(self.dim)

    def f(self, x):
        return .5 * x.T @ self.Q @ x
    
    def g(self, x):
        return self.Q @ x

    def sample_init_point(self):
        return sample_x0_centered_disk(self.dim, self.R)


class QuadBadAccel(object):
    def __init__(self, dim, mu=1, L=2, R=1):
        self.dim = dim
        self.mu = mu
        self.L = L
        self.R = R

        self.x0 = np.zeros(dim)
        self.x0[0] = R

        self.f_star = 0
        self.x_star = np.zeros(dim)

        self.Q = np.diag(mu + np.random.uniform(high=L-mu, size=(dim,)))

    def f(self, x):
        return .5 * x.T @ self.Q @ x
    
    def g(self, x):
        return self.Q @ x

    def sample_init_point(self):
        return sample_x0_centered_disk(self.dim, self.R)


def quad_samples(cfg):
    log.info(cfg)
    np.random.seed(cfg.seed.full_samples)

    df = []

    params = {
        't': cfg.eta / cfg.L,
        'K_max': cfg.K_max,
        'q': cfg.mu / cfg.L, 
    }

    if cfg.alg == 'grad_desc':
        algo = gradient_descent
    elif cfg.alg == 'nesterov_grad_desc':
        algo = nesterov_accelerated_gradient
    elif cfg.alg == 'nesterov_fgm':
        algo = nesterov_fgm
    else:
        log.info('invalid alg in cfg')
        exit(0)

    for i in trange(cfg.sample_N):
        h = Quad(cfg.dim, mu=cfg.mu, L=cfg.L, R=cfg.R)
        # h = QuadBadAccel(cfg.dim, mu=cfg.mu, L=cfg.L, R=cfg.R)
        # x0 = h.x0
        x0 = h.sample_init_point()
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


def quad_pep(cfg):
    log.info(cfg)
    if cfg.alg == 'grad_desc':
        algo = gradient_descent
    elif cfg.alg == 'nesterov_grad_desc':
        algo = nesterov_accelerated_gradient
    elif cfg.alg == 'nesterov_fgm':
        algo = nesterov_fgm
    else:
        log.info('invalid alg in cfg')
        exit(0)

    # objs = ['obj_val', 'grad_sq_norm', 'opt_dist_sq_norm']
    objs = ['obj_val']
    # objs = ['opt_dist_sq_norm']

    res = []

    # quad_pep_subproblem(cfg, algo, 1, objs[0])
    for k in range(cfg.K_min, cfg.K_max + 1):
        for obj in objs:
            tau, solvetime = quad_pep_subproblem(cfg, algo, k, obj)

            res.append(pd.Series({
                'K': k,
                'obj': obj,
                'val': tau,
                'solvetime': solvetime,
            }))
            df = pd.DataFrame(res)
            df.to_csv(cfg.pep_fname, index=False)


def quad_pep_subproblem(cfg, algo, k, obj, return_problem=False):
    problem = PEP()
    # func = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=cfg.mu, L=cfg.L)
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=cfg.mu, L=cfg.L)
    xs = func.stationary_point()
    fs = func(xs)
    x0 = problem.set_initial_point()

    params = {
        't': cfg.eta / cfg.L,
        'K_max': k,
        'q': cfg.mu / cfg.L, 
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

    # start = time.time()
    # pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')
    pepit_tau = problem.solve(wrapper='cvxpy', solver='MOSEK')
    # pepit_tau = problem.solve(wrapper='mosek')
    # solvetime = time.time() - start

    solvetime = problem.wrapper.prob.solver_stats.solve_time

    log.info(pepit_tau)
    return pepit_tau, solvetime


def quad_dro(cfg):
    log.info(cfg)

    if cfg.alg == 'grad_desc':
        algo = gradient_descent
    elif cfg.alg == 'nesterov_grad_desc':
        algo = nesterov_accelerated_gradient
    elif cfg.alg == 'nesterov_fgm':
        algo = nesterov_fgm
    else:
        log.info('invalid alg in cfg')
        exit(0)

    if cfg.dro_obj == 'expectation':
        N = cfg.training.expectation_N
        num_clusters = cfg.num_clusters.expectation
        dro_obj = 'expectation'

    elif cfg.dro_obj == 'cvar':
        N = cfg.training.cvar_N
        num_clusters = cfg.num_clusters.cvar
        dro_obj = 'cvar'

    else:
        log.info('invalid dro obj')
        exit(0)

    eps_vals = np.logspace(cfg.eps.log_min, cfg.eps.log_max, num=cfg.eps.logspace_count)
    alpha = cfg.alpha

    np.random.seed(cfg.seed.train)
    quad_funcs = []
    for i in range(N):
        q = Quad(cfg.dim, mu=cfg.mu, L=cfg.L, R=cfg.R)
        # q = QuadBadAccel(cfg.dim, mu=cfg.mu, L=cfg.L, R=cfg.R)
        quad_funcs.append(q)
    
    res = []
    sample_df_list = []

    for k in range(cfg.K_min, cfg.K_max + 1):
        samples = []
        problem = quad_pep_subproblem(cfg, algo, k, cfg.dro_pep_obj, return_problem=True)
        # problem.solve(wrapper='cvxpy', solver='CLARABEL')
        mosek_params = {
            # 'intpntCoTolDfeas': 1e-7,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-7,
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-7,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-7,
        }
        pepit_tau = problem.solve(
            wrapper='cvxpy',
            solver='MOSEK',
            mosek_params=mosek_params,
        )
        log.info(f'----pep problem solved at k={k}----')

        for i in range(N):
            h = quad_funcs[i]
            # x0 = h.x0
            x0 = h.sample_init_point()
            xs = h.x_star
            fs = h.f_star

            params = {
                't': cfg.eta / cfg.L,
                'K_max': k,
                'q': cfg.mu / cfg.L, 
            }

            G, F = generate_trajectories(h.f, h.g, x0, xs, fs, algo, params)
            # log.info(F.shape)
            samples.append((G, F))
            sample_df_list.append(pd.Series({
                'i': i,
                'K': k,
                'obj_val': F[-1] - F[0],
                'grad_sq_norm': G[-1, -1],
            }))
        sample_df = pd.DataFrame(sample_df_list)
        sample_df.to_csv('samples.csv', index=False)
        # log.info(samples)

        DR = DROReformulator(
            problem,
            samples,
            dro_obj,
            'clarabel',
            precond=True,
            precond_type=cfg.precond_type,
            mro_clusters=num_clusters,
        )

        for eps_idx, eps in enumerate(eps_vals):
            log.info(eps_idx)
            log.info(eps)

            DR.set_params(eps=eps, alpha=alpha)
            out = DR.solve()
            # dro_feas = DR.extract_dro_feas_sol_from_mro(eps=eps, alpha=alpha)
            if num_clusters is not None:
                dro_feas = DR.extract_dro_feas_sol_from_mro(eps=eps, alpha=alpha)
            else:
                dro_feas = out['obj']

            res.append(pd.Series({
                'K': k,
                'eps_idx': eps_idx,
                'eps': eps,
                'alpha': alpha,
                'mro_sol': out['obj'],
                'solvetime': out['solvetime'],
                'dro_feas_sol': dro_feas,
            }))
        
            df = pd.DataFrame(res)
            df.to_csv(cfg.dro_fname, index=False)


def quad_lyap(cfg):

    log.info(cfg)

    if cfg.alg == 'grad_desc':
        algo = gradient_descent
    elif cfg.alg == 'nesterov_grad_desc':
        algo = nesterov_accelerated_gradient
    elif cfg.alg == 'nesterov_fgm':
        algo = nesterov_fgm
    else:
        log.info('invalid alg in cfg')
        exit(0)

    N = cfg.training.lyap_N
    eps_vals = np.logspace(cfg.eps.log_min, cfg.eps.log_max, num=cfg.eps.logspace_count)
    alpha = cfg.alpha

    np.random.seed(cfg.seed.train)
    quad_funcs = []
    params = {
        't': cfg.eta / cfg.L,
        'K_max': cfg.K_max,
        'q': cfg.mu / cfg.L, 
    }

    samples = []
    for i in range(N):
        q = Quad(cfg.dim, mu=cfg.mu, L=cfg.L, R=cfg.R)
        # q = QuadBadAccel(cfg.dim, mu=cfg.mu, L=cfg.L, R=cfg.R)
        quad_funcs.append(q)
        x0 = q.sample_init_point()
        xs = q.x_star
        # fs = q.f_star

        x, g, f = algo(q.f, q.g, x0, xs, params)
        x = x[1:]
        g = g[1:]
        f = f[1:]
        sample_i = {
            'x': x,
            'g': g,
            'f': f,
        }
        samples.append(sample_i)
    
    # compute rho
    # for now use obj value objective and initial distance

    GF = []
    for i in range(N):
        sample = samples[i]
        G, F, q = compute_sample_rho(sample)
        GF.append((G, F))
        # log.info(q)
    
    # exit(0)

    # dro_eps = .01
    # # lyap_res = gd_lyap(cfg.mu, cfg.L, cfg.eta / cfg.L, 1, GF, dro_eps)
    # lyap_res = gd_lyap_nobisect(cfg.mu, cfg.L, cfg.eta / cfg.L, 1, GF, dro_eps)
    # log.info(lyap_res)

    dro_eps = .01
    dro_eps_vals [1e-4, 1e-3, 1e-2, 1e-1]

    alpha_vals = np.linspace(1, .05, 20)
    print(alpha_vals)
    one_minus_alphas = []
    rhos = []
    for alpha in alpha_vals:
        # lyap_res = gd_lyap(cfg.mu, cfg.L, cfg.eta / cfg.L, 1, GF, dro_eps, cvar_alpha=alpha)
        lyap_res = gd_lyap_nobisect(cfg.mu, cfg.L, cfg.eta / cfg.L, 1, GF, dro_eps, cvar_alpha=alpha)
        log.info(lyap_res)
        one_minus_alphas.append(1 - alpha)
        rhos.append(lyap_res)
    
    plt.plot(one_minus_alphas, rhos)
    plt.xlabel('one minus alpha')
    plt.ylabel('rho')
    # plt.show()
    plt.savefig('rho_plot.pdf')

def compute_sample_rho(sample):
    x, g, f = sample['x'], sample['g'], sample['f']
    rho_max = 0
    K = len(x) - 1
    q = 0
    # print('----')
    for i in range(K):
        xiplus1 = x[i+1]
        f_iplus1 = f[i+1]
        xi = x[i]
        # rho_i = f_iplus1 / np.linalg.norm(xi) ** 2
        rho_i = np.linalg.norm(xiplus1) ** 2 / np.linalg.norm(xi) ** 2
        if rho_i > rho_max:
            rho_max = rho_i
            q = i
        # print(rho_i)
    print(rho_max, q)
    G_half = np.array([x[q], g[q], g[q+1]])
    return G_half @ G_half.T, np.array([f[q], f[q+1]]), q

# def quad_lyap(cfg):

#     res = []
#     sample_df_list = []

#     for k in range(cfg.K_min, cfg.K_max + 1):
#         log.info(f'----k={k}----')
#         samples = []

#         for i in range(N):
#             h = quad_funcs[i]
#             # x0 = h.x0
#             x0 = h.sample_init_point()
#             xs = h.x_star
#             fs = h.f_star

#             params = {
#                 't': cfg.eta / cfg.L,
#                 'K_max': k,
#                 'q': cfg.mu / cfg.L, 
#             }

#             G, F = generate_trajectories(h.f, h.g, x0, xs, fs, algo, params)
#             # log.info(F.shape)
#             samples.append((G, F))
#             sample_df_list.append(pd.Series({
#                 'i': i,
#                 'K': k,
#                 'obj_val': F[-1] - F[0],
#                 'grad_sq_norm': G[-1, -1],
#             }))
#         sample_df = pd.DataFrame(sample_df_list)
#         sample_df.to_csv('samples.csv', index=False)

#         dro_eps = 0.1
#         lyap_res = gd_lyap(cfg.mu, cfg.L, cfg.eta / cfg.L, k, samples, 0.1)
#         log.info(lyap_res)
