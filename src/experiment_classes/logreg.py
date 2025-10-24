import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import time
from tqdm import trange
from sklearn.datasets import load_breast_cancer
from .utils import gradient_descent, nesterov_accelerated_gradient, nesterov_fgm, generate_trajectories, sample_x0_centered_disk
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from reformulator.dro_reformulator import DROReformulator
# from ucimlrepo import fetch_ucirepo
from .lyap_classes.gd import gd_lyap, gd_lyap_nobisect

log = logging.getLogger(__name__)


def sigmoid(z):
    return 1/(1 + np.exp(-z))


class LogReg(object):

    def __init__(self, sample_frac=0.01, delta=0.1, R=1):
        self.delta = delta

        full_X, full_y = load_breast_cancer(return_X_y=True)

        # default_of_credit_card_clients = fetch_ucirepo(id=350)
        # X = default_of_credit_card_clients.data.features 
        # y = default_of_credit_card_clients.data.targets 
        # full_X = X.to_numpy()
        # full_y = y.to_numpy().reshape(-1,)
        
        self.full_X = full_X
        self.full_y = full_y

        self.samp_X, self.samp_y = self.sample_normalized(sample_frac=sample_frac)
        self.solve_optimal_values()
        self.compute_mu_L()

        self.x0 = np.zeros(self.samp_X.shape[1])
        self.x0[0] = R
        self.R = R
        self.dim = self.samp_X.shape[1]

    def sample(self, sample_frac=0.8):
        X = self.full_X
        y = self.full_y

        sample_size = int(0.8 * X.shape[0])
        idx = np.random.choice(X.shape[0], size=sample_size, replace=False)

        return X[idx], y[idx]

    def sample_normalized(self, sample_frac=0.8):
        X_samp, y_samp = self.sample(sample_frac=sample_frac)
        means = X_samp.mean(axis=0)
        std_devs = X_samp.std(axis=0)

        X_samp_normalized = (X_samp - means) / std_devs
        ones_column = np.ones((X_samp.shape[0], 1))
        X_samp_with_ones = np.hstack((X_samp_normalized, ones_column))
        return X_samp_with_ones, y_samp

    def solve_optimal_values(self):
        X, y = self.samp_X, self.samp_y
        m, n = X.shape
        beta = cp.Variable(n)
        log_likelihood = cp.sum(
            cp.multiply(y, X @ beta) - cp.logistic(X @ beta)
        )
        obj = - 1 / m * log_likelihood + 0.5 * self.delta * cp.sum_squares(beta)
        problem = cp.Problem(cp.Minimize(obj))
        problem.solve()

        self.x_opt = beta.value
        self.f_opt = problem.value

    def compute_mu_L(self):
        X = self.samp_X
        m = X.shape[0]

        XTX_eigvals = np.real(np.linalg.eigvals(X.T @ X))
        lambd_max = np.max(XTX_eigvals)
        L = lambd_max / (4 * m) + self.delta
        mu = self.delta
        
        self.mu, self.L = mu, L

    def f(self, z):
        X, y = self.samp_X, self.samp_y
        m = X.shape[0]
        z = z + self.x_opt
        log_likeli = np.sum(np.multiply(y, X @ z) - np.logaddexp(0, X @ z))
        return - 1 / m * log_likeli + 0.5 * self.delta * z.T @ z - self.f_opt

    def grad(self, z):
        X, y = self.samp_X, self.samp_y
        m = X.shape[0]
        z = z + self.x_opt
        return 1 / m * X.T @ (sigmoid(X @ z) - y) + self.delta * z

    def sample_init_point(self):
        return sample_x0_centered_disk(self.dim, self.R)

def logreg_samples(cfg):
    log.info(cfg)
    log.info(cfg)
    np.random.seed(cfg.seed.full_samples)

    df = []

    params = {
        't': cfg.eta, # NOT cfg.eta / cfg.L
        'K_max': cfg.K_max,
        'q': cfg.delta / cfg.L, 
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

    L_vals = []

    for i in trange(cfg.sample_N):
        lr = LogReg(sample_frac=cfg.sample_frac, delta=cfg.delta, R=cfg.R)
        # x0 = lr.x0
        x0 = lr.sample_init_point()
        # xs = lr.x_opt
        # fs = lr.f_opt
        xs = np.zeros(lr.samp_X.shape[1])
        fs = 0

        L_vals.append(lr.L)

        x_stack, g_stack, f_stack = algo(lr.f, lr.grad, x0, xs, params)
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

    log.info(f'mu:{cfg.delta}')
    log.info(f'L:{np.max(L_vals)}')

    plot_worst_case(df_to_save, 'obj_val', cfg)


def plot_worst_case(df, col, cfg):
    worst_cases = df[['K', col]].groupby(['K']).max()
    plt.plot(range(1, cfg.K_max + 1), worst_cases)
    plt.yscale('log')
    plt.title(col)
    plt.savefig('worstcases.pdf')


def logreg_pep(cfg):
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

    mu = cfg.delta
    L = cfg.L

    res = []
    for k in range(cfg.K_min, cfg.K_max + 1):
        for obj in objs:
            tau, solvetime = logreg_pep_subproblem(cfg, mu, L, algo, k, obj)

            res.append(pd.Series({
                'K': k,
                'obj': obj,
                'val': tau,
                'solvetime': solvetime,
            }))
            df = pd.DataFrame(res)
            df.to_csv(cfg.pep_fname, index=False)


def logreg_pep_subproblem(cfg, mu, L, algo, k, obj, return_problem=False):
    problem = PEP()
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    xs = func.stationary_point()
    fs = func(xs)
    x0 = problem.set_initial_point()

    params = {
        't': cfg.eta, # NOT cfg.eta / cfg.L
        'K_max': k,
        'q': cfg.delta / cfg.L, 
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
    pepit_tau = problem.solve(wrapper='cvxpy', solver='CLARABEL')
    # solvetime = time.time() - start

    solvetime = problem.wrapper.prob.solver_stats.solve_time

    log.info(pepit_tau)
    return pepit_tau, solvetime


def logreg_dro(cfg):
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
    logreg_funcs = []
    for i in trange(N):
        lr = LogReg(sample_frac=cfg.sample_frac, delta=cfg.delta, R=cfg.R)
        logreg_funcs.append(lr)
    
    res = []

    for k in range(cfg.K_min, cfg.K_max + 1):
        samples = []
        problem = logreg_pep_subproblem(cfg, cfg.delta, cfg.L, algo, k, cfg.dro_pep_obj, return_problem=True)
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
            lr = logreg_funcs[i]
            # x0 = lr.x0
            x0 = lr.sample_init_point()
            # xs = lr.x_opt
            # fs = lr.f_opt
            xs = np.zeros(lr.samp_X.shape[1])
            fs = 0

            params = {
                't': cfg.eta, # NOT cfg.eta / cfg.L
                'K_max': k,
                'q': cfg.delta / cfg.L, 
            }

            G, F = generate_trajectories(lr.f, lr.grad, x0, xs, fs, algo, params)
            # log.info(F.shape)
            samples.append((G, F))

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


def logreg_lyap(cfg):
    log.info(cfg)

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
    logreg_funcs = []
    params = {
        # 't': cfg.eta / cfg.L,
        't': cfg.eta,
        'K_max': cfg.K_max,
        'q': cfg.delta / cfg.L, 
    }

    samples = []
    for i in trange(N):
        lr = LogReg(sample_frac=cfg.sample_frac, delta=cfg.delta, R=cfg.R)
        logreg_funcs.append(lr)
        x0 = lr.sample_init_point()
        xs = np.zeros(lr.dim)

        x, g, f = algo(lr.f, lr.grad, x0, xs, params)
        x = x[1:]
        g = g[1:]
        f = f[1:]
        sample_i = {
            'x': x,
            'g': g,
            'f': f,
        }
        samples.append(sample_i)
    
    GF = []
    for i in range(N):
        sample = samples[i]
        G, F, q = compute_sample_rho(sample)
        GF.append((G, F))

    dro_eps = .001
    dro_eps_vals = [1e-4, 1e-3, 1e-2, 1e-1]

    alpha_vals = np.linspace(1, .05, 20)
    print(alpha_vals)
    one_minus_alphas = []
    rhos = []
    for alpha in alpha_vals:
        # lyap_res = gd_lyap(cfg.delta, cfg.L, cfg.eta / cfg.L, 1, GF, dro_eps, cvar_alpha=alpha)
        lyap_res = gd_lyap_nobisect(cfg.delta, cfg.L, cfg.eta / cfg.L, 1, GF, dro_eps, cvar_alpha=alpha)
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
        fi = f[i]
        rho_i = f_iplus1 / fi
        # rho_i = np.linalg.norm(xiplus1) ** 2 / np.linalg.norm(xi) ** 2
        if rho_i > rho_max:
            rho_max = rho_i
            q = i
        # print(rho_i)
    print(rho_max, q)
    G_half = np.array([x[q], g[q], g[q+1]])
    return G_half @ G_half.T, np.array([f[q], f[q+1]]), q


def main():
    np.random.seed(0)

    delta = 0.1

    lr = LogReg(delta=delta)
    # X, y = lr.sample_normalized()
    X, y = lr.samp_X, lr.samp_y

    m, n = X.shape
    beta = cp.Variable(n)
    log_likelihood = cp.sum(
        cp.multiply(y, X @ beta) - cp.logistic(X @ beta)
    )
    obj = - 1 / m * log_likelihood + 0.5 * delta * cp.sum_squares(beta)
    problem = cp.Problem(cp.Minimize(obj))
    problem.solve()
    print(beta.value)

    def f(z):
        # log_likeli = np.sum(np.multiply(y, X @ z) - np.log1p(np.exp(X @ z)))
        log_likeli = np.sum(np.multiply(y, X @ z) - np.logaddexp(0, X @ z))
        return - 1 / m * log_likeli + 0.5 * delta * z.T @ z

    def grad(z):
        return 1 / m * X.T @ (sigmoid(X @ z) - y) + delta * z

    beta_k = np.zeros(n)
    fp_resids = []
    for _ in range(1000):
        beta_new = beta_k - 0.1 * grad(beta_k)
        fp_resids.append(np.linalg.norm(beta_new - beta_k))
        beta_k = beta_new
    print(beta_k)
    # print(fp_resids)
    XTX_eigvals = np.real(np.linalg.eigvals(X.T @ X))
    lambd_max = np.max(XTX_eigvals)
    L = lambd_max / (4 * m) + delta
    print(L)

    print(problem.value)
    print(f(beta.value))

    # f(x) -> f(x + x^\star) - f^\star

    print('--------')

    beta_k = np.ones(n)
    for _ in range(1000):
        beta_k = beta_k - 0.1 * lr.grad(beta_k)
    print(beta_k)
    print(lr.f(beta_k))
    print(lr.grad(beta_k))

    default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
    # data (as pandas dataframes) 
    X = default_of_credit_card_clients.data.features 
    y = default_of_credit_card_clients.data.targets 
    X = X.to_numpy()
    Y = y.to_numpy()
    print(X.shape, y.shape)

if __name__ == '__main__':
    main()
