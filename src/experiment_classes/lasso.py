import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sp
import logging
import time
from tqdm import trange

# from .utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, generate_trajectories
from .utils import sample_x0_centered_disk
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
    # log.info(f'single x sol with lambda = {cfg.lambd}: {x.value}')
    # log.info(f'opt value = {res}')

    R = np.linalg.norm(x.value)
    return x.value, R


def simulate_alg(cfg, x0, A, b, x_opt, L, lu, piv, alg='ista'):
    K_max = cfg.K_max
    x_ls = sp.linalg.lu_solve((lu, piv), A.T @ b)
    lambd = cfg.lambd
    def f1(x):
        return .5 * np.linalg.norm(A @ x - b) ** 2 - .5 * np.linalg.norm(A @ x_ls - b)

    def f2(x):
        return lambd * np.linalg.norm(x, 1)

    def f(x):
        return f1(x) + f2(x)

    def grad_f1(x):
        return A.T @ (A @ x - b)

    f_opt = f(x_opt)

    # only doing obj_val for this experiment

    if alg == 'ista':
        gamma = cfg.eta / L

        xk = x0
        # F_vals = [f(x0) - f_opt]
        F_vals = []
        for _ in range(K_max):
            xnew = soft_threshold(xk - gamma * grad_f1(xk), gamma * lambd)
            F_vals.append(f(xnew) - f_opt)
            xk = xnew
    elif alg == 'fista':
        gamma = cfg.eta / L

        xk = x0
        yk = x0
        beta_k = 1
        F_vals = []
        for _ in range(K_max):
            xkplus1 = soft_threshold(yk - gamma * grad_f1(yk), gamma * lambd)

            beta_kplus1 = .5 * (1 + np.sqrt(1 + 4 * beta_k ** 2))
            ykplus1 = xkplus1 + (beta_k - 1) / beta_kplus1 * (xkplus1 - xk)

            F_vals.append(f(xkplus1) - f_opt)

            xk = xkplus1
            yk = ykplus1
            beta_k = beta_kplus1
    elif alg == 'optista':
        thetas = [1]
        for _ in range(K_max-1):
            thetas.append(.5 * (1 + np.sqrt(1 + 4 * thetas[-1] ** 2)))
        thetas.append(.5 * (1 + np.sqrt(1 + 8 * thetas[-1] ** 2)))

        gammas = []
        for i in range(K_max):
            gamma_i = (2 * thetas[i] / thetas[K_max] ** 2) * ((thetas[K_max] ** 2) - (2 * thetas[i] ** 2) + thetas[i])
            gammas.append(gamma_i)

        xk = x0
        yk = x0
        zk = x0
        F_vals = []
        for k in range(K_max):
            # xtilde = x[k] - gammas[k] / L * f1.gradient(y[k])
            # x[k+1], g[k+1], f[k+1] = proximal_step(xtilde, f2, gammas[k] / L)
            # z[k+1] = y[k] + 1 / gammas[k] * (x[k+1] - x[k])
            # y[k+1] = z[k+1] + (thetas[k] - 1) / thetas[k+1] * (z[k+1] - z[k])  + thetas[k] / thetas[k+1] * (z[k+1] - y[k])
            
            xkplus1 = soft_threshold(xk - gammas[k] / L * grad_f1(yk), gammas[k] * lambd / L)
            zkplus1 = yk + 1 / gammas[k] * (xkplus1 - xk)
            ykplus1 = zkplus1 + (thetas[k] - 1) / thetas[k+1] * (zkplus1 - zk) + thetas[k] / thetas[k+1] * (zkplus1 - yk)

            F_vals.append(f(xkplus1) - f_opt)

            xk = xkplus1
            zk = zkplus1
            yk = ykplus1

    else:
        raise NotImplementedError
        exit(0)

    return F_vals


def lasso_samples(cfg):
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

    x0 = np.zeros(cfg.n)

    np.random.seed(cfg.seed.out_of_sample)
    sample_b = []
    sample_xopt = []
    max_R = 0

    df = []

    for i in trange(cfg.N):
        b_samp = generate_single_b(cfg, A)
        xopt_samp, R_samp = solve_single_cvxpy(cfg, A, b_samp)
        max_R = max(max_R, R_samp)

        sample_b.append(b_samp)
        sample_xopt.append(xopt_samp)

        F_vals = simulate_alg(cfg, x0, A, b_samp, xopt_samp, L, ATA_lu, ATA_piv, alg=cfg.alg)
        # log.info(F_vals)

        for k in range(1, cfg.K_max + 1):
            df.append(pd.Series({
                'i': i,
                'K': k,
                'obj_val': F_vals[k-1],
            }))

        if i % 1000 == 0:
            log.info(f'saving at i={i}')
            df_to_save = pd.DataFrame(df)
            df_to_save.to_csv(cfg.sample_fname, index=False)
    
    log.info(f'maximum sample radius: {max_R}')

    df_to_save = pd.DataFrame(df)
    df_to_save.to_csv(cfg.sample_fname, index=False)


def lasso_pep(cfg):
    log.info(cfg)

    obj = 'obj_val'

    A = generate_A(cfg)
    log.info(A)
    ATA = A.T @ A

    ATA_eigvals = np.real(np.linalg.eigvals(ATA))
    L = np.max(ATA_eigvals)
    if cfg.m >= cfg.n:
        mu = np.min(ATA_eigvals)
    else:
        mu = 0
    log.info(f'L: {L}, mu: {mu}')

    res = []
    for k in range(cfg.K_min, cfg.K_max + 1):
        tau, solvetime = pep_subproblem(cfg, k, mu, L, cfg.R, alg=cfg.alg)
        res.append(pd.Series({
            'K': k,
            'obj': obj,
            'val': tau,
            'solvetime': solvetime,
        }))
        df = pd.DataFrame(res)
        df.to_csv(cfg.pep_fname, index=False)


def pep_subproblem(cfg, K, mu, L, R, return_problem=False, alg='ista'):
    problem = PEP()
    lambd = cfg.lambd

    f1 = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=mu, L=L, reuse_gradient=True)
    f2 = problem.declare_function(ConvexLipschitzFunction, M=lambd * np.sqrt(cfg.n), reuse_gradient=True)
    # f2 = problem.declare_function(ConvexFunction, reuse_gradient=True)
    func = f1 + f2

    xs = func.stationary_point()
    x0 = problem.set_initial_point()

    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)

    if alg == 'ista':
        gamma = cfg.eta / L

        x = [x0 for _ in range(K+1)]
        g = [None for _ in range(K+1)]
        f = [None for _ in range(K+1)]

        for k in range(K):
            y = x[k] - gamma * f1.gradient(x[k])
            x[k+1], g[k+1], f[k+1] = proximal_step(y, f2, gamma)
    elif alg == 'fista':
        gamma = cfg.eta / L

        x = [x0 for _ in range(K+1)]
        y = [x0 for _ in range(K+1)]
        g = [None for _ in range(K+1)]
        f = [None for _ in range(K+1)]
        beta_k = 1

        for k in range(K):

            xtilde = y[k] - gamma * f1.gradient(y[k])
            x[k+1], g[k+1], f[k+1] = proximal_step(xtilde, f2, gamma)

            beta_kplus1 = .5 * (1 + np.sqrt(1 + 4 * beta_k ** 2))
            y[k+1] = x[k+1] + (beta_k - 1) / beta_kplus1 * (x[k+1] - x[k])

            beta_k = beta_kplus1
    elif alg == 'optista':
        x = [x0 for _ in range(K+1)]
        y = [x0 for _ in range(K+1)]
        z = [x0 for _ in range(K+1)]
        g = [None for _ in range(K+1)]
        f = [None for _ in range(K+1)]

        thetas = [1]
        for _ in range(K-1):
            thetas.append(.5 * (1 + np.sqrt(1 + 4 * thetas[-1] ** 2)))
        thetas.append(.5 * (1 + np.sqrt(1 + 8 * thetas[-1] ** 2)))

        gammas = []
        for i in range(K):
            gamma_i = (2 * thetas[i] / thetas[K] ** 2) * ((thetas[K] ** 2) - (2 * thetas[i] ** 2) + thetas[i])
            gammas.append(gamma_i)

        for k in range(K):
            xtilde = x[k] - gammas[k] / L * f1.gradient(y[k])
            x[k+1], g[k+1], f[k+1] = proximal_step(xtilde, f2, gammas[k] / L)
            z[k+1] = y[k] + 1 / gammas[k] * (x[k+1] - x[k])
            y[k+1] = z[k+1] + (thetas[k] - 1) / thetas[k+1] * (z[k+1] - z[k])  + thetas[k] / thetas[k+1] * (z[k+1] - y[k])
    else:
        raise NotImplementedError
        exit(0)

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

    # clarabel_params = {
    #     'tol_feas': 1e-7,
    # }

    # pepit_tau = problem.solve(
    #     wrapper='cvxpy',
    #     solver='CLARABEL',
    #     # clarabel_params=clarabel_params,
    #     tol_feas=1e-5,
    # )

    # pepit_tau = problem.solve()
    log.info(f'pepit_tau at K={K}: {pepit_tau}')

    solvetime = problem.wrapper.prob.solver_stats.solve_time
    log.info(f'solvetime: {solvetime}')

    return pepit_tau, solvetime


def single_trajectory(cfg, K, A, b, x_opt, x0, lu, piv, L, alg='ista'):
    x_ls = sp.linalg.lu_solve((lu, piv), A.T @ b)
    lambd = cfg.lambd

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

    if alg == 'ista':
        gamma = cfg.eta / L

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
    elif alg == 'fista':
        gamma = cfg.eta / L

        xk = x0
        yk = x0
        beta_k = 1
        for _ in range(K-1):
            xtilde_kplus1 = yk - gamma * grad_f1(yk)
            xkplus1 = soft_threshold(xtilde_kplus1, gamma * lambd)
            gkplus1 = (xtilde_kplus1 - xkplus1) / gamma

            beta_kplus1 = .5 * (1 + np.sqrt(1 + 4 * beta_k ** 2))
            # y[k+1] = x[k+1] + (beta_k - 1) / beta_kplus1 * (x[k+1] - x[k])
            ykplus1 = xkplus1 + (beta_k - 1) / beta_kplus1 * (xkplus1 - xk)
            grad_f1_ykplus1 = grad_f1(ykplus1)

            G += [gkplus1, grad_f1_ykplus1]
            F += [f2(xkplus1), f1(ykplus1)]

            xk = xkplus1
            yk = ykplus1
            beta_k = beta_kplus1
        
        xtilde = yk - gamma * grad_f1(yk)
        xk = soft_threshold(xtilde, gamma * lambd)
        gk = (xtilde - xk) / gamma
        # last iteration the y update is irrelevant
        subgrad_f_xk = grad_f1(xk) + gk
        f2_k = f2(xk)
        f_k = f(xk)

        G += [gk, subgrad_f_xk, x_ls]
        F += [f2_k, f_k]
    elif alg == 'optista':
        thetas = [1]
        for _ in range(K-1):
            thetas.append(.5 * (1 + np.sqrt(1 + 4 * thetas[-1] ** 2)))
        thetas.append(.5 * (1 + np.sqrt(1 + 8 * thetas[-1] ** 2)))

        gammas = []
        for i in range(K):
            gamma_i = (2 * thetas[i] / thetas[K] ** 2) * ((thetas[K] ** 2) - (2 * thetas[i] ** 2) + thetas[i])
            gammas.append(gamma_i)
        
        xk = x0
        yk = x0
        zk = x0

        for k in range(K-1):
            # xkplus1 = soft_threshold(xk - gammas[k] / L * grad_f1(yk), gammas[k] * lambd / L)
            # zkplus1 = yk + 1 / gammas[k] * (xkplus1 - xk)
            # ykplus1 = zkplus1 + (thetas[k] - 1) / thetas[k+1] * (zkplus1 - zk) + thetas[k] / thetas[k+1] * (zkplus1 - yk)
            xtilde_kplus1 = xk - gammas[k] / L * grad_f1(yk)
            xkplus1 = soft_threshold(xtilde_kplus1, gammas[k] * lambd / L)
            gkplus1 = (xtilde_kplus1 - xkplus1) / (gammas[k] / L)

            zkplus1 = yk + 1 / gammas[k] * (xkplus1 - xk)
            ykplus1 = zkplus1 + (thetas[k] - 1) / thetas[k+1] * (zkplus1 - zk) + thetas[k] / thetas[k+1] * (zkplus1 - yk)

            grad_f1_ykplus1 = grad_f1(ykplus1)
            G += [gkplus1, grad_f1_ykplus1]
            F += [f2(xkplus1), f1(ykplus1)]

            xk = xkplus1
            zk = zkplus1
            yk = ykplus1

        xtilde = xk - gammas[K-1] / L * grad_f1(yk)
        xk = soft_threshold(xtilde, gammas[K-1] * lambd / L)
        gk = (xtilde - xk) / (gammas[K-1] / L)
        # last iteration the y and z updates are irrelevant
        subgrad_f_xk = grad_f1(xk) + gk
        f2_k = f2(xk)
        f_k = f(xk)

        G += [gk, subgrad_f_xk, x_ls]
        F += [f2_k, f_k]
    else:
        raise NotImplementedError
        exit(0)

    # F += [f(xk) - f(x_opt), f1(x_ls)]

    G = np.array(G)
    F = np.array(F)

    # if K == 2:
        # print(F)
        # exit(0)

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
    x_test_opt, _ = solve_single_cvxpy(cfg, A, b_test)

    R = cfg.R
    log.info(f'radius: {R}')

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
    x0 = np.zeros(cfg.n)

    np.random.seed(cfg.seed.in_sample)

    sample_b = []
    sample_xopt = []
    for _ in trange(N):
        b_samp = generate_single_b(cfg, A)
        xopt_samp, _ = solve_single_cvxpy(cfg, A, b_samp)

        sample_b.append(b_samp)
        sample_xopt.append(xopt_samp)

    res = []
    sample_df_list = []
    for k in range(cfg.K_min, cfg.K_max + 1):
        samples = []
        problem = pep_subproblem(cfg, k, mu, L, R, return_problem=True, alg=cfg.alg)
        # problem.solve(wrapper='cvxpy', solver='MOSEK')
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
            b_samp = sample_b[i]
            xopt_samp = sample_xopt[i]

            G, F = single_trajectory(cfg, k, A, b_samp, xopt_samp, x0, ATA_lu, ATA_piv, L, alg=cfg.alg)
            samples.append((G, F))
            # log.info((G, F))
            sample_df_list.append(pd.Series({
                'i': i,
                'K': k,
                'obj_val': F[-1] - F[0],
            }))
        sample_df = pd.DataFrame(sample_df_list)
        sample_df.to_csv('samples.csv', index=False)

        DR = DROReformulator(
            problem,
            samples,
            dro_obj,
            'clarabel',
            precond=True,
            precond_type=cfg.precond_type,
            mro_clusters=num_clusters,
            obj_vec_cutoff=2,
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

    # K = 5
    # x0 = np.zeros(cfg.n)

    # problem = pep_subproblem(cfg, K, mu, L, R, return_problem=True, alg=cfg.alg)
    # tau = problem.solve(wrapper='cvxpy', solver='MOSEK')
    # log.info(f'----pep problem solved at k={K} with tau={tau}----')

    # G, F = single_trajectory(cfg, K, A, b_test, x_test_opt, x0, ATA_lu, ATA_piv, L, alg=cfg.alg)
    # # print(G, F)
    # # for constr in problem._list_of_constraints_sent_to_wrapper[1:]:
    # #     A_cons, b_cons, c_cons = expression_to_matrices(constr.expression)
    # #     print('---')
    # #     print(A_cons, b_cons, c_cons)
    # #     print(constr.equality_or_inequality)
    # #     print(np.trace(A_cons @ G) + b_cons @ F + c_cons)

    # samples = [(G, F)]
    # DR = DROReformulator(
    #     problem,
    #     samples,
    #     cfg.dro_obj,
    #     'clarabel',
    #     precond=True,
    #     precond_type=cfg.precond_type,
    #     mro_clusters=None,
    #     obj_vec_cutoff=2,
    # )

    # eps = 1e-3
    # alpha = 0.1

    # DR.set_params(eps=eps, alpha=alpha)
    # out = DR.solve()
    # log.info(out['obj'])
