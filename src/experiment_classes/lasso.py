import diffcp_patch  # noqa: F401  # COO->CSC fix for diffcp/clarabel
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import logging
import time
import jax
import jax.numpy as jnp
from tqdm import trange

# from .utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, generate_trajectories
from .utils import sample_x0_centered_disk, build_eps_vals
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction, ConvexLipschitzFunction, ConvexFunction
from PEPit.primitive_steps import proximal_step
from PEPit.tools.expressions_to_matrices import expression_to_matrices
from reformulator.dro_reformulator import DROReformulator
from learning.pep_constructions import (
    construct_ista_pep_data, construct_fista_pep_data, ista_pep_data_to_numpy,
)
from learning.trajectories import (
    problem_data_to_ista_trajectories, problem_data_to_fista_trajectories,
)

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)


def orthonormal_cols(rows, cols, rng):
    """rows x cols matrix with orthonormal columns (rows >= cols)."""
    Q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return Q[:, :cols]


def generate_A(cfg):
    """Spiked-MP design A = U diag(s) V^T with a DEGENERATE two-level spectrum of
    A^T A: eigenvalues {bulk_q*L (mult r-n_spikes), L (mult n_spikes)} on the
    r=min(m,n) dim range (normalized so the largest eigenvalue is exactly L).
    The coherent small bulk at bulk_q*L is what makes FISTA ripple."""
    rng = np.random.default_rng(cfg.seed.A)
    m, n = cfg.m, cfg.n
    r = min(m, n)

    eigs = np.full(r, cfg.bulk_q)
    eigs[-cfg.n_spikes:] = 1.0
    eigs = np.sort(eigs)
    eigs *= cfg.spike_L / eigs[-1]      # largest eigenvalue exactly spike_L

    s = np.sqrt(eigs)
    U = orthonormal_cols(m, r, rng)
    V = orthonormal_cols(n, r, rng)
    A = (U * s) @ V.T
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
    res = prob.solve(solver=cp.CLARABEL)
    # log.info(f'single x sol with lambda = {cfg.lambd}: {x.value}')
    # log.info(f'opt value = {res}')

    R = np.linalg.norm(x.value)
    return x.value, R


def lasso_obj(A, b, lambd, x):
    """f(x) = 0.5||A x - b||^2 + lambd ||x||_1."""
    return 0.5 * np.linalg.norm(A @ x - b) ** 2 + lambd * np.linalg.norm(x, 1)


def least_squares_sol(A, b):
    """Min-norm least-squares solution; robust for wide (singular A^T A)."""
    return np.linalg.lstsq(A, b, rcond=None)[0]


def compute_lambda_R(cfg, A):
    """Set the family's absolute lambda and initial radius R from reference data.

    lambda = lambda_frac * median_b ||A^T b||_inf  (standard lasso scale);
    R = max_b ||x*(b)||. Deterministic (seed.in_sample) so samples/pep/dro agree.
    Mutates cfg.lambd and cfg.R, and returns (lambd, R).
    """
    rng = np.random.default_rng(cfg.seed.in_sample)
    ref_b = []
    for _ in range(cfg.N):
        x_samp = rng.normal(size=(cfg.n,)) * rng.binomial(1, cfg.p_xsamp_nonzero, size=(cfg.n,))
        ref_b.append(A @ x_samp + cfg.noise_eps * rng.normal(size=(cfg.m,)))

    lambd = float(cfg.lambda_frac * np.median([np.max(np.abs(A.T @ b)) for b in ref_b]))
    cfg.lambd = lambd

    R = float(max(np.linalg.norm(solve_single_cvxpy(cfg, A, b)[0]) for b in ref_b))
    cfg.R = R
    log.info(f'computed lambda={lambd}, R={R}')
    return lambd, R


def simulate_alg(cfg, x0, A, b, x_opt, L, lu, piv, alg='ista'):
    K_max = cfg.K_max
    x_ls = least_squares_sol(A, b)  # min-norm LS (A^T A is singular for wide m<n)
    lambd = cfg.lambd
    def f1(x):
        # TODO: does this need to be ** 2 ?
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
    compute_lambda_R(cfg, A)   # sets cfg.lambd, cfg.R from the family
    ATA = A.T @ A

    ATA_lu, ATA_piv = None, None  # x_ls now via least_squares_sol (A^T A singular for wide)

    ATA_eigvals = np.real(np.linalg.eigvals(ATA))
    L = np.max(ATA_eigvals)

    if cfg.m >= cfg.n:
        mu = np.min(ATA_eigvals)
    else:
        mu = 0
    log.info(f'L: {L}, mu: {mu}')

    x0 = np.zeros(cfg.n)
    alpha_vals = list(cfg.alpha_vals)
    n_repeats = cfg.cross_val_repeats
    max_R = 0

    # Repeat the whole sampling experiment n_repeats times (independent reseeds; A, lambd, R fixed)
    # to build a distribution of per-K empirical mean / CVaR / worst summaries for eps cross-val.
    dist = []
    for j in trange(n_repeats):
        np.random.seed(cfg.seed.out_of_sample + j)
        df = []
        for i in range(cfg.N):
            b_samp = generate_single_b(cfg, A)
            xopt_samp, R_samp = solve_single_cvxpy(cfg, A, b_samp)
            max_R = max(max_R, R_samp)

            F_vals = simulate_alg(cfg, x0, A, b_samp, xopt_samp, L, ATA_lu, ATA_piv, alg=cfg.alg)

            for k in range(1, cfg.K_max + 1):
                df.append(pd.Series({
                    'i': i,
                    'K': k,
                    'obj_val': F_vals[k-1],
                }))

        df_to_save = pd.DataFrame(df)
        summary = summarize_per_k(df_to_save, alpha_vals, 'obj_val', cfg.K_max)

        if j == 0:
            # Repeat-0 artifacts reproduce the legacy single-run outputs.
            df_to_save.to_csv(cfg.sample_fname, index=False)
            plot_worst_case(df_to_save, 'obj_val', cfg)
            summary.to_csv('sample_summary.csv', index=False)
            log.info(summary)
            plot_sample_summary(summary, cfg)

        dist.append(summary.assign(repeat=j))

    log.info(f'maximum sample radius: {max_R}')

    dist_df = pd.concat(dist, ignore_index=True)
    dist_df = dist_df[['repeat'] + [c for c in dist_df.columns if c != 'repeat']]
    dist_df.to_csv('sample_summary_dist.csv', index=False)


def plot_worst_case(df, col, cfg):
    worst_cases = df[['K', col]].groupby(['K']).max()
    plt.figure()
    plt.plot(range(1, cfg.K_max + 1), worst_cases)
    plt.yscale('log')
    plt.title(col)
    plt.savefig('worstcases.pdf')
    plt.close()


def compute_empirical_cvar(values, alpha):
    """Empirical CVaR at level alpha: mean of the worst (largest) alpha-fraction."""
    n_tail = max(1, int(np.ceil(alpha * len(values))))
    return float(np.mean(np.sort(values)[-n_tail:]))


def summarize_per_k(df_to_save, alpha_vals, col, K_max):
    """Per-K mean / worst / cvar_<a> summary rows for one experiment."""
    rows = []
    for k in range(1, K_max + 1):
        vals = df_to_save.loc[df_to_save['K'] == k, col].to_numpy()
        row = {'K': k, 'mean': float(np.mean(vals)), 'worst': float(np.max(vals))}
        for a in alpha_vals:
            row[f'cvar_{a}'] = compute_empirical_cvar(vals, a)
        rows.append(row)
    return pd.DataFrame(rows)


def default_alpha(alpha_vals):
    """Default CVaR level for single-alpha plots: 0.05 if present, else the middle value."""
    return 0.05 if 0.05 in alpha_vals else alpha_vals[len(alpha_vals) // 2]


def plot_sample_summary(summary, cfg):
    """Plot empirical mean, CVaR(alpha), and worst-case of f(x_K)-f* vs K.

    Produces two figures: one overlaying every alpha in cfg.alpha_vals, and one
    using only the default alpha.
    """
    Ks = summary['K']
    alpha_vals = list(cfg.alpha_vals)

    # All-alpha version.
    plt.figure()
    plt.plot(Ks, summary['mean'], label='mean')
    for a in alpha_vals:
        plt.plot(Ks, summary[f'cvar_{a}'], label=f"CVaR (alpha={a})")
    plt.plot(Ks, summary['worst'], label='worst-case', linestyle='--')
    plt.yscale('log')
    plt.xlabel('iteration K')
    plt.ylabel('f(x_K) - f*')
    plt.title(f"Lasso {cfg.alg}: empirical metrics")
    plt.legend()
    plt.savefig('sample_summary_all_alpha.pdf')
    plt.close()

    # Default-alpha version.
    a = default_alpha(alpha_vals)
    plt.figure()
    plt.plot(Ks, summary['mean'], label='mean')
    plt.plot(Ks, summary[f'cvar_{a}'], label=f"CVaR (alpha={a})")
    plt.plot(Ks, summary['worst'], label='worst-case', linestyle='--')
    plt.yscale('log')
    plt.xlabel('iteration K')
    plt.ylabel('f(x_K) - f*')
    plt.title(f"Lasso {cfg.alg}: empirical metrics")
    plt.legend()
    plt.savefig('sample_summary.pdf')
    plt.close()


def lasso_pep(cfg):
    log.info(cfg)

    obj = 'obj_val'

    A = generate_A(cfg)
    log.info(A)
    compute_lambda_R(cfg, A)   # sets cfg.lambd, cfg.R (consistent with samples/dro)
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

    f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L, reuse_gradient=True)
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

    pepit_tau = problem.solve(
        wrapper='cvxpy',
        solver='CLARABEL',
        tol_feas=1e-5,
    )

    # pepit_tau = problem.solve()
    log.info(f'pepit_tau at K={K}: {pepit_tau}')

    solvetime = problem.wrapper.prob.solver_stats.solve_time
    log.info(f'solvetime: {solvetime}')

    return pepit_tau, solvetime


def single_trajectory(cfg, K, A, b, x_opt, x0, lu, piv, L, alg='ista'):
    x_ls = least_squares_sol(A, b)  # min-norm LS (A^T A is singular for wide m<n)
    lambd = cfg.lambd

    def f1(x):
        # TODO: does this need to be ** 2 ?
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
    compute_lambda_R(cfg, A)   # sets cfg.lambd, cfg.R (consistent with samples/pep)
    ATA = A.T @ A

    ATA_eigvals = np.real(np.linalg.eigvals(ATA))
    L = float(np.max(ATA_eigvals))

    if cfg.m >= cfg.n:
        mu = float(np.min(ATA_eigvals))
    else:
        mu = 0.0
    log.info(f'L: {L}, mu: {mu}')

    R = cfg.R
    log.info(f'radius: {R}')

    if cfg.alg not in ('ista', 'fista'):
        raise NotImplementedError(
            f"DRO custom PEP construction supports 'ista'/'fista' only, got '{cfg.alg}'")

    if cfg.dro_obj == 'expectation':
        N = cfg.training.expectation_N
        num_clusters = cfg.num_clusters.expectation
        measure = 'expectation'
    elif cfg.dro_obj == 'cvar':
        N = cfg.training.cvar_N
        num_clusters = cfg.num_clusters.cvar
        measure = 'cvar'
    else:
        log.info('invalid dro obj')
        exit(0)

    eps_vals = build_eps_vals(cfg)
    alpha_vals = list(cfg.alpha_vals)
    # alpha only affects the cvar objective; expectation ignores it (single pass).
    alphas_to_run = alpha_vals if measure == 'cvar' else [alpha_vals[0]]
    x0 = np.zeros(cfg.n)
    gamma = cfg.eta / L

    np.random.seed(cfg.seed.in_sample)

    sample_b = []
    sample_xopt = []
    sample_fopt = []
    for _ in trange(N):
        b_samp = generate_single_b(cfg, A)
        xopt_samp, _ = solve_single_cvxpy(cfg, A, b_samp)

        sample_b.append(b_samp)
        sample_xopt.append(xopt_samp)
        sample_fopt.append(lasso_obj(A, b_samp, cfg.lambd, xopt_samp))

    res = []
    sample_df_list = []
    for k in range(cfg.K_min, cfg.K_max + 1):
        # stepsizes + custom PEP data + matching trajectory fn for horizon k
        if cfg.alg == 'ista':
            stp = jnp.full(k, gamma)
            pep_data = ista_pep_data_to_numpy(construct_ista_pep_data(
                gamma, mu, L, R, k, pep_obj=cfg.pep_obj, composition_type='final'))

            def traj_fn(b, xo, fo, stp=stp, k=k):
                return problem_data_to_ista_trajectories(
                    stp, A, b, x0, xo, fo, cfg.lambd, k, return_Gram_representation=True)
        else:  # fista
            betas_t = [1.0]
            for _ in range(k):
                betas_t.append(0.5 * (1 + np.sqrt(1 + 4 * betas_t[-1] ** 2)))
            beta = jnp.array(betas_t)
            stp = (jnp.full(k, gamma), beta)
            pep_data = ista_pep_data_to_numpy(construct_fista_pep_data(
                gamma, beta, mu, L, R, k, pep_obj=cfg.pep_obj, composition_type='final'))

            def traj_fn(b, xo, fo, stp=stp, k=k):
                return problem_data_to_fista_trajectories(
                    stp, A, b, x0, xo, fo, cfg.lambd, k, return_Gram_representation=True)

        A_obj_np, b_obj_np = pep_data[0], pep_data[1]

        samples = []
        for i in range(N):
            G, F = traj_fn(sample_b[i], sample_xopt[i], sample_fopt[i])
            G, F = np.asarray(G), np.asarray(F)
            samples.append((G, F))
            emp = float(np.trace(A_obj_np @ G) + b_obj_np @ F)  # PEP objective on the sample
            sample_df_list.append(pd.Series({'i': i, 'K': k, 'obj_val': emp}))
        pd.DataFrame(sample_df_list).to_csv('samples.csv', index=False)

        DR = DROReformulator(
            pep_data,
            samples,
            measure,
            'clarabel',
            precond=cfg.precond,
            precond_type=cfg.precond_type,
            mro_clusters=num_clusters,
        )
        log.info(f'----dro reformulator built at k={k}----')
        for eps_idx, eps in enumerate(eps_vals):
            for alpha_idx, alpha in enumerate(alphas_to_run):
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
                    'alpha_idx': alpha_idx,
                    'alpha': alpha,
                    'mro_sol': out['obj'],
                    'solvetime': out['solvetime'],
                    'dro_feas_sol': dro_feas,
                }))

                df = pd.DataFrame(res)
                df.to_csv(cfg.dro_fname, index=False)

