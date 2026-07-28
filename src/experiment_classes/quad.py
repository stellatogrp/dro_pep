import diffcp_patch  # noqa: F401  # COO->CSC fix for diffcp/clarabel
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import time
import cvxpy as cp
from tqdm import trange

from .utils import marchenko_pastur, gradient_descent, nesterov_accelerated_gradient, nesterov_fgm, generate_trajectories, sample_x0_centered_disk, generate_P_fixed_mu_L, build_eps_vals
from reformulator.dro_reformulator import DROReformulator
from learning.pep_constructions import (
    construct_gd_pep_data, construct_fgm_pep_data, pep_data_to_numpy,
)
from learning.trajectories import (
    problem_data_to_gd_trajectories, problem_data_to_nesterov_fgm_trajectories,
)
from .lyap_classes.gd import gd_lyap, gd_lyap_nobisect

jax.config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)


def rejection_sample_MP(dim, mu, L):
    Q = marchenko_pastur(dim, mu, L)
    eigvals = np.real(np.linalg.eigvals(Q))
    if mu > np.min(eigvals) or L < np.max(eigvals):
        # print('reject sample')
        return rejection_sample_MP(dim, mu, L)
    return Q


class Quad(object):
    """Offset MP quadratic: f(x) = 0.5 (x - x*)^T Q (x - x*), start x0 = 0.

    Q ~ Marchenko-Pastur (rejection-sampled into [mu, L]); x* ~ N(0, I_dim).
    The optimum is x* with f* = 0, and the initial radius R = ||x0 - x*|| = ||x*||
    is determined by the draw (not controlled).
    """

    def __init__(self, dim, mu=0, L=1, x_star=None):
        self.mu = mu
        self.L = L

        self.Q = rejection_sample_MP(dim, mu, L)
        self.dim = self.Q.shape[0]

        if x_star is None:
            x_star = np.random.normal(0, 1, self.dim)
        self.x_star = x_star

        self.x0 = np.zeros(self.dim)
        self.f_star = 0.0

    @property
    def R(self):
        return float(np.linalg.norm(self.x_star))

    def f(self, x):
        d = x - self.x_star
        return .5 * d.T @ self.Q @ d

    def g(self, x):
        return self.Q @ (x - self.x_star)


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


# =============================================================================
# Shared helpers for the offset-quadratic custom PEP/DRO pipeline
# =============================================================================

def xstar_rng(cfg):
    """RNG stream for x* draws, independent of the (global) stream used for Q.

    Shared by compute_R and quad_dro so the DRO samples' x* are a prefix of the
    reference set ⇒ R = max||x*|| over ref_N draws bounds every DRO sample, and
    the PEP initial condition ||x0 - x*||^2 <= R^2 stays feasible."""
    return np.random.default_rng(cfg.seed.R_ref)


def compute_R(cfg):
    """Set the family's initial radius R = (1+margin) * max_{ref_N} ||x*||.

    x* ~ N(0, I_dim); R = ||x0 - x*|| = ||x*|| since x0 = 0. Deterministic
    (seed.R_ref) so samples/pep/dro share one radius envelope. Mutates cfg.R.
    The small margin keeps samples strictly inside the radius (boundary samples
    make the small-eps DRO SDP numerically marginal)."""
    rng = xstar_rng(cfg)
    R = float(max(np.linalg.norm(rng.normal(size=(cfg.dim,))) for _ in range(cfg.ref_N)))
    cfg.R = R * 1.001
    log.info(f'computed radius R={cfg.R}')
    return cfg.R


def nesterov_betas(K_max):
    """Standard FGM momentum: t_0=1, t_{k+1}=0.5(1+sqrt(1+4 t_k^2)),
    beta_k=(t_k-1)/t_{k+1}. Returns array of length K_max (beta[0]=0)."""
    t = 1.0
    betas = []
    for _ in range(K_max):
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        betas.append((t - 1) / t_next)
        t = t_next
    return np.array(betas)


def simulate_quad(h, alg, gamma, beta, K_max):
    """Numpy forward sim in shifted coords (z = x - x*, z0 = -x*) mirroring the
    custom construction dynamics exactly. Records final-iterate metrics for
    k=1..K_max. Returns (obj_val, grad_sq_norm, opt_dist_sq_norm) lists."""
    Q = h.Q
    obj, gsq, dsq = [], [], []
    if alg == 'grad_desc':
        z = -h.x_star
        for _ in range(K_max):
            z = z - gamma * (Q @ z)
            gz = Q @ z
            obj.append(0.5 * float(z @ gz))
            gsq.append(float(gz @ gz))
            dsq.append(float(z @ z))
    elif alg == 'nesterov_grad_desc':
        x_prev = -h.x_star
        y = -h.x_star
        for k in range(K_max):
            x_new = y - gamma * (Q @ y)
            y = x_new + beta[k] * (x_new - x_prev)
            x_prev = x_new
            gx = Q @ x_new
            obj.append(0.5 * float(x_new @ gx))
            gsq.append(float(gx @ gx))
            dsq.append(float(x_new @ x_new))
    else:
        raise NotImplementedError(f"simulate_quad: unsupported alg '{alg}'")
    return obj, gsq, dsq


def solve_pep_sdp(pep_data):
    """Solve the worst-case PEP SDP from a custom pep_data 9-tuple with CLARABEL.

    maximize <A_obj, G> + b_obj.F  s.t.  G >= 0 and
    <A_vals[i], G> + b_vals[i].F + c_vals[i] <= 0 for all i."""
    (A_obj, b_obj, A_vals, b_vals, c_vals, *_) = pep_data
    dimG = A_obj.shape[0]
    dimF = b_obj.shape[0]
    G = cp.Variable((dimG, dimG), symmetric=True)
    F = cp.Variable(dimF)
    cons = [G >> 0]
    for i in range(A_vals.shape[0]):
        cons.append(cp.trace(A_vals[i] @ G) + b_vals[i] @ F + c_vals[i] <= 0)
    prob = cp.Problem(cp.Maximize(cp.trace(A_obj @ G) + b_obj @ F), cons)
    prob.solve(solver=cp.CLARABEL)
    solvetime = prob.solver_stats.solve_time
    return prob.value, solvetime


def _build_pep_and_traj(cfg, k, gamma, mu, L, R, beta_full):
    """Return (pep_data_np, traj_fn) for the configured alg at horizon k.

    traj_fn(h) runs the matching custom trajectory on shifted coords
    (z0 = -h.x_star, zs = 0, fs = 0) and returns (G, F)."""
    zs = jnp.zeros(cfg.dim)
    fs = 0.0
    if cfg.alg == 'grad_desc':
        pep_data = pep_data_to_numpy(construct_gd_pep_data(
            gamma, mu, L, R, k, pep_obj=cfg.dro_pep_obj, composition_type='final'))
        stp = jnp.full(k, gamma)

        def traj_fn(h, stp=stp, k=k):
            return problem_data_to_gd_trajectories(
                stp, jnp.asarray(h.Q), jnp.asarray(-h.x_star), zs, fs, k,
                return_Gram_representation=True)
    elif cfg.alg == 'nesterov_grad_desc':
        beta = jnp.asarray(beta_full[:k])
        pep_data = pep_data_to_numpy(construct_fgm_pep_data(
            gamma, beta, mu, L, R, k, pep_obj=cfg.dro_pep_obj, composition_type='final'))
        stp = (jnp.full(k, gamma), beta)

        def traj_fn(h, stp=stp, k=k):
            return problem_data_to_nesterov_fgm_trajectories(
                stp, jnp.asarray(h.Q), jnp.asarray(-h.x_star), zs, fs, k,
                return_Gram_representation=True)
    else:
        raise NotImplementedError(
            f"custom PEP/DRO supports 'grad_desc'/'nesterov_grad_desc' only, got '{cfg.alg}'")
    return pep_data, traj_fn


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
    """Plot empirical mean, CVaR(alpha), and worst-case of the metric vs K.

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
    plt.ylabel(cfg.dro_pep_obj)
    plt.title(f"Quad {cfg.alg}: empirical metrics")
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
    plt.ylabel(cfg.dro_pep_obj)
    plt.title(f"Quad {cfg.alg}: empirical metrics")
    plt.legend()
    plt.savefig('sample_summary.pdf')
    plt.close()


def quad_samples(cfg):
    log.info(cfg)
    # R is not needed for the empirical forward sim (simulate_quad ignores it).

    gamma = cfg.eta / cfg.L
    beta = nesterov_betas(cfg.K_max)
    alpha_vals = list(cfg.alpha_vals)
    n_repeats = cfg.cross_val_repeats

    # Repeat the whole sampling experiment n_repeats times (independent reseeds) to build a
    # distribution of per-K empirical mean / CVaR / worst summaries for eps cross-validation.
    dist = []
    for j in trange(n_repeats):
        np.random.seed(cfg.seed.full_samples + j)
        df = []
        for i in range(cfg.sample_N):
            h = Quad(cfg.dim, mu=cfg.mu, L=cfg.L)
            obj, gsq, dsq = simulate_quad(h, cfg.alg, gamma, beta, cfg.K_max)
            for k in range(1, cfg.K_max + 1):
                df.append(pd.Series({
                    'i': i,
                    'K': k,
                    'obj_val': obj[k-1],
                    'grad_sq_norm': gsq[k-1],
                    'opt_dist_sq_norm': dsq[k-1],
                }))

        df_to_save = pd.DataFrame(df)
        summary = summarize_per_k(df_to_save, alpha_vals, cfg.dro_pep_obj, cfg.K_max)

        if j == 0:
            # Repeat-0 artifacts reproduce the legacy single-run outputs.
            df_to_save.to_csv(cfg.sample_fname, index=False)
            plot_worst_case(df_to_save, cfg.dro_pep_obj, cfg)
            summary.to_csv('sample_summary.csv', index=False)
            log.info(summary)
            plot_sample_summary(summary, cfg)

        dist.append(summary.assign(repeat=j))

    dist_df = pd.concat(dist, ignore_index=True)
    dist_df = dist_df[['repeat'] + [c for c in dist_df.columns if c != 'repeat']]
    dist_df.to_csv('sample_summary_dist.csv', index=False)


def quad_pep(cfg):
    log.info(cfg)
    if cfg.alg not in ('grad_desc', 'nesterov_grad_desc'):
        raise NotImplementedError(
            f"custom PEP supports 'grad_desc'/'nesterov_grad_desc' only, got '{cfg.alg}'")

    compute_R(cfg)
    mu, L, R = cfg.mu, cfg.L, cfg.R
    gamma = cfg.eta / L
    beta_full = nesterov_betas(cfg.K_max)

    res = []
    for k in range(cfg.K_min, cfg.K_max + 1):
        pep_data, _ = _build_pep_and_traj(cfg, k, gamma, mu, L, R, beta_full)
        tau, solvetime = solve_pep_sdp(pep_data)
        log.info(f'----pep SDP solved at k={k}: tau={tau}----')

        res.append(pd.Series({
            'K': k,
            'obj': cfg.dro_pep_obj,
            'val': tau,
            'solvetime': solvetime,
        }))
        df = pd.DataFrame(res)
        df.to_csv(cfg.pep_fname, index=False)


def quad_dro(cfg):
    log.info(cfg)

    if cfg.alg not in ('grad_desc', 'nesterov_grad_desc'):
        raise NotImplementedError(
            f"custom DRO supports 'grad_desc'/'nesterov_grad_desc' only, got '{cfg.alg}'")

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

    compute_R(cfg)
    mu, L, R = cfg.mu, cfg.L, cfg.R
    gamma = cfg.eta / L
    beta_full = nesterov_betas(cfg.K_max)

    eps_vals = build_eps_vals(cfg)
    alpha_vals = list(cfg.alpha_vals)
    # alpha only affects the cvar objective; expectation ignores it (single pass).
    alphas_to_run = alpha_vals if measure == 'cvar' else [alpha_vals[0]]

    # x* from the same stream as compute_R (prefix ⇒ all samples satisfy ||x*|| <= R);
    # Q drawn fresh per instance from the global stream (seed.train).
    np.random.seed(cfg.seed.train)
    rng_x = xstar_rng(cfg)
    quad_funcs = [Quad(cfg.dim, mu=cfg.mu, L=cfg.L, x_star=rng_x.normal(size=cfg.dim))
                  for _ in range(N)]
    max_sample_R = max(h.R for h in quad_funcs) * 1.001
    if max_sample_R > cfg.R:
        log.warning(f'sample radius {max_sample_R} exceeds reference R {cfg.R}; '
                    f'raising R (increase ref_N >= N to keep pep/dro consistent)')
        cfg.R = max_sample_R
        R = cfg.R

    res = []
    sample_df_list = []

    for k in range(cfg.K_min, cfg.K_max + 1):
        pep_data, traj_fn = _build_pep_and_traj(cfg, k, gamma, mu, L, R, beta_full)
        A_obj_np, b_obj_np = pep_data[0], pep_data[1]

        samples = []
        for i in range(N):
            G, F = traj_fn(quad_funcs[i])
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
    eps_vals = build_eps_vals(cfg)
    alpha = cfg.alpha_vals[0]

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

def quad_learn_dro(cfg):
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
    
    
